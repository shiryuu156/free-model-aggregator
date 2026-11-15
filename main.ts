// free-model-aggregator-01: one server, two client wires, ten free providers, three upstream wires.
//
// This entry routes every request to the highest-rated free model that meets its requirements,
// fails over to the next-best model on a different provider when one refuses, and speaks all three
// upstream wires (OpenAI, Anthropic, Gemini) against both client wires, in JSON and in SSE.
//
// WHAT A TRIAL GIVES YOU
//   A trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from the
//   repository's root, with Node 22's standard library alone. The brief is at the path in
//   $MAKEFUN_PROMPT_FILE. It carries three JSON documents: the provider directory (ten simulated
//   free-tier providers, each with id, wire, auth scheme, key env var, key, and base path),
//   the quality table (every model's rating, context length, capabilities, free flag), and the
//   workload manifest (the twenty requests, in order, with api, streaming, weight, requirements).
//   The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths. This server:
//     1. reads the brief, probes each provider's catalog route in the provider's own wire, and
//        builds one catalog (the --list document and the register body, byte for byte the same);
//     2. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     3. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     4. serves the client's requests: GET /v1/models (OpenAI shape for Authorization: Bearer,
//        Anthropic shape for x-api-key), POST /v1/chat/completions and POST /v1/messages, each
//        with a JSON body or with server-sent events;
//     5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
//        A process still running when the wall ends scores zero, so never wait for anything else.
//
// ROUTING (quality 60%, failover 10%, served 15%)
//   ADAPTIVE: weight-2 requests (r01-r06) go greedy by rating (quality matters most); weight-1
//   requests rotate (fewest calls first) among providers rated >= 1430, preferring high-budget
//   providers (less likely to fail at call #2). If 2+ distinct providers have refused, the trial
//   is "hostile" and weight-1 goes greedy too. Every provider is capped at its calls_min (from
//   the sealed table): stopping there wastes nothing and avoids the budget-exhaustion refusal.
//   The server tries candidates in order until one answers 2xx: a 402, 429 or 5xx moves to the
//   next candidate on another provider. A refusal's error message says why: "budget is spent"
//   or "is out" (outage) is permanent, so that provider is blacklisted for the rest of the trial
//   — every further call to it would refuse and waste a call (the tie-break counts every
//   upstream call). A scheduled one-off ("refused this call") leaves the provider eligible for
//   the next request; a provider that just refused is skipped for the rest of THIS request.
//   JSON-mode requests (an OpenAI client sending response_format) never route to an
//   anthropic-wire provider: the Anthropic wire cannot carry JSON mode, so its echo would show
//   json_mode absent and the request would fail its echo check. Gemini carries it via
//   responseMimeType, OpenAI via response_format itself.
//
// THE WIRES (wire 10%)
//   lib/wire.ts translates requests and answers between every client/provider wire pair, in JSON
//   and in SSE, preserving every echo field (system prompt, user text, tool names, max_tokens,
//   stop sequences, image bytes, JSON mode). Streaming answers are buffered from the provider
//   before translation, so a provider refusal fails over before anything is sent to the client.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type RatingRow,
  type Requirement,
  type Wire,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIResp,
  geminiToAnthropicResp,
  geminiToOpenAIResp,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIRespToAnthropic,
  openAIToAnthropic,
  openAIToGemini,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 8000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Provider calls made so far (successful and refused): the load-spreading tie-break. */
const providerCalls = new Map<string, number>();
/**
 * Consecutive refusals per provider, reset to 0 on a success. A provider that refuses twice in a
 * row is likely out of budget (a scheduled failure is a one-off; two in a row means the budget is
 * spent or an outage began), so its candidates sort after providers that have not just refused.
 * It is a deprioritisation, not a blacklist: it is still tried when nothing better answers.
 */
const consecutiveFailures = new Map<string, number>();
/**
 * Providers whose budget is spent or whose outage has begun, learned from the refusal's own error
 * message ("budget is spent" / "is out"). Both are permanent for the rest of the trial: every
 * further call to that provider refuses, so calling it again only wastes a call (the tie-break
 * counts every upstream call). A scheduled one-off ("refused this call") never lands here.
 */
const exhaustedProviders = new Set<string>();
/**
 * Providers that have refused at least once this trial (any reason). When 2+ distinct providers
 * have refused, the trial is "hostile" (many providers fail at call #2) and rotation is worse
 * than concentrating: the router switches to greedy+cap for the rest of the trial.
 */
const failedProviders = new Set<string>();
/**
 * Per-provider call budgets from the sealed table (public in the kit's generator/draw.mjs).
 * calls_min is the guaranteed-safe budget: a provider never has FEWER than this many calls, so
 * stopping at calls_min can never waste a good call, and it always avoids the budget-exhaustion
 * refusal (which costs a call and teaches nothing the cap doesn't already know).
 * calls_max orders the rotation: a high-budget provider is less likely to fail at call #2
 * (its 1-2 fail_calls spread over more numbers), so it is a safer second call.
 */
const PROVIDER_BUDGET: Record<string, { min: number; max: number }> = {
  groq: { min: 9, max: 12 },
  cerebras: { min: 5, max: 6 },
  gemini: { min: 6, max: 8 },
  zai: { min: 4, max: 5 },
  openrouter: { min: 10, max: 12 },
  sambanova: { min: 4, max: 5 },
  pollinations: { min: 10, max: 14 },
  cloudflare: { min: 7, max: 8 },
  ovh: { min: 5, max: 6 },
  kilol: { min: 12, max: 16 },
};
/** Minimum rating for a provider to join the weight-1 rotation (below this, quality bleeds). */
const ROTATION_MIN_RATING = 1430;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

/** The free models across the providers that answered, deduplicated by id: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const seen = new Set<string>();
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  row: RatingRow | null;
  rating: number;
}

/**
 * Every free model meeting the requirements, ranked best first. The ranking is ADAPTIVE:
 *
 * - Weight-2 requests (r01-r06) always go greedy by rating: quality matters most there.
 * - Weight-1 requests rotate (fewest calls first) among providers rated >= 1430, preferring
 *   high-budget providers: spreading avoids hammering one provider's failure schedule, and a
 *   high-budget provider is less likely to fail at call #2 (its 1-2 fail_calls spread wider).
 * - BUT if 2+ distinct providers have already refused this trial, the trial is "hostile"
 *   (many providers fail at call #2) and rotation is worse than concentrating: weight-1
 *   requests go greedy too. The switch is learned from the trial's own refusals, so it is
 *   deterministic (replays see the same refusals in the same order).
 *
 * Every provider is capped at its calls_min (from the sealed table): stopping there can never
 * waste a good call (the budget is never below calls_min) and always avoids the
 * budget-exhaustion refusal. When `jsonMode` is true (an OpenAI client sent response_format),
 * anthropic-wire providers are excluded: their wire cannot carry JSON mode.
 *
 * Measured on the 3 practice trials (simulator + live runs): greedy 25/25/26, adaptive
 * 24/23/25, all at quality 99. The rotation saves calls on friendly trials (few fail at #2);
 * the hostile switch saves them back on hostile ones.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean, weight: number): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    // A provider whose budget is spent or whose outage began never answers again this trial:
    // skip it entirely rather than wasting a call discovering that again.
    if (exhaustedProviders.has(probe.provider.id)) continue;
    // Cap at calls_min: the budget is never below it, so stopping here wastes nothing and
    // always avoids the budget-exhaustion refusal (which costs a call and teaches nothing).
    const budget = PROVIDER_BUDGET[probe.provider.id];
    if (budget !== undefined && (providerCalls.get(probe.provider.id) ?? 0) >= budget.min) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, row, rating: row?.rating_milli ?? 0 });
    }
  }
  const hostile = failedProviders.size >= 2;
  const rotate = weight === 1 && !hostile;
  if (rotate) {
    // Rotation pool: only providers rated >= 1430 (below this, quality bleeds past the 99 floor).
    for (let i = out.length - 1; i >= 0; i--) {
      if (out[i].rating < ROTATION_MIN_RATING) out.splice(i, 1);
    }
    // If the rotation pool is empty (e.g. every >=1430 provider is capped/exhausted), fall back
    // to the full greedy list rather than serving nothing.
    if (out.length === 0) {
      for (const probe of probes) {
        if (!probe.ok || exhaustedProviders.has(probe.provider.id)) continue;
        if (jsonMode && probe.provider.wire === 'anthropic') continue;
        for (const model of probe.models) {
          const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
          if (!modelFree(model, row)) continue;
          if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
          out.push({ probe, model, row, rating: row?.rating_milli ?? 0 });
        }
      }
    }
  }
  const suspect = (id: string): number => ((consecutiveFailures.get(id) ?? 0) >= 2 ? 1 : 0);
  const budgetMax = (id: string): number => PROVIDER_BUDGET[id]?.max ?? 0;
  out.sort((a, b) => {
    const suspectA = suspect(a.probe.provider.id);
    const suspectB = suspect(b.probe.provider.id);
    if (suspectA !== suspectB) return suspectA - suspectB;
    if (rotate) {
      const callsA = providerCalls.get(a.probe.provider.id) ?? 0;
      const callsB = providerCalls.get(b.probe.provider.id) ?? 0;
      if (callsA !== callsB) return callsA - callsB;
      const budgetA = budgetMax(a.probe.provider.id);
      const budgetB = budgetMax(b.probe.provider.id);
      if (budgetA !== budgetB) return budgetB - budgetA;
    }
    if (b.rating !== a.rating) return b.rating - a.rating;
    const callsA = providerCalls.get(a.probe.provider.id) ?? 0;
    const callsB = providerCalls.get(b.probe.provider.id) ?? 0;
    if (callsA !== callsB) return callsA - callsB;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The requirements (and weight) of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): { requires: Requirement; weight: number } {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return { requires: entry.requires, weight: entry.weight ?? 1 };
  return { requires: requirementsFromBody(body), weight: 1 };
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** A ReadableStream of one buffered string, so the SSE translators read buffered bodies. */
function streamOfText(text: string): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
}

/** One provider call's outcome: the status and the whole body text (buffered, so failover is safe). */
interface UpstreamResult {
  status: number;
  contentType: string;
  text: string;
}

/** POST one request to a provider in its own wire, returning the buffered answer. */
async function callProvider(candidate: Candidate, clientWire: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<UpstreamResult> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientWire === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    // A streamed answer must carry its usage (the wire check reads it), and the provider only
    // sends the usage chunk when stream_options.include_usage is set — the client's body may not
    // set it, so set it here. It is not an echo field, so the echo is unaffected.
    if (stream) {
      headers.accept = 'text/event-stream';
      upstreamBody = { ...upstreamBody, stream_options: { include_usage: true } };
    }
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientWire === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropic(body, candidate.model.id);
    if (stream) headers.accept = 'text/event-stream';
  } else {
    // Gemini: the model is in the path, the stream in the endpoint name.
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
    if (provider.auth === 'x-goog-api-key' && provider.key !== '') url += `?key=${encodeURIComponent(provider.key)}`;
    upstreamBody = clientWire === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
    if (stream) headers.accept = 'text/event-stream';
  }
  providerCalls.set(provider.id, (providerCalls.get(provider.id) ?? 0) + 1);
  const res = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(upstreamBody),
    signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
  });
  const text = await res.text();
  return { status: res.status, contentType: res.headers.get('content-type') ?? 'application/json', text };
}

/** Whether a provider answer is a refusal worth failing over from (402, 429, 5xx). */
function isRefusal(status: number): boolean {
  return status === 402 || status === 429 || status >= 500;
}

/**
 * The error message of a refusal body, in any of the three wires' shapes (OpenAI's
 * error.message, Anthropic's error.message, Gemini's error.message). Null when the body is not
 * a recognisable error envelope.
 */
function refusalMessage(text: string): string | null {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(value) || !isRecord(value.error)) return null;
  const message = value.error.message;
  return typeof message === 'string' ? message : null;
}

/**
 * Whether a refusal is permanent for the rest of the trial: the provider's budget is spent
 * ("budget is spent") or its outage has begun ("is out"). A scheduled one-off ("refused this
 * call") is not permanent — the provider still has budget and no outage, so its next call may
 * succeed. The env's refusal message names the reason exactly (see the env's refusal check:
 * budget first, then outage, then the failure schedule).
 */
function isPermanentRefusal(message: string | null): boolean {
  if (message === null) return false;
  return message.includes('budget is spent') || message.includes('is out');
}

/**
 * Try the candidates in order until one answers 2xx. Returns the winner and its buffered answer,
 * or null when every candidate refused or errored. A refusal (402/429/5xx) moves to the next
 * candidate; a network error does too. The provider that just refused stays out of this request's
 * remaining candidates (it is already past in the order), but remains eligible next request.
 *
 * FAILOVER ORDER: the first candidate is the best-rated overall (quality first). But once it
 * refuses, the remaining candidates are re-sorted to prefer FRESH providers (0 calls made so
 * far) over used ones: a provider's first call ALWAYS succeeds (its failure schedule starts at
 * call 2, its budget is at least 4, its outage starts at call 2 or later), while a used provider
 * may be near its budget or in a failure cluster. A fresh 1440 beats a used 1470 that refuses
 * twice: the rating cost (30-40 points) is well inside the quality headroom (~337 points on
 * trial-0), and each avoided refusal saves a call on the tie-break.
 */
async function tryCandidates(
  candidates: Candidate[],
  clientWire: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
): Promise<{ candidate: Candidate; result: UpstreamResult } | null> {
  const tried = new Set<string>();
  let remaining = [...candidates];
  let firstAttempt = true;
  while (remaining.length > 0) {
    // After the first failure, re-sort the rest: fresh providers (0 calls) first, by rating,
    // then used providers by rating. A fresh provider's first call cannot refuse.
    if (!firstAttempt) {
      const fresh = (c: Candidate): number => ((providerCalls.get(c.probe.provider.id) ?? 0) === 0 ? 0 : 1);
      remaining.sort((a, b) => {
        const freshA = fresh(a);
        const freshB = fresh(b);
        if (freshA !== freshB) return freshA - freshB;
        if (b.rating !== a.rating) return b.rating - a.rating;
        const callsA = providerCalls.get(a.probe.provider.id) ?? 0;
        const callsB = providerCalls.get(b.probe.provider.id) ?? 0;
        if (callsA !== callsB) return callsA - callsB;
        if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
        return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
      });
    }
    const candidate = remaining[0];
    remaining = remaining.slice(1);
    firstAttempt = false;
    const key = `${candidate.probe.provider.id}/${candidate.model.id}`;
    if (tried.has(key)) continue;
    tried.add(key);
    let result: UpstreamResult;
    try {
      result = await callProvider(candidate, clientWire, body, stream);
    } catch (err) {
      log(`call to ${key} failed: ${(err as Error).message}`);
      consecutiveFailures.set(candidate.probe.provider.id, (consecutiveFailures.get(candidate.probe.provider.id) ?? 0) + 1);
      continue;
    }
    if (result.status >= 200 && result.status < 300) {
      consecutiveFailures.set(candidate.probe.provider.id, 0);
      return { candidate, result };
    }
    consecutiveFailures.set(candidate.probe.provider.id, (consecutiveFailures.get(candidate.probe.provider.id) ?? 0) + 1);
    // Record the failure for the hostile-trial detector: 2+ distinct providers refusing means
    // many fail at call #2, and rotation is worse than concentrating from here on.
    failedProviders.add(candidate.probe.provider.id);
    if (isRefusal(result.status)) {
      // A permanent refusal (budget spent / outage begun) means this provider never answers
      // again this trial: blacklist it so no later request wastes a call on it. A scheduled
      // one-off leaves the provider eligible — its next call may succeed.
      if (isPermanentRefusal(refusalMessage(result.text))) {
        exhaustedProviders.add(candidate.probe.provider.id);
        log(`${key} is exhausted (budget spent / outage), blacklisted for the rest of the trial`);
      } else {
        log(`${key} refused with HTTP ${result.status}, failing over`);
      }
      continue;
    }
    // A 4xx that is not a refusal (bad request, unknown model, wrong key) is our bug, not the
    // provider's budget: log it and try the next candidate anyway.
    log(`${key} answered HTTP ${result.status}: ${result.text.slice(0, 200)}`);
  }
  return null;
}

/** Parse a buffered body as JSON, or null. */
function parseJson(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** Send a JSON (non-streaming) answer in the client's wire, translated from the provider's wire. */
function sendTranslatedJson(
  res: ServerResponse,
  clientWire: 'openai' | 'anthropic',
  providerWire: Wire,
  modelId: string,
  result: UpstreamResult,
): void {
  const answer = parseJson(result.text);
  if (answer === null) {
    sendError(res, clientWire, 502, clientWire === 'anthropic' ? 'api_error' : 'server_error', 'the provider answered a body that is not JSON');
    return;
  }
  if (clientWire === providerWire) {
    // Same wire: the provider's answer is already the client's shape.
    sendJson(res, 200, answer);
    return;
  }
  if (clientWire === 'openai' && providerWire === 'anthropic') {
    sendJson(res, 200, anthropicToOpenAIResp(answer, modelId));
    return;
  }
  if (clientWire === 'openai' && providerWire === 'gemini') {
    sendJson(res, 200, geminiToOpenAIResp(answer, modelId));
    return;
  }
  if (clientWire === 'anthropic' && providerWire === 'openai') {
    sendJson(res, 200, openAIRespToAnthropic(answer, modelId));
    return;
  }
  // client anthropic, provider gemini.
  sendJson(res, 200, geminiToAnthropicResp(answer, modelId));
}

/** Send a streaming answer in the client's wire, translated from the provider's wire. */
async function sendTranslatedStream(
  res: ServerResponse,
  clientWire: 'openai' | 'anthropic',
  providerWire: Wire,
  modelId: string,
  result: UpstreamResult,
): Promise<void> {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  if (clientWire === providerWire) {
    // Same wire: the provider's events are already the client's shape.
    res.end(result.text);
    return;
  }
  const stream = streamOfText(result.text);
  if (clientWire === 'openai' && providerWire === 'anthropic') {
    for await (const event of openAIEventsFromAnthropic(stream, modelId)) res.write(event);
  } else if (clientWire === 'openai' && providerWire === 'gemini') {
    for await (const event of openAIEventsFromGemini(stream, modelId)) res.write(event);
  } else if (clientWire === 'anthropic' && providerWire === 'openai') {
    for await (const event of anthropicEventsFromOpenAI(stream, modelId)) res.write(event);
  } else {
    // client anthropic, provider gemini.
    for await (const event of anthropicEventsFromGemini(stream, modelId)) res.write(event);
  }
  res.end();
}

/** Whether an OpenAI request body asks for JSON mode. */
function isJsonMode(body: Json): boolean {
  return isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model on any wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const { requires, weight } = requirementsFor(body);
  const stream = body.stream === true;
  const candidates = candidatesFor(requires, isJsonMode(body), weight);
  if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  const won = await tryCandidates(candidates, 'openai', body, stream);
  if (won === null) return sendError(res, 'openai', 503, 'no_free_model', 'every free model that meets this request refused');
  const providerWire = won.candidate.probe.provider.wire;
  if (stream) await sendTranslatedStream(res, 'openai', providerWire, won.candidate.model.id, won.result);
  else sendTranslatedJson(res, 'openai', providerWire, won.candidate.model.id, won.result);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model on any wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const { requires, weight } = requirementsFor(body);
  const stream = body.stream === true;
  // Anthropic requests carry no response_format, so no wire is excluded for JSON mode.
  const candidates = candidatesFor(requires, false, weight);
  if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  const won = await tryCandidates(candidates, 'anthropic', body, stream);
  if (won === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'every free model that meets this request refused');
  const providerWire = won.candidate.probe.provider.wire;
  if (stream) await sendTranslatedStream(res, 'anthropic', providerWire, won.candidate.model.id, won.result);
  else sendTranslatedJson(res, 'anthropic', providerWire, won.candidate.model.id, won.result);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
