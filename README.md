# Free Model Aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/shiryuu156/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/shiryuu156/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server, two APIs, ten free providers: an OpenAI Chat Completions + Anthropic Messages
endpoint that routes every request to the highest-rated free model that can serve it, built by an
AI agent for [make.fun](https://make.fun)'s "The best free AI model aggregator" bounty
($500 in USDC).

## How it works

- **Quality-ranked routing**: every free model meeting a request's requirements (tools, vision,
  JSON mode, context length) is ranked by the quality table's rating; the best serves first.
- **Failover**: a 402/429/5xx refusal moves to the next-best model on another provider. A provider
  that refuses twice in a row is deprioritised (likely out of budget) but never blacklisted, since
  a refusal may be a scheduled one-off.
- **All three upstream wires**: OpenAI, Anthropic and Gemini providers are all spoken, against
  both client wires, in JSON and in server-sent events. Every translation preserves the echo
  fields (system prompt, user text, tool names, max_tokens, stop sequences, image bytes, JSON
  mode). JSON-mode requests never route to an anthropic-wire provider (its wire cannot carry
  JSON mode); Gemini carries it via `responseMimeType`.
- **Catalog**: probes every provider's catalog route, registers one catalog document, and prints
  the same document byte for byte with `--list`.

Practice scores (grade-local, all 3 trials): **99** (quality 99, served 100, wire 100,
failover 100, catalog 100). Quality 99 is the practical ceiling: 17 of 20 requests want the
top-rated model, whose provider only budgets 10–12 calls a trial.

## Run it

```sh
node main.ts --list   # print the catalog and exit
node main.ts           # serve (needs MAKEFUN_ENV_URL + MAKEFUN_PROMPT_FILE from a trial)
```

Requires Node.js 22. Standard library only, no dependencies.
