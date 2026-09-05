---
uuid: "51774a9d-a722-4f32-a3f1-65a61c6e2c19"
title: "turbospark-server"
summary: "One in-process backend behind three chat-shaped wire formats (OpenAI, Anthropic, Ollama) plus legacy completions and responses. One runner, one mutex, requests queue"
tags: ["crate", "server"]
depends_on: ["8f3c2b1a-6d4e-4f9a-9c1b-2e5a7d8f3c01"]
source: "crates/server/CLAUDE.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## What does turbospark-server do?

An Axum HTTP server speaking four wire formats against one or more local
backends: OpenAI `/v1/chat/completions`, legacy `/v1/completions` and
`/v1/responses`, Anthropic `/v1/messages` (+ `count_tokens`), Ollama
`/api/{tags,version,show,chat,generate}`, plus `/v1/models` and `GET
/health`. Every generation endpoint supports non-streaming and SSE (Ollama:
NDJSON). Wire types come from the vendored `anyllm_translate` crate: an
Anthropic request translates into the OpenAI shape the shared
`handler::plan`/`handler::exec` core understands, then translates back.

`RealChatModel` (macOS only) holds exactly ONE `RealForwardRunner` behind a
`Mutex`, so concurrent requests to the same model queue rather than run in
parallel. `ScriptedChatModel` is the portable, canned-completion backend
every integration test drives, and the only one available off macOS.

## Don't

- Don't assume a knob you'd expect per-request (rate cap, speculation,
  guardrails, API key, prefix reuse) can vary by request. All resolve ONCE
  at `RealChatModel::open` and apply process-wide, since one runner exists
  per process and a power/safety policy is a property of the machine.
- Don't encode a chat-templated prompt with `add_bos: true`. The template
  already emits the literal `<bos>`, so chat paths call
  `encode(&prompt, false)`. Only `/v1/completions` (no template) passes
  `true`.
- Don't assume `top_k: 0` is a safe default. `ShapingConfig::new` rejects
  `top_p < 1.0` with `temperature > 0` and `top_k == 0`, which would 400
  every standard OpenAI request setting only `top_p`. The default is 64.
- Don't look for a `seed` field on the OpenAI request struct. It has none.
  `anyllm_translate` sweeps unmapped fields (`seed`, `logprobs`, `n`) into a
  `#[serde(flatten)] extra` map, and `build_config` reads it back via
  `request.extra.get("seed")`. Drop that lookup and every seeded request
  silently becomes unseeded, with no error anywhere.
- Don't treat Anthropic SSE as OpenAI SSE with different JSON. Clients
  dispatch on the `event:` line (OpenAI's stream never sets one), the
  stream ends at `message_stop` with no `[DONE]`, and `StreamingTranslator`
  is a state machine over the exact OpenAI chunk order (role, content,
  finish). Out-of-order chunks malform the Anthropic stream without
  erroring.
- Don't reach for `anyllm_translate`'s `middleware` feature here. It wants
  a `backend_url` and forwards over `reqwest` (this backend is in-process)
  and pulls in axum 0.8 against this crate's 0.7. Call
  `translate_request`/`translate_response`/`new_stream_translator` directly.

## Also here

Tool-call guardrails (request buffering) have their own page:
[[crate-server-guardrails]]. Also in this crate: a `ModelRegistry` for
multi-model routing, `ServerEvent`/`ServerObserver` reporting,
client-disconnect cancellation, and `--session-slots` KV-prefix reuse
across interleaved conversations, each documented in its own numbered
gotcha in `crates/server/CLAUDE.md`.
