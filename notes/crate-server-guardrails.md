---
uuid: "8f3c2b1a-6d4e-4f9a-9c1b-2e5a7d8f3c01"
title: "turbospark-server: tool-call guardrails"
summary: "Guardrails are on by default. A request carrying tools is BUFFERED, not streamed, because a repair verdict needs the whole turn before it can be judged"
tags: ["crate", "server", "guardrails"]
source: "crates/server/CLAUDE.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## How do the tool-call guardrails work, and why do tool requests feel slow?

`src/guardrails.rs` wraps the `forge-guardrails` crate (default features
off): it rescues a tool call the `StructuredAssistantDecoder` couldn't
parse out of raw text, checks a parsed call's arguments against the
request's own schema, and re-asks ONCE with a nudge if either fails.
`--guardrails off` restores the pre-guardrails path exactly.

**The buffering is inherent, not a shortcut.** A verdict needs the whole
turn: a call worth rescuing is indistinguishable from prose until the turn
ends. `stream_response` forks: a tool-carrying request goes to
`buffered_stream_response`, everything else keeps the live `stream_blocking`
path byte for byte. The cost is time-to-first-token becoming
time-to-last-token for a tool turn (roughly 5-10s at 20-45 tok/s), accepted
since the alternative helps nobody: a proxy in front would buffer the same
traffic identically anyway.

## Don't

- Don't expect the retry to append tokens to the failed turn. It
  RE-RENDERS the whole prompt: a nudge has to arrive as a fresh `user` turn
  through the checkpoint's own template, because appended tokens would land
  inside whatever channel the model was last writing in and come out wrong
  on every dialect. `run_guarded` takes the whole `ChatCompletionRequest`
  for this reason.
- Don't return a rescued call without validating it. The first version did
  exactly that and it's backwards: a call recovered from unparseable markup
  is the one MOST likely to have arguments the model also got wrong.
  `inspect` rescues first, then validates, and a rescued call that fails
  validation gets re-asked rather than sent. The bug was caught by this
  crate's own test returning a call with empty arguments.
- Don't test a retry with `ScriptedChatModel` directly. `with_producer`
  replays the same scripted answer on every call, so a retry gets the first
  turn's response again. `tests/guardrails.rs`'s `TwoTurnModel` also
  overrides `produce_prefill` to a no-op, because a retry's prompt is
  longer (it carries the failed turn plus the nudge) and a producer sized
  for the first generation runs out mid-prefill on the second.
- Don't expect a second `--guardrails`-style flag for the retry budget. It's
  fixed at ONE retry, because the request queue is serial (one runner per
  process) and a second generation doubles the worst-case mutex hold.
- Don't add a per-request guardrails override. It's process-level like the
  rate cap, speculation, and `--api-key`, on the same reasoning plus one of
  its own: a per-request field would let any client opt its own traffic out
  of the repair the deployment chose.
- Don't assume this crate's default `rust-version` applies here.
  `forge-guardrails` needs 1.87 against the workspace's 1.82 floor, so
  `crates/server/Cargo.toml` overrides `rust-version` literally rather than
  inheriting it. This is the one crate in the workspace that does.

See [[crate-server]] for the rest of this crate.
