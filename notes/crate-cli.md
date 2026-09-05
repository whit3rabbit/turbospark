---
uuid: "04ecf0c4-95ea-40d2-ba78-a721c138e58d"
title: "turbospark-cli"
summary: "Two binaries: turbospark-check (decode) and turbospark-model (catalog/download). Two binaries, no lib target, so tests can only drive the built binaries as black-box processes"
tags: ["crate", "cli"]
depends_on: ["b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e02"]
source: "crates/cli/CLAUDE.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## What does turbospark-cli do?

Two process entry points in one crate. `turbospark-check` parses `argv` via
`turbospark-invocation`, applies exit status and stream routing, and drives
GPU decode through `RealForwardRunner` on macOS. `turbospark-model` is the
catalog and download surface, backed by `turbospark-catalog`, and decides
nothing itself: it resolves rows and picks column widths, `turbospark-catalog`
reaches every verdict.

Three settings resolve ONCE per process, inside `open_session`, and cannot
change mid `--chat` session: power profile, expert-cache-slots, and
max-context. Each goes through the same shape: a pure `invocation::` enum
mapped onto a `runtime::` enum in exactly one place, because `invocation`
touches no OS state and `runtime` needs both the machine and the install.

## Don't

- Don't run only the greedy smoke test and call a decode/KV/head change
  verified. `argmax` is invariant under monotone distribution transforms, so
  a broken sampler distribution passes greedy every time. Always run sampled
  too.
- Don't diff a stdout md5 against a historic hash without normalizing the
  resolved-request block first. `--expert-cache-slots` and `--max-context`
  both default to `auto`, so the printed block reads `Auto` instead of a
  number even when the generated text is byte-identical.
- Don't look for `--speculative`'s policy logic in this crate. It moved to
  `runtime::speculation_policy` in 2026-08-21 when `turbospark-server` needed
  the same three decisions and this crate (two binaries, no lib target)
  couldn't expose a line of it. `open_session` only maps enums and calls it.
- Don't verify a speculation refusal message at this binary's sampled
  default (T=0.2). The SAMPLED refusal resolves before the drafter's, so
  you'll see "acceptance is exact only at temperature 0" instead of the
  message you meant to check. Use `--temperature 0`.
- Don't skip an empty text delta before it reaches `ChannelSplit`. Every
  Harmony frame token (`<|channel|>`, `<|message|>`, etc.) renders to an
  empty string, so skipping empties upstream means the reasoning/answer
  split never triggers and `gpt-oss` output prints as one run of
  markup-laced content with no error anywhere.
- Don't append a `--image` to the prompt. It PREPENDS to the last user turn
  on purpose, matching the reference template's `[image, text]` ordering.
  Appending shifts every mRoPE position and produces a different prompt for
  the same request.
