---
uuid: "1bd9d722-cd6e-4ab8-8775-10b2fb841f76"
title: "turbospark-window-fit"
summary: "Pure, stateless conversation-window turn dropping (fit_conversation_window). Turn 0 and the newest user turn are pinned and never dropped"
tags: ["crate", "window-fit"]
source: "crates/window-fit/CLAUDE.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## What does turbospark-window-fit do?

One function: `fit_conversation_window`. It drops the oldest eligible turns
from a conversation until the measured length fits caller-supplied bounds.
It's pure and stateless: no I/O, no tokenization, no state held between
calls. Length measurements come from the caller, not from this crate.

## Don't

- Don't expect the optional leading system/instruction turn (turn 0) or
  the newest user message turn to ever be dropped. Both are pinned
  unconditionally by the fitting rule, whatever the length bound.
- Don't feed it a token count from somewhere other than the tokenizer this
  conversation will actually be rendered with. It trusts the caller's
  length measurement completely and has no way to sanity-check it.
