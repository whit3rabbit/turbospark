---
uuid: "96040d84-192f-406a-8cd1-ab44c3a559ab"
title: "turbospark-tokenizer: stop sets and reasoning frames"
summary: "Harmony's stop set has three inverted-naming members. A structured decoder's initial channel must come from the rendered prompt, not the --reasoning flag, or it silently swallows or leaks the reply"
tags: ["crate", "tokenizer"]
source: "crates/tokenizer/CLAUDE.md"
created: "2026-09-04"
updated: "2026-09-04"
depends_on: ["a8a695d0-3b88-4c92-baff-736945ac0e94"]
---

## Why does a dialect's stop set or reasoning output look wrong?

- **Harmony (gpt-oss)** stops on `<|return|>` (a real answer) and
  `<|call|>` (a tool invocation), with `<|endoftext|>` as the base EOS.
  `<|end|>` is deliberately NOT a stop token, it only closes the system and
  user turns inside a rendered prompt. Dropping `<|call|>` from the stop
  set doesn't error. The model just generates straight past its own tool
  call, reading as a rambling model rather than a stop-set bug.
- **A Harmony tool call comes out of `finish()`, not `consume()`.**
  `<|call|>` is itself a stop token, so `run_raw_completion` breaks the
  loop before the decoder ever sees it. A caller driving only `consume()`
  silently drops every tool call: no error, no markup, just an empty turn.
- **A ChatML generation prompt opens the `<think>` frame itself** when
  thinking is on (it ends `...assistant\n<think>\n`). A decoder that always
  starts in `Channel::Visible` reports the whole scratchpad as the answer
  with zero bytes on the reasoning stream. The decoder's initial state has
  to come from scanning the rendered PROMPT (backwards, since tool
  instructions can contain balanced `<think></think>` pairs of their own),
  not from whether `--reasoning` was passed, because a template can enable
  thinking without prefilling the tag.
- **Muse Glimmer reasons on every turn by default.** Its template calls
  `render_reasoning()` unconditionally from the system message, so its
  decoder arm is built unconditionally like Harmony's rather than gated
  behind `--reasoning`.
- **Harmony and Muse Glimmer share `<|start|>` and `<|message|>` and
  nothing else.** `detect_dialect`'s probe order and marker count are both
  load-bearing: Harmony's arm requires `<|channel|>` as a third marker, or
  a Muse Glimmer checkpoint resolves to Harmony and then fails to load on
  a missing `<|startoftext|>`.

## Don't

- Don't assume dropping a stop token or misreading a channel marker gets
  caught by a test. The common failure mode here is fluent, coherent-
  looking WRONG output (a rambling model, an empty reply, reasoning leaked
  as the answer), not a crash.
- Don't add a new frame-based dialect without first asking "where does the
  rendered generation prompt leave the model," then checking whether ANY
  decoder arm exists for it at all. Muse Glimmer originally had none, and
  its `to=self` scratchpad printed as the reply on every single turn.
