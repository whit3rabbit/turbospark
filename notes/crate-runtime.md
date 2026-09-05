---
uuid: "5df86045-4367-4c05-b50d-2fcc3173c2fd"
title: "turbospark-runtime"
summary: "The decode engine (macOS only). LogitProducer::produce returns raw logits never probabilities, and family selection keys on ArchConfig.family, never tensor naming"
tags: ["crate", "runtime"]
depends_on: ["e2d4f6a8-1c3b-4e5d-9f7a-8b6c4d2e0a19", "f3e5a7c9-2d4b-4f6e-8a9c-1b7d5e3f2c0a"]
source: "crates/runtime/CLAUDE.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## What does turbospark-runtime do?

It owns the raw-completion generation loop (`raw_completion.rs`), the
`LogitProducer` trait every decode backend implements, and
`RealForwardRunner`, the real Metal forward-pass engine (macOS only, gated
per module). `#![forbid(unsafe_code)]` is enforced. Each model family gets
its own decode flow under `families/<name>/` (gemma4, gptoss, llama,
museglimmer, qwen, synthetic), selected once at open by `ArchConfig.family`
and never re-derived from tensor names, because several families share
tensor naming conventions.

## Don't

- Don't return probabilities from `LogitProducer::produce`. It must return
  raw, unnormalized logits. `selection::select` softmaxes internally, and
  returning probabilities produces `softmax(softmax(z))`, which destroys
  temperature reweighting while argmax and top-k still look correct. This
  bit the real Gemma 4 head once.
- Don't assume `produce_prefill` and `produce` can skip the same work.
  `produce_prefill` may skip the output head (only the last prompt token's
  logits are read), but any override must still advance every other
  per-token side effect: KV cache, position, and the command buffer
  commit-and-wait. The wait is what stops the next token overwriting
  scratch the GPU is still reading.
- Don't identify a family from tensor names. Gemma 4 and Qwen 3.6 both
  carry `language_model.model.embed_tokens.weight`. Flow selection reads
  `ArchConfig.family` exactly once, at open.
- Don't reset only the KV cache on a linear-attention layer. A GDN layer
  keeps its whole history in `GdnStateManager`'s delta-rule state and conv
  tail, which the KV cache holds nothing for. Resetting one without the
  other leaks the previous generation's context invisibly (output stays
  finite and deterministic, just wrong).
- Don't add a `&mut self` call between binding `let real = self.real.as_ref()`
  and using it, in any `families/*/mod.rs` per-token function. It's E0502.
  Re-bind `real` immediately after the `&mut self` call, which is the
  existing idiom throughout these files.
- Don't treat moving a family's decode flow between files as risk-free
  because "nothing changed." A pure file split once inserted a Gemma-style
  sandwich norm into the Qwen flow (Qwen has none), producing token soup
  with the whole workspace suite green, because the synthetic Qwen fixture
  has untrained weights and cannot distinguish meaningful garbage from
  broken garbage. Only that family's quality gate caught it. A code MOVE
  earns the same real-model gates as a code CHANGE, plus the family's
  quality gate.

See [[crate-runtime-2]] for the env-var dispatch seams and expert-cache
correctness constraints, and [[crate-runtime-3]] for speculative decoding.
