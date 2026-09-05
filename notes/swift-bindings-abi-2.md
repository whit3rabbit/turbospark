---
uuid: "adde586f-5363-4d77-b5bb-e8f89be9494b"
title: "Swift/C ABI: session defaults that silently change behavior"
summary: "GenerateOptions.temperature defaults to 0.2, so speculation never engages unless a caller explicitly sends temperature 0. The fallback is silent by design"
tags: ["swift", "ffi", "abi"]
depends_on: ["cdc6f0dd-077e-49bd-9ed3-d22415ae0c4d"]
source: "docs/SWIFT_BINDINGS.md"
created: "2026-09-05"
updated: "2026-09-05"
---

## Which OpenOptions/GenerateOptions defaults will surprise me?

`OpenOptions` and `GenerateOptions` default to automatic, which is right
for a GUI, but several defaults are worth understanding rather than
accepting: getting them wrong produces silently different behavior, not an
error.

## Don't

- Don't assume speculative decoding engages with default options.
  Acceptance requires temperature 0 exactly (`argmax(target) == proposal`),
  but `GenerateOptions.temperature` defaults to 0.2. An app that never
  explicitly sends temperature 0 never speculates, and the fallback to the
  sequential loop is silent BY DESIGN (a per-turn warning would fire on the
  normal case). Read `session.info.speculation` once, at the session
  level, to know what's actually possible.
- Don't leave `powerProfile: nil` while benchmarking. `nil` asks the OS,
  and Low Power Mode silently selects `.efficiency`, which becomes part of
  your measured result rather than a setting you chose.
- Don't assume `expertCacheSlots: .auto` can make a session slower, and
  don't assume it's independent of `loadGuard`. Auto climbs from the
  shipped default of 16 toward the largest count that fits available
  headroom, floored at 16, so it can only add throughput. Every PUBLISHED
  footprint figure for this engine was measured under `.relaxed`, not
  under whatever `.auto` resolves to on your machine.
- Don't call `TurboSparkCatalog.recommend` with a different `loadGuard`
  than the one you'll open with. They share one memory budget by
  construction. Recommending under `.relaxed` while opening under
  `.strict` promises a fit the loader then refuses, in the one place a
  user can't see the two disagree.
- Don't expect prefix reuse or chunked prefill to be configurable, or to
  always apply. Both run unconditionally with no flag, and both fall back
  to the plain path silently rather than erroring: prefix reuse on a
  session's first turn, a diverged render, or a family it can't help.
  Chunked prefill when the turn carries an image. Read
  `GenerationResult.reusedPrefixTokens` to see what actually happened.
- Don't read a non-nil `session.info.speculation.reason` as evidence
  something is broken. It's set whenever a caller might reasonably expect
  speculation and it isn't active, including cases where `.auto`
  deliberately declined a drafter it FOUND (the checkpoint's own MTP head
  measured faster than the DFlash2 alternative it passed over).
- Don't reach for these bindings expecting tool calling. It's the one
  thing `crates/server` has that this binding doesn't.
  `StructuredAssistantDecoder` is constructed here with an empty tool
  allowlist, so a Harmony `commentary` body arrives labeled as reasoning
  instead of as a tool call.
