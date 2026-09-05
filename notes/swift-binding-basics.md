---
uuid: "50158806-656a-4a09-8bc1-39dc3801244b"
title: "Swift binding: package layout and build mechanics"
summary: "TurboSpark is a class with a serial queue, never an actor, because cancel() must run while generate() is in flight. make swift-lib must run once before any Swift build"
tags: ["swift", "binding"]
source: "swift/CLAUDE.md"
created: "2026-09-05"
updated: "2026-09-05"
---

## What is turbospark's Swift binding, and how do I build it?

`swift/TurboSpark` is a thin SwiftPM binding over the C ABI in `crates/ffi`
(async/await, `AsyncThrowingStream`, Codable wire types). `swift/TurboSparkApp`
is the SwiftUI app built on it. Both are macOS/Apple Silicon only and link
the engine IN PROCESS: no HTTP, no IPC, no server. Read
`crates/ffi/CLAUDE.md` first for any change crossing the boundary. That
crate's own gotchas and this file's are two halves of one contract.

`TurboSparkSession` is a CLASS with a private serial `queue`, deliberately
never an `actor`. An actor method can't run while another is in flight, so
`cancel()` on an actor would suspend behind the generation it's cancelling
and take effect only once the model finished on its own, meaning the Stop
button would appear to do nothing on exactly the long turns it exists for.
Everything except `cancel()` runs on the serial queue, which also
satisfies the C layer's one-generation-at-a-time contract. Do not
"modernize" this into an actor.

```sh
make swift-lib          # stages crates/ffi's .a + header into Sources/CTurboSpark
make swift-test         # binding suite, no model, ~1s, checks turbospark.h itself
make swift-test-real MODEL=~/models/gemma4.gturbo   # + end-to-end arm, minutes
```

## Don't

- Don't skip `make swift-lib` on a fresh checkout or worktree. Both the
  staticlib and header are gitignored copies staged from `crates/ffi`, and
  `swift build` fails on a missing header until it's run once.
- Don't expect a rebuilt `.a` under unchanged Swift sources to trigger a
  relink. SwiftPM doesn't treat the staticlib as a build input (it arrives
  through an unsafe `-L` linker flag), so the test target silently keeps
  linking the PREVIOUS library. `scripts/swift-lib.sh` ends with a
  `find ... -exec touch {} +` over both packages for exactly this reason,
  load-bearing and not to be tidied away.
- Don't be alarmed by `ld: warning: search path 'Sources/CTurboSpark' not
  found` on every app build. A SwiftPM `-L` flag resolves against the
  package being BUILT, not the one that declared it, so the link succeeds
  on the app's own copy anyway. Expected, not a regression.
- Don't touch `AppModel.swift` (the base file) for anything beyond stored
  properties and `init`. It drifted to 863 lines by growing accessors
  instead, which let four gating predicates each miss a term a sibling
  already carried. New behavior goes in a new `AppModel+<Domain>.swift`
  extension (`ls Sources/TurboSparkApp/State/AppModel+*.swift` for the
  current list).
- Don't assume a model's proposed tool call is parsed in more than one
  place. `Tools/Core/ToolCallParser` is the ONLY parser. `AppModel` and
  `SubagentRunner` carried duplicate copies until 2026-09-04, which is why
  three separate bugs turned out to be one fix applied to only one copy.
  Each caller keeps its own permission GUARD, but never its own parser.

See [[swift-binding-api-gotchas]] for the Catalog/SessionInfo/streaming API
quirks (Gotchas 5-9).
