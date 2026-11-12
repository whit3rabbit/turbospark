# Stage 5: Initial zoom-out report (OC-13)

Target: docs/designs/opencode2-parity/OC-13-engine-version-handshake.md (full file, 160 lines). Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed.

## Zoom-out (ZO)

ZO-1 Blocker. Lines 49-52, 68-71, 88-94, 109-110, 134-136. The FFI "pairing check" already exists: `TS_ABI_VERSION` in `turbospark.h:60`, `ABI_VERSION` in `crates/ffi/src/abi.rs`, `ts_abi_version()` in `crates/ffi/src/api/core.rs`, a Rust test `abi_version_matches_the_header`, and Swift `verifyABIOnce()` called from `TurboSparkServer.start`, `TurboSparkSession` and the tokenizer. The draft never mentions it and says the FFI is unguarded. Fix: delete Goal 5, CUJ 3, plan item 4 and Open Question 2, or reduce to "verify `verifyABIOnce` covers session open".

ZO-2 Major. Lines 19-37, 74-80. The attach path is asserted, not shown. `ServerDaemonLaunch.swift` only builds spawn args; `TurboSparkServer.swift` wraps the in-process server. The draft names no concrete skewed request shape and does not say the app ever attaches to a daemon it did not start (Non-Goal at lines 59-61 says it always launches its own). Fix: name the real attach or status call site (likely `TurboSparkDaemon.status()`) and one concrete skew failure, or cut scope to a version field plus display.

ZO-3 Major. Lines 41-46, 98-110. Three fields plus a major/minor predicate and a new module, with no owner for `protocolVersion` or the app minimum. Daemon and app ship from one repo, so `CARGO_PKG_VERSION` equality or minimum would do. Contradicts the "corner, not a flow" non-goal. Fix: expose engine version only, compare to the app's own, drop `protocolVersion` and `engineBuild` until a real skew case appears.

ZO-4 Major. Lines 125-133, 107-108. Plan item 2 names a nonexistent `routes.rs` (health is `/health` in `crates/server/src/lib.rs`, `handler::health`). Item 3 puts the probe in the in-process server wrapper; the existing seam is `TurboSparkDaemonStatus` and `TurboSparkDaemon.status()` in `Daemon.swift`. Fix: retarget items 1-3 to `handler::health` and `Daemon.swift`.

ZO-5 Nit. Lines 125-147. Eight files and a mutation check for what INDEX sizes as small. Fix: re-scope, then trim tests to one decode test and one comparison test.

ZO-6 Nit. Lines 3-9. Parity framing borrows OpenCode's separately versioned sidecar. The local reason is a Sparkle update leaving an old daemon running. Fix: reword motivation to that scenario.

## Inventive catalyst (IC)

IC-1 Blocker (standard-level call). Lines 20-22, 49-52, 108-110, 134-136. Same as ZO-1: the FFI half duplicates `verifyABIOnce` and `crates/ffi/src/abi.rs`. Fix: cite the existing tripwire; note a gap only if one is found.

IC-2 Major. Lines 19-26, 28-37, 74-80. The attach path is not shown to exist; `Daemon.swift` (`TurboSparkDaemonStatus`, `status()`) is the real surface and is never mentioned. CUJ 1 (foreign daemon) contradicts Non-Goal 3. Fix: ground the doc in `Daemon.swift` and state whether foreign-daemon attach is possible.

IC-3 Major. Lines 41-42, 98-101, 125-129. Route is hedged ("or equivalent"). `/health` is exempt from the API key, so adding version and build there is an information-disclosure choice the draft does not make. Fix: name the route and state whether version fields are served unauthenticated.

IC-4 Major. Lines 43-44, 63-67. Field and rule redundancy. `buildInfo()` already exposes `abiVersion` and crate version. Suggested contradiction elimination: one monotonic `protocolVersion` integer with a minimum; absent field (tolerant `decodeIfPresent` decode) means older daemon; engine version display-only. Liability: bump discipline, and absent conflates old with non-conforming. Fix: adopt one integer or the engine-version-only form.

IC-5 Major. Lines 56-61, 77-80. Refuse-and-ask-to-restart where the app can already relaunch its own daemon via `ServerDaemonLaunch.args`. Suggested: offer or perform restart through the existing path, refusal as fallback. Liabilities: killing a daemon serving other clients (needs consent or ownership check), restart loops (cap at one). Fix: state the restart policy.

IC-6 Nit. Lines 114-117, 140-141. 21-language localization is the main cost; "small" sizing is not credible as written. Fix: re-size after scope cuts.

IC-7 Nit. Lines 3-9, 100-105. Parity target is exact predicates but the draft adopts a minimum rule; rationale sits only in Open Questions. Fix: state in Goals why exact-match was rejected.

IC-8 Nit. Lines 149-156. Manual test needs an older daemon; no way to produce one. Fix: use a stubbed payload via the file 8 test.
