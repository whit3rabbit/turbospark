# OC-13: Engine and daemon version handshake

Finding: Section 8, "Typed protocol contract, versioned SDK, and
in-process SDK host - Missing in Swift" in
[OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md).
Parity targets: OpenCode's
[service-version predicates](https://github.com/anomalyco/opencode/blob/dev/packages/client/src/service-version.ts)
and
[protocol contract](https://github.com/anomalyco/opencode/blob/dev/packages/protocol/src/client.ts).

* Author: whit3rabbit
* Status: Partial (daemon version stamp, predicate, and menu bar warning implemented; HTTP health fields and Rust-side predicate deferred)
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Add a version and capability handshake between the Swift app and the
serving surfaces it can attach to: the background daemon
([ServerDaemonLaunch.swift](swift/TurboSparkApp/Sources/TurboSparkApp/Server/ServerDaemonLaunch.swift),
[TurboSparkServer.swift](swift/TurboSpark/Sources/TurboSpark/TurboSparkServer.swift))
and any future attach path. OpenCode enforces exact version predicates
at sidecar discovery; TurboSpark's in-process FFI is compile-time
coupled (same build, no skew possible), so the handshake only guards
the surfaces that can skew: the out-of-process daemon.

## The Problem

The FFI contract is a generated C header plus hand-written wrappers;
capability questions are answered at runtime by the engine, but the
daemon attach path has no version check at all. A user-started daemon
from an older install can serve an app that expects newer request
shapes, and the failure appears as confusing request errors instead of
one clear refusal. Recent work already stamps model formats (the
gturbo manifest minor check and producer block), but the serving
surface itself is unguarded.

## Goals

* The daemon's existing status/health surface gains explicit fields:
  `protocolVersion`, `engineVersion`, `engineBuild`.
* The Swift attach path probes before use and enforces a compatibility
  rule: matching major, daemon minor at or above the app's minimum.
* Mismatch is one refusal with a localized message naming both
  versions and the fix (restart the daemon from this app).
* The Server pane shows the daemon's version when attached and a
  warning state on mismatch with a Restart action.
* The FFI path gains a defensive pairing check: the generated header
  and the Rust FFI carry a contract revision constant, asserted at
  session open, so a stale linked library cannot pass silently. This
  is a tripwire, not a protocol.

## Non-Goals

* No schema-shared IDL or generated Swift client (the C header stays
  the contract; this draft adds runtime guards, not codegen).
* No auto-download or self-update of the daemon binary.
* No breaking-change negotiation beyond the major/minor rule; the
  app always launches a compatible daemon itself, so mismatch is a
  corner, not a flow.

## Open Questions

* **Compatibility rule**: exact-minimum (daemon minor >= app minimum,
  lean) versus exact-match. Exact-match turns every daemon update into
  a hard failure; the minimum rule tolerates a newer daemon.
* **Contract revision storage**: a constant in the generated header
  versus a header comment convention. Lean: a named constant emitted by
  the generator, so drift is a compile error, not a review catch.

## Critical User Journeys

### CUJ 1: stale user-started daemon

* **Given**: a daemon started earlier from an older install; the app
  updated.
* **When**: the app attaches.
* **Then**: attach refuses with both versions named and a Restart
  action that launches the app's own daemon.

### CUJ 2: healthy path

* **Given**: matching versions.
* **When**: the app attaches.
* **Then**: attach proceeds; the Server pane shows the version.

### CUJ 3: stale linked library

* **Given**: a dev build where the Swift wrapper and the Rust FFI
  drifted.
* **When**: a session opens.
* **Then**: a fatal, explicit contract-revision assertion fires with
  both revisions in the message, pointing at the build mismatch.

## Technical Plan (Architecture)

```text
App attach path
  1 GET status/health -> {protocolVersion, engineVersion, engineBuild}
  2 compatibility check (major equal, minor >= minimum)
     | ok                    | mismatch
     v                       v
  proceed + show version  refuse + localized message + Restart action
```

The daemon side serves the fields from its existing status handler;
the probe is one request on the attach path. The FFI revision is a
pair of constants (header, Rust) generated together and compared at
`ts_session` creation.

## Non-Functional Requirements

* **Performance**: one probe request on attach; the FFI check is a
  constant compare.
* **Localization**: refusal, warning, and restart copy in 21 languages
  in the same change.
* **Diagnostics**: refusal and mismatch log lines carry both versions
  for support reproduction.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `crates/server/src/version.rs` (Create)
   * **Rationale**: the version payload builder and the compatibility
     predicate, pure and tested.
2. `crates/server/src/routes.rs` or equivalent handler file (Modify)
   * **Rationale**: expose the fields on the existing status/health
     route.
3. `swift/TurboSpark/Sources/TurboSpark/TurboSparkServer.swift` (Modify)
   * **Rationale**: the probe, the predicate application, and the
     typed refusal.
4. `crates/ffi/` generator and `swift/TurboSpark/Sources/TurboSpark/` header consumer (Modify)
   * **Rationale**: the contract revision constant pair and the assert
     at session open.
5. `swift/TurboSparkApp/Sources/TurboSparkApp/Server/ServerPaneView.swift` (Modify)
   * **Rationale**: version display, mismatch warning state, Restart
     action.
6. `swift/TurboSparkApp/Localization/Localizable.xcstrings` (Modify)
   * **Rationale**: all new strings, 21 languages, same change.
7. `crates/server/tests/version_handshake.rs` (Create)
   * **Rationale**: predicate matrix (major/minor combinations).
     Mutation check: drop the major-equality rule and assert its test
     fails.
8. `swift/TurboSparkApp/Tests/TurboSparkAppTests/ServerAttachGuardTests.swift` (Create)
   * **Rationale**: probe refusal paths with a stubbed payload.

### Verification Plan

* **Build**: `cargo build --workspace` and `make swift-lib`.
* **Tests**: the new suites plus `cargo test --workspace` for the
  server crate changes.
* **Manual**: launch an intentionally older daemon (previous build)
  and confirm the refusal and Restart action; confirm the healthy path
  and the version display.

--------------------------------------------------------------------------------

## Implementation Record

Implemented 2026-11-14. The EGM review (docs/egm-reviews/opencode2-parity/OC-13/)
found the draft's FFI tripwire redundant, its attach path undefined, and its
`/health` posture unstated. The implementation follows the review's own cut:
show and check the daemon engine version, with no new unauthenticated surface.

### Open Question resolutions

* **Compatibility rule**: minimum-minor (same major, daemon minor at or
  above the app's minimum), per the draft's lean. The app minimum is
  `DaemonEngineCompatibility.minimumEngineVersion` (`"0.8"`), raised in the
  same change that makes the app depend on newer daemon behavior. A
  pre-stamp daemon is `unknownVersion` (a warning, never a pass).
* **Contract revision storage**: moot. The tripwire already exists:
  `TS_ABI_VERSION` in the header, `ABI_VERSION` in `crates/ffi/src/abi.rs`,
  `ts_abi_version()`, and `TurboSparkRuntime.verifyABIOnce()` (called by
  session, tokenizer, and server opens, throwing rather than asserting so a
  release app does not crash). Goal 5, CUJ 3, and plan item 4 are satisfied
  by reuse and need no change.

### What changed

* `crates/ffi/src/api/daemon.rs`: the spawner stamps `engineVersion`
  (`CARGO_PKG_VERSION`) into `server.meta`; `ts_daemon_status_json` returns
  it (null for a pre-stamp meta). The app bundle ships the FFI library and
  `turbospark-server` together, so the spawner's version is the server
  binary's. Test: `model_meta_tests::engine_version_reads_the_stamp_and_treats_absence_as_unknown`
  (mutation-checked: removing the empty-string filter fails it).
* `swift/TurboSpark/Sources/TurboSpark/Daemon.swift`:
  `TurboSparkDaemonStatus.engineVersion`, `compatibility()`, and the pure
  `DaemonEngineCompatibility` predicate. Tests:
  `DaemonEngineCompatibilityTests` (matrix incl. numeric minor compare,
  major mismatch both ways, unknown, malformed, older-daemon decode, and the
  build satisfying its own minimum). Mutation-checked: dropping the major
  rule fails the major test.
* `swift/TurboSparkApp/.../Server/DaemonVersionLine.swift` (new) and
  `ServerMenuBarView.swift`: a version line in the background daemon menu,
  and a mismatch or unknown-version warning line naming both versions. The
  existing Restart Background Daemon item (which carries the user's launch
  args) is the remedy. Tests: `ServerAttachGuardTests`.
* `Localizable.xcstrings`: 3 new keys, 21 languages each.

### Decisions and deviations

* **Probe source**: the stamped meta file, not an HTTP `/health` read. A
  daemon is a local, app-managed process whose status already comes from the
  meta file, and there is no foreign-daemon attach path in the app (the
  review's B2), so no HTTP probe exists to guard.
* **Scope cut to display plus warning**, per the review: no hard refusal,
  because nothing attaches to the daemon over HTTP. The warning appears
  where the daemon is shown (menu bar daemon section); there is no daemon
  display in the Server pane today.
* **Predicate lives in Swift**, not `crates/server/src/version.rs`: the
  only consumer is the app. `crates/server/tests/version_handshake.rs` is
  not created.
* **Diagnostics**: the warning text carries both versions; no log line was
  added (the menu re-renders, so a log there would repeat).

### Deferred

* **`protocolVersion` / `engineBuild` and version fields on `GET /health`**:
  `/health` is exempt from authentication and deliberately omits build
  metadata (`handler::health`). Adding version fields reverses that and is
  an auth-posture decision. Needed only if a non-app client must read the
  version over HTTP. Decide first whether an unauthenticated engine version
  is acceptable on a loopback-only default.
* **Hard attach refusal and a Restart action inside an attach flow**: no
  attach flow exists. Revisit with any foreign-daemon attach (OC-12).
* **Manual check with an older daemon** (Verification Plan): not run; the
  stale and unstamped cases are covered with stubbed payloads. To run it,
  start a daemon from a pre-change build, open the menu, and expect the
  unknown-version warning.

--------------------------------------------------------------------------------

*Drafted with EGM*
