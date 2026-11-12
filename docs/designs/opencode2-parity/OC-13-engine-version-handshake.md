# OC-13: Engine and daemon version handshake

Finding: Section 8, "Typed protocol contract, versioned SDK, and
in-process SDK host - Missing in Swift" in
[OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md).
Parity targets: OpenCode's
[service-version predicates](https://github.com/anomalyco/opencode/blob/dev/packages/client/src/service-version.ts)
and
[protocol contract](https://github.com/anomalyco/opencode/blob/dev/packages/protocol/src/client.ts).

* Author: whit3rabbit
* Status: Draft
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

*Drafted with EGM*
