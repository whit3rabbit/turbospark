# Stage 7: Critic report (OC-13)

Target: lines 1-160 (FeatureRange = whole file). One review round, no remediation, convergence not assessed. egm-design (DC) and egm-structural (ST) used feature-spec-standard; egm-style (SY) had no standard. All three completed.

## Design (DC)

DC-1 Blocker. Lines 1-160. No Alternatives Considered and no Risks & Mitigations section. Missing alternatives: per-request version header, capability flags, extending `ts_server_info_json`/`ServerInfo`, doing nothing, pinning the daemon to the app bundle. Fix: add both sections with concrete rejection reasons.
DC-2 Blocker. Lines 19-26, 59-61, 74-80. Self-contradiction on whether an attach path exists. Line 60 says the app always launches its own daemon; CUJ 1 attaches to a user-started one. `ServerDaemonLaunch.swift` only builds spawn args; `TurboSparkServer.swift` is the in-process server. Port discovery and auth are not described. Fix: name the real attach code or design it, and reconcile line 60 with CUJ 1.
DC-3 Major. Lines 41-43, 100, 107. Probe surface ambiguous. `GET /health` is exempt from the API key, so version and build would be exposed to any local process. Fix: pick one surface, state auth posture.
DC-4 Major. Lines 41-48, 63-67. Rule underspecified: field meanings, who bumps minor, where the app minimum lives, newer-major handling, additive-only assumption. Fix: add bump policy and compatibility table.
DC-5 Major. Lines 28-37, 74-80, 45. An older daemon lacks the new fields, so the probe sees an absent field; behavior undefined, and "names both versions" cannot hold. Fix: absent field means refuse with a "pre-handshake daemon" message.
DC-6 Major. Lines 45-46, 77-80. Restart unspecified: stopping a daemon the app may not own, port collision, differing launch options, no stop mechanism or consent. Fix: define stop mechanism, confirmation, and whether foreign daemons are stopped.
DC-7 Major. Lines 49-52, 88-94, 109-110. FFI tripwire duplicates `TS_ABI_VERSION`/`verifyABIOnce()`; a fatal assertion would crash release apps. Fix: extend the existing check or drop it; use a thrown error.
DC-8 Major. Lines 125-136. Target files vague or unverified ("routes.rs or equivalent", `version.rs`, generator); the probe belongs in a daemon-client type (`Daemon.swift`). Fix: real paths.
DC-9 Nit. Lines 63-67. "exact-minimum" mislabeled and already assumed in Goals. Fix: resolve or drop from Goals.
DC-10 Nit. Line 116. "21 languages" without string key count. Fix: count the keys.
DC-11 Nit. Lines 112-120. No timeout, retry or unreachable-daemon state. Fix: add probe failure behavior.

## Structural (ST)

ST-1 Blocker. Whole document. No Alternatives Considered (same as DC-1). Fix: add the H2.
ST-2 Blocker. Whole document. No Risks & Mitigations (false-positive refusal, unauthenticated version leak, probe timeout, fatal assertion in release, field drift). Fix: add the H2.
ST-3 Major. Lines 21-22. Swift links resolve relative to the doc folder and break; INDEX uses `../../../`. Fix: add the prefix.
ST-4 Major. Lines 128, 134. Placeholder and directory-only paths. Fix: exact file paths.
ST-5 Major. Lines 44-45, 63-70, 102. Open Questions hold a rule that Goals and the plan already commit to. Fix: move decided rule out of Open Questions.
ST-6 Nit. Line 28. "The Problem" vs "Problem Statement". Fix: rename.
ST-7 Nit. Lines 3-13. Unlabeled preamble. Fix: add a Context block.
ST-8 Nit. Lines 63-70. "exact-minimum" vs "exact-match" confusion. Fix: rename "minimum-minor".
ST-9 Nit. Lines 149-156. No Swift test command. Fix: add it.
ST-10 Nit. Lines 72-94. No failure CUJ (unreachable daemon, timeout, malformed payload). Fix: add one.

## Style (SY)

SY-1 Major. Lines 65-67. "Exact-minimum" is contradictory, "lean" undefined, question already answered. Fix: rename, state recommendation, close.
SY-2 Major. Lines 65-67, 70-71. "Every daemon update becomes a hard failure" is unmeasured and conflicts with "mismatch is a corner". Fix: make conditional or drop "every".
SY-3 Major. Lines 31-37. Claims about current attach behavior and "confusing request errors" have no evidence. Fix: cite code and one concrete failing request.
SY-4 Major. Lines 36-37. "Serving surface is unguarded" overstated; current status payload not stated. Fix: say what the endpoint returns today with a file reference.
SY-5 Major. Lines 24-26, 49-52. "Cannot pass silently" is absolute and conflicts with "no skew possible". Fix: "guards a mismatched dev build at session open"; reword to "no skew in release builds".
SY-6 Major. Lines 60-61. Figurative "corner, not a flow". Fix: state that mismatch occurs only with a user-started or stale daemon.
SY-7 Major. Lines 41-48, 100-104, 129. Inconsistent names for the endpoint; `engineBuild` undefined; compared field unnamed. Fix: name the endpoint and define each field.
SY-8 Nit. Lines 22-25, 53-56. "Any future attach path" speculative. Fix: drop or point to OC-12.
SY-9 Nit. Lines 25-26, 40. Vague "surfaces". Fix: say "out-of-process daemon".
SY-10 Nit. Lines 53-57, 69-71. Codegen scope unclear (generator emits constant). Fix: state it.
SY-11 Nit. Lines 114-120. Performance line is not measurable. Fix: drop or bound.
SY-12 Nit. Lines 116, 140-141. "21 languages" repeated; INDEX already requires it. Fix: reference once.
SY-13 Nit. Lines 143-145. Mutation check owner and procedure unstated. Fix: say manual one-time.
SY-14 Nit. Lines 151-156. How to obtain an older daemon unstated. Fix: stub payload fixture.
