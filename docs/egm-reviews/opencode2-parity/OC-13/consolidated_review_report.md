# Consolidated review report: OC-13 engine and daemon version handshake

Target: docs/designs/opencode2-parity/OC-13-engine-version-handshake.md (full file, 160 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed.
Source IDs: ZO and IC = zoom_out_report (zoom-out, inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 4, Major 9, Nit 10.

## Blocker

B1. Lines 49-52, 68-71, 88-94, 109-110, 134-136. The FFI revision tripwire already exists as `TS_ABI_VERSION`, `ts_abi_version()` and `verifyABIOnce()`; a fatal assertion would also crash release apps. Fix: delete Goal 5, CUJ 3, plan item 4 and Open Question 2, or reduce to reuse of the existing check. [ZO-1, IC-1, CO-7, DC-7, S8-B1]
B2. Lines 19-37, 59-61, 74-80. Self-contradiction: line 60 says the app always launches its own daemon, CUJ 1 attaches to a user-started one; no attach code is shown and no concrete skew failure is named. Fix: name the real call site (`Daemon.swift status()`), one skew failure, and whether foreign-daemon attach exists; if not, cut scope to display. [ZO-2, IC-2, CO-1, DC-2, S8-B2]
B3. Whole file. No Alternatives Considered section. Fix: add one rejecting exact-match, per-request header, capability flags, `ServerInfo` extension, do nothing, auto-restart, each with a reason. [DC-1, ST-1, S8-B3]
B4. Whole file. No Risks & Mitigations section. Fix: add one covering false-positive refusal, unauthenticated version leak, probe timeout, release-build crash, field drift. [DC-1, ST-2, S8-B3]

## Major

M1. Lines 41-46, 63-67, 98-110. Three fields, a predicate and a new module with no owner for `protocolVersion` or the app minimum; fields and rule undefined. Fix: expose engine version only and compare to the app's own, or add a field table and bump policy. [ZO-3, IC-4, CO-3, DC-4, SY-7, S8-M1]
M2. Lines 125-136, 107-108. Target files wrong or vague (`routes.rs` does not exist; health is `handler::health` in `crates/server/src/lib.rs`; probe belongs in `Daemon.swift`). Fix: real paths, retarget items 1-3. [ZO-4, CO-8, DC-8, ST-4, S8-M2]
M3. Lines 41-43, 98-101, 125-129. `/health` is key-exempt; auth posture of version fields unstated. Fix: name the route and state posture. [IC-3, DC-3, S8-M3]
M4. Lines 45-46, 56-61, 77-80. Restart unspecified: stop mechanism, ownership, failure, differing launch options. Fix: define them, with consent and a one-retry cap. [IC-5, CO-2, DC-6, S8-M4]
M5. Lines 28-37, 74-80. Older daemon returns no new fields; behavior undefined and breaks "names both versions". Fix: absent means "pre-handshake daemon" with a distinct refusal. [DC-5, S8-M5]
M6. Lines 63-71. Open Question 1 mislabeled "exact-minimum" and already decided in Goals and plan. Fix: rename "minimum-minor", move to Goals, keep only real open items. [CO-4, DC-9, ST-5, ST-8, SY-1, S8-M6]
M7. Lines 31-37, 24-26, 49-52, 60-61. Uncalibrated claims: unguarded surface and "confusing errors" uncited, "cannot pass silently" absolute vs "no skew possible", figurative "corner, not a flow", unmeasured "every daemon update". Fix: cite code, soften to "no skew in release builds", state mismatch occurs only with stale daemon. [SY-2 to SY-6]
M8. Lines 22, 107-110, 49-53. `ts_session` undefined and FFI check described inconsistently (who reads which constant, compile error vs runtime). Fix: define terms or remove with B1. [CO-6, CO-7]
M9. Lines 21-22. Swift links resolve relative to the doc folder and break. Fix: add `../../../`. [ST-3, S8-N2]

## Nit

N1. Lines 3-9, 22. Parity framing borrows OpenCode's sidecar; local reason is a Sparkle update leaving an old daemon. "Sidecar" unglossed. Fix: reword motivation, gloss once. [ZO-6, IC-7, CO-10, S8-N1]
N2. Lines 112-120. No probe timeout, retry or unreachable-daemon state; performance line not measurable. Fix: add failure behavior, drop or bound. [DC-11, SY-11]
N3. Lines 72-94. No failure CUJ. Fix: add one. [ST-10, S8-N3]
N4. Lines 114-117, 125-147. Not "small": eight files, 21 languages; "21 languages" unlisted and repeated. Fix: re-size after cuts, reference INDEX once, count keys. [ZO-5, IC-6, CO-12, DC-10, SY-12, S8-N3]
N5. Lines 142-156. No Swift test command; mutation check manual or automated unstated; no way to produce an older daemon. Fix: add command, say manual, use a stub payload. [ZO-5, IC-8, CO-11, ST-9, SY-13, SY-14]
N6. Line 28, lines 3-13. "The Problem" vs "Problem Statement"; unlabeled preamble. Fix: rename, add Context block. [ST-6, ST-7]
N7. Lines 35-36. "gturbo manifest ... producer block" unglossed. Fix: link or gloss. [CO-9]
N8. Lines 22-26, 40, 53-57, 69-71. "Any future attach path" speculative, "surfaces" vague, codegen scope unclear. Fix: drop or point to OC-12, say "out-of-process daemon", state generator scope. [SY-8, SY-9, SY-10]
N9. Lines 100-104, 41-42. "status/health surface" has no route, method or auth. Fix: give the route once. [CO-13, SY-7]
N10. Lines 3-9, 100-105. Why exact-match was rejected is stated only in Open Questions. Fix: state in Goals. [IC-7]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-6, IC-1 to IC-8.
- comprehension_report.md: CO-1 to CO-13.
- critic_report.md: DC-1 to DC-11, ST-1 to ST-10, SY-1 to SY-14.
- intermediate_zoom_out_report.md: S8-B1 to S8-B3, S8-M1 to S8-M6, S8-N1 to S8-N3. Confirms all Stage 5 findings open. Recommends resolving B2 then B1 first; if no foreign-daemon attach exists the document shrinks to showing the daemon engine version with an optional warning. Scope-adding asks (field table and bump policy, failure CUJ, consent flow) should follow that cut.
- Merge notes: DC-1 split into B3 and B4; SY-2 to SY-6 merged into M7; the Style reviewer's finding 15 (boilerplate footer) was dropped as no change needed.
