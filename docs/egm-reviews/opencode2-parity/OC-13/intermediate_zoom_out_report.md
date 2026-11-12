# Stage 8: Intermediate zoom-out report (OC-13)

Reviewer: egm-zoom-out (S8, completed). Target unchanged since Stage 5, so every Stage 5 finding remains open. Re-confirmed: `verifyABIOnce()` is called from TurboSparkSession.swift:76, TurboSparkServer.swift:418 and TurboSparkTokenizer.swift:25; `TurboSparkDaemonStatus` and `status()` live in Daemon.swift:5 and :40.

S8-B1 Blocker. Lines 49-52, 68-71, 88-94, 109-110, 134-136. FFI tripwire duplicates `TS_ABI_VERSION` and `verifyABIOnce()`; a fatal assertion would crash release apps. Fix: delete Goal 5, CUJ 3, plan item 4 and Open Question 2, or reduce to reuse of the existing check.
S8-B2 Blocker. Lines 19-37, 59-61, 74-80. Document contradicts itself on whether an attach path exists; no concrete skewed request named. Fix: name the call site (`Daemon.swift status()`), one skew failure, and whether foreign-daemon attach exists; if not, cut scope to showing the version.
S8-B3 Blocker. Whole file. Missing Alternatives Considered and Risks & Mitigations (blocker on the structure standard; under zoom-out the alternatives section defends simplicity). Fix: add both, listing engine-version-only, do nothing, and automatic restart as alternatives.
S8-M1 Major. Lines 41-46, 98-110. Three fields, a predicate and a new module exceed need; app and daemon ship from one repo; `protocolVersion` and the minimum have no owner. Fix: expose engine version only, compare to the app's own, drop the rest until real skew appears.
S8-M2 Major. Lines 125-136. Target files wrong or vague (`routes.rs` does not exist; health is `handler::health` in `crates/server/src/lib.rs`; probe belongs in `Daemon.swift`). Fix: retarget items 1-3, drop item 4.
S8-M3 Major. Lines 41-43, 100. `/health` is key-exempt; auth posture of the version fields unstated. Fix: name the route and state posture.
S8-M4 Major. Lines 45-46, 77-80. Restart unspecified (stop mechanism, ownership, failure). Fix: state them; consider relaunch via `ServerDaemonLaunch.args` with a one-retry cap.
S8-M5 Major. Lines 28-37, 74-80. Older daemon returns no new fields; undefined behavior and breaks "names both versions". Fix: absent means "pre-handshake daemon", distinct refusal message.
S8-M6 Major. Lines 63-70. Open Question 1 mislabeled and already decided in Goals. Fix: rename "minimum-minor" and move to Goals.
S8-N1 Nit. Lines 3-9. OpenCode sidecar framing is not the local reason (Sparkle update leaves old daemon). Fix: reword motivation.
S8-N2 Nit. Lines 21-22. Swift links lack `../../../`. Fix: add prefix.
S8-N3 Nit. Lines 114-117, 125-147. Sizing is not "small"; no failure CUJ, timeout state, or Swift test command. Fix: re-size after cuts, trim tests, add one failure CUJ.

Scope guidance: adding asks (compatibility table and bump policy CO-3/DC-4, failure CUJ DC-11/ST-10, stop and consent flow DC-6, pre-handshake message DC-5, Context block ST-7, string-key count DC-10) should be adopted only after cutting to engine-version-only. Neutral clarity fixes: CO-1, CO-5, CO-6, CO-9, CO-10, ST-3, ST-6, ST-8, SY-1. The Critic's alternatives (per-request header, capability flags, `ServerInfo` extension) should be listed and rejected, not built. Recommended order: resolve S8-B2 then S8-B1; if no foreign-daemon attach exists the document shrinks to showing the daemon's engine version with an optional mismatch warning.
