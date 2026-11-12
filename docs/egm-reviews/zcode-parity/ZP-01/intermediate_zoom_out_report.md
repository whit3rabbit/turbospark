# Stage 8: Intermediate Zoom Out report (ZP-01)

Reviewer: egm-zoom-out (completed; persona and zoom-out standard read; Stages 5-7 reports, INDEX.md and ZP-90 read; ZCODE_COMPARISON.md not re-read). The document was not edited, so every Stage 5 finding (ZO-1 to ZO-14) remains open. Re-confirmed: `SettingsPanes.swift` does not exist (pane is `Components/CronJobsSettingsPaneView.swift`); no `beginActivity`, `IOPMAssertion` or idle-time reader in the Swift tree.

## Blocker

B1. Idle signal undecided (lines 59-69 vs 15-19, 86-89, 128-135, 173-174); an assertion is not an idle detector. Fix: take option (b), a user "allow off-peak now" toggle, optionally plus the existing screen-lock and screens-sleep observers (`ProfileVaultCoordinator.swift`, `RootView.swift:199`). Defer HID idle, thermal, battery, AC signals. Do not adopt IC-1's signal gate with hysteresis and revocation in v1.

B2. Where tasks run unresolved; pointer to ZP-02 is stale, prerequisite is ZP-23 (INDEX 78-79, 98, 102-104). Fix: v1 runs in the originating chat only and only when that chat is selected and idle, as cron does; cite ZP-23 as the future relaxation. No scratch or dedicated chats.

B3. Cap and store locking contradictory (lines 46-48, 126-132, 162-163, 189-193); raised from Major because DC-3 and ST-5 independently found it unimplementable. Fix: independent dispatcher with its own store, timer and lock, atomic check-and-insert for the cap, drop the "same lock" language and the CronScheduler modification.

## Major

M1. Power API named three ways (lines 18-19, 186-188). Pick `ProcessInfo.beginActivity`, delete the `IOPMSleepDisabled` hedge; keep-awake is an output of admission, not part of the gate.
M2. Shutdown-path reuse overstated, crash unsupported (lines 91-98, 141-144, 197-199). Rule: a persisted `running` record at launch becomes `interrupted`. Drop `admitted`. No boot id or heartbeat.
M3. Unattended-run policy missing (lines 46-48, 100-107, 210-215). State one ladder position matching cron tools; a blocked approval ends the task as interrupted or failed so the assertion is released; model-origin tasks cannot enqueue tasks. Do not build a "needs attention" state or watchdog.
M4. Knob proliferation and staleness semantics (lines 44-46, 75-77, 193-204). Ship the toggle only, fixed cap of 20, no staleness skip (or opt-in per-task expiry with no default).
M5. Priority vs FIFO undefined (line 38 vs 39). Remove priority and "optional constraints".
M6. Broken or unsupported cross-references (lines 54-55, 75, 111-112). ZP-90 1.4 is the remote relay, ZCode's cap of 20 is verified only for cron, "CronScheduler owns pane integration" is unsupported.
M7. Alternatives thin and attributed to "the author" (lines 146-163). Link the roadmap, add pure user toggle and `NSBackgroundActivityScheduler`; skip the XPC helper. Alt B's rationale conflicts with the shared-tick plan (resolved by B3).
M8. No Risks and Mitigations section. Keep short: user returns mid-run, crash between claim and run, remote-provider prompts leaving the machine, queue growth, re-running a side-effecting prompt. Skip metrics, logging, schema versioning.
M9. Re-run semantics undefined (`interrupted` has no outgoing transition). State that re-run requeues at the tail. Reject IC-5 auto-requeue.

## Nit

N1. Line 205 wrong filename; "(Modify; basename...)" annotations inconsistent.
N2. Lines 17-19, 28-31: ZCode uses Electron `powerSaveBlocker`.
N3. Lines 178-224 too low-level (claim-write order, mutation checks, CSyntext note).
N4. Lines 234-235 unverified counts; reference the shared constraint (INDEX 115-117).
N5. Lines 3-6: cite ZCode `docs/off-peak-task/spec.md`; "Cron parity target" is odd.
N6. Lines 59-77: mark which option the body assumes (INDEX 20-21).
N7. Lines 66-69: "no defensible single answer" contradicts "pick one".
N8. Lines 22, 109: header names differ from the standard.
N9. Lines 173-176: state what happens after assertion failure.
N10. Lines 167-169: "never leave the machine" false once ZP-22 remote providers are used.

## Later-stage additions

Reject or defer as scope growth: IC-1 (signal gate with revocation), IC-3 (watchdog), IC-5 (auto-requeue), DC-6 (boot id, heartbeat), DC-9 XPC helper, DC-11 (schema versioning, metrics, logging), comprehension jargon requests (link INDEX instead).
Adopt as ambiguity removal: name one power API, fix ZP-02 to ZP-23, state one ladder position, add short Risks section, decide the idle signal.

Overall: solid objective, server half rightly rejected, opt-in default-off. Not implementable by a fresh reader. The three Blockers each have a small fix; together they shrink ZP-01 to a toggle-gated, single-store FIFO queue running in the originating chat, with a launch-time recovery sweep and one keep-awake API.
