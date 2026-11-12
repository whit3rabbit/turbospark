# Consolidated review report: ZP-01 off-peak idle tasks

Target: docs/designs/zcode-parity/ZP-01-off-peak-idle-tasks.md (full file, 242 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed.
Source IDs: ZO = zoom_out_report (zoom-out), IC = zoom_out_report (inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 4, Major 11, Nit 12.

## Blocker

B1. Lines 59-69 (also 15-19, 39, 86-89, 128-135). Idle signal is undecided yet assumed elsewhere; a no-sleep assertion is not an idle detector. Fix: decide option (b) user toggle, optionally plus the existing screen-lock observers; name the signal. [ZO-1, IC-1, CO-1, DC-1, SY-1, S8-B1]
B2. Lines 70-74, 81, 86-88. Where tasks run is unresolved and cites ZP-02; the prerequisite is ZP-23 (sequenced after ZP-01). Fix: v1 runs in the originating chat only, when selected and idle; cite ZP-23 as the later relaxation. [ZO-5, IC-4, CO-2, DC-2, ST-4, SY-9, SY-22, SY-23, S8-B2]
B3. Lines 46-48, 126-132, 156-163, 189-193. Cap "same store lock as the scheduler" contradicts the separate `offpeak_tasks.json`, the "shared JSON store" diagram label and "store-adjacent". Fix: independent dispatcher with its own store and lock, atomic check-and-insert cap, drop the cron change. [ZO-4, IC-6, CO-3, DC-3, ST-5, SY-10, S8-B3]
B4. Lines 146-175 (missing section). No Risks & Mitigations section, required by the standard. Fix: add a short section (user returns mid-run, crash after claim, unattended injection, remote-provider prompts, queue growth, re-running side-effecting prompts). [ST-1, DC standards gap, S8-M8]

## Major

M1. Lines 18-19, 61, 185-188. Power API named three ways; `IOPMSleepDisabled` is not an assertion type. Fix: use `ProcessInfo.beginActivity`; keep-awake is an output, not part of the gate. [ZO-2, CO-4, DC-7, SY-2, S8-M1]
M2. Lines 91-98, 141-144, 197-199. Recovery "reuses the shutdown path" but that path recovers chat rows and does not run on a crash. Fix: launch-time sweep turns a persisted `running` record into `interrupted`; drop `admitted`. [ZO-6, IC-5, CO-6, DC-6, SY-4, S8-M2]
M3. Lines 46-48, 100-107, 210-215. No unattended-run policy and two stacked ladder rules. Fix: one ladder position matching cron tools; blocked approval ends the task and releases the assertion; model-origin tasks cannot enqueue tasks. [ZO-8, IC-3, CO-9, DC-8, SY-18, S8-M3]
M4. Lines 44-46, 75-77, 193-204. Four new knobs without defaults; staleness imported from ZCode ticket expiry defeats "run eventually". Fix: toggle only, fixed cap of 20, no staleness skip or opt-in expiry. [ZO-7, IC-2, IC-7, CO-7, CO-8, DC-10, SY-7, S8-M4]
M5. Lines 38-39, 94-95, 141-144. Priority undefined under FIFO; `interrupted` has no outgoing transition; behavior when the user returns mid-run undefined. Fix: drop priority and "constraints"; re-run requeues at the tail; state mid-run behavior. [CO-5, DC-5, SY-6, S8-M5, S8-M9]
M6. Lines 54-55, 75, 111-112. Broken references: ZP-90 1.4 is the remote relay, ZCode's cap of 20 is verified only for cron, "CronScheduler owns pane integration" is unsupported. Fix: cite correctly or drop. [ST-3, ST-6, SY-3, CO-10, S8-M6]
M7. Lines 146-163. Alternatives attributed to "the author"; roadmap unlinked; missing `NSBackgroundActivityScheduler` and pure toggle. Fix: concrete rationales, link roadmap, add both. [DC-9, ST-2, SY-14, S8-M7]
M8. Lines 111, 186, 190-193. Shared tick vs own timer unresolved; 20 s poll coarse. Fix: own timer (follows from B3). [DC-4]
M9. Lines 128-134. Existing screen-lock and screens-sleep observers (`ProfileVaultCoordinator.swift`, `RootView.swift:199`) are not reused or mentioned. Fix: build the lock signal on them. [ZO-3]
M10. Lines 170-176. Power NFR omits AC/thermal/battery and does not decide dispatch after assertion failure. Fix: state AC requirement or accepted drain, and whether dispatch proceeds on failure. [DC-7, SY-5, CO-13]
M11. Lines 14-20, 221-238. No task and run-entry data model; verification omits idle gate, tool ladder, cap race, stale skip, crash and model tools. Fix: add minimal schema and matching tests. [DC-11, SY-21]

## Nit

N1. Lines 205, 210. `SettingsPanes.swift` does not exist (pane is `Components/CronJobsSettingsPaneView.swift`); "basename" annotations inconsistent. Fix: full verified paths. [ZO-9, DC-12, ST-9, SY-17, S8-N1]
N2. Lines 17-19, 28-32, 153-154. ZCode uses Electron `powerSaveBlocker`; the gate is not "power-assertion-based". Fix: reword. [ZO-10, ST-8, SY-11, SY-12, S8-N2]
N3. Lines 178-224. Over-detailed (claim-write order, mutation checks, tick, CSyntext note); mutation bullet mis-nested and lacks expected failure. Fix: trim or restate. [ZO-11, ST-10, SY-19, SY-20, S8-N3]
N4. Lines 234-235. "ten gates" and "21 languages" unverified. Fix: reference the INDEX constraint. [ZO-12, DC-13, SY-20, CO-12, S8-N4]
N5. Lines 3-6. Does not cite ZCode `docs/off-peak-task/spec.md`; "Cron parity target" wording odd. Fix: cite and reword. [ZO-13, ST-11, S8-N5]
N6. Lines 59-77. Open questions already carry assumed defaults; INDEX says record rather than silently choose. Fix: mark assumed option. [ZO-14, DC-15, S8-N6]
N7. Lines 66-69. "No defensible single answer" contradicts "pick one". Fix: drop the clause. [SY-8, S8-N7]
N8. Lines 22, 109. Headers differ from standard names. Fix: rename to "Problem Statement" and "Proposed Architecture". [ST-7, S8-N8]
N9. Lines 167-169. "Never leave the machine" is absolute; false with ZP-22 remote providers. Fix: say off-peak adds no network path. [SY-16, S8-N10]
N10. Lines 31-33, 173-174. "Useful on its own" and "opt-in posture" are uncited. Fix: cite or drop. [SY-13, SY-15]
N11. Lines 19, 77. Naming inconsistency (TurboSpark vs TurboSparkApp); "profile plumbing" cost unestimated. Fix: unify and drop or estimate. [DC-14]
N12. Lines 1-243 vocabulary. "Off-peak" never defined; jargon undefined. Fix: one defining sentence, link INDEX. [CO-11, CO-12]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-14 (zoom-out), IC-1 to IC-7 (inventive catalyst). Feeds B1-B3, M1-M4, M9, N1-N6.
- comprehension_report.md: CO-1 to CO-13. Feeds B1-B3, M1-M7, M10, N4, N12.
- critic_report.md: DC-1 to DC-15 (design), ST-1 to ST-11 (structural), SY-1 to SY-23 (style). Feeds all Blockers and Majors, N1-N11.
- intermediate_zoom_out_report.md: S8-B1 to S8-B3, S8-M1 to S8-M9, S8-N1 to S8-N10. Confirms all Stage 5 findings open; recommends rejecting scope-adding asks (IC-1, IC-3, IC-5, DC-6, DC-9 XPC, DC-11 schema/metrics) and keeping ambiguity-removal fixes.
- Merge notes: ZO-4, IC-6, DC-3, ST-5 merged into B3; CO numbers follow the order of the comprehension findings list.
