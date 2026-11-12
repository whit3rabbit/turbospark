# Stage 7: Critic report (ZP-01)

Loop status: one round only (no remediation allowed in this run, so no second round and no convergence check). All three reviewers completed (egm-design and egm-structural against feature-spec-standard, egm-style with no standard). Direct dispatch, no synthesizer. No reviewer failed.
Context: INDEX.md, ZP-90, ZCODE_COMPARISON.md (lines 1-237).

## egm-design (DC)

Verdict: not ready to implement. Standard present: Problem, Goals, Non-Goals, Architecture, Alternatives, NFR. Missing: Risks & Mitigations.

DC-1 Blocker. Lines 59-69, 18, 38-39, 128-135. Idle heuristic undecided yet assumed by Goals, CUJ 1 and diagram. A no-idle-sleep assertion prevents sleep and is not an idle detector. Real signals (CGEventSource idle time, HIDIdleTime, screen-lock notification, thermal/battery) are not named.
DC-2 Blocker. Lines 70-74. Where tasks run is unresolved and load-bearing. Generation is single-active (finding 2.1, `pendingUserMessages`); behavior when the user is generating, or the target chat is deleted, is undefined. Points to ZP-02, but the prerequisite is ZP-23, which INDEX sequences after ZP-01.
DC-3 Blocker. Lines 46-48, 24-34, 126-132, 191-193. Cap "serialized through the same store lock as the scheduler", but cron has no cap and the new file `offpeak_tasks.json` shares no lock. Cap enforcement across UI, model tool and recovery needs an explicit atomic check-and-insert.
DC-4 Major. Lines 111, 186, 190-193. Tick shared or own is undecided; `CronScheduler` is `@unchecked Sendable` with NSLock and Timer; 20 s is coarse.
DC-5 Major. Lines 142-145. Transitions for `admitted` and `running` undefined; behavior if the user returns mid-run undefined; staleness anchor, default and relaunch behavior undefined.
DC-6 Major. Lines 91-98, 120-135. Crash recovery claims reuse of the shutdown path, which does not run on a crash. A stale "running" claim cannot be told from a live one (no boot id or heartbeat). Re-running a partially executed side-effecting prompt is not discussed.
DC-7 Major. Lines 170-172, 186-188. Two power APIs offered; `IOPMSleepDisabled` is not an assertion type. Display sleep vs system sleep, battery drain, no AC or thermal gate, and fan controller interaction unaddressed.
DC-8 Major. Lines 176-224. Model-enqueued prompts run unattended with no stated permission mode: auto-deny or auto-allow tool approvals, blocking forever, self-queuing loops, prompt-injection amplification. Suggest model-origin tasks need user approval before first admission, or run read-only.
DC-9 Major. Lines 146-163. "Rejected by author" attribution is forbidden by the standard. Roadmap not linked. Missing alternatives: `NSBackgroundActivityScheduler`, pure user toggle, separate process or XPC helper.
DC-10 Major. Lines 36-49, 59-77. Cap, idle threshold and staleness bound have no defaults, so settings cannot be built. "Priority" conflicts with FIFO.
DC-11 Major. Lines 14-20, 29-34, 221-238. No data model or schema/versioning for tasks and run entries; tests omit idle gate, tool ladder, cap race, quit path; no success metric or logging.
DC-12 Nit. Lines 205, 210. `SettingsPanes.swift` did not resolve at the stated path.
DC-13 Nit. Lines 233-235. "21 languages" and "ten parity gates" unverified.
DC-14 Nit. Lines 77, 19. Unestimated "profile plumbing" cost; naming inconsistency TurboSpark / TurboSparkApp.
DC-15 Nit. Open Questions vs Technical Plan: INDEX says open questions are recorded not silently chosen, yet the plan silently chose (a).

## egm-structural (ST)

Heading map: H1, 9 H2 sections; no skipped levels, fences balanced. Fails the standard as written.

ST-1 Blocker. Lines 146-175. No Risks & Mitigations section (required). Missing failure modes: user returns mid-run, sleep/wake, GPU and memory contention, battery/thermal, unattended injection and self-queuing, queue growth, crash between claim-write and run, corrupt store.
ST-2 Major. Lines 146-163. Alt A cites "the roadmap's exclusion policy" without a link (INDEX links `.kiro/steering/roadmap.md`). Alt B's reason restates the design. Admission options a/b/c and the assertion API are never evaluated as alternatives.
ST-3 Major. Lines 54-55. Says server admission is analyzed in ZP-90 under finding 1.4; ZP-90's 1.4 is the mobile/web remote-control relay and never mentions off-peak or finding 1.2. Broken cross-reference.
ST-4 Major. Lines 70-74, 81. Stale pointer to ZP-02; concurrent chats were adopted as ZP-23 (INDEX 51, 78-79; ZP-90 164-166). No ZP-23 dependency stated; INDEX sequences ZP-01 (step 2) before ZP-23 (step 6).
ST-5 Major. Lines 116-139 vs 47-48, 156-163, 189-193. Store model inconsistent: "shared JSON store" vs new `offpeak_tasks.json` (cron uses `cron_jobs.json`) vs "same store lock" vs "store-adjacent". Diagram arrows ambiguous; Gate and Keeper not attached to boxes.
ST-6 Major. Lines 75, 47-48, 203. "ZCode caps at 20 jobs" is the cron cap (automation-create-limit.md), not stated for off-peak in finding 1.2. Cite or say borrowed. Cap placement open yet treated as a setting.
ST-7 Nit. Lines 22, 109. Headers differ from the standard's names ("Problem Statement", "Proposed Architecture").
ST-8 Nit. Lines 61, 31, 186-188. `IOPMSleepDisabled` is not an assertion API; line 61 calls an assertion "idle detection"; line 31 says "IOPMAssertion-style" where finding 1.2 says powerSaveBlocker.
ST-9 Nit. Lines 205, 210. "(Modify; basename...)" annotations and basename paths inconsistent with the other entries.
ST-10 Nit. Lines 221-223. "Mutation checks" bullet is at the wrong nesting level.
ST-11 Nit. Lines 3-6. "Cron parity target" wording is odd (cron is finding 1.1).

## egm-style (SY)

No Blockers. Did not verify Swift sources.

SY-1 Major. Lines 61-69, 39-41, 83-89, 130-135, 183-188. Option (a) treats the assertion as an idle input; elsewhere it is the keep-awake. Question reads as both open and decided. Option (c) wording ambiguous.
SY-2 Major. Lines 185-188, 18. Power API named three ways; `IOPMSleepDisabled` unverified (real constants are `kIOPMAssertionTypePreventUserIdleSystemSleep`).
SY-3 Major. Lines 111-112. "CronScheduler already owns ... settings-pane integration" unsupported by finding 1.1.
SY-4 Major. Lines 98, 197-199. "Recovery reuses it" overstates reuse: the store claim produces the interrupted task row, the shutdown path produces the interrupted chat row.
SY-5 Major. Lines 175-176. Power NFR gives no decision on dispatch after assertion failure.
SY-6 Major. Lines 94-95, 42-43, 141-142. State machine has no transition out of `interrupted`; re-run semantics (new run vs requeue, FIFO position) unspecified.
SY-7 Nit. Lines 74-77. "20 is reasonable" is subjective; ZCode's 20 is verified only for cron.
SY-8 Nit. Lines 66-69. "No defensible single answer" contradicts "pick one"; drop first clause.
SY-9 Nit. Lines 71-75. Stale ZP-02 pointer; cite ZP-23.
SY-10 Nit. Lines 38-48, 127. Priority undefined under FIFO; "same store lock" and "shared JSON store" contradict the new file.
SY-11 Nit. Lines 18-19, 153-154. Gate is not "power-assertion-based".
SY-12 Nit. Lines 29-32. "IOPMAssertion-style" vs powerSaveBlocker.
SY-13 Nit. Lines 31-33. "useful on its own" unmeasured.
SY-14 Nit. Lines 156-163, 191-192. Alt B rationale ("shared tick hides assertions") conflicts with plan to share the tick.
SY-15 Nit. Lines 173-174. "Opt-in posture for background machinery" uncited.
SY-16 Nit. Lines 167-169. "Never leave the machine" is absolute; remote-provider chats (ZP-22) could send prompts off-device.
SY-17 Nit. Lines 205, 210. Unexplained "basename" shorthand.
SY-18 Nit. Lines 212-215. Two stacked permission rules may conflict; INDEX requires one stated ladder position.
SY-19 Nit. Lines 221-223. Mutation checks lack an expected failure.
SY-20 Nit. Lines 227-235. Unexplained `CSyntext.o` note; "21 languages" unsourced.
SY-21 Nit. Lines 236-238. Verification omits stale-skip, cap, crash recovery and model tools; no release gate stated.
SY-22 Nit. Lines 55-57. No statement of what happens when the user is typing or generating in the target chat; finding 2.1 says non-selected chats drain only when selected and idle.
SY-23 Nit. Line 1 and INDEX. State the single-active-chat constraint to match INDEX ordering.
