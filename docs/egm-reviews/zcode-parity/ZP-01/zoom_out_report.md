# Stage 5: Zoom Out and Inventive Catalyst report (ZP-01)

Target: docs/designs/zcode-parity/ZP-01-off-peak-idle-tasks.md (full file)
Context read: INDEX.md, ZCODE_COMPARISON.md (lines 1-237 only), ZP-90.
Reviewers: egm-zoom-out (completed), egm-inventive-catalyst (completed). Both read persona and standard.

## Zoom Out findings (egm-zoom-out)

Code-search facts the reviewer used: no `beginActivity`, `IOPMAssertion`, `idleSystemSleepDisabled`, or HID idle reader in /swift. Screen-lock and screens-sleep observers already exist in `State/ProfileVaultCoordinator.swift` and `App/RootView.swift:199`. The Scheduled Tasks pane is `Components/CronJobsSettingsPaneView.swift`. `SettingsPanes.swift` does not exist. `CronScheduler` has its own `NSLock` and `writeLock`.

### Blocker

ZO-1. Lines 59-69 vs 15-19, 86-89, 128-134, 173-174. The admission signal, the feature's defining mechanism, is left as an open question ("pick one before implementation") while the Overview, CUJ 1 and diagram assume option (a) or (c). Simplest answer is (b), a user toggle, optionally plus the existing screen-lock observers.

### Major

ZO-2. Lines 61-69, 123, 129-135, 186-188. An IOPMAssertion/beginActivity only prevents sleep; it cannot detect idleness. The doc calls it "IOPMAssertion-based idle detection" and offers two APIs; `IOPMSleepDisabled` is not an assertion type. Name one API (`ProcessInfo.beginActivity`) and separate admission from keep-awake.

ZO-3. Lines 128-134. Reuse is not addressed: existing screen-lock and screens-sleep observers could supply a "screen locked" signal. Reference them or say why insufficient.

ZO-4. Lines 47-48, 111-114, 120-137, 162-163, 191-193. Structure contradicts itself: "second dispatcher inside the same module" vs "separate but store-adjacent"; diagram says "shared JSON store" but a new `offpeak_tasks.json` is created; tick sharing is an unresolved either/or; the cap "serialized through the same store lock as the scheduler" is impossible across two stores with no stated cross-object lock. Pick one structure; simplest is an independent dispatcher with its own store, timer and lock.

ZO-5. Lines 55-56, 70-74, 86-88, 194-199. Execution path unspecified and inconsistent with Non-Goals. The open question points at ZP-02 for the concurrent-chats decision, but INDEX.md and ZP-90 show it was adopted as ZP-23. Under today's single-active-chat model an off-peak run into a non-selected chat is not possible. Decide: originating chat only (like cron), or depend explicitly on ZP-23.

ZO-6. Lines 91-98, 141-144. "Recovery reuses the shutdown path" overstates reuse: that path (AppShutdownCoordinator) recovers chat transcript rows, not task-queue state, which needs its own persisted run state. Seven states are listed; `admitted` and `running` are not distinguished by any behavior. Consider dropping `admitted`.

ZO-7. Lines 44-46, 75-77, 193-204. Four new knobs (toggle, idle threshold, staleness bound, job cap) at once; staleness came from ZCode's ticket expiry and is speculative locally; cap is itself an open question; cron has none of these. Ship toggle only with fixed cap, add others on demand.

ZO-8. Lines 46-48, 100-107, 210-215. Model-enqueued tasks run a model-authored prompt unattended with the user away; no statement of which tools and approvals apply. Tool ladder is stated twice with competing rules ("low/medium" and "follow cron tool treatment"). INDEX.md requires one ladder position. Pick one, preferably same as the cron tools, and add an unattended-run policy.

### Nit

ZO-9. Line 205. `SettingsPanes.swift` does not exist; the pane is `Components/CronJobsSettingsPaneView.swift`.
ZO-10. Lines 17-19, 28-31. ZCode keep-awake is Electron `powerSaveBlocker` per ZCODE_COMPARISON 1.2, not "IOPMAssertion-style".
ZO-11. Lines 178-224. Several details are too low-level for a design doc (claim-write ordering, mutation checks, 20 s tick, CSyntext shim note).
ZO-12. Lines 234-235. Restates localization counts ("ten gates", "21 languages") that can drift; reference the shared constraint instead.
ZO-13. Lines 3-6. Does not cite ZCode's `docs/off-peak-task/spec.md` (ZCODE_COMPARISON line 69), the provenance for the semantics being ported.
ZO-14. Lines 59-77 vs INDEX.md 15-16. Each open question already has an assumed default in the body; mark the assumed default.

Good: clear objective, server ticket half rejected with reasoning, Alternative B defensible, opt-in default-off, no Rust or protocol work.

Caveat: ZCODE_COMPARISON.md read only to line 237.

## Inventive Catalyst findings (egm-inventive-catalyst)

IC-1 (Major). Open Questions admission signal, Technical Plan, Goals. Idle detection is framed as unresolvable and deferred; a no-sleep assertion is an output, not an idleness input. Admission is gate-only: nothing yields when the user returns, so local Metal inference contends with foreground work. Resolution: make the gate a pure function over pushed signals (HID idle seconds, screen lock, AC vs battery, thermalState, low-power mode, app generation-busy), add revocation that interrupts and re-queues with attempt preserved, toggle as master consent. Liabilities: thrash (needs hysteresis and minimum slice), more test surface (stub signal port).

IC-2 (Major). Goals misfire semantics, state machine. Cron-style staleness skip imported into an availability queue, contradicting Alternative B; a task enqueued Friday is skipped Monday, defeating "run eventually". Resolution: opt-in per-task expiry with no default TTL measured from admission, or a "stale, needs confirmation" state instead of a silent skip.

IC-3 (Major). CUJ 3, Technical Plan, NFR Power. Unattended execution with no unattended policy: a run can stall on pendingToolCall approval, holding the no-sleep assertion forever; no run timeout or watchdog. Resolution: blocked approval becomes a "needs attention" state that releases the assertion and notifies; deny-by-default for side-effecting tools unless pre-authorized at enqueue; per-task wall-clock watchdog owns assertion release.

IC-4 (Major). Open Questions where tasks run, CUJ 1. ZP-02 reference is stale (ZP-23 is the prerequisite; INDEX sequences ZP-01 before ZP-23). ZP-01 must work under single-active-chat guards where an unattended run can block or hijack the foreground chat. Resolution: per-task scratch run record with summary posted to target chat, or v1 dispatches only when target chat is selected and idle. Fix the ZP-02 reference.

IC-5 (Major). Goals interrupted recovery, CUJ 2. Uniform manual re-run turns an unattended queue into manual work after any quit or crash. Resolution: bounded auto-requeue (persisted attempt counter before dispatch, default 1 retry) for tasks marked idempotent or read-only.

IC-6 (Nit). Diagram, Target Files 1-2. "Shared JSON store" vs separate file; "same store lock" not shared; tick left either/or. Resolution: a small store actor owning both files' writes and the cap, or fully separate stores.

IC-7 (Nit). Target File 1, Open Questions cap, Target File 4. Hedge between `beginActivity` and a non-existent `IOPMSleepDisabled`; four knobs for unproven needs; "ten parity gates" and "21 languages" should be verified against `LocalizationParityTests`. Fix cap at 20, expose toggle and threshold only, pick one API.

Dependency note: doc cites ZP-02 where ZP-23 is the prerequisite; reconcile INDEX and ZP-01.
