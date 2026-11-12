# Stage 6: Comprehension report (ZP-01)

Reviewer: egm-comprehension (completed). It read the document alone, no standard, no extra context.

## What this proposes

Add a second automation domain to the TurboSpark Swift app: an "off-peak" task queue. A user, or the model via tools, enqueues a prompt (with target chat, priority, optional constraints). It is run later, in FIFO order, only when the machine is judged idle. While any such task runs, the app holds a macOS no-idle-sleep assertion. Interrupted tasks are recovered on relaunch and can be re-run. Tasks older than a staleness bound are skipped and recorded. A new `OffPeakTaskDispatcher` sits beside `CronScheduler`, with its own JSON store, a state machine, an admission gate, a power-assertion wrapper, a job cap, settings, a pane section and tools. It is default off. Server-ticketed admission, as in ZCode, is excluded.

## How the existing system works, per this document

`CronScheduler` has a 20-second timer over a JSON store and fires prompts into the originating chat. It has a Scheduled Tasks settings pane. The generation path (the "AppModel submit path") already has interrupt and queue semantics, and a shutdown path that turns an in-flight reply into an "interrupted row". Stores use `AppJSONStore` with quarantine rules. Tools are registered in a risk "ladder" with mode gating, and cron already has a tool treatment. ZCode (the reference product) uses a server ticket queue plus a keep-awake assertion.

## Findings

1. Blocker. Scope: Open Questions "Admission signal"; Overview and Goals; CUJ 1; Architecture (`OffPeakAdmissionGate`). The central mechanism, what "idle" means, is explicitly undecided ("pick one before implementation") yet Goals, CUJ and diagram treat it as settled. The three options conflict. IOPMAssertion is called "idle detection" but elsewhere is a keep-awake assertion. "N minutes" and "heuristic threshold" are unspecified.

2. Blocker. Scope: Open Questions "Where off-peak tasks run"; Goals ("target chat"); CUJ 1/3; Architecture. Execution location (originating, dedicated or scratch chat) is undecided and depends on "the concurrent-chats decision (see ZP-02 dependencies)", which is not linked or summarized. Unclear whether a task can run while the user is in that chat, and whether idle means the whole app or that chat.

3. Major. Scope: Architecture diagram vs text; Implementation items 1 and 2. The diagram shows a "shared JSON store" but names a new `offpeak_tasks.json`. The cap is "serialized through the same store lock as the scheduler" but no lock is defined. Item 2 leaves the tick shared-or-own open, and Alternative B cites "the shared tick hides off-peak's power assertions". Relation between the two dispatchers and diagram arrows are hard to follow.

4. Major. Scope: Implementation item 1 power mechanism. `beginActivity(.idleSystemSleepDisabled)` or `IOPMAssertion` via `IOPMSleepDisabled` is left either/or, though Overview and Goals say IOPMAssertion. The APIs are not equivalent and no criterion is given.

5. Major. Scope: Goals ("priority"); state machine; FIFO. Tasks have a priority field but admission is strictly FIFO; priority's effect is never stated. "Optional constraints" undefined. "admitted" and transitions are not described. One-at-a-time claiming is implied but not stated as a goal. "Claim" is not defined beyond "written to the store before generation starts".

6. Major. Scope: CUJ 2; Problem; Goals ("persisted run state"). Recovery relies on the "existing shutdown path" and "interrupted row", neither explained nor linked. Unclear whether a crash (no shutdown path) is covered; Goals promise crash recovery but CUJ 2 covers only quit. "Persisted run state written at dispatch" vs the store claim-write is ambiguous.

7. Major. Scope: Goals ("misfire semantics", "configurable staleness bound"). Misfire, the default bound, what age is measured from (enqueue?), and where the "recorded outcome" is shown are undefined. An idle-gated queue may routinely age out tasks; not discussed.

8. Major. Scope: Open Questions "Job cap"; Goals; Implementation item 4. Cap is "20 default, confirm whether in settings" yet item 4 already adds a cap setting. Counting scope (all states or active only) and behavior at the cap (error to the model?) unspecified.

9. Major. Scope: Implementation item 6 (tool ladder). "Ladder position", "low risk/medium", "always ask in read-only mode" and "existing cron tool treatment" assume an undocumented permission scheme. Tool names and parameters not given. "Model-origin marker" (CUJ 3) not specified.

10. Major. Scope: Links and references. ZCODE_COMPARISON.md ("Finding 1.2"), ZP-90 and ZP-02 are referenced but not summarized. "Excluded cloud category" and "roadmap's exclusion policy" justify rejecting Alternative A with no definition or link. "Cron parity target" unexplained.

11. Nit. Scope: Title and Overview; CUJ 1 example. "Off-peak" is used for idle-time work and "peak" is never defined. Unclear whether the CUJ 1 chat must contain the transcript.

12. Nit. Scope: Verification Plan; items 5 and 6 notes. Undefined jargon: "ten parity gates", `LocalizationParityTests`, "21 languages", `make design-check`, "CSyntext.o shim requirement", "catalog", "AppJSONStore quarantine rules", "basename". "System profiler power assertions" lacks a command. "Mutation checks" wording is cryptic.

13. Nit. Scope: Non-Functional Requirements "Power". The assertion-failure path does not say whether the task is still dispatched or what the user sees. Does the default-off toggle govern the model tools too?
