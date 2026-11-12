# Stage 5: Initial zoom out report (ZP-23)

Reviewers: egm-zoom-out (persona read: agents/zoom_out.md, standard zoom-out-standard.md), egm-inventive-catalyst (persona read: agents/inventive_catalyst.md). Both completed.

## Zoom-out (ZO)

Verdict: not ready to implement as written. Context is sufficient; root cause, blast radius and reuse claims are weak.

ZO-1 Major, lines 111-139, 175-193. Single-active behavior comes from per-turn singletons on AppModel (`generating`, `submitting`, `pendingToolCall`; about 25 guard sites across at least 10 files: +Chat, +Projects, +SlashParity, +Tools, +Memory, +BangCommands, +History, +Agents, +SplitView, BrowserPermissionApprovalCoordinator). The plan lists only AppModel+Chat and a few files. Without moving per-turn state into a per-chat object, the coordinator sits on top of the same singleton. Name the per-chat turn-state holder as the structural change and say which guards stay global.

ZO-2 Major, lines 66-74, 113-139. The decode-lane question is a gating unknown, yet the plan commits to a Swift round-robin scheduler as "the core mechanism". Per-session KV and sampler state surviving interleaved steps is asserted (line 163), not shown. Promote engine-concurrency verification to a phase-0 gate, make the plan conditional or split it. Cheaper option: turn-level lease, but then "background chat keeps generating" is false.

ZO-3 Major, lines 49-50, 75-77, 161-164. `ResidencyPolicy` does not exist in Swift (only `ModuleResidencyPolicy` in the ZImage pipeline). "Hardware-fit logic" and "expert cache caps" have no named types. Existing primitives (`WorkflowRunEngine` actor, subagent serialization) were not considered.

ZO-4 Major, lines 56-64, 175-193. Missing touch points: `SplitChatPanes.swift` (line 85) and `AppModel+SplitView.swift` (line 78) carry the same blocked-switch text and guard. Split view, projects (three guards), title-generation foreground-idle rule and cron dispatch into a non-selected chat (line 34) have no section. Removing the overlay string is a 21-locale catalog removal.

ZO-5 Major, lines 89-109. Concurrent approvals mean tool execution in non-selected chats. Shared mutable surfaces (workspace files, background shells, browser automation coordinator, permission prompt UI, `AppToolPermissionEngine` state) are not addressed; two chats editing the same files is a data-loss path. INDEX requires a ladder position statement; absent. Shutdown ordering when two flushes share the lease is not described.

ZO-6 Nit, lines 175-209. Test matrix and mutation checks are over-detailed for a design.

ZO-7 Nit, lines 75-77, 194-197. "2 is safe on 16 GB-class machines" is unmeasured; a settings field plus hardware-aware bound may be YAGNI for v1.

ZO-8 Nit. CUJ 3 refusal reason is a literal string while line 59 says "named reason"; Alternative B omits the real alternative (turn-level lease); INDEX says drafts were written against `feat/classlru-replay-arm`, guard line numbers confirmed on main only.

Holds up: problem statement is accurate; dropping multi-model and engine batching is the right cut; ZP-02/ZP-21 sequencing is coherent.

## Inventive catalyst (IC)

IC-1 Major, lines 71-74, 135-139, 167-169. False compromise: "round-robin with foreground priority" cannot both let background chats progress and not steal the interactive budget. Strict priority starves background; fair round-robin halves foreground cadence. Define a weighted schedule with a foreground floor and a background minimum quota, or throttle background on foreground inter-token latency.

IC-2 Major, lines 66-74, 135. Per-step session switching cost (KV, sampler, Metal command scheduling) is unmeasured, and Alternative A's rejection rests on the same premise. Add a spike gate with pre-written thresholds, or switch at chunk boundaries (tens of tokens).

IC-3 Major, lines 48-49, 75-77, 161-164. A fixed count stands in for KV memory; "without new memory risk" is asserted. Behavior of a third chat at the cap (queue, refuse, evict) and memory pressure mid-run is undefined. Admit on KV bytes with count as ceiling; add a bounded admission queue with a visible waiting state.

IC-4 Major, lines 89-95, 118. Unstated whether awaitingTool/awaitingQuestion chats count against the cap or hold KV. Either two idle approvals block all generation or memory is oversubscribed. State that only `generating` counts; release lease and KV on awaiting states and re-prefill on resume.

IC-5 Major, lines 51-53, 63-64. Two running chats in one project can edit the same files or race on git state; ZP-21 does not cover it. Add a per-workspace write lease or conflict check, bounded against approval waits.

IC-6 Major, lines 55-59, 96-102. Model switch refused while any chat generates on the model gives no recourse. Offer "stop A and switch", a deferred switch at drain, or per-chat model pinning.

IC-7 Major, lines 51-53, 104-109, 191-193. Crash, force-quit or OOM with two live generations leaves half-written chats; "atomic per chat" has no mechanism. Add a per-chat in-flight marker and a launch-time reconciliation to interrupted rows.

IC-8 Nit, lines 136-139. Do a parent chat and its subagents count as one slot or several? Say one slot per top-level chat.

IC-9 Nit, lines 31-35. Cron/off-peak dispatch into a non-selected chat is motivation with no goal, CUJ or test; route through the same admission path.

IC-10 Nit, line 75. 16 GB-class cap claim is unmeasured, conflicts with lines 219-220; reword as hypothesis.

Not false compromises: GPU decode serialization, single resident model, rejection of Alternative B.
