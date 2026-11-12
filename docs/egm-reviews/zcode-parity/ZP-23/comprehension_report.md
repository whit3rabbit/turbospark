# Stage 6: Comprehension report (ZP-23)

Reviewer: egm-comprehension. Persona read: agents/comprehension.md. Target read alone, lines 1-224, no standard.

## What this proposes

Let several chats generate at the same time. The user can leave a generating chat, the background chat keeps going, and its sidebar row shows state through "ZP-02" badge states. A new Swift component, ChatGenerationCoordinator, owns one "engine lease" (a single decode lane). Active chats' sessions step through that lane alternately (round-robin, foreground prioritized), up to a settings cap (default 2, hardware-aware upper bound). Global single-active guards become per-chat guards. Model switching is refused while any chat generates on the current model. Quit flushes every running chat into an interrupted row.

## How the existing system works, per this document

TurboSpark allows only one active chat. selectChat, createChat and selectPreviousChat refuse while a chat is generating, submitting or has a pending tool call. Sidebar rows are disabled with a message. Per-chat queues drain only when that chat is selected and idle. Subagents already serialize against foreground work. Shutdown already has a per-chat interrupted-row flow; ResidencyPolicy, expert cache caps and hardware-fit logic exist. The document does not say whether the engine can run two generations at once (Open Question).

## Findings

CO-1 Major, lines 66-74 vs 111-139. The engine lease is called "the core mechanism" while the Open Question says its feasibility is unknown. A reader cannot tell decision from contingency. State the decision and the conditional branch.

CO-2 Major, lines 3-9, 18-24, 85-86, 170. ZP-02 and ZP-21 are not summarized or linked; the dependency direction is confusing (line 21-22 says prerequisite for ZP-02, line 18-19 consumes ZP-02 states). Add one-line definitions and links.

CO-3 Major, lines 49, 118, 161-169. ResidencyPolicy, expert cache caps, hardware-fit logic, "KV per session" are undefined; "no new memory risk" is unexplained; the 16 GB-class rationale has no numbers.

CO-4 Major, lines 31-33, 56-61, 118 vs 120-125. Per-chat guard semantics are vague: what deleteChat on a generating chat does (cancel, refuse, wait) and what a third submit at the cap does (refuse, queue, error) are never stated; no CUJ covers the cap.

CO-5 Major, lines 96-102 vs 57-59, 123. CUJ 3 refusal reason "generating on this model" is always true when any chat generates (all chats share the model). It also conflates today's chat-switch guard with a model-switch guard.

CO-6 Major, lines 51-53, 104-109, 165-166. "Ordered flush", "atomic per chat", "interrupted row", "Continue/Retry" are undefined; AppShutdownCoordinator is named but the file list shows AppModel+Persistence only.

CO-7 Major, lines 45-47 vs 89-94. "Steering", "turn boundary" and queue model undefined; no CUJ covers a background queue draining; whether a draining queue counts against the cap is unstated; cron dispatch (lines 33-34) is not tied to a goal or test.

CO-8 Major, lines 135-139. Whether a subagent counts toward the cap (one chat plus subagent consuming it all) is not stated; "already serialize" is asserted without explanation.

CO-9 Nit, lines 143-150, 220. "No-unmeasured-gates rule" and "bench gates" undefined and unlinked; Alternative A's "per-step overhead only" has no evidence.

CO-10 Nit, line 165. "Sampling owns normalization" and "no two chats share a session" are unconnected to any mechanism.

CO-11 Nit, lines 63-64. "Workspace tab" undefined; the non-goal cannot be interpreted.

CO-12 Nit, lines 118, 177-209. MaxConcurrentGenerations name, UI location and persisted key not given; "settings field" vs "settings-visible"; file-list Type/Scope fields too terse.

CO-13 Nit, lines 170-171, 215-220. "21 languages" not located; manual hardware step has no pass criteria beyond "interactive".

No Blockers: every section could be paraphrased.
