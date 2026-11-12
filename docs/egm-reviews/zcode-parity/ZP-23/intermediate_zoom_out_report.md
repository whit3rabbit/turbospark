# Stage 8: Intermediate zoom out report (ZP-23)

Reviewer: egm-zoom-out. Persona read: agents/zoom_out.md; standard zoom-out-standard.md. Document unchanged since Stages 5-7, so no earlier finding is mitigated. Source IDs: S8.

S8-B1 Blocker, lines 66-74 vs 111-139. Committed design rests on an unverified premise (can the FFI step two sessions on one resident model). Make a phase-0 spike with a pass/fail criterion and mark the coordinator contingent. Resolves ZO-2, CO-1, DC-B2, SY-5, ST-3.
S8-M1 Major, lines 111-139, 175-193. Name a per-chat turn-state holder as the structural change; add a blast-radius note and touch points (SplitChatPanes, AppModel+SplitView, projects). Do not enumerate the 25 guard sites.
S8-M2 Major, lines 48-50, 75-77, 161-164. For v1: constant cap of 2, no settings field, no KV-byte admission. Define behavior at the cap (refuse or queue visibly).
S8-M3 Major, lines 89-109. Policy line: background chats do not auto-run tools; a chat reaching awaitingTool parks and releases the lease and KV. Only `generating` counts against the cap. Add the ladder-position statement INDEX requires. A write-lease subsystem belongs to ZP-21.
S8-M4 Major, lines 135-139, 167-169. State one scheduling policy: strict foreground priority, background advances when foreground is idle. If the spike shows switch cost, use chunk-boundary switching.
S8-M5 Major, lines 141-157. Add the turn-level lease as an alternative and spike fallback; replace "rejected by author" with technical reasons; one line each for engine-per-chat and KV swap.
S8-M6 Major, lines 104-109, 165-166. Reuse the existing interrupted-row flow; state crash reconciliation is out of scope or name the existing launch path.
S8-M7 Major, lines 3-9, 18-24. One line on ZP-23 providing state and ZP-02 rendering it; add links.
S8-N1 Nit, lines 96-102. CUJ 3 string always true when any chat generates; reframe, name chats, mark placeholder.
S8-N2 Nit, lines 173-220. Collapse test matrix and mutation checks.
S8-N3 Nit, lines 28-35. Route cron dispatch through the same cap in one sentence or remove.
S8-N4 Nit, lines 135-139. One slot per top-level chat, subagents inside it.
S8-N5 Nit, lines 18-24, 66-77, 161-171. Mark unmeasured claims as hypotheses or cite crates/runtime.
S8-N6 Nit, lines 159-171. Convert into a short Risks and Mitigations section.
S8-N7 Nit. Carve-out for the "Different" classification versus INDEX "Missing" framing.

Scope growth to reject: KV-byte admission and waiting queue (IC-3), weighted scheduler (IC-1), write-lease subsystem (IC-5), model pinning/deferred switch (IC-6), new in-flight marker and reconciler (IC-7), idle KV swap/LRU (DC-M6), rollout/metrics apparatus (DC-N5), hardware-aware settings bound.
