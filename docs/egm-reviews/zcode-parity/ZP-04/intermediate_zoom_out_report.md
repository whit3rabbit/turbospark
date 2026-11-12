# Stage 8: intermediate zoom out report (ZP-04)

Target: docs/designs/zcode-parity/ZP-04-subagent-coordinator-messaging.md (full file, 193 lines)
Reviewer: egm-zoom-out (S8). Completed. Inputs: zoom_out_report.md, comprehension_report.md, critic_report.md.

Verdict: still a draft at the wrong altitude in two directions. It names a new component without checking what exists, and it leaves its core decision open while specifying a file plan. Shrink first, add mitigations second. Reuse re-checked: `SubagentRunner.run` takes `priorHistory` (State/SubagentRunner.swift:292) and returns `transcript`; `pendingTaskNotifications` exists (AppModel+Generation.swift:541, AppModel+Queue.swift:22).

## Blocker

S8-1. Lines 61-67, 36-47, 110-117, 158-162. Resume persistence unmade while Goals, CUJ 2 and plan assume different answers. Fix: decide in-memory (option a), keep the transcript with the existing background-run entry, prune with it, defer (b). Do not adopt IC-1 (durable log plus warm cache): two layers with no shown need, overlaps ZP-03.

## Major

S8-2. Lines 100-103, 153-157. `SubagentMessageHub` is a second queue beside `pendingTaskNotifications`; risks out-of-order delivery; Critic mitigations would grow it into a mailbox subsystem. Fix: add a message case to the existing queue, reuse drain and escaping, keep a small per-child counter.

S8-3. Lines 17, 36-47, 163-166. Two features bundled. A foreground child blocks its coordinator, so next-step delivery happens only after the child ends and CUJ 1 fails. Fix: ship SendMessage first or split into phases; scope RespondToCoordinator to background children or state foreground behavior.

S8-4. Lines 167-171. Injection seam hedged and wrongly described. Fix: name the AppModel turn-tail drain, drop SystemReminders; this also answers idle-coordinator delivery (DC-2), so reject IC-2 as a separate rate-limited wake feature unless the drain proves insufficient.

S8-5. Lines 143-145, 68-69. Suggested bounds in two places, silent loss. Fix: fixed values in NFRs only, truncate oversize with a marker, return "mailbox full" to the child; reject coalescing and aggregate caps for now.

S8-6. Lines 138-148. Prompt injection treated as labeling. Fix (justified, cheap): escape carrier tags with the existing escaping, state content is untrusted data, add one test; no trust model.

S8-7. Lines 121-136. Alternatives weak; "Rejected by author" (ST-1) applies. Fix: replace with spawn-fresh-child-with-prior-result and retain-versus-persist.

S8-8. Lines 140-142, 163-166. Ladder positions inconsistent, not in ladder vocabulary. Fix: one table (tool, tier, mode); allow RespondToCoordinator in read-only; resumed child inherits agent-file tools and the deny-on-ask gate and counts as the same run for depth.

S8-9. Lines 42-44, 86-90. SendMessage identifier and state cases unspecified. Fix: running child returns an error to wait; unknown or pruned id returns not-found; no hub state machine (reject IC-4).

## Nit

S8-10. Line 50. Stray "Non-Go". Delete.
S8-11. Lines 163-165. Malformed label; unassigned RespondToCoordinator file; AgentTools.swift already holds `stop_agent`.
S8-12. Lines 40-41, 106, 146-147. Carrier naming and "step boundary" drift. Define once.
S8-13. Lines 3-7, 17-23. "Resolves the contradiction" overstated; link kiro brief; note ZP-03 and ZP-23 overlap.
S8-14. Lines 94-112. Diagram arrow for SendMessage wrong.
S8-15. Lines 172-185. Trim mutation detail; add tests for final-result delivery, escaping, oversize.

Scope guidance. Add: persistence decision, real alternatives, escaping, permission table, SendMessage edge rules, fixed bounds with visible drop signal, named seam, a short Risks list. Do not add: hub state machine, rate-limited wake, aggregate caps and coalescing, warm cache over durable storage, a large Risks section.
