# Stage 5: initial zoom out report (ZP-04)

Target: docs/designs/zcode-parity/ZP-04-subagent-coordinator-messaging.md (full file, 193 lines)
Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed. No Blockers raised.

Reuse evidence found by ZO: `SubagentRunner.run` already takes `priorHistory` and returns `transcript` (State/SubagentRunner.swift); a per-chat parked queue `pendingTaskNotifications` exists (State/AppModel.swift:746-749) drained at the turn tail (State/AppModel+Queue.swift:130), with tag escaping in State/AppModel+Subagents.swift. Searches for `SendMessage`, `RespondToCoordinator`, `SubagentMessageHub` found nothing, so finding 3.5 "absent" holds.

## Zoom-out findings

ZO-1 Major. Lines 100-103, 153-157. New `SubagentMessageHub` reinvents the existing pending-notification queue; a second queue risks out-of-order delivery against completion notifications; "outbound" queues are never explained. Fix: add a `subagent-message` case to the existing queue and reuse its drain and escaping.

ZO-2 Major. Lines 167-171. Carrier injection is placed in `Generation/SystemReminders.swift` "or the prompt assembly extension" and called "the same injection point as task-notifications". Task-notifications are injected from AppModel.swift, AppModel+Queue.swift and AppModel+Subagents.swift. Fix: name the real seam, drop SystemReminders.

ZO-3 Major. Lines 61-67, 110-117, 158-162. Resume persistence is left open though `priorHistory` plus the returned `transcript` already provide option (a); option (b) is premature. Fix: decide (a), keep the transcript with the `backgroundAgentRuns` entry, bound retention by pruning, defer (b); a new resume path may be unnecessary.

ZO-4 Major. Lines 17, 36-47, 163-166. Two features with different risk are bundled. Edge cases are unspecified: SendMessage to a running, killed, errored or pruned child; a foreground child whose coordinator is blocked, so "next model step" delivers only after the child ends and defeats CUJ 1. Fix: scope RespondToCoordinator to background children or specify foreground behavior; consider two phases, SendMessage first.

ZO-5 Nit. Lines 50, 52-59. Stray "Non-Go" duplicates Non-Goals. Fix: delete line 50.

ZO-6 Nit. Lines 163-165. "Type / Spawn-tool schema" is a template mixup; AgentTools.swift already holds `stop_agent` beside spawn. Fix: use "Type / Scope" and state the pattern.

ZO-7 Nit. Lines 68-69, 143-145, 178-180. 4 KiB and depth-8 appear as both Open Question and NFR; oldest-drop loses the earliest message silently; mutation-check detail is low altitude. Fix: pick one place, surface a drop marker, trim.

## Inventive catalyst findings

IC-1 Major. Lines 63-67, 114-117, 43. Resume persistence framed as either/or; Goals say "persisted transcript", plan says "retained or persisted". Fix: persist as the source of truth with a bounded warm cache over it and a retention rule (note ZO-3 argues for the simpler in-memory option).

IC-2 Major. Lines 45-46, 121-128. Only step-boundary delivery; no handling when the coordinator is idle or the message is urgent, so a message can wait indefinitely. Fix: let the hub request a wake for an idle coordinator, keep boundary injection when running, rate-limit wakes.

IC-3 Major. Lines 143-145, 179-180. Oldest-drop on cap 8 silently discards critical messages and gives the child no stop signal. Fix: return a "mailbox full" result to the child, coalesce, keep the first message verbatim, hard-truncate at 4 KiB with a marker.

IC-4 Major. Lines 42-44, 158-162, 50-57. SendMessage to a running or concurrently resuming child is unspecified; whether a resume counts against depth caps and concurrency is unclear. Fix: hub owns per-child state (running, finished, resuming) with a serial inbox; SendMessage to a running child enqueues for its next turn boundary.

IC-5 Major. Lines 146-148, 140. Carrier injection is a prompt-injection boundary treated only as a labeling problem; child text may carry untrusted content into the more privileged coordinator. Fix: escape carrier-like tags, mark content as untrusted data, add a test.

IC-6 Nit. Lines 50, 52-59, 164, and file list. Duplicate Non-Go text; no file listed for the RespondToCoordinator tool definition and executor. Fix: remove line 50, list the file.

IC-7 Nit. Lines 139-142, 165-166, 68. Read-only exclusion for RespondToCoordinator looks backwards (it writes nothing to the workspace); SendMessage "medium risk" ignores that resume re-grants tool use; 4 KiB and 8 are only "suggested", so tests cannot assert them. Fix: justify or allow in read-only, state whether resume re-prompts or inherits the gate, fix the values.

IC-8 Nit. Lines 17-23, INDEX sequencing item 4. Overview claims to "resolve" a factual contradiction; should record the code search and mark the kiro brief for correction. Persistence overlap with ZP-03 rollout logging not noted. Fix: reword, note dependency.
