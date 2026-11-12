# Stage 6: comprehension report (ZP-04)

Target: docs/designs/zcode-parity/ZP-04-subagent-coordinator-messaging.md (full file, no standard, no Goldfish context)
Reviewer: egm-comprehension. Completed.

## What this proposes

TurboSpark subagents today are one-shot: they only report through a final result and a completion `<task-notification>`. The document proposes two new tools. `RespondToCoordinator` is available only to a child and lets it send a message mid-task while it keeps running. `SendMessage` lets the coordinator continue a finished child with its prior context. A new component, `SubagentMessageHub`, queues child messages per child and delivers them into the coordinator's next model step as attributed `<subagent-message>` carriers. A resume path in `SubagentRunner` re-enters a finished child's loop. Whether the context for resuming is kept in memory or persisted to disk is left as an open question.

## How the existing system works, per this document

A coordinator launches a child through `SubagentRunner`, which runs the child in an isolated session with tools declared in its agent file's frontmatter. It enforces depth caps and a "deny-on-ask" permission gate. The child's result comes back as a `<task-notification>`, which is injected into the coordinator's prompt at some injection point. A "closure-bound port pattern" is used in the launch flow, and the coordinator's tool "capture" is where its tool set is defined. The app's "steering pills" deliver input at turn boundaries. The permission system has a "ladder" with risk levels and modes. The document does not explain any of these in detail. It also says "the original launch result stays intact in the transcript" without describing the transcript.

## Findings

CO-1 Blocker. Lines 63-67, 114-117, 42-44. The core mechanism of `SendMessage` cannot be paraphrased as one design. Open Questions give options (a) retained and (b) persisted and say "The author decides". Goals say the child "resumes from its own persisted transcript" (option b), the file plan says "retained context" (option a). The meaning of "original launch result stays intact in the transcript" (lines 89-90) is unclear in either.

CO-2 Major. Lines 50-55. The non-goal is stated twice; "Non-Go:" at line 50 is a leftover and an undefined term.

CO-3 Major. Lines 8-12, 21-23. "SubagentRunner", "kiro brief", "the newer audit" and finding 3.5 are referenced but only the comparison file is linked. The contradiction the document says it resolves cannot be understood, and how it is resolved is not stated. "Kiro" is undefined.

CO-4 Major. Lines 40-41, 106-107, 146-147. Carrier naming is inconsistent: "`<subagent-message>`-equivalent", `<subagent-message name="...">`, and "child agent type and launch id". The attribute set is never specified. "Launch id" is undefined.

CO-5 Major. Lines 105-108, 168-171, 45-46. "Same injection point as task-notifications" and "next model step boundary" are used as if known. Behavior when the coordinator is idle or finished is not stated (line 45-46 covers only "busy"). Line 167 leaves the target file undecided ("SystemReminders.swift or the prompt assembly extension").

CO-6 Major. Lines 140-142, 164-166. "Ladder", "allow in all modes except read-only" and "medium risk" are project-specific and undefined. Why a message tool is blocked only in read-only mode, and why SendMessage is medium risk, are unexplained. "The existing gate" (line 141) is ambiguous with "deny-on-ask gate". What happens when a deny-on-ask child calls RespondToCoordinator is unstated.

CO-7 Major. Lines 114-116, 49-50, 165-166. "Closure-bound port pattern", "the coordinator's capture" and "the launch flow" are undefined. How child-only binding is achieved is unclear.

CO-8 Major. Lines 153-157, 98-103. The file plan says "inbound/outbound queues" but the diagram shows one per-child mailbox for child-to-coordinator messages. What the outbound queue carries is unstated, and SendMessage goes straight to the runner in the diagram, not through the hub.

CO-9 Major. Lines 121-129, 138-148. Missing behavior definitions: the identifier SendMessage accepts (name or launch id); SendMessage to a running, unknown or discarded child; whether the coordinator can reply to a mid-task message. The name RespondToCoordinator suggests a reply semantic but the tool is one-way.

CO-10 Major. Lines 68-69, 143-145. 4 KiB and depth 8 are only "suggest" values. Behavior for an oversize message (reject, truncate, error) is unstated. Oldest-drop is visible only in a log event.

CO-11 Nit. Lines 163-164. Heading reads "Type / Spawn-tool schema" where others use "Type / Scope"; "Proposed Changes" label missing.

CO-12 Nit. Lines 56-58. "cross-session-context spec" and "cross-session mailbox adoption" are unlinked and undefined.

CO-13 Nit. Lines 13-35. "ZCode" is introduced only as a link target; "coordinator" is never defined.

CO-14 Nit. Lines 127, 135. Alternatives are "rejected by author" with reasons that restate precedent. "Steering pills" is an undefined UI term.

CO-15 Nit. Lines 172-185. Verification names `make swift-lib` and "the app scheme" without location or scheme name; "mutation checks" is undefined.
