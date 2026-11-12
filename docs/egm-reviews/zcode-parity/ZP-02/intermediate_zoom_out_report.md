# Stage 8: Intermediate zoom out report (ZP-02)

Reviewer: egm-zoom-out (completed). Target unchanged since Stage 5 (203 lines), so every Stage 5 finding remains open and no simplification has happened.

Code-search evidence: no chat-level revision, `lastReadAt`, `unreadAt` or `isUnread` exists (revision fields appear only on MemoryLedger, AppArtifact and audio types), so the "chat archive revision the organizer already reads" does not exist. `AppModel.swift:628` has `pendingToolCall` and `:635` has `pendingToolCallChatID`, so a per-chat identity for a pending approval already exists (a reuse point; DC-7 called it global).

S8-1 (Blocker) Lines 17-21, 38-40, 58-62, 85, 163-172. Countdown, snooze, auto-resolve and its settings (file items 4, 5) rest on an unresolved Open Question and default off, so the countdown does nothing. Empty payload on an approval is undefined and a privilege escalation if read as approve. Cut countdown, snooze, three settings fields and the AppModel+Questions change; ship the static pending pill; put auto-resolve in its own draft. Removes CO-1, CO-4, DC-2, DC-6, DC-10, SY-7, IC-7, IC-8.

S8-2 (Major) Lines 38-40, 110-114, 146-151. CAS is copied from a multi-client model; single process here. Use one persisted last-read watermark on AppChat (message count or `lastReadAt` versus `updatedAt`). Answers CO-3, DC-4, DC-8, IC-2. If CAS stays, name the real revision source and add it to the file list.

S8-3 (Major) Lines 94-108, 146-151. Drop ChatBadgeState.swift as a persisted model: one field on AppChat, priority slot as a small pure function in ChatSidebarOrganizer. Rename heading to "Proposed Architecture"; drop item 7 (a verification gate).

S8-4 (Major) Lines 36, 103-104, 154-155. Pending source is vague. Reuse `pendingToolCall` plus `pendingToolCallChatID` now; define a small read-only per-chat provider (running, openApproval, openQuestion, changed-since-read) that ZP-23 implements later, so ZP-02 is testable before it. The question-state source is not named in the doc.

S8-5 (Major) Lines 44-45, 47-54, 83-85, 113-114. Stale dependency text. State a hard ordering matching INDEX lines 78-80 and what ships if ZP-23 slips; remove "any future concurrent-chats world" and "exist for the concurrent future"; add real Non-Goals (error state, reordering, dock badge work); link ZP-90 and the desktop-shell-hardening brief.

S8-6 (Major) Lines 36-37, 88-92, 100-105. Pending > unread > running contradicts ZP-23 CUJ 1 (spinner while generating, dot at completion). Define unread as turn completed while unselected, or put running above unread. State error as a deliberate exclusion. Do not add an error state here.

S8-7 (Major) Lines 63-66, 69-76. Resolve the unread question now; recommended: reply completion while unselected, with a rationale beyond "matches chat conventions". Closes CO-2, DC-5, ZO-6, IC-5.

S8-8 (Major) Lines 116-130. Alternatives are strawmen. Replace with watermark versus CAS, static pill versus countdown pill, and a pointer to the split-out auto-resolve draft. (Stage 7 rated this Blocker; kept as Blocker in the consolidated report.)

S8-9 (Major) Lines 25-32, 134-136, 183-185, 195. Existing-behavior claims lack file or symbol cites (`rowIcon`, AppChat `updatedAt`); "21 languages" and "ten gates" uncited; persistence location and decode default for old chats (no unread after upgrade) missing. Two or three sentences, no migration tooling.

S8-10 (Nit) Lines 134-136. Name the non-color cue (dot shape plus accessibility label).

S8-11 (Nit) Lines 9-11, 17-21, 96-108. "State machine" without transitions, "sized for" filler, Status Draft beside "adopted".

S8-12 (Nit) Lines 173-180, 194. After simplification tests shrink to priority, watermark advance and re-raise, never-on-selected, pill for approval and question, decode default. Move the mutation note inside the item schema; justify `make design-check-full`.

S8-13 (Nit) Lines 121-123, 128-130. Fix "across generation sessions" to "across app relaunches" if Alternatives are kept.

## Earlier asks to reject as scope growth

DC-9 / ST-2 full Risks section (two risk bullets in NFR suffice after S8-1, S8-2); DC-6 snooze keyboard and VoiceOver path; DC-12 countdown, snooze and separate migration tests; CO-3 / DC-4 revision granularity and coalescing (artifact of CAS); DC-8 cleanup, branching and cron cases beyond one line; IC-4 spinner-plus-dot overlay redesign; IC-8 named timed-out result and any new auto-resolve semantics; DC-1 system-notification alternative; IC-1 phase-2 spec inside this doc; CO-8 defining repo jargon (link instead); SY-10 and SY-11 as separate passes.

Net: with S8-1 to S8-4 applied the file list drops from seven to four items (AppChat field, ChatSidebarOrganizer, ChatSidebarChatRowView, tests), with no settings and no question-lifecycle change.
