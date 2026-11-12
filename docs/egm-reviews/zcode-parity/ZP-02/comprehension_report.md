# Stage 6: Comprehension report (ZP-02)

Reviewer: egm-comprehension (completed). Read the document alone.

## What this proposes

Chat sidebar rows get a single status slot with four states in priority order: pending-interaction pill (open approval card or open question), unread dot, running spinner, default icon. Unread is a persisted per-chat flag, cleared by a compare-and-set "markRead" that only succeeds if the chat's revision still matches what the reader saw, so new content landing in between re-raises the dot. The pending pill shows a countdown fill tied to an "auto-resolve window", and clicking it snoozes (extends the window once). What happens when the countdown ends is left as an open question. The work touches a new badge-state file, the sidebar organizer, the row view, settings, the question lifecycle, and tests, plus localization in 21 languages.

## How the existing system works, per this document

The sidebar has no unread concept. The only row cue is a progress flame icon on the selected, running chat. Pending tool approvals appear only as cards inside the transcript. Only the selected chat generates today, so concurrent-write races happen only at chat-switch boundaries. A separate spec, "desktop-shell-hardening", already covers the dock badge. The chat archive has a revision the organizer already reads. A view named ChatSidebarChatRowView renders rows, and the sidebar organizer feeds it state. Questions have a dismissal path that the transcript, hooks and task-notifications all consume.

## Findings

CO-1 (Major) Lines 58-62, 41-42, 83-85, 167-172. The auto-resolve window is the core of the countdown, yet its meaning is undecided and the doc contradicts itself. Goals state the countdown as settled; Open Questions say the app has no such semantic. Unclear: what the countdown ends in, the default window length, whether it applies to approvals or only questions. "Empty answers payload" is unexplained and meaningless for an approval card (approve or deny?). Line 169 "same dismissal path as the user declining" fits only questions.

CO-2 (Major) Lines 63-66 vs 69-76. Unread definition is open, but CUJ 1 ("generation appends content") silently picks the broader option. Nothing says whether a pending state leaves an unread flag behind after it clears.

CO-3 (Major) Lines 110-114, 150-155. The revision token is never defined: what it is, who increments it, whether it bumps on every content change, where the observed revision is captured (selection or render time), and where the flag and stored revision persist. Lines 108 and 148-149 describe only a flag, so the data model is incomplete.

CO-4 (Major) Lines 17-18, 41-42, 85-86, 160, 164. Snooze is underspecified: "interval" undefined (window length or a separate snooze extension setting), unclear whether approvals can be snoozed, what the pill shows after use, and click is both navigate (CUJ 2) and snooze (line 17).

CO-5 (Major) Lines 87-92, 36-37, 76. CUJ 3 says the spinner shows on the selected row, but background chats also run under ZP-23; the Technical Plan says "running: chat is generating". The "yields to unread" statement is confusing since unread never shows on the selected row.

CO-6 (Major) Lines 51-54, 83-84, 113-114, 44-45, 196-197. Inconsistent about whether concurrent chats exist (adopted versus "today only the selected chat generates" versus "future"). ZP-90, "finding 2.1", "finding 1.4" and ZCODE_COMPARISON.md are referenced as known; only ZP-23 is linked.

CO-7 (Major) Lines 29-32, 44-45. Dock badge relationship is vague: a goal feeds another spec's badge with no link, interface or target file in this plan; owner of the connection unclear.

CO-8 (Nit) Lines 63, 160, 164, 184-185, 195. Undefined project terms: desktop-shell-hardening spec, token ratchet, DESIGN.md, "ten gates" (not listed), MacAppSettings, task-notifications, hooks, Reduce Motion rules.

CO-9 (Nit) Lines 134-136. "Never color alone" without naming the non-color cue for the dot and the pill.

CO-10 (Nit) Lines 96-108, 154. ChatBadgeState block mixes type description with pseudo-code; unclear whether "pending" is computed or persisted; "pending state from the app model" is not described elsewhere.

CO-11 (Nit) Lines 121-123, 128-130. Alternatives rejected "by author" with little reasoning; A's "across generation sessions" means app relaunches.

CO-12 (Nit) Lines 179-180. "Mutation checks" is terse; reader may not know it means manually inverting code to prove tests catch it.
