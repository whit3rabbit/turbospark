# Consolidated review report: ZP-02 unread and pending badges

Target: docs/designs/zcode-parity/ZP-02-unread-pending-badges.md (full file, 202 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed. A stage 5 spawn of the comprehension reviewer hit "Concurrent subagent limit reached" once and succeeded on retry.
Source IDs: ZO = zoom_out_report (zoom-out), IC = zoom_out_report (inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 2, Major 11, Nit 11.

## Blocker

B1. Lines 116-130. Alternatives are "Rejected by author" strawmen, with no concrete trade-off. Fix: replace with real competitors (last-read watermark versus CAS, static pill versus countdown pill, pointer to a separate auto-resolve draft) with cost-based rejections. [DC-1, ST-1, S8-8, ZO-8, SY-9]
B2. Lines 17-19, 41-42, 58-62, 85, 163-172. Countdown, snooze and auto-resolve rest on an unresolved Open Question and default off; auto-resolving an approval with an empty payload is undefined and an escalation risk through the permission ladder. Fix: cut countdown, snooze, settings and the AppModel+Questions change; ship the static pill; split auto-resolve into its own draft. [DC-2, S8-1, ZO-2, IC-1, CO-1, CO-4, SY-1, ST-3, IC-8]

## Major

M1. Lines 38-40, 110-114, 146-155. CAS has no real revision source (none exists on chats), is built for multi-writer clients, and may re-raise on a streaming selected chat. Fix: one persisted last-read watermark on AppChat; unread = content past the marker. [ZO-1, IC-2, IC-6, CO-3, DC-4, S8-2]
M2. Lines 94-108, 142-186. New ChatBadgeState.swift mixes derived and persisted state; storage layer, decode default for old chats and file-list gaps unstated. Fix: one AppChat field, slot as a pure function in ChatSidebarOrganizer, name storage and default (no unread after upgrade). [ZO-3, DC-8, ST-8, S8-3, S8-9]
M3. Lines 44-45, 47-54, 83-85, 113-114, 24-32. Stale and hidden ZP-23 dependency; ZP-02 is inert before ZP-23. Fix: state hard ordering per INDEX and what ships if it slips; remove "future concurrent world" wording. [ZO-4, IC-3, CO-6, DC-7, DC-14, ST-4, ST-5, SY-3, SY-4, S8-5]
M4. Lines 36-38, 87-92, 100-105. Priority contradicts ZP-23 CUJ 1 (streaming background chat shows dot, not spinner); spinner scoped to selected row; error state dropped silently. Fix: define unread at turn completion or put running above unread, define background spinner, record error exclusion. [ZO-5, IC-4, DC-3, ST-6, CO-5, SY-11, S8-6]
M5. Lines 63-76. Unread definition left open while CUJ 1 picks the broad one; tool-only churn. Fix: choose reply completion while unselected and align CUJ and tests. [ZO-6, IC-5, CO-2, DC-5, SY-2, S8-7]
M6. Lines 36, 103-104, 154-155. Pending source vague; `pendingToolCall` and `pendingToolCallChatID` (AppModel.swift:628, 635) exist; question source unnamed; pill scope for approvals versus questions and a selected chat unclear. Fix: reuse them behind a small read-only per-chat provider. [DC-7, S8-4, ST-7, CO-10]
M7. Lines 17, 41-42, 83-86, 160. Click is both navigate and snooze; snooze unit and persistence undefined. Fix: resolved by B2 (drop snooze), otherwise specify the gesture split. [CO-4, DC-6, SY-10]
M8. Whole document. No Risks section. Fix: two NFR risk bullets (old-chat default, many rows lighting at once), not a full section, once B2 and M1 are applied. [DC-9, ST-2] (S8 judged a full section scope growth.)
M9. Lines 136-139. "No polling" conflicts with the countdown timer. Fix: disappears with B2; otherwise state one shared tick on visible pill rows. [DC-10, SY-7, IC-7]
M10. Lines 30, 44, 127. Dock badge spec unlinked and not found under this worktree's `.kiro/specs`; owner of the shared count unnamed. Fix: link the brief per INDEX, say who owns the count, or drop the goal. [CO-7, ST-9, ZO-4, DC-11, SY-4]
M11. Lines 25-33, 134-136, 183-185, 195. Claims about existing behavior and "21 languages" and "ten gates" lack file, symbol or source. Fix: cite `rowIcon`, AppChat `updatedAt` and the gate source once. [SY-5, SY-6, CO-8, DC-13, S8-9]

## Nit

N1. Lines 134-136. Non-color cue unnamed. Fix: dot shape plus accessibility label. [CO-9, S8-10]
N2. Lines 9-11, 17-21. "State machine" without transitions, "sized for" filler, Draft versus "adopted". Fix: reword to "priority slot". [SY-8, S8-11, DC-14]
N3. Lines 121-130. Vague pros and "across generation sessions". Fix: "across app relaunches". [SY-9, CO-11, ST-12, S8-13]
N4. Lines 83-92. "Navigation no longer stops anything", "interval" versus "extension", "yields". Fix: one term, plain wording. [SY-10]
N5. Line 94. Heading differs from standard. Fix: "Proposed Architecture". [ST-10]
N6. Lines 179-180. Mutation checks outside item schema and unexplained. Fix: nest and explain in one clause. [ST-11, CO-12]
N7. Lines 173-180, 194. Tests omit decode default and re-raise on selected-and-streaming; `design-check-full` unjustified. Fix: align tests with the simplified design. [DC-12, S8-12, ZO-7]
N8. Lines 47-54. Non-Goals thin; one bullet is a dependency. Fix: list error state, reordering, dock work. [DC-11, ST-4]
N9. Lines 96-108. Pseudo-code block mixes types and logic. Fix: describe data flow only. [CO-10, ZO-7]
N10. Line 181. Item 7 "Unchanged, must pass" is a gate, not a change. Fix: move to Verification Plan. [ST-8]
N11. Line 94 and sibling set. Heading pattern differs from ZP-23. Fix: align. [ST-10]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-8, IC-1 to IC-8. Feeds B1, B2, M1-M5, M9, M10, N7, N9.
- comprehension_report.md: CO-1 to CO-12. Feeds B2, M1, M3-M7, M10, M11, N1, N3, N6, N9.
- critic_report.md: DC-1 to DC-14, ST-1 to ST-12, SY-1 to SY-11. Feeds all Blockers and Majors, N2-N11.
- intermediate_zoom_out_report.md: S8-1 to S8-13. Confirms all Stage 5 findings open; recommends cutting countdown, snooze, settings and the new model file (file list from seven items to four) and rejecting DC-9 full section, IC-4 overlay redesign, IC-8 new semantics, DC-1 system-notification alternative and DC-6 snooze accessibility path as scope growth.
- Merge notes: DC-1, ST-1 and S8-8 merged into B1 (Stage 8 rated Major, Stage 7 Blocker; kept Blocker). DC-7 merged into M6 and M3. DC-2 and S8-1 merged into B2.
