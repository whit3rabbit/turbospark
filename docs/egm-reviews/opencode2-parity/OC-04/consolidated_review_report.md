# Consolidated review report: OC-04 session tabs

Target: docs/designs/opencode2-parity/OC-04-session-tabs.md (full file, 192 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round, so convergence was not assessed. Stage 9 skipped by preset. All reviewers completed; none failed (two spawns were retried after "Concurrent subagent limit reached" and then succeeded).
Source IDs: ZO and IC = zoom_out_report (zoom-out, inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 4, Major 12, Nit 6.

## Blocker

B1. Whole file, 73-84. No Alternatives Considered; siblings ZP-21 and ZP-23 have it. Fix: add the section, first entry an open-chat id list plus MRU in the sidebar, then split-panes-only and drafts-on-`AppChat.draft`. [DC-1, ST-1, ZO-8, S8-M4]
B2. Whole file, 136-145. No Risks & Mitigations section, required by the standard. Fix: add a short section (plaintext drafts, deleted chats, ghost chats, Cmd+W change, tab growth). [DC-1, ST-2, S8 caveat]
B3. Lines 46-47, 114, 131-134. Separate persisted draft store bypasses the `makeChatArchive` ghost filter and sits beside the vault and profile model. Fix: ghost chats never appear in tabs, drafts or closed entries; reuse the archive builder; add a test. [DC-2, ZO-6, IC-5, S8-M1]
B4. Lines 56-59. Cmd+W is system-reserved in OC-01 yet claimed as close tab; last-tab, draft-only and Settings cases undefined (last close would quit the app). Fix: define Cmd+W as a fixed command, update OC-01's reserved list, state last-tab behavior. [IC-1, DC-6, ZO-5, ST-3, S8-M3]

## Major

M1. Lines 43-49, 88-94, 114-116, 124-127. Two sources of truth for drafts: `AppChat.draft` vs the tab store vs CUJ 3; `materializeDraftChatIfNeeded` and the single transient draft not reconciled. Fix: draft tab is a lazily materialized empty chat row; no second draft store. [ZO-1, IC-4, DC-3, CO-1, S8-M1]
M2. Lines 24-27, 53-55, 68-69, 81-83, 113-134. Ownership overlap with ZP-21's per-tab slice and with split panes (tab and pane for the same chat, promote, close). Fix: session tabs a field of ZP-21's slice; one tab per chat; panes exclude the selected tab's chat. [ZO-3, IC-3, IC-11, DC-7, DC-8, ST-4, CO-5, CO-8, S8-M2]
M3. Lines 24-27, 50-52, 129, 138-139. ZP-23 prerequisite not stated; behavior when switching or closing a generating or approval-pending tab undefined. Fix: ZP-23 hard prerequisite; interim refused switch with existing reason. [ZO-4, IC-7, DC-8, S8-M5]
M4. Lines 56-59, 165, 167-172. Option+Arrow collides with composer word navigation; catalog timing contradicts ("once both land" vs unconditional file 7); no reopen path stated. Fix: non-colliding cycle chords (confirm existing prev/next binding first), stable ids now, rebinding after OC-01. [IC-2, IC-14, DC-6, ST-3, CO-6, SY-7, S8-M3, S8-N2]
M5. Lines 113-129, 161-162. `selectedTabID` vs `selectedChatID` can disagree; restore order, shutdown flush, project mismatch unspecified. Fix: chat archive owns `selectedChatID`; restore after `loadChats` and `loadProjects`. Drops away largely if M1 is adopted. [IC-3, IC-6, DC-4]
M6. Lines 84, 113-116. Closed stack is "session-scoped" and "bounded 20" lean, yet stored in the persisted store. Fix: in-memory only, relaunch clears it. [IC-9, DC-9, CO-4, ZO-2, S8-M1]
M7. Lines 47-49, 88-94, 127-129. Promotion undefined: failure, atomicity, no-model and guarded states; "no flash" unmeasurable. Fix: one sentence: swap kind only after the chat row exists; define the test. [DC-10, CO-2, IC-13, SY-4]
M8. Lines 50-52, 98-102, 151-154. Reopen edge cases (position out of range, chat already open, chat deleted while open or closed) and silent drops. Fix: prune invalid ids at restore and reopen. [IC-10, CO-3, DC-9]
M9. Lines 52-54, 138-139. Per-tab scroll promised with no storage; "no reload from disk" NFR vacuous (all chats resident). Fix: scroll in-memory, not restored; delete the NFR. [IC-12, DC-5, ZO-7, CO-12, S8-N1]
M10. Lines 77-80, 119-121. Strip hides at one tab plus one draft so Cmd+T looks inert; "never truncate" conflicts with caps. Fix: show at two or more tabs; one line on overflow into the labeled menu. [IC-8, CO-7, DC-11, SY-6, CO-11, S8-N3]
M11. Lines 68-69, 73-84. Open Questions carry decisions the plan already commits to (placement, split layout, size 20); same status recorded twice. Fix: move decided items to Goals and Non-Goals. [ST-5, SY-5, CO-7, S8-N3]
M12. Lines 31-39. Unverifiable "ranks this gap fourth", "its report" without antecedent, "whole model is absent" overclaims given split panes. Fix: cite the section or soften. [SY-1, SY-2, CO-10, S8-N4]

## Nit

N1. Lines 3-10, 32-34. Swift links are repo-relative and do not resolve from the docs folder. Fix: `../../../swift/...`. [ZO-9, IC-15, DC-12, ST-8, S8-N4]
N2. Lines 4-10. Upstream links point to mutable `dev`. Fix: pin to 7ce35879b8. [ST-10, DC-12]
N3. Lines 175-188. Tests omit ghost, one-tab-per-chat, shutdown flush; scheme and DESIGN.md section 9 and localization gates not named. Fix: add one-tab-per-chat test, name scheme and link. Larger test expansion rejected by S8. [IC-16, DC-13, ST-7, SY-8, CO-13]
N4. Lines 147-179. File entries are Rationale-only unlike sibling template; item 5 silent on previous/next with a draft tab. Fix: one format; state draft behavior. [ST-6]
N5. Lines 1-10, 33-39, 120-121. Undefined project terms and draft-tab title rule. Fix: skip glossary, add one-liners only where needed. [CO-9, CO-11]
N6. Lines 20-22, 28, 56-59. Filler ("browser-style", "worth having"), "Cmd"/"option" capitalization, unsourced "macOS convention". Fix: tighten or drop. [SY-3, SY-7, ST-9]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-9, IC-1 to IC-16. Feeds B1, B3, B4, M1-M10, N1, N3.
- comprehension_report.md: CO-1 to CO-13. Feeds B4, M1-M4, M6-M12, N3, N5.
- critic_report.md: DC-1 to DC-13, ST-1 to ST-10, SY-1 to SY-8. Feeds all Blockers and Majors, N1-N6.
- intermediate_zoom_out_report.md: S8-M1 to M5, S8-N1 to N4. Confirms Stage 5 root causes; recommends rejecting scope-adding fixes (full risk expansion, schema versioning and migration, delete observer and toast, confirm-on-close, reopen chord table, tab cap, expanded tests, glossary). Adopting M1 through M5 removes about 20 findings.
- Caveats: S8-N2 says the claim that Cmd+Shift+[ and ] already cycle chats (ZO-5) was not found in the catalog; unverified. IC read only the first 136 of 521 lines of OPENCODE2_COMPARISON.md. S8 advised keeping Risks short; B2 stays a Blocker because the standard requires the section.
