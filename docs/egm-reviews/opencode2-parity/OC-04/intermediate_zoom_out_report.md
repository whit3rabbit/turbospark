# Stage 8: Intermediate zoom out report (OC-04 session tabs)

Reviewer: egm-zoom-out on the full file with zoom_out_report.md, comprehension_report.md and critic_report.md as input. Completed. Source IDs below use S8-.

Verdict: not simple enough yet. Stage 5 root-cause findings stand. Most Stage 6 and 7 findings are the same problems seen from other angles, and many proposed fixes add scope.

## Findings that stand

S8-M1 Major. Draft tabs and the closed stack are a second store over data that already exists (`AppChat.draft`, single transient draft, split-pane symbols confirmed present). It is the root of the ghost bypass, two sources of truth for drafts and selection, the persisted-vs-session contradiction, and reopen edge cases (collapses ZO-1, ZO-2, IC-4, CO-1, DC-3, DC-9, CO-4). Fix: open tabs are an ordered list of chat ids, a draft tab is a lazily materialized empty chat row, Recently Closed is an in-memory id list reopened via the existing select path.
S8-M2 Major. Ownership with ZP-21, ZP-23 and split panes undefined (ZO-3, IC-3, IC-11, DC-7, DC-8, ST-4, CO-5, CO-8). Fix: one sentence each: session tabs are a field of ZP-21's per-tab slice; a chat is in at most one tab; panes stay window-level and exclude the selected tab's chat.
S8-M3 Major. Chords (ZO-5, IC-1, IC-2, DC-6, ST-3, CO-6). Cmd+W reserved in OC-01; Option+Arrow collides with the composer. Fix: define last-tab Cmd+W once, pick non-colliding cycle chords, add stable catalog ids now and expose rebinding after OC-01.
S8-M4 Major. Missing Alternatives Considered (DC-1, ST-1, ZO-8). Fix: add the section; first entry is the cheapest option, an open-chat id list plus MRU in the sidebar.
S8-M5 Major (new). Tabs only pay off if switching is never blocked by generation. Fix: state ZP-23 is a hard prerequisite and the interim behavior is a refused switch with the existing reason.

## Fixes to reject as scope growth

- Full Risks & Mitigations expansion (ST-2): keep a three-line note that drafts reuse the chat archive and its ghost filter. (Caveat from the consolidator: the standard requires the section, so keep at least the short form.)
- Schema version, migration, persistence work (DC-4); shutdown flush and ordered restore (IC-6); draft debounce design (IC-5): side effects of the second store.
- Delete-while-open observer and toast (IC-10, DC-9): prune invalid ids at restore and reopen.
- Confirm-on-close for dirty drafts (IC-7); new reopen chord and chord table (DC-6, CO-6); tab-count cap, minimum widths, tooltip and overflow-order rules (DC-11, SY-6, CO-11).
- Expanded test plan and new deleted-chat CUJ (IC-16, DC-13, ST-7): add only a one-tab-per-chat test.
- Glossary, "resident" definition, promotion choreography (CO-9, CO-12, CO-2, DC-10, IC-13); extra NFR sections (ST-7).

## New findings

S8-N1 Nit. Per-tab scroll goal (line 53) has no mechanism. Fix: mark in-memory, not restored; delete the "no reload from disk" NFR (ZO-7, IC-12, DC-5, CO-12).
S8-N2 Nit. The Stage 5 claim that Cmd+Shift+[ and ] already cycle chats was not found in `KeyboardShortcutCatalog.swift` under selectNextChat or selectPreviousChat; unverified. Fix: confirm where prev/next chat is bound before reusing.
S8-N3 Nit. Strip hide rule makes Cmd+T look inert. Fix: show at two or more tabs; move decided placement (a) and no per-tab split layout out of Open Questions (IC-8, CO-7, DC-11, ST-5, SY-5).
S8-N4 Nit. Cheap fixes: links to `../../../swift/...`, antecedent for "its report", cite or soften "ranks fourth" and "whole model absent" (SY-1, SY-2, CO-10), pin upstream links to 7ce35879b8 (ST-10). Skip capitalization and filler items.

Net: adopting S8-M1 to M5 cuts the plan from 9 files to about 5 and removes roughly 20 findings without answering each.
