# Stage 6: Comprehension report (ZP-21 workspace tabs)

Reviewer: egm-comprehension (naive reader, target only, no context files). Completed. Whole file, lines 1-191.

## Paraphrase (kept)

The proposal adds a tab strip to the one main window, one tab per open workspace (project). A reserved "settings" tab id would exist, though Open Questions lean toward not using it yet. The tab list, selected tab and each tab's "availability" persist across launches. Switching tabs restores each workspace's selected chat, scroll position and composer draft. To do that, the per-project state now held once on AppModel moves into a per-tab store keyed by tab id, and AppModel reads the selected tab's slice. The work is gated on concurrent background chats (ZP-23), so a tab can keep generating while another is shown.

How the existing system works per the document: single window showing one project at a time; switching changes sidebar filter, git pane and file surfaces; that context lives once on AppModel and views read it there; persistence uses AppJSONStore files quarantined on corruption; one active chat per workspace today; split panes already show side-by-side chats within one workspace.

## Findings

CO-1 Major. Goal 2, Non-Goals. "Availability" is undefined and absent from the persisted tab model (id, workspaceID, reservedSettings?).
CO-2 Major. Goal 1 vs Open Questions "Settings tab vs window". "In this mode" undefined; Goal reads committed while the question leans the other way.
CO-3 Major. CUJ 1, Verification, Goal 4. Concurrency is called adopted yet CUJ 1 says "if adopted" and Verification says "both Open Question answers"; no such open question remains.
CO-4 Major. CUJ 3, Open Questions. "Tab switch guard" and "per-chat guards" undefined; CUJ 3 describes no guard while the tests and file 2 route "guards".
CO-5 Major. Technical Plan. "Unchanged in most surfaces" not enumerated; which state is per tab vs global is unclear; "per-workspace UI state" undefined.
CO-6 Major. CUJ 1, tab strip. How a workspace opens or closes, what the plus entry does, closing a tab with a running chat, duplicate tabs on one workspace, deleted or moved workspace on restore, and which tab the single-tab fallback shows are all unstated.
CO-7 Major. Goal 4 "without schema changes". Which schema is unstated; the tab model holds no chat data.
CO-8 Major. Overview, file 4. Strip "above the sidebar" is a surprising, unexplained layout versus a window-wide strip.
CO-9 Major (testability). CUJ 1 "restore exactly" vs NFR "lightweight state (not live render trees)". Scroll anchor capture and replay undescribed; no measurable target.
CO-10 Nit. Undefined project terms (AppModel, AppJSONStore, quarantine rules, selectChat-family, ZP-02 states, token ratchet, shortcut catalog, ZP-90, finding 2.1); badge states never listed.
CO-11 Nit. "NSWindowTab" is likely not the real API name (NSWindow tabbing); "creation order" after close and reopen unstated.
CO-12 Nit. Strip view placed in `Generation/`; "21 languages" unlisted; tab overflow behavior missing.
CO-13 Nit. "Sequence ZP-23 before this draft" is awkward; presumably means implement ZP-23 first.

Verdict: no Blockers; the shape is understandable. Majors are Goals, Open Questions and CUJs disagreeing, and undefined key terms.

Persona file read: /Users/whit3rabbit/.claude/plugins/cache/egm/egm/0.1.1/agents/comprehension.md
