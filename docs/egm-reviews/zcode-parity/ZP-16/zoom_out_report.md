# Zoom out report (Stage 5): ZP-16 command center palette

Target: docs/designs/zcode-parity/ZP-16-command-center-palette.md (full file, 181 lines). Both reviewers completed.

## Zoom-out (egm-zoom-out), IDs ZO-n

Verified present: ChatSearchOverlayView (mounted RootView.swift:135), ChatSearch, ProjectFileIndex (2000/8/20 s match), KeyboardShortcutCatalog, PluginManager, Cmd+K at TurboSparkApp.swift:463 posting `.showChatSearch`, BuiltInSlashCommand.all has 39 entries.

* ZO-1 Major, lines 44, 156-159. Cmd+Shift+P is unbound, but neighbour Cmd+Shift+K is Clear Chat History (TurboSparkApp.swift:479). Plan edits only KeyboardShortcutCatalog.swift; TurboSparkApp.swift and RootView.swift are missing from Target Files. A second notification or a payload on `.showChatSearch` is needed for "pre-filtered to commands".
* ZO-2 Major, lines 38-39, 96-97, 144-151. Command provider underspecified. Built-ins exist (`BuiltInSlashCommand.all`, `SlashCommandMenuGroup.grouped`); plugin skills exist as `<plugin>:<skill>` AppSkill values (nested namespaces possible) via `PluginManager.pluginSkills`. "Menu-reachable actions" have no registry (hand-written Buttons in TurboSparkApp.swift): that is the real new design surface. "Enter runs the export flow" has no dispatch path.
* ZO-3 Major, lines 107-109. "No new scoring claims" contradicts ChatSearch, which is unscored AND-of-substring in recency order. Commands and files have no recency, so prefix-position ranking is new. All scope needs a cross-provider merge rule; chats first would keep CUJ 3.
* ZO-4 Major, lines 3, 20-22. Cited kiro spec `chat-runtime-qol` not found under .kiro; INDEX.md line 43 repeats the claim. Fix the reference or drop it.
* ZO-5 Major, lines 70-76, 41-43, 133. File-to-composer handoff: the existing mention path is a completion inside the composer (ComposerAutocompleteEngine/Controller), no standalone insert API found, so it is a new seam. `entries(for:)` can start a walk on a cold or stale cache; use `cachedEntries` on open. Behavior with no open project undefined. Search history has no storage location and "workspace" is undefined.
* ZO-6 Major, lines 35-36. "Three scope tabs" lists four (All, Commands, Chats, Files).
* ZO-7 Nit, lines 43, 56-58, 165-166. Persisted history is speculative (problem statement does not need it) and overlaps home view option (c). Cmd+Shift+P is parity-driven.
* ZO-8 Nit, line 142. View file bundles parser, rows, history and router; use a pure SwiftUI-free query parser (ChatSearch.Query precedent).
* ZO-9 Nit, lines 49-50. Commands and Files need a matcher; reuse `ChatSearch.Query` and say so.
* ZO-10 Nit, lines 111-126. Alternative B is a strawman; real alternative is extending the overlay in place with prefixes only. Protocol with three implementations is mild over-abstraction.
* ZO-11 Nit, lines 135-136, 173-174. "21 languages" unverified; INDEX cites "ten localization parity gates". Cite the gate by name.
* ZO-12 Nit, lines 165-166. Mutation checks are implementation-level detail for a design doc.

## Inventive catalyst (egm-inventive-catalyst), IDs IC-n

No Blocker contradictions. Spot checks as above; ZCODE_COMPARISON.md only grepped (finding 8.5, line 729), so ZCode behavior claims unverified.

* IC-1 Major, line 35 vs 36-40, 175. Goals say three tabs and list four; plan and verification also say "three scopes". State count and whether All is a tab or default view.
* IC-2 Major, CUJ 3 (78-85) vs Non-Goals (49-50) vs plan (107-109). All scope "blends" chats, commands, files with no merge order, yet promises chat behavior unchanged. Chat ordering could be displaced.
* IC-3 Major, CUJ 3 "plain-Enter on a chat row". Which row is pre-selected in All is dodged; either chats first or muscle memory changes. State which.
* IC-4 Major, lines 44, 156-159. Cmd+Shift+K adjacency; shortcut and presenter files (TurboSparkApp.swift, RootView.swift) missing from file list; `.showChatSearch` needs a sibling or payload.
* IC-5 Major, Technical Plan (false compromise, ranking). Resolution: render All as fixed sections (Chats, Commands, Files), each in its own native order with a per-section cap (for example 5); no cross-provider score. Liabilities: vertical space, keyboard and VoiceOver traversal across sections.
* IC-6 Major, NFR lines 132-134 vs 20 s freshness. "No new I/O on open" is false after 20 s idle (full scan on open path). Resolution: serve stale cache, refresh in background, update rows with stable selection.
* IC-7 Nit, CommandProvider. Hand-aggregating builds a second command registry beside the menu and slash table. Resolution: one descriptor registry (title, binding, enabled predicate), or a read-only adapter first. Enabled state must match menu `.disabled` logic (for example Clear Chat History while running).
* IC-8 Nit, router failure handling. Disabled commands, deleted files in history, disabled plugins after history saved are unhandled. Validate history lazily; show disabled commands with a reason.
* IC-9 Nit, CUJ 2 vs line 41. Index cap (2000 entries, depth 8) silently misses files in large repos; state it or show a truncation hint.
* IC-10 Nit, Overview 20-22 and alternatives. "Rejected by author" gives no technical reason. State: Help menu search cannot return chats or inject composer references.
* IC-11 Nit, Open Questions. Home view and history share the empty state; resolve together.
* IC-12 Nit, lines 38-40 vs 132-134. Plugin command discovery may read manifests; say PluginManager list is in memory or define a fallback.
