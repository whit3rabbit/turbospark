# Comprehension report (Stage 6): ZP-16 command center palette

Reviewer: egm-comprehension (read only the target). Completed. IDs CO-n.

## Paraphrase

Replace the Cmd+K chat-search overlay with a palette that searches commands, chats and workspace files, with scope tabs and prefixes (`>` commands, `#` chats, `@` files). Plain text searches everything. Cmd+Shift+P opens it filtered to commands. Each workspace keeps up to 20 persisted searches. Selecting a row runs the command, opens the chat, or attaches the file to the composer as an `@file:` reference. The existing overlay uses `ChatSearch` (recency order) over an in-memory archive; `ProjectFileIndex` is a bounded walk (2000 entries, depth 8, 20 s freshness); `KeyboardShortcutCatalog` is the single shortcut source; `PluginManager` supplies plugin commands.

## Findings

* CO-1 Major, lines 35-36, 91. "Three scope tabs" lists four. Unclear whether history is a tab or a section.
* CO-2 Major, lines 44, 138-166. Cmd+Shift+P "pre-filtered to commands" undefined (Commands tab, typed `>`, or filter on All); clearability and tests missing.
* CO-3 Major, lines 54-58, 91, 142-145. Home view undecided yet a "history section" is planned. History contents (queries, selected results, both), storage and display undefined; empty state untested.
* CO-4 Major, lines 107-109, 78-85. Ranking "per-scope recency and prefix-position" unclear: commands and files have no recency; All-scope merge rule missing; "blends" conflicts with "chat behavior preserved".
* CO-5 Major, lines 38, 62-68, 101-105. "Enter runs the export flow" assumes every command can run from the palette; commands with arguments, composer context or dialogs not handled; no list or source for "menu-reachable actions"; sync with menu and plugin state undefined.
* CO-6 Major, lines 38-40, 98. "When plugins are enabled" undefined (global, per plugin, per workspace); disabled or failing plugins unhandled; PluginManager not linked.
* CO-7 Major, lines 73-76, 132-134. Index cap and depth silently hide files in large repos; stale index behavior undefined; "no new I/O on open" conflicts with 20 s freshness.
* CO-8 Major, lines 89-105, 142-155. Router location ambiguous (separate box in diagram, inside CommandPaletteView.swift in file plan); "common scored row type" never defined; provider protocol shape (sync or async, cancellation, limits) missing.
* CO-9 Major, lines 85, 152-155. Whether ChatSearchOverlayView.swift is deleted, gutted or kept as a shell is unclear.
* CO-10 Major, lines 20-22. `chat-runtime-qol` kiro spec and its "Partial row" cited with no link; supersession justified only by author decision.
* CO-11 Major, lines 41-42, 134, 168-177. Only number is "under 100 ms" with no test; verification is manual; DESIGN.md and `make design-check` unlinked.
* CO-12 Major, lines 130-131, 174. "21 languages" with no catalog named; prefix characters and shortcuts may vary by layout or locale; prefix collisions (file named `#notes`, chat query starting with `@`) uncovered.
* CO-13 Nit, lines 3-7. ZCode never described as the reference product.
* CO-14 Nit, lines 26-29. "Roughly 40" slash commands unsourced.
* CO-15 Nit, line 121. "NSPredicate-based search" does not match the described AppKit help search.
* CO-16 Nit, lines 156-159. "Palette help entries" and "single-source rule" undefined.
* CO-17 Nit, lines 165-166. Mutation-check wording needs a one-clause explanation.
* CO-18 Nit, lines 160-164. Tests omit history persistence across relaunch and per-workspace isolation.

No Blocker found.
