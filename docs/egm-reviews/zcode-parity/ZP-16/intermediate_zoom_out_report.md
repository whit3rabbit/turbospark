# Intermediate zoom out report (Stage 8): ZP-16 command center palette

Reviewer: egm-zoom-out on full file. Completed. Target unchanged since Stage 5, so every prior finding is still open. No Blocker from this lens. IDs S8-n.

## Still open, fix (low cost, text only)

* S8-1 Major, 35-36, 175. Tab count: four tabs (All plus three provider scopes); fix "all three scopes".
* S8-2 Major, 19-22, INDEX 43. `chat-runtime-qol` spec does not exist; link the real spec or reduce to "supersedes an earlier deferral, per author decision 2026-11-11".
* S8-3 Major, 44, 156-159. Add TurboSparkApp.swift (binding) and RootView.swift (presenter) to Target Files; state the pre-filter mechanism (optional payload on `.showChatSearch` that pre-types `>`); note Cmd+Shift+K is Clear Chat History.
* S8-4 Major, 78-85, 107-109. All renders fixed sections (Chats, Commands, Files), each in native order with a per-section cap, no cross-provider score; first chat row pre-selected so Enter keeps old behavior; delete the "no new scoring claims" parenthetical.
* S8-5 Major, 38-40, 96-102, 144-151. Ship slash commands, plugin commands, and a small fixed table of named menu actions sharing the menu's enabled predicate; Enter on a disabled command is a no-op; commands needing arguments insert into the composer.
* S8-6 Major, 70-76, 132-134. Read `cachedEntries` on open, refresh in background; state the 2000/depth 8 cap, empty Files scope with no project, directory handling; the composer handoff is a new seam, add a small insert entry point in the autocomplete controller and list that file.
* S8-7 Major, 54-58. Record lean (c) as default, non-blocking; decide history with it.
* S8-8 Nit, 135-136, 173-174. "21 languages" versus INDEX "ten localization parity gates"; cite the gate by name.
* S8-9 Nit, 26. Replace "roughly 40" with the registry count (39 per Stage 5).
* S8-10 Major, 121, 111-126. "NSPredicate-based" mislabels the AppKit help search; replace "Rejected by author" with one technical reason (help search cannot return chats or inject composer references). ST-2 rates this Blocker; this lens rates Major.
* S8-11 Nit, 165-166. Move mutation checks to the Verification Plan or delete.

## Scope-adding fixes, recommend rejecting

* Full Risks section (ST-1, DC-8): accept three lines at most (overlay regression covered by parity test, stale index served from cache, history local only).
* New alternatives (DC-2), full history spec (DC-9, SY-14, CO-18; consider cutting history, ZO-7), formal prefix grammar (DC-11; one sentence), extra CUJs (ST-9; one for Cmd+Shift+P), measurement harness and VoiceOver audit and release gate (ST-8, DC-17; mark 100 ms as a target), ZCode tab rename note (DC-13), locale handling of prefixes (CO-12), full provider protocol shape (CO-8; say providers are synchronous reads of in-memory or cached state), and polish items (IC-8, IC-12, DC-15, ST-10, ST-11, SY-8, SY-10 to SY-12).
* Accept one line each for the extra tests (ST-6, DC-14).

## New findings

* N-1 Major, 152-155 and Target Files. AppModel+SlashParity.swift:88 also posts `.showChatSearch`; keep the notification and its no-payload default unchanged, add the pre-filter as an optional payload, or add that file to the touch list. Also Cmd+K posts at TurboSparkApp.swift:459 and RootView.swift:173 consumes it (line numbers drifted from earlier reports).
* N-2 Nit, 152-155. Pick replace-in-place or delete-and-create for ChatSearchOverlayView.swift; avoid leaving a dead shell.
* N-3 Nit, 90-91, 142-145. Put the prefix and scope parser in a SwiftUI-free type reusing `ChatSearch.Query`, not in the view file.
* N-4 Nit, 89-109. Provider protocol is mild over-abstraction; keep it a plain enum or struct of closures, no plugin-extensible provider API.

Net: one editing pass of about ten text corrections plus N-1, and a decision to defer or shrink history. No new sections needed beyond a short risks note.
