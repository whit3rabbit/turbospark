# Swift chat search and the Cmd+K command center

What keyword search over previous chats is, what it searches, and the two
rules it must not break. The engine is `Generation/ChatSearch.swift` (pure,
no SwiftUI); the UI is `Generation/CommandPaletteView.swift` (the command center,
which replaced the chat-only overlay) plus `ChatSearchResultRowView.swift`.
The palette's own engine is `Generation/PaletteProviders.swift`.

## The command center (ZP-16)

Cmd+K opens one palette over commands, chats and project files, with four
scope tabs (All, Commands, Chats, Files). The FIRST character of the field
is a prefix that overrides the tab: `>` commands, `#` chats, `@` files; a
`#` or `@` anywhere else is literal, and deleting the prefix returns to the
tab. Tab and Shift+Tab switch scope. Cmd+Shift+P ("Run Command...") posts
`.showChatSearch` with a `>` seed in `userInfo`; a post with no payload
(Cmd+K, `/resume`) keeps the old toggle behavior.

- All stacks capped sections (6 rows each) in a fixed order: Chats,
  Commands, Files. Chats come first so the first row is still the top chat
  and Enter opens it as before. With an empty field this is the home view:
  recent chats, commands, then the last 5 recent searches; no files.
- Commands lists the built-in slash commands, plugin commands
  (`<plugin>:<command>`, enabled and invocable only, from the composer's
  own resolved skill list), the navigation sections (with their chords),
  every Settings pane and Check for Updates (only while the menu item is
  enabled). Enter RUNS a command only from a fixed list of argument-free,
  non-destructive local commands (`PaletteActionRouter.runImmediately`) and
  `/stats`, `/help`, `/export`; every other command is inserted into the
  composer draft (`/clear`, `/reset`, `/delete` included) so the normal
  submit-time gates apply.
- Files lists `ProjectFileIndex` (2000 entries, depth 8, so deeper files
  are invisible here, as they are to `@`). It reads the CACHED listing on
  open and refreshes in the background; with no project the scope is empty.
  Enter appends the `@path` mention token to the draft (switching to Chat
  first) and focuses the composer; it never opens the file.
- History is per workspace (project id, or `global`), 20 entries, most
  recent first, stored in `command_palette_history.json` through
  `AppJSONStore`. It records the raw query when a result is chosen, is read
  once on first open, stays on this machine, and can be cleared from the
  home view.

## The surface (chat search)

`Cmd+K` toggles a centered modal over any section (RootView mounts it; the
Chat menu's "Command Center..." item posts `.showChatSearch`, the same key
opens and closes it). The dialog lists results in plain recency order: chat
title, a snippet of the first matching text with the matched tokens in the
accent color, a match count, and a relative date bucket (Today / Past week /
Past month / Older). Arrows move the selection, Enter opens the chat, Esc
closes. Opening a hit switches to the chat section and, when the chat
belongs to another project, moves the project selection first --
`filteredChats` is project-scoped, so without that the chat would open
without appearing in the sidebar.

The sidebar's own "Filter chats..." field is UNCHANGED: it matches title and
the 60-char preview only. The dialog is the deep search.

## What is searched

Everything textual a chat carries, one searchable entry per blob:

- every message's `content` (any role) -- the content tier;
- every message's `reasoning` trace;
- tool calls (`name`, argument values, `rawInvocation`) and tool results
  (`output`);
- draft attachments (`fileName` + `extractedText`).

Not searched: the draft text itself, todos, artifacts, `contextSummary`
(the rows it summarizes stay in the transcript and are searched directly).

## Matching semantics

Deliberately the unsloth studio shape. Documents are built once per dialog
open from the in-memory `chats` array -- every committed turn enters that
array at the moment it persists, so a search needs no disk I/O and no index
store -- and each blob is lowercased once. A query is trimmed, lowercased,
split on whitespace, and a blob matches iff EVERY token is a substring
(AND semantics, no operators, no phrases). The empty query returns every
chat as a snippet-less recency row. There is NO scoring: recency is the
whole order, and the snippet comes from the first matching CONTENT entry,
falling back to any entry, then the title.

While the dialog is open, a `$chats` emission schedules a rebuild after
300 ms (the same coalescing idea as the persist debounce), so commits that
land mid-dialog become searchable.

## The two rules

1. **Ghost exclusion is by the FLAG.** `ChatSearch.buildDocuments` skips
   every `isGhost` row and never touches `GhostChatVault`. A ghost row
   carries no plaintext today, but the exclusion must not DEPEND on that: a
   match would leak that a temporary chat exists. Pinned by
   `testGhostChatsNeverProduceDocumentsEvenWhenTheyCarryMessages`, whose
   ghost fixture deliberately carries messages.
2. **A search result may not lie about why it matched.** The snippet is cut
   from the original-cased text around the earliest token occurrence, with
   every token occurrence inside the window highlighted; the ellipses are
   real truncation marks, not decoration.

## Keyboard surface

`Cmd+K` is free only because "Clear Chat History" is `Cmd+Shift+K`; the
catalog row lives beside it in `KeyboardShortcutCatalog.chat` and is held
against the menu by `KeyboardShortcutCatalogTests`. The dialog's own keys
are hidden buttons with window-level `.keyboardShortcut`s (up/down/default/
cancel actions) rather than a key monitor, so they keep working while the
search field is first responder.
