# Swift ghost mode (temporary chats)

Added 2026-09-05. A ghost chat (`AppChat.isGhost`) is a conversation that
leaves nothing on disk: quit, crash and profile switch all discard it.
This page is the home for the two-layer design and the three rules a future
change must not break.

Read this before touching `AppModel+Ghost.swift`, `GhostChatVault`,
`AppModel+Persistence.swift`'s `makeChatArchive`, or any code that appends
to a chat's `messages` array directly.

## Two layers, only the first is a guarantee

A ghost chat lives only in memory. Both archive-construction points
(`makeChatArchive` in `AppModel+Persistence.swift`, reached by
`persistChats` and `persistChatsDebounced` and therefore by all ~30 save
sites) filter `isGhost` rows, so quit, crash and profile switch all leave
nothing on disk. That filter is the guarantee.

The second layer, `GhostChatVault` (AES-GCM under a per-launch key), is
defense in depth. A ghost row's `messages`, `todos`, `draft`,
`contextSummary` and `skillState` are ALWAYS empty -- the content is sealed
in the vault and moves only through `mutateTurnMessages(for:_:)` /
`mutateGhostPayload(for:_:)` / the `turnMessages(for:)`-family accessors in
`AppModel+Ghost.swift`.

## Three things a future change must not break

**A ghost row stays empty.** Any new code that appends to
`chats[i].messages` directly writes plaintext a transcript reading the
vault will never see (the debug assert in `makeChatArchive` catches it at
the next persist). Route the write through `mutateTurnMessages` instead.

**An emptiness check on a ghost row is a lie by design.** `createChat`'s
reuse branch excludes ghosts for exactly this reason, and the sidebar's
history filter goes through `chatHasTranscript` rather than checking
`messages.isEmpty`.

**Ghost turns dispatch no `UserPromptSubmit` hook.** The hook receives the
prompt text and a hook script may log it, which is a trace a user who asked
for a chat that leaves no trace has not consented to.

## Where ghost state surfaces elsewhere

- **Compaction** (`swift/docs/SWIFT_COMPACTION.md`): a ghost chat's summary
  and boundary live sealed in the vault too, read through
  `compactionState(chatID:)` and written through `setStoredCompaction`.
- **Memory quick-save** (`swift/docs/SWIFT_MEMORY.md`): a ghost chat's `#`
  quick-save appends the note through `mutateTurnMessages` (sealed, never on
  the row) but still writes the memory FILE -- ghost hides the transcript,
  not the store.
- **Chat search** (`swift/docs/SWIFT_CHAT_SEARCH.md`): excluded by the
  `isGhost` flag, never by an emptiness check, so a ghost row carrying
  messages (a fixture, or a future bug) still produces no search document.

## Tests

Ghost-specific cases live beside each feature's own suite
(`MemoryFeatureTests`'s ghost quick-save case, `ChatSearchTests`'s
`testGhostChatsNeverProduceDocumentsEvenWhenTheyCarryMessages`,
`CompactionTests`'s ghost round trip) rather than in one file, because each
feature's write path is the thing under test.
