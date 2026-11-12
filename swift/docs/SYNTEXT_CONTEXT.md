# Syntext context engine: conversation recall

Conversation recall keeps a long chat's earlier turns reachable after they
leave the model's window. Each chat's turns are journaled to a local
`SyntextContext` store. When some of them are no longer in the prompt, the
ones relevant to the person's newest message come back as cited evidence.

It is on by default: Settings > General > Context Compaction, "Recall earlier
messages in long chats".

Related docs:
- [SYNTEXT.md](SYNTEXT.md): the same library's code indexing (`grep_search`),
  the package pin, and the dual-staticlib link. The pin rule there applies
  here too.
- [SWIFT_MEMORY.md](SWIFT_MEMORY.md): approved memory claims, a different
  feature. See "Memory vs the context engine" below.
- [SWIFT_COMPACTION.md](SWIFT_COMPACTION.md): the summary that replaces
  compacted turns. Recall complements it and does not replace it.
- [PROFILE_VAULT.md](PROFILE_VAULT.md) and
  [SWIFT_GHOST_MODE.md](SWIFT_GHOST_MODE.md): why recall is off for
  protected profiles and Ghost chats.

## Memory vs the context engine

Two features put remembered text into a prompt. They answer different
questions and store different things.

| | Memory (`MemoryLedgerStore`) | Conversation recall (`ChatRecallStore`) |
|---|---|---|
| Holds | Short approved claims (facts, preferences, decisions) | The verbatim turns of one chat |
| Reach | Every chat in the profile, plus the attached project | Only the chat that wrote the turns |
| Gets in by | Explicit saves, reviewed model proposals, optional extraction | Automatically, from the transcript |
| Storage | The SQLCipher profile database (encrypted) | A plaintext syntext store under the profile root |
| Default | Model memory off | On, except Ghost chats and protected profiles |
| In the prompt | Recalled claims in the system prompt | `<evidence>` blocks on the newest user message |
| When | Every turn | Only when turns of this chat are outside the prompt |
| Removed by | Forgetting a claim, with suppression of the source | Deleting or clearing the chat, or editing its turns |

Use memory for what should carry across conversations. Recall only stops a
single conversation from losing what it already said. Neither reads the
other's store.

## How it works

1. **Record.** `ChatRecallPlanner.recordable` picks the rows worth keeping:
   the person's non-synthetic user messages and the model's prose answers.
   Tool turns, guardrail retries, hook nudges, and reasoning are left out.
   Text is NUL-stripped, CRLF-normalized, and capped at 32 KB per turn.
2. **Reconcile.** The journal is brought in line with the transcript rather
   than hooked into every writer. Before each turn, and 750 ms after a write
   through `mutateTurnMessages`, `mutateRecoveryState`, or `rewindTo`, the
   recordable rows are compared with `SyntextConversation.history()`. New rows
   are appended. Any divergence (edit, retry, rewind, variant switch) forgets
   the chat and records it again.
3. **Fit first.** `executeGenerationTurn` builds and fits the prompt exactly as
   it would without recall (compaction, microcompaction, window fit).
4. **Decide.** `ChatRecallPlanner.earliestRetainedRow` finds the oldest
   transcript row the fitted prompt still carries. The window fitter drops
   messages, not turns, so this works from the projection's source rows,
   skipping the leading system message and the pinned-instruction block.
   Recorded turns from that row on are `keptTurns`. If every recorded turn is
   kept, recall stops here and the request is byte-identical to one without
   recall.
5. **Make room.** Evidence gets at most 1,024 tokens or an eighth of the
   prompt budget. If the fitted prompt leaves less headroom than that, it is
   refit at the smaller bound, which can drop more old turns.
6. **Recall.** `ChatRecallStore.evidence` calls `SyntextContextEngine.prepare`
   with the chat's conversation, `keptTurns`, and the budget. The engine
   searches the chat's scope (lexical BM25F plus a bounded entity walk),
   excludes the kept turns, and packs whole passages into sanitized,
   trust-labelled `<evidence>` blocks. Recalled turns are labelled
   `trust="memory"`.
7. **Inject.** The evidence goes on the model-bound copy of the person's
   newest message: syntext's evidence policy, the blocks, then
   `Question: <the message>`. It is never put in the system message (that
   would break KV prefix reuse) and never stored in the transcript.
8. **Check.** The final prompt is counted with the session's own counter. If
   it would not fit, the turn uses the original fit.

## Integration points

| Piece | Where |
|---|---|
| Pure decisions (recordable rows, reconcile plan, kept turns, injection, budget) | `State/ChatRecall/ChatRecallPlanner.swift` |
| Store service: one serial queue, store, engine, open conversations, cache | `State/ChatRecall/ChatRecallStore.swift` |
| `AppModel` glue: eligibility, reconcile scheduling, forgetting, the recall call | `State/AppModel+ConversationRecall.swift` |
| The turn: eligibility captured once, reconcile before the turn, recall after the fit | `State/AppModel+Generation.swift` (`executeGenerationTurn`) |
| Transcript writes that schedule a reconcile | `mutateTurnMessages`, `mutateRecoveryState` (`AppModel+Ghost.swift`), `rewindTo` (`AppModel+SlashParity.swift`) |
| Forgetting | `deleteChat`, `clearOutput` (`AppModel+Chat.swift`) |
| Setting | `MacAppSettings.conversationRecallEnabled`, `AppModel.conversationRecallEnabled`, `GeneralSettingsPaneView` |
| Protection | `ProfileVaultCoordinator.protect` purges the store; `loadSettings` purges a leftover one |
| Shutdown | `AppModel.shutdown` closes the handles |

Design notes:
- **One serial `DispatchQueue`, not an actor.** Every syntext call blocks (an
  fsync, a store scan). Blocking inside Swift's cooperative pool starves
  unrelated tasks. The serial order also guarantees that the pre-turn
  reconcile lands before that turn's evidence request.
- **Per-chat scope.** Each chat is memory mode
  `.app("turbospark.chat.<uuid>")` with the chat id as the session. Syntext
  grants a conversation's whole scope to a request, not only its session, so
  a shared scope would let one chat recall another's turns.
- **The model's own tokenizer.** The engine counts with a
  `TurboSparkTokenizer` opened from the loaded model's path, one engine per
  model. It never calls the session's counter, because generation holds the
  session queue for the whole turn.
- **Bounded wait.** Evidence that is not ready within 1.5 seconds is skipped
  for that request. A blocking FFI call cannot be cancelled, so it is left to
  finish rather than waited on.
- **Cached per request.** Evidence is cached per chat, kept-turn count,
  budget, model, and question, so every step of a tool loop sees the same
  prompt.

## Storage and gates

- **Location:** `<AppStorageRoot>/context-recall/`, per profile. The folder is
  mode 0700 and excluded from backups. Files: `manifest.json`, `journal.log`,
  `seg-*.ctxs`, `write.lock`, and temporary files.
- **Plaintext.** Syntext has no encryption, and the journal and segments hold
  turn text as written. That is why the gates below exist.
- **Ghost chats** are never recorded or recalled. Ghost Mode's promise is that
  nothing reaches disk.
- **Passphrase-protected profiles** have no store. Protecting a profile purges
  it, every call re-checks protection, and launch purges any store left by a
  crash in between. A local-key profile keeps its database key beside the
  database, so a plaintext store there exposes nothing new to someone who can
  already read the folder.
- **Turning the setting off** deletes the store. It is derived data, so
  turning it back on rebuilds it lazily, chat by chat, at each chat's next
  turn.
- **Opened without fsync.** The transcript is the source of truth, a torn
  journal tail is cut at the next open, and reconcile re-records the gap.

## Limitations

- **Lexical retrieval only.** No embedder is wired in. Syntext's own
  evaluation at a 4K window, without embeddings, put paraphrased questions at
  42% of answers recalled. A question phrased very differently from the turn
  that answered it may miss.
- **Prose only.** Tool calls and results, reasoning, and images are not
  recorded. A fact that lived only in a tool result is not recallable here.
  The tool-observation store and `ctx_inspect` cover tool output instead.
- **Per chat only.** No recall across chats. That is memory's job.
- **Main turns only.** Subagent runs (`SubagentRunner`) and the bounded
  SKILL.state prompt do not use recall.
- **Small budget.** At most 1,024 tokens of evidence per request, and none
  when less than about 96 tokens of headroom remain.
- **Only after turns leave the prompt.** Short chats never use it. The first
  turn after compaction is where it starts to matter.
- **Plaintext at rest** on local-key profiles, readable by anything that can
  read the profile folder.
- **Deleted, not shredded.** Forgetting removes journaled turns at once.
  Compacted turns leave the disk at the debounced forced compaction about 2
  seconds later. Removal unlinks files and does not overwrite them, and local
  APFS snapshots can keep earlier copies.
- **Rewrites are full re-records.** An edit early in a long chat forgets and
  records every turn again.
- **One writer per store.** A second app process on the same profile cannot
  open the store, and recall is silently off there.
- **Long turns are truncated** to 32 KB in the journal. Large pasted
  attachments are only partly recallable.
- **Not used yet:** syntext's agent tools (`context_search`, `context_read`,
  `context_outline`), embeddings, and receipts with source drift checks.

## Verification

`swift/TurboSparkApp/Tests/TurboSparkAppTests/ChatRecallTests.swift`:
- `ChatRecallPlannerTests` covers recordable rows, normalization, the
  reconcile plan, kept-turn arithmetic past a summary and a pin, the injection
  target, the byte-identical prompt when nothing is outside, the fallback when
  the final prompt would not fit, the reserve refit, and the setting's decode.
- `ChatRecallStoreTests` runs the real syntext store in a temporary folder. It
  covers the round trip after a reopen, the 0700 folder, the protected and
  disabled gates, per-chat isolation, kept turns excluded from evidence,
  forgotten text gone from the disk after compaction, and the rebuild after a
  rewrite.

```bash
cd swift/TurboSparkApp && swift test --filter ChatRecall
```

A green suite does not show that recall helps answers. That needs a long chat
on a real model, past compaction, checking that a fact from a compacted turn
reaches the prompt and the reply.
