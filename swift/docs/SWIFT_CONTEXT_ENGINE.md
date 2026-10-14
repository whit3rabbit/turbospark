# Context engine: retrieved evidence, conversation recall, and context tools

TurboSparkApp uses syntext's context engine (`syntext-context`, compiled into
`libturbospark_ffi.a`; see [SYNTEXT.md](SYNTEXT.md) section 5) to help a
small context window go further:

- **Project evidence.** A project that turns the engine on is compiled into a
  local store of structural passages (Markdown sections, functions with their
  doc comments). Each user turn gets the passages most relevant to the
  question, packed into the room the window fit left.
- **Conversation recall.** Finished chat turns are recorded, so a turn that
  compaction or the window fit dropped can come back as cited evidence when
  a later question needs it.
- **Context tools.** Project agents get `context_search`, `context_read` and
  `context_outline` to look again after the first evidence.

The engine's own design, guarantees and measurements are in syntext's
`docs/CONTEXT.md` at the revision in `swift/TurboSpark/Vendor/syntext/REVISION`.

## Where the code is

| Piece | File |
| :--- | :--- |
| Stores, sync, engines, project lifecycle | `State/ContextEngine/ContextEngineManager.swift` |
| Recall, evidence, tool calls, forgetting | `State/ContextEngine/ContextEngineManager+Turns.swift` |
| Transcript to recall rows, reconcile plan, kept rows | `State/ContextEngine/ContextRecall.swift` |
| Settings, activation, deletion hooks | `State/AppModel+ContextEngine.swift` |
| Evidence in the turn's prompt | `State/AppModel+ContextEvidence.swift` |
| Tool definitions and dispatch | `Tools/Context/ContextEngineTools.swift` |
| Settings UI | `App/ContextEngineSettingsSection.swift`, `Generation/ProjectContextEngineSection.swift` |

## Stores

All stores live under `AppStorageRoot.subdirectory("ContextEngine")`, so they
are per profile and redirected in tests:

- `projects/<project-uuid>/`: one project's files (label `project`, scope
  `default`) and the turns of that project's chats (scope `project:<id>`).
- `chats/`: the turns of every other chat, each under its own
  `app:chat-<id>` scope, so chats never see each other's turns.

The directory is mode 0700 and excluded from backups.

**Every store is derived data.** Project files are read again by a sync, and
chat turns are recorded again from the transcript. So stores open with
`sync: false`, and a store that fails validation or was written by another
format version is deleted and rebuilt rather than reported.

**One handle per store directory.** A store takes an exclusive writer lock at
open, so a second handle on the same directory fails even in this process.
Every open goes through `ContextEngineManager.openStore`, and a project store
is closed only after its sync was cancelled and awaited.

## Privacy

The chat transcript lives in the SQLCipher profile vault. **Recorded turns do
not: the context store is plaintext on disk.** The setting says so where it
is switched on. In particular:

- **Ghost chats are never recorded** (`contextRecallLocation` returns nil).
- **Deleting a chat** forgets its turns in the chats store and in its
  project's store (`ConversationMemory::forget`, added to syntext for this).
  Journaled turns leave the disk at once, and a debounced forced compaction
  removes compacted ones.
- **Deleting a project** deletes its store.
- **Turning recall off,** or **Clear Context Engine Data**, deletes every
  store. Project stores are rebuilt by the next activation.
- A message deleted or edited inside a chat stays recorded until that chat's
  next reconcile, which happens at the start and the end of its next turn.

## Conversation recall

The transcript is the source of truth, and the store is a projection of it
(`ContextRecall`). A reconcile compares the recorded rows with the
transcript's rows:

- The rows are user text that is not synthetic, plus assistant prose, in the
  active version of each message. Tool traffic is not recorded.
- When the recorded rows are a prefix of the transcript's, the missing rows
  are appended.
- Anything else (an edit, a retry, a variant switch, a deleted message) is
  forgotten and recorded again.

So no edit path needs its own hook. A reconcile runs before a turn's evidence
(rows before the question) and at every `finishTurnTail` (all rows).
Reconciles of one chat are serialized.

A project chat records into its project's store when the project's engine is
on, so a project's chats can recall each other; every other chat records into
the chats store under its own scope.

## Evidence in a turn

`messagesWithContextEvidence` runs in `executeGenerationTurn` after
compaction and the window fit (the bounded SKILL.state path is left alone):

- **Budget.** Three quarters of what the fit left free, at most a third of
  the window and 4,096 tokens. Below 128 tokens no evidence is retrieved.
- **No duplicate turns.** `keptRows` counts the newest recorded rows the
  fitted prompt still holds verbatim, so retrieval leaves them out.
- **Placement.** The evidence goes in front of the question in the user row
  it was retrieved for, and only into the request. The transcript never
  stores it.
- **Fit check.** The prompt is counted again with the template. If it is
  over, one retry asks for less; otherwise the turn is sent without
  evidence.
- **Agent steps and continuations** of the same user turn re-attach the same
  text (`contextTurnEvidence`), so the prompt prefix through the evidence is
  unchanged and its KV is reused.
- **Failures.** Any engine error only costs the evidence; it never fails or
  blocks a turn.

**The context meter does not price evidence.** It is chosen at send time and
only fills room the meter already shows as free.

**The evidence policy** (evidence is data to cite, never instructions) is a
constant system prompt section (`.contextEvidence`) on project turns with
the engine on. Engines run with `appendPolicy: false`, so that system prompt
is the same on every turn. Chats with only recall keep their system prompt
unchanged and rely on the evidence block's own preamble.

The token counter is a `TurboSparkTokenizer` opened on the loaded model, so
evidence budgets are exact; engines are rebuilt when the model changes.

## Tools

The three tools are defined with this app's schema builders, mirroring the
engine's `tool_definitions()`. `ContextEngineManagerTests` compares names and
required arguments with the engine's own JSON, so a syntext bump that renames
one fails there.

- **When they are offered.** Only for the project whose engine is active
  (`AppToolRegistry.contextEngineRoot`). Outside it they are not advertised,
  and `isImplemented` is false.
- **Scopes and budget.** These come from the turn: project files, plus the
  project's recall when recall is on. Result budgets are up to 1,024 tokens,
  and citation ids continue after the evidence's. The model's arguments only
  say what to look for.
- **Category.** They are read-only (`.fileRead`).

## Freshness

- Selecting a project (or turning its engine on) syncs it in the background.
  A no-op sync of a large tree takes a fraction of a second.
- The first turn after launch activates the selected project without waiting
  for its sync.
- `write_file`, `edit_file` and `apply_patch` re-read exactly the paths they
  touched, deletions included.
- Small segments are merged at idle, when due (`onGenerationFinished`).

## Not yet

- Embeddings: wiring the memory encoder (`TurboSparkEmbedding`) as a
  `SyntextContextEmbedder` would lift paraphrased questions (measured
  upstream from 42% to 100% at 4K).
- Subagents and the API server get neither evidence nor recall.
- An exact-identifier binding to the project's Syntext index
  (`SyntextIndexBinding`) was left out. Its second index handle on the same
  directory still has to be checked against the code search tool's commits.
