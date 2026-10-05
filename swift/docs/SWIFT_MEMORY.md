# Swift memory

TurboSpark stores approved memory in the active encrypted SQLCipher profile.
`MemoryLedgerStore` owns stable claim IDs, status, scope, confidence, evidence,
review state, source cursors, suppression records, reflection drafts, and run
receipts. `ProfileDatabase.saveMemoryLedger` commits the ledger snapshot,
claim/evidence tables, FTS rows, and Markdown projections in one transaction.
Embeddings are model-keyed derived rows and can be rebuilt.

The logical Markdown layout is included in profile export:

```text
memory/ledger.json
memory/profile/MEMORY.md
memory/profile/daily/YYYY-MM-DD.md
memory/profile/bank/<kind>.md
memory/profile/legacy/IMPORTED_MEMORY.md
memory/profile/dreams/YYYY-MM-DD.md
memory/profile/ALIGNMENT_SYNTHESIS.md
memory/projects/<key>/memory/MEMORY.md
memory/projects/<key>/memory/<topic>.md
memory/projects/<key>/legacy/IMPORTED_TOPICS.md
```

The open export contains readable projections and the ledger. Encrypted profile
backups copy the SQLCipher database and restore the ledger with it. External
backups already made cannot be changed by an in-app forget action.

## Capture and review

`/memory` opens the Settings manager. `# text` and `/memory text` save an
explicit user claim to the profile.
`/memory project text` saves to the attached project. These commands work
without a loaded model or model memory enabled. The `memory` tool stages model-issued saves and forget
requests for review. Settings can approve, edit, merge, replace, dismiss, or
forget claims even when model memory is disabled.

Automatic extraction has its own setting and defaults off. While the app is
open and unlocked, the hourly job examines non-ghost chats idle for at least
ten minutes. It supplies bounded user messages to a local model and accepts
structured proposals only when the cited message ID and exact quote match
the supplied source. Invalid claims and secret-shaped text are rejected.
Successful empty runs advance the source cursor; failed or interrupted runs
retry. Foreground generation cancels background capture.

Existing Markdown is imported once as source-unverified legacy claims. Its
original text remains in the ledger and archive projection. It is not part of
prompt recall until the user approves it.

## Recall

The actual current user message or subagent task is the search query. The
claim-ID keyed index combines FTS and optional semantic recall, with lexical
fallback if the encoder or embedding index is missing or stale. Main-turn
semantic waiting is bounded at 200 ms. The Rust FFI holds one serialized
encoder instance, avoiding a model reload on each call. A turn caches one
recall result for its prompt assembly, context estimates, and retries.

Only approved active profile and attached-project claims are visible to the
agent. The curated profile sheet and project index have prompt caps, and
recalled claims already on the sheet are excluded. `memory_search` returns
claim IDs and concise text. `memory_explain` returns evidence for claims in
the current scope.

## Forget and reflection

Forgetting retracts a claim and linked proposals, removes evidence quotes,
Markdown projections, FTS rows, vectors, affected reflection drafts, and
receipts. It retains source-ID and content-hash suppression metadata so an
unchanged preserved chat message is not learned again. Edits or deletion of
source messages invalidate evidence and return unsupported active claims to
review. The original chat remains in history.

The nightly job runs once per local day after 02:00 while the app is open,
unlocked, idle, and a local model is available. It stores a dated reflection
and reviewable guidance. Raw prose is never injected. Approved standing
guidance remains subordinate to the current user request.

## Verification

Run `swift test --filter MemoryFeatureTests` in `swift/TurboSparkApp` for
focused tests, then `make swift-lib`, `make swift-test`, and `make swift-app`.
The FFI cache also needs Rust workspace checks and the portable cross-target
gate in `.claude/docs/verification.md`. A real installed encoder is needed
to establish cold and repeated-call latency and resident memory.
