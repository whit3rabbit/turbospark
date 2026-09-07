# Swift message editing (retry, edit, branch)

Message edit / regenerate / branch in TurboSparkApp, ported from Unsloth
Studio's chat surface (edit composer, regenerate, branch picker, thread
fork) at the scope this app's flat transcript allows. The three operations
share one mental model:

- **Retry** regenerates the last response in place. The replaced reply
  becomes an inactive version of the new one, navigable with `< n / m >`.
- **Edit** rewrites the LAST user prompt in place, drops the response after
  it, and re-runs. The old prompt and (when it was plain prose) the old
  response both survive as versions.
- **Branch** takes an EARLIER user prompt and forks the conversation into a
  new chat containing the prefix up to that prompt, with the edit applied,
  then continues there. The original conversation is untouched.

Read this before touching `AppModel+MessageEditing.swift`, the `alternates`
field on `AppChatMessage`, or the hover-bar editing affordances.

## No engine changes

The prompt is rebuilt from the transcript on every turn
(`buildAppendOnlyHistory`), and `executeGenerationTurn(step: 0, chatID:)`
re-enters cleanly, so all three operations are transcript surgery plus a
re-run call. Rust is untouched; no cargo gate applies.

## The one persisted field

`AppChatMessage.alternates: [AppChatMessage]` holds the inactive versions,
oldest first. The struct's own fields ARE the active version, so every
existing reader (prompt assembly, transcript, sidebar preview, export)
keeps reading what it always read. Rules that keep it sane:

- A pushed copy is FLATTENED (its own `alternates` dropped): the archive
  stays one level deep.
- Every version keeps its own `createdAt`. The union of `alternates` plus
  the active fields, sorted by `createdAt`, is the version list; the active
  one is found by timestamp-and-content match, falling back to newest. That
  is what gives the switcher stable `n / m` numbering without a stored
  position field.
- The transcript row's `id` NEVER changes across a swap. It is the
  ForEach identity; following the activated version's id would rebuild the
  row mid-navigation.
- The field is read with `decodeLossyArray` in the tolerant decoder, per
  the Gotcha 13 rule on this struct (`swift/CLAUDE.md`): an archive written
  before the field existed must decode, not quarantine.

Ghost chats get all of this for free: the vault payload stores
`[AppChatMessage]`, so versions seal and unseal with the messages.
Branching is the one operation REFUSED in Ghost Mode, because a branch is
by construction a new persisted chat and copying vault contents into one
would launder a temporary conversation onto disk.

## Anchors and what Retry v1 covers

The anchor is the last REAL user prompt: role `.user` with empty
`toolResults` and no `<user-memory-input>` wrap. Tool-result turns and
`#` quick-saves are `.user` rows too, and neither is something a response
can be regenerated against.

Retry v1 applies only when the chain after the anchor is exactly ONE
assistant message with no tool calls and no tool results. A tool chain
hides the button rather than half-working: restoring it as a single
alternate would have to carry every row, and re-rolling an agent's work is
a different feature. Recorded as a follow-up, not an oversight.

## Seeding: how a replaced response survives

`pendingResponseVariants: [UUID: [AppChatMessage]]` parks the replaced
prose response between the operation and the re-run. Three exits keep a
stale entry from grafting old text onto a turn it was never written for:

1. `finishProseTurn` seeds it as the new reply's `alternates` and removes
   the entry.
2. `finishCancelled` seeds it onto the partial append the same way, so a
   stopped Retry still leaves the old reply navigable beside the partial
   one.
3. A turn that came back as tool calls drops it (`AppModel+Generation`), and
   an ordinary submission clears it up front (`AppModel+Submission`).

Without all four touchpoints a stale entry would attach old text to some
later unrelated turn.

## Paired variant switching

An edit creates the prompt variant and its response variant together, so
stepping a USER prompt's version also steps the response row after it by
the same delta, clamped to that response's own range. Without the pairing,
navigating back to the old prompt would pair it with a reply written for
the new one.

## The compaction boundary clamp

Assembly skips rows below `compactedMessageCount`. A summary covering the
prompt a retry or edit is about to re-run would send the model a turn with
no prompt in it, so `clampStoredCompaction(chatID:toRow:)` brings the
boundary down to the prompt row first. Rows below keep their summary; the
prompt row goes back live. `branchedChat` applies the same clamp to the
copy (`min(boundary, messageIndex)`) and carries the summary text over,
which stays accurate: the prefix rows it describes are byte-identical
copies.

`setStoredCompaction` moved from `private` to internal for this -- the
clamp writes through the same ghost-aware primitive instead of restating
its two arms.

## Where the affordances live

`MessageActionBarView` grew optional parameters (`editAction`,
`branchAction`, `retryAction`, `variantStep` + position/count); the row
decides what to offer, the bar stays a dumb pill strip.
`MessageEditingViews.swift` holds the switcher, the in-place composer
(Escape cancels, Save re-runs), and the branch sheet. Guards reuse the
codebase idiom: `!generating, !submitting, pendingToolCall == nil` plus a
loaded session, and bounded-state (SKILL.state) projects refuse all three,
because their prompt comes from `skillState`, not from transcript rows.

## Non-goals, recorded deliberately

Retry on tool-call/agent chains; editing assistant messages; full tree
branching (versions exist only on the last prompt / last response); variant
navigation on messages that never had variants.

## Tests

`MessageEditBranchTests.swift` runs without a session, through the static
cores (`retryApplied`, `editApplied`, `branchedChat`, `applyVariantStep`,
`lastPromptAnchorIndex`, `isSingleProseResponse`) -- the same split
CompactionTests uses. All seven are mutation-checked; the anchor-detection
fixture puts the tool-result row LAST, because with a real prompt after it
`lastIndex` picks the prompt either way and the test survives its own
mutation (found by the mutation check itself).
