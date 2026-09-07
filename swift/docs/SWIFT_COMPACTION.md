# Swift context compaction (auto-compact)

Context compaction in TurboSparkApp: when a conversation's prompt approaches
the model's context window, the older turns are summarized by the model
itself and the summary rides in the prompt in their place. The transcript
rows are never deleted -- the app, the archive and a ghost vault keep the
full conversation; only the PROMPT disagrees, by exactly the summarized
prefix. Default on.

Read this before touching `AppChatCompaction`, the boundary arithmetic, or
quoting an auto-compact threshold. Mirrors Claude Code's
`services/compact/` in shape (a threshold, a structured summary prompt, a
boundary, a status-bar countdown) at the scale this app actually runs at.

## The trigger

Auto-compaction fires PRE-SEND, inside `executeGenerationTurn`'s task before
the window fit, when

```
measured * 5 >= usable * 4        (AppChatCompaction.shouldAutoCompact)
usable   = maxContext - reservedForNew
```

`measured` is `session.countTokens` over the assembled history; `usable`
subtracts what the reply needs, because a prompt that "fits" at the full
window leaves no word for the answer. The fraction is of the USABLE window,
not the raw one -- Claude Code's flat 13k buffer is sized for ~200k contexts
and would be noise at the 4k-8k windows this engine ships.

Every failure is FAIL-OPEN: a skipped token count, a rejected threshold or a
failed summarizer falls through to the window fit the turn would have run
anyway (`fitWindow` drops oldest turns; the no-room error remains the last
resort for a conversation compaction cannot help, e.g. when the recent tail
IS the whole conversation). Compaction must never be a new way for a turn
to die. There is deliberately no circuit breaker: a failed attempt is one
warning toast and a retry on the next turn.

## The boundary

- `AppChat.compactedMessageCount` (vault twin: `GhostChatPayload`) counts
  leading rows the summary replaces in the prompt. Slicing is by MESSAGE
  ROW, so a tool call and its result can never be split -- they live on the
  same row.
- `keepRecent` (setting, clamped to 1...8, default 2) rows stay verbatim.
- Both prompt builders -- `buildAppendOnlyHistory` AND `updateTokenEstimate`
  -- skip rows at or before the boundary and inject the summary through ONE
  shared helper (`insertingSummaryInjection`), so the meter prices the
  prompt that would actually be sent. A third assembly point must do the
  same, or the two will describe different conversations.
- The injection is a USER-role message (`<context_summary>` wrapper) placed
  after the system message: a mid-history `.system` message is refused
  outright by the fallback renderers and by real templates that require
  alternating roles.
- Re-compacting folds the previous summary in via `<previous_summary>` in
  the summarizer prompt rather than stacking two summaries.
- `clearOutput` resets the boundary and the summary, for normal and ghost
  chats alike.

## The summarizer

`performCompaction` renders the covered rows to a plain-text transcript
(roles named, tool calls recorded even when the row's prose is empty,
images named, results kept to their row), caps it at 100k chars
(head 60k + tail 40k with an omission marker -- the summarizer runs on the
SAME window that just overflowed), and sends a two-message prompt asking
for exactly these sections:

```
Primary request and intent:
Key technical concepts and decisions:
Files, paths and code touched:
Errors and fixes:
Pending tasks and open questions:
Current state of the work:
Next step:
```

Generation settings are the boring ones -- reasoning off, temperature 0.2,
700 new tokens -- because every reasoning token is one taken from the
summary. A `PreCompact` lifecycle hook fires first (ids only, so a ghost
chat's privacy rule is untouched). The flag `isCompacting` is up for the
duration; the status bar shows "Compacting conversation".

## The manual entry point

`/compact [focus]` in the composer runs the same path by hand, with the
optional text as the summarizer's focus instruction. It refuses while a
turn is generating (the session's serial queue is single-tenant) and runs
on `submissionTask`, so Stop cancels a summarizer that will not finish.

## The ghost rule

Compaction state is CONVERSATION CONTENT. For a ghost chat it lives sealed
in `GhostChatPayload` (`contextSummary`, `compactedMessageCount`) and the
row stays empty, exactly like messages -- read it through
`compactionState(chatID:)`, write it through `setStoredCompaction`, never
touch the row fields directly, or a summary leaks to `chats_archive.json`.

## Settings and UI

| Setting | Default | Where |
|---|---|---|
| `autoCompact` | true | Settings > General > Context Compaction |
| `compactionKeepRecentTurns` | 2 | same section, picker over `keepRecentRange` |

The status bar's context meter counts down "% until auto-compact" once the
used fraction comes within 40 points of the trigger, and names the
summarizer state while it runs. It reads `AppChatCompaction`'s constants --
a second fraction spelled there would drift from the real trigger.

## Deliberate differences from Claude Code

- A proportional trigger instead of flat token buffers (window sizes differ
  by 50x).
- Nothing is ever deleted from the transcript; Claude Code also rewrites
  its in-memory list. The app keeps one source of truth and makes only the
  prompt disagree.
- No post-compact file re-reads or attachment re-announcements: the app has
  no tool-offering surface to restore state for.
- No circuit breaker, no reactive compact on a hard overflow: fail-open and
  the existing no-room error cover it.

## Excluded paths

The bounded SKILL.state prompt (`buildSkillStateHistory`) is O(1) in step
count and never compacts; subagent histories are short-lived and never do.
The engine and the FFI are untouched -- compaction is purely app-side
orchestration over `countTokens` and `generate`.

## Verifying a change to this

```sh
cd swift/TurboSparkApp && swift test --filter CompactionTests   # offline, no Metal
```

The cases cover the threshold arithmetic, the boundary slicing (including
tool pairing across it), the injection placement, legacy-archive decode,
the ghost round trip, `clearOutput`, transcript rendering and the
summarizer prompt. The summarizer's generate call itself is the one
untested seam -- it needs a live model; drive it once by hand with
`/compact` against a real install when touching `performCompaction`.
