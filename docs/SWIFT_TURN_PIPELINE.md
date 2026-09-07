# The turn pipeline extras: the message queue and system reminders

Two mechanisms that sit between the composer and a turn, both ported from
Claude Code's `src/utils` at this app's scale: the message QUEUE
(`messageQueueManager.ts`) and system REMINDERS (`attachments.ts`). They
are documented together because they share one drain point -- the tail of a
generation turn -- and one rule about what may start a turn from it.

Read this before touching `AppModel+Queue.swift`, `SystemReminders.swift`,
the `canQueue`/`canRunOrQueue` gating, or the drain calls at the
generation-turn tail.

## The message queue

A prompt submitted while its chat is busy is PARKED, not dropped. Before
this landed, Return was swallowed whenever `canRun` was false: a user who
typed into a long agent turn had to notice, wait, and send again.

The contract, in order:

- `canRun` is unchanged: idle end to end (not generating, not submitting,
  no open, no approval card, session loaded, draft non-empty).
- `canQueue` (`AppModel+Gating.swift`, pure terms in `canQueueTerms`) is
  true when the ONLY failing terms are the busy ones: `generating ||
  submitting || pendingToolCall != nil`, plus session and draft. An open in
  flight queues NOTHING -- nothing drains at the end of an open, so a
  prompt parked there would sit forever.
- `run()` checks `canQueue` at the old `canRun` guard and calls
  `enqueueCurrentDraft(chatID:)`, which moves the trimmed draft and the
  draft attachments out of the chat row (the ghost draft clears through
  the vault payload) into `pendingUserMessages[chatID]`.
- Drain: the generation-turn tail calls
  `drainPendingUserMessagesIfIdle(chatID:)` BEFORE the task-notification
  drain. A prompt the user typed mid-turn is the older intent. Both drains
  re-check idleness, so whichever fires second parks cleanly until the
  next tail.
- Drain goes THROUGH `run()`, never around it: the entry is restored into
  the composer (text into `promptText`, attachments onto the row) and the
  ordinary submission path runs. That is what makes a queued `/compact`
  dispatch as a command, `@` mentions resolve, and the `UserPromptSubmit`
  hook fire.
- Only the FIRST entry goes per tail; the rest wait for the turn it
  starts. The composer shows the parked entries as pills; clicking one
  pulls it back into the draft (`restoreQueuedMessage`) for editing.
- `deleteChat` discards that chat's queue. The queue is in-memory only and
  never persisted, the same exposure class as the composer itself; ghost
  chats queue like any other.

Known edge, deliberately accepted: the drain fires only from a turn tail
AND only when the chat is selected (`canInjectTaskNotification`'s
`selectedChatID == chatID` term). A prompt parked for a chat the user
switches away from waits there until that chat is frontmost at some
future tail.

## System reminders

Claude-Code-style `<system-reminder>` blocks computed at prompt assembly
and appended to the model-bound copy of the LAST user message.
`SystemReminders.reminder(todos:messages:planModeActive:)` is pure; the
injection lives in `buildAppendOnlyHistory`.

Two state sources fire:

- **Todos**: the chat has items that are neither completed nor cancelled,
  and at least `todoStaleTurnThreshold` (2) assistant steps have passed
  since the last message carrying a `todowrite`/`todo_write` tool call
  (`assistantTurnsSinceLastTodoWrite` scans the transcript; a chat whose
  todos arrived any other way counts every assistant step, which is the
  right answer -- they are at least as stale). The reminder lists the
  items and says to keep the list current.
- **Plan mode**: while `PlanModeExecutor.isPlanModeActive(for: chatID)`,
  every prompt restates the read-only constraint and names
  `exit_plan_mode` as the way out. Before this existed the flag was
  write-only from the model's seat: nothing in prompt assembly read it.

The contract that matters: **reminders are ephemeral by construction.**
They are computed fresh on every assembly pass, appended to the history
array only, and the persisted transcript row keeps exactly what the user
sent. A reminder that qualified on step 3 leaves no residue when it no
longer qualifies on step 4 -- which is what lets the staleness rule be
"state of the world right now" rather than bookkeeping.

The SKILL.state history path deliberately gets NO reminders: its prompt is
O(1) in step count with its own state patch format (`docs/SKILL_STATE.md`),
and a per-turn reminder would defeat the bound it exists for.

## Tests

`MessageQueueTests` (predicate term matrix, enqueue/restore/drain,
deletion), `SystemRemindersTests` (the fire/no-fire matrix and the
transcript-untouched assertion). All mutation-checked.
