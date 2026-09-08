# The turn pipeline extras: the message queue and system reminders

Two mechanisms that sit between the composer and a turn, both ported from
Claude Code's `src/utils` at this app's scale: the message QUEUE
(`messageQueueManager.ts`) and system REMINDERS (`attachments.ts`). They
are documented together because they share the drain points of a
generation turn -- the agent loop's step boundaries and, at the tail, the
idle moment before the next turn -- and one rule about what may start a
turn from them.

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

Known edge, deliberately accepted: the tail drain fires only when the chat
is selected (`canInjectTaskNotification`'s `selectedChatID == chatID`
term). A prompt parked for a chat the user switches away from waits there
until that chat is frontmost at some future tail. The steer boundary drain
below has no such term and is not gated on the selection at all.

## Steer delivery at step boundaries

Second delivery point, added 2026-09-07, ported from opencode v2's session
contract (`specs/v2/session.md`), where it is the DEFAULT delivery mode:
`steer` delivers parked input at the next safe step boundary of a RUNNING
turn, while `queue` waits for idle. Claude Code behaves the same way, and
the reason is the UX: a user who types "stop, use Python instead" into a
twenty-step agent turn wants the running turn to incorporate it, not to
watch the original instruction finish first.

The boundary is `continueOrStop(afterStep:chatID:project:)`
(`AppModel+AgentLoop.swift`), which every tool outcome funnels through, and
the drain is `deliverSteersAtBoundary` (`AppModel+Queue.swift`):

- Order between the two points: BOUNDARY first, tail second. Entries that
  arrive during the final step catch no boundary and remain the tail
  drain's, exactly as before. With no queue the boundary drain is a
  nil check and `continueOrStop` behaves as it always did.
- The gate: no `pendingToolCall` (an approval card means the loop is
  parked BETWEEN steps, and delivery waits for the boundary the approved
  or denied call lands on) and no pending cancel (a steer must not outrun
  the Stop).
- The assembly mirrors `run()`'s submission body on purpose -- mention
  resolution first, then attachments off the restored row, then the hook,
  then sanitization -- minus the turn lifecycle around it. The draft was
  captured at enqueue time, the chat exists and is titled, and the next
  step's `buildAppendOnlyHistory` carries the appended user row to the
  model naturally.
- Hook semantics: `UserPromptSubmit` fires per steered entry (ghost chats
  dispatch none, `run()`'s no-trace rule). A block or
  `continue: false` RE-PARKS the entry at the front and stops the batch,
  closest to `run()`'s "nothing appended, draft kept": the pill can still
  pull it back, and the tail drain later submits it through `run()`,
  where the same verdict becomes the ordinary error. Later entries stay
  ahead of anything enqueued during the awaits, so order is preserved.
- Batch rule: every entry goes at one boundary, and the step allowance
  resets ONCE for the batch (`continueAgentLoop(step: 0)`), so fresh user
  input also supersedes the step cap and the Stop consultation at the
  same boundary -- no `Stop` hook is asked about a turn the user just
  added to.
- The boundary drain is NOT gated on chat selection, unlike the tail
  drain: the turn is running in its own chat, and a steer lands in the
  transcript rather than in visible UI, so `canInjectTaskNotification`'s
  selection term does not apply.

Tests: the steer cases in `MessageQueueTests` (delivery, order, the two
gates, attachment inlining) and
`testAQueuedSteerAtTheCapBoundaryDeliversAndSkipsTheStopConsultation` in
`AgentLoopLifecycleTests`, which pins the seam through `denyPendingToolCall`
with a blocking Stop hook as the discriminator. All mutation-checked.
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

**The goal loop rides both of these seams** (`swift/docs/SWIFT_GOALS.md`
owns the details): `reminder` gained a `goal:` parameter, so an active
`/goal` restates its condition, iteration count and latest not-met reason
in a `<goal>` section on every assembly; and `run()`'s submission task
resets the goal's per-prompt counters -- the stall pause and the idle
check-in cap -- on every accepted prompt. The goal's own evaluation arm
sits at the top of `dispatchStopAndContinueIfBlocked`, making it the
third consumer of the stop seam beside the user's Stop hooks and the
steer reset.

## Tests

`MessageQueueTests` (predicate term matrix, enqueue/restore/drain,
deletion, and the steer boundary cases), `AgentLoopLifecycleTests` (the
steer-beats-Stop seam at the step cap), `SystemRemindersTests` (the
fire/no-fire matrix and the transcript-untouched assertion). All
mutation-checked.
