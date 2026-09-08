# The /goal loop

A per-chat goal loop ported from Claude Code's built-in `/goal` (its
telemetry name is `tengu_joyful_globe`): the user states a completion
condition, and at every point the agent would stop, an evaluator model
judges the conversation against it. Not met means the loop CONTINUES on
the evaluator's reason; met or impossible clears the goal with a verdict
row; background work defers evaluation with backoff check-ins; repeated
tool-free replies stall-pause it.

Read this before touching `AppModel+Goal.swift`, `ChatGoal.swift`,
`GoalEvaluator.swift`, `GoalBannerView.swift`, the top of
`dispatchStopAndContinueIfBlocked`, or quoting a check-in interval, the
idle cap, or the stall threshold.

## The seams, in the order a turn hits them

- **Set**: `/goal <condition>` dispatches in `run()`'s never-generate band
  (above `canQueue`, beside `/memory`), so status and clear work with no
  model and a set-while-busy parks the directive turn in the queue. The
  set arm stores the goal and then submits the condition itself through
  `run()` -- CC's "setting a goal immediately starts a turn on it", with
  mentions, hooks and queueing all inherited for free.
- **Evaluate**: the top of `dispatchStopAndContinueIfBlocked`
  (`AppModel+AgentLoop.swift`) calls `handleGoalAtStop` BEFORE the user's
  own Stop hooks -- exactly where CC's goal hook sits among the blockable
  turn-end hooks. The seam fires when the steps run out AND at every
  prose reply, which is CC's "when Claude tries to stop".
- **Continue**: a not-met verdict appends a status row and calls
  `continueAgentLoop(step: 0)`. Step 0 is deliberate: the goal is
  standing user intent and the evaluation round is the unit of work now;
  continuing at `resumeStep` would make every post-cap round evaluate
  once per single step. The goal's continuation returns `true` ("the
  turn re-entered") without consuming the Stop-hook 8-reentry cap; the
  user's Stop hooks are consulted only at the turn the goal actually
  ends (met, impossible, stall, or evaluator error).
- **Reset**: every accepted user prompt (in `run()`'s submission task)
  lifts a stall pause and zeroes the idle check-in count. CC's cap is
  "three idle check-ins per goal BETWEEN YOUR PROMPTS".

## Deferral and check-ins

`runningGoalTasks(chatID:)` flattens the two registries that defer
evaluation -- background agents (`backgroundAgentRuns`, this chat,
running) and background shells (`BackgroundShellManager.runningRecords`)
-- the same enumeration `stopBackgroundWork(forDeletedChat:)` uses. Any
running work means NO evaluation on that pass. The pure decision is
`GoalPolicy.deferralPass`, a port of CC's new-run rule: work STARTED
after the last deferral pass restarts the backoff, but only when the
last pass is itself older than one base interval.

Two delivery paths, CC's two:

- **Turn end**: at the seam, a check-in due by backoff is injected as a
  `<goal_checkin>` user row and the loop continues on it.
- **Idle timer**: the pipeline's FIRST recurring timer
  (`goalIdleTimerTasks`, a MainActor `Task`). It exists for the quiet
  case nothing else covers: background shells have NO completion
  notification, so a deferral whose work drains while the app is idle
  would otherwise wait forever. On fire it re-checks everything: busy or
  unselected chat re-arms short, the idle cap re-arms WITHOUT injecting
  (CC's "tick re-armed only"), the third delivery announces the pause,
  and drained work gets the "no longer running" text and starts a turn
  (`executeGenerationTurn(step: 0)`, the task-notification shape).

Constants (all CC's published values, in `GoalPolicy`): first check-in
30 min, doubling per check-in, capped at 4x (30 m -> 1 h -> 2 h -> 2 h);
3 idle check-ins per goal between prompts; stall threshold 3; evaluator
timeout 30 s.

## The evaluator

`GoalEvaluator` + `runGoalEvaluation`: a side `session.generate` in the
`performCompaction` shape (reasoning off, temperature 0.2, 200-token
cap), judged against the transcript since the last evaluation
(`goalEvalMessageCounts`), racing a 30 s timeout. The judge never runs
tools; it answers one JSON object (`ok` / `reason` / `impossible`),
parsed tolerantly (first `{` through the last `}`).

**A nil verdict is TRANSIENT** -- timeout, cancellation, malformed JSON.
The goal stays, the turn ends normally, and the next stop re-evaluates.
Only an unloaded model clears the goal on its own (CC's unrecoverable
error, the one case this app has); the toast names it and says to run
`/goal` again.

**Stall detection replaces an iteration cap** (CC's design): three
consecutive EVALUATION ROUNDS whose transcript slice carried no tool
call or tool result pause the goal. A round, not a step -- one round can
hold five tool steps.

## State and persistence

The row field `AppChat.goal` (tolerant decode; ghost chats carry theirs
in the vault payload, and a goal alone counts as ghost content) is the
source of truth; `activeGoals` is the published mirror, kept in step by
`updateGoal` and hydrated by `restoreGoalsFromRows()` in `loadChats()`.
Restore keeps only `condition` and `setAt` (`restoredForRelaunch`) --
CC's resume rule: counters, timer and token baseline reset. Chat
deletion runs `teardownGoal`, which is what stops the idle timer from
re-arming forever over a dead chat.

The model's copy of the goal is the assembly-time `<goal>` reminder
section (`SystemReminders.goalSection`, condition + iterations + latest
reason + paused flag), appended like the todo and plan reminders to the
model-bound copy of the last user turn. That is how the goal survives
compaction: verdict rows scroll away, the reminder recomputes every
assembly. The SKILL.state history path gets none, like every reminder.

## Deviations from Claude Code, recorded

- **The judge is the main model**, not a Haiku-class small model: this
  app has one loaded session. Evaluation cost therefore rides the main
  model's throughput and budget; the `tokensSpent` field on the goal is
  the display for it.
- No `CLAUDE_CODE_GOAL_CHECKIN_MINUTES` knob: the intervals are
  constants in `GoalPolicy`. A settings key is the natural follow-up if
  anyone wants to tune them.
- `/clear` has no equivalent here (this app has no clear command); the
  goal clears via `/goal clear` or the banner's Stop button, and dies
  with the chat.

## UI

`GoalBannerView` sits in the transcript after `TaskChecklistPanelView`
(outside the streaming row, same rationale): condition, elapsed time,
iteration count, latest reason, the deferral/paused/evaluating states,
and the Stop button. Its labels are the six catalog keys this feature
added (`Goal active`, `Stop goal`, `Iterations`, `Evaluating`, `Waiting
for background work`, `Paused`), full 21-language parity. Verdict rows,
check-in rows and toasts are plain English transcript content, like the
guardrail nudge and `<task-notification>`, not catalog keys. One trap
the parity scanner caught during review: a bare `Text("\(count)")`
literal is flagged -- use `Text(verbatim:)` for computed numbers.

## Tests

`GoalPolicyTests` (backoff matrix, deferral new-run rule and due-ness,
idle cap, check-in text shapes, condition clipping, verdict parsing
including prose-wrapped and impossible-over-ok, status rows, restore
stripping), `GoalStateTests` (archive round-trip, the pre-goal and
wrong-typed decodes, ghost payload + `hasContent`, reminder sections,
elapsed formatting), plus the `/goal` assertions in
`ComposerAutocompleteTests`' pinned meta-command set and the
autocomplete row list. All mutation-checked: backoff cap exponent, idle
cap, impossible priority, the decode read-back line, and the reminder's
condition line each redden exactly their own cases.
