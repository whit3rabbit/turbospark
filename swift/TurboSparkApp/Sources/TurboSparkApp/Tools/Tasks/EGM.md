# Tasks

## Overview
Model-facing tools for delegating and tracking work: spawning subagents, a per-chat task list, a TodoWrite checklist, and a `batch` transport that runs several tool calls concurrently. The directory holds tool schemas (`OpenAITool` definitions), Codable payload types, and the executors that tolerate sloppy argument shapes from different model families. Task state is process-memory only; nothing here persists to disk.

## Interface
- Subagent tool: `AgentInput`, `AgentOutput` (`completed`, `asyncLaunched`, `remoteLaunched` payloads), and `AgentToolDefinitions.agent` / `.stopAgent` / `.all` [AgentTools.swift:AgentToolDefinitions]. `stop_agent` is named separately from `TaskStop` because `TaskStop` belongs to `TaskManager`.
- Task list: `TaskManager.shared` with `createTask`, `getTask`, `listTasks(chatID:)`, `updateTask`, `stopTask`, the `onTasksUpdated` callback, and static executors `executeCreate/Get/List/Update/Stop/Output` [TaskManager.swift:TaskManager].
- Task payload types and schemas: `TaskCreateInput`, `TaskGetInput`, `TaskUpdateInput`, `TaskStopInput`, `TaskOutputInput`, their outputs, `TaskItemInfo`, and `TaskItemToolDefinitions` (TaskCreate, TaskGet, TaskUpdate, TaskList, TaskStop, TaskOutput, TodoWrite; `.all`) [TaskItemTools.swift:TaskItemToolDefinitions].
- Checklist: `TodoItem` (status predicates `isCompleted`, `isInProgress`, `isPending`, `isCancelled`), `TodoWriteInput`, `TodoWriteOutput` [TaskItemTools.swift:TodoItem]; `TodoWriteExecutor.parseTodos`, `.execute`, and the global `onTodosUpdated` hook [TodoWriteExecutor.swift:TodoWriteExecutor].
- Batch: `BatchToolDefinitions.batch` / `.all` [BatchToolDefinitions.swift:BatchToolDefinitions]; `BatchToolExecutor.parseItems`, `.execute`, `.maxBatchSize` (25), `BatchItem` [BatchToolExecutor.swift:BatchToolExecutor]; internal `MutationGate` actor.
- Child directories: none.

## Invariants
- `batch` must stay a transport only. Each child is re-checked: caller tool filter (`AppToolRegistry.callerToolFilter`), PreToolUse hooks, `AppToolPermissionEngine`, and for `.ask` a PermissionRequest hook that must return allow. A batch has no approval UI, so approval of the outer call never covers a child [BatchToolExecutor.swift:execute].
- Recursive `batch` is rejected (case-insensitive name match), empty arrays are rejected, and more than 25 items is rejected [BatchToolExecutor.swift:execute].
- Children whose category is `.fileWrite` or `.terminal` run one at a time through `MutationGate` (FIFO) so concurrent edits to one path cannot lose a write; read-only children run in parallel. Results are re-sorted by original index before formatting [BatchToolExecutor.swift:MutationGate]. The `batch` schema text says the same (it no longer claims every child runs concurrently) [BatchToolDefinitions.swift].
- `AgentInput.isolation` is decoded for tolerance but no executor honors it, so the `Agent` schema does not advertise it [AgentTools.swift].
- Batch parameters are flattened to `[String: String]`; nested arrays/objects are re-encoded as sorted-key compact JSON to match the non-batch path [BatchToolExecutor.swift:AnyCodableValue].
- `TodoWriteExecutor.execute` throws (leaving the list unchanged) when parsing yields nothing unless the caller passed an explicit empty list (`"[]"` or `{"todos": []}`). Do not relax this: a truncated payload would otherwise wipe the checklist while reporting success [TodoWriteExecutor.swift:isExplicitClear].
- Todo ids without a model-supplied id are positional (`todo-N`), with collision suffixes, because row identity in the checklist view must be stable across updates. `TodoItem.activeForm` falls back to `content` [TodoWriteExecutor.swift:extractItems, TaskItemTools.swift:TodoItem].
- `activeForm` is optional in the TodoWrite schema on purpose; models that never send it must still validate [TaskItemTools.swift:todoWrite].
- `TaskManager` guards `tasks` and `chatTasks` with an `NSLock`; `onTasksUpdated` is invoked after the lock is released. A nil `chatID` maps to the `"default"` list key. `listTasks(chatID: nil)` returns every task sorted by creation date [TaskManager.swift:TaskManager].
- `TaskUpdate` with status `deleted` removes the task from the store and from every chat list (the returned snapshot has status `deleted`) [TaskManager.swift:updateTask].
- `createTask` checks the generated 8-char id against the store and regenerates on collision (full UUID after 16 tries) so an insert never overwrites another task; `idGenerator` is internal for tests [TaskManager.swift:createTask].
- `TaskOutput` (`executeOutput`, async) reads only the `output` stored on the task-list entry. With `block` true it polls until status is terminal (completed/done/stopped/failed/cancelled/deleted) or `timeout` ms (default 10000, capped at 600000) pass, then appends a timed-out note. It does NOT read a background shell or subagent: tasks are not linked to `BackgroundShellManager` or the subagent registry. `block` and `timeout` are optional in the schema; only `task_id` is required [TaskManager.swift:executeOutput].
- `TaskManager.stopTask` only sets status to `"stopped"`; it does not cancel any process or subagent. `TaskItemInfo`/`TaskUpdateInput` carry fields (`blocks`, `blockedBy`, `owner`) that `TaskManager` does not store.
- `executeGet`, `executeOutput` read tasks by id globally, not scoped to a chat [TaskManager.swift:executeGet].
- Error codes under domain `TurboSparkTool`: 35-38 batch, 50-54 task executors, 80 TodoWrite.

## Side Effects
- Batch execution dispatches PreToolUse, PermissionRequest, PostToolUse, PostToolUseFailure hooks via `AppHookExecutionEngine.shared`, reads `SessionApprovalStore.shared`, and runs arbitrary child tools through `AppToolRegistry.execute` [BatchToolExecutor.swift:execute].
- `TaskManager` mutates in-memory dictionaries and fires `onTasksUpdated` [TaskManager.swift:createTask].
- `TodoWriteExecutor` fires the static `onTodosUpdated` callback [TodoWriteExecutor.swift:execute]. Task IDs and default todo IDs use `UUID()`; timestamps use `Date()`.
- No disk, network, Keychain, UserDefaults, or process spawn in this directory itself.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
