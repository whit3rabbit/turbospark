# Swift tools: execution, containment, and adding or removing a tool

`swift/TurboSparkApp` runs model-proposed tool calls in process: file reads
and writes, a shell, web search and fetch, skills, subagents, user-defined
JSON tools, and MCP servers. This page is the map for a developer who has to
add a tool, change one, or take one out. It says where each piece lives,
which lists a tool name has to appear in, and which tests pin the result.

It does not restate what other pages own. `docs/PERMISSION_GATE.md` owns the
terminal command classifier and every number about it, and its "Where it
hooks" section is the decision ladder for a shell command. `swift/CLAUDE.md`
Gotchas 11, 29, and 30 record why the gate and the containment have the shape
they do, and its `state#N` ledger is the incident history behind every
parenthesised `state#` below.

All paths on this page are relative to
`swift/TurboSparkApp/Sources/TurboSparkApp/` unless they start with `Tests/`
or `docs/`.

## 1. The pipeline as it runs

A tool call passes through five stages, and the permission gate is NOT one
of the registry's. It sits in the caller.

1. **Parse.** `Tools/Core/ToolCallParser` pulls calls out of the reply. It
   is the only parser, with two callers, `State/AppModel+AgentLoop` and
   `State/SubagentRunner`. Each keeps its own guard and neither keeps its
   own parser. On the main loop, `Tools/Guardrails/ForgeGuardrailsEngine`
   runs first (`State/AppModel+Generation.swift`), rescuing a call the
   parser could not read and checking its arguments against the schema the
   turn advertised through `AppToolCatalog.tools(for:)`.
2. **Gate, in the caller.** `PreToolUse` hooks run first and fail CLOSED
   (state#40). Then `Tools/Core/AppToolPermissionEngine.evaluate`, which
   asks `Tools/Core/ToolRiskClassifier` for a risk assessment and applies
   eight ordered steps: strict read-only mode, category deny, high-risk
   always asks, permissive mode, session approval, MCP `autoApprove`, ask
   mode, auto mode. For a shell command the classifier consults
   `Tools/Terminal/TerminalCommandClassifier.isAutoApprovable` (a positive
   allowlist) and `Tools/Core/CommandGate` (advisory reason on, veto off).
   A `.ask` raises the approval card. A `.deny` is recorded as refused.
3. **Refuse or run.** `AppToolRegistry.execute` does exactly one check of
   its own before the switch: a tool that resolves a path or spawns a
   process is refused by name in a chat with no project (section 3). Then
   the `case` for the name runs. Nothing in `execute` consults the
   permission engine, so a direct call to it is ungated. The unit tests
   rely on that.
4. **Return.** The result goes back to the model as a `.tool` message and
   never as a mid-history `system` one (state#32, state#74).
5. **Render.** `Generation/ToolCallDiffFormatter.summarize` builds the
   card's one-line summary by tool name, and `Generation/ToolCallCardView`
   carries one name-keyed branch, the todo checklist.

The two callers differ on purpose:

| | Main loop | `SubagentRunner` |
|---|---|---|
| A `.ask` verdict | approval card | refused, no card (state#18) |
| Hooks | all events | `PreToolUse` and `PostToolUse`, never `PermissionRequest` (state#68) |
| Advertised tools | the project's `tools(for:)` slice | the same slice, not `allTools` (state#47) |
| Nesting | n/a | `subagentDepth` bounds `agent` calling `agent` |
| `chatID` | the turn's chat | threaded through, so a checklist lands on the right chat |

## 2. Where a tool name has to appear

The vocabulary is several hand-maintained lists that describe one thing, and
each omission is a distinct, silent failure. `AppToolRegistry+Vocabulary`
says so in its header for the four it holds. The full set is longer.

| Surface | File | Missing from it means |
|---|---|---|
| a `*Definitions.all` list, appended to an `AppToolCatalog` collection and the right `tools(for:)` profiles | `Tools/<Domain>/*.swift`, `Tools/Registry/AppToolCatalog.swift` | the model never sees the tool |
| `AppToolRegistry.supportedToolNames` | `Tools/Registry/AppToolRegistry+Vocabulary.swift` | `isImplemented` filters it out of every advertised list, and a call to it errors as unimplemented |
| `AppToolRegistry.workspaceRootedToolNames` | same file | in a projectless chat it runs against the `/dev/null` placeholder root (state#70) |
| `AppToolCatalog.category(for:)` | `Tools/Registry/AppToolCatalog.swift` | the `default` arm returns `.automation`, so the call is gated on the wrong permission switch (state#71) |
| the `switch` in `execute` | `Tools/Registry/AppToolRegistry.swift` | the honest "not implemented by this client" error (T5) |
| the always-safe arm of `ToolRiskClassifier.assessRisk` | `Tools/Core/ToolRiskClassifier.swift` | a read-only tool is assessed like an unknown one |
| `AppHookToolNameAliases.claudeCodeToApp` | `Tools/Hooks/AppHookToolNameAliases.swift` | a hook matcher written for the Claude Code spelling never fires on this tool |
| `ToolCallDiffFormatter.summarize` | `Generation/ToolCallDiffFormatter.swift` | a generic card summary (cosmetic) |

Two names look like surfaces and are not. `AppToolRegistry.standardTools`
has no reader: `systemPromptAddendum(for:tools:)` ignores its `tools`
argument and builds the prompt from `tools(for:)`. And `mcp__server__tool`
names are never listed anywhere. `isImplemented`, the rooted check, and the
`default` arm all recognise the prefix dynamically.

## 3. Rules the registry enforces itself

**No project means no root, and no root means no file or shell tool.** The
refusal is by name: `workspaceRootedToolNames`, plus every `mcp__` name,
plus every custom tool (state#70). `skill`, `todowrite`, `agent` and the
web tools need no root and still run. There is no defensible default root.
`/` makes the containment check a no-op and `~` holds every credential on
the machine (Gotcha 30).

**Containment is on the FILE tools only.** `resolveSecurePath` refuses
absolute and `~` paths and compares the resolved target against the resolved
root with symlinks followed on both sides. `AppToolSandbox.validateWritePath`
adds a write-path allow and deny list. `run_command` takes no path and a
shell leaves the root by its own means, so for it the permission gate is the
whole story (Gotcha 11).

**Every read is bounded before it happens.** `AppFileReadLimits` caps a
single read at 16 MiB and checks the size before loading. `searchCode` stops
at 5,000 files, 64 MiB, or 40 matches. `compactOutput` keeps 25 head and 65
tail lines of anything over 120. `ProcessExecutor` (internal, not public)
carries a timeout and an output cap.

The app is unsandboxed. Nothing here is a sandbox, and `AppToolSandbox` is a
path and domain policy rather than one.

**A failure throws, so `isError` follows.** A non-zero exit from
`run_command` throws with the combined output (state#81). A stale file under
`edit_file` throws because `FileSnapshotStore` saw it change since the last
read. A `CancellationError` is caught separately and reaches the model as
"Stopped by the user before it finished. Do not retry" rather than as
`Error: cancelled` (state#64).

**No fabricated success (T5).** The `default` arm throws for a name it does
not implement, and `AppToolCatalog` filters every advertised list through
`isImplemented` so the model is never offered a tool with no executor.
Section 8 records the three arms that still violate this.

**Arguments are `[String: String]` and models spell keys loosely.** Every
arm reads several spellings (`path` or `file_path`, `command` or `cmd`,
`patch_text` or `patch`). `TodoWriteExecutor.parseTodos` accepts four JSON
shapes. Read `end_line` and `limit` from their own keys: one is an absolute
bound and the other a count, and collapsing them once returned 520 lines for
a 21-line request.

## 4. Directory layout

Everything a tool call passes through is under `Tools/`. `Registry/` is the
five files adding a tool touches.

| Path | What is in it |
|---|---|
| `Tools/Registry/AppToolRegistry.swift` | `execute`: the rooted refusal, the `switch`, `executeMcpCall` |
| `Tools/Registry/AppToolRegistry+Handlers.swift` | the file and shell handlers: `resolveSecurePath`, `listDirectory`, `readFile`, `writeFile`, `editFile`, `searchCode`, `runCommand`, plus `AppFileReadLimits` and `compactOutput` |
| `Tools/Registry/AppToolRegistry+Vocabulary.swift` | `supportedToolNames`, `workspaceRootedToolNames`, `isImplemented`, `standardTools` |
| `Tools/Registry/AppToolTypes.swift` | `AppToolCategory`, `AppToolCall`, `AppToolResult`, `AppToolDefinition` |
| `Tools/Registry/AppToolCatalog.swift` | the eight `OpenAITool` collections, `allTools`, `tools(for:)`, `category(for:)`, `systemPromptAddendum` |
| `Tools/Core/OpenAIToolSchema.swift` | `OpenAITool`, `JSONSchema`, the serializer |
| `Tools/Core/ToolCallParser.swift` | the one parser, XML block first and the markdown form as fallback |
| `Tools/Core/AppToolPermissionEngine.swift` | `evaluate`, `ToolPermissionDecision`, `SessionApprovalStore` |
| `Tools/Core/ToolRiskClassifier.swift` | `assessRisk`, sensitive-path list, `assessTerminalCommand` (the classifier's hook point) |
| `Tools/Core/CommandGate.swift`, `CommandLexicalFeatures.swift`, `HashedFeatureVectorizer.swift` | the bundled logistic model: weight loading, features, scoring. `docs/PERMISSION_GATE.md` owns these |
| `Tools/Core/AppToolSandbox.swift` | write-path and domain allow and deny configuration |
| `Tools/Core/ProcessExecutor.swift` | bounded subprocess launcher: timeout, output cap, one reader thread per pipe (state#66) |
| `Tools/Hooks/` (11 files) | the hook engine: models, store and discovery, matcher, command runner, decision aggregator, stdin payload, tool-name aliases |
| `Tools/Guardrails/ForgeGuardrailsEngine.swift` | tool-call rescue and schema check before parsing. Not the memory guardrails (Gotcha 24) |
| `Tools/Custom/` | `CustomToolDefinition`, `CustomToolParser`, `CustomToolManager` (scopes and directories), `CustomToolExecutor` |
| `Tools/MCP/` | `McpClientEngine` (stdio and SSE), `McpServerSpec`, `McpTools` (resource tool schemas), `McpResourceExecutor`, `ProjectMcpDetector` |
| `Tools/File/` | `FileReadWriteTools`, `FileSearchTools`, `ApplyPatchTool` (schemas), `NotebookEditExecutor`, `SnipExecutor`, `SendUserFileExecutor`, `ApplyPatchExecutor`, `FileSnapshotStore` |
| `Tools/Terminal/` | `TerminalTools` (schema only), `TerminalCommandClassifier` (the auto-approve allowlist, and `isCollapsible` which is presentation only) |
| `Tools/Web/` | `WebTools` (schemas), `WebSearchExecutor` (Exa, Parallel, Brave, SearXNG), `WebFetchExecutor` |
| `Tools/Tasks/` | `AgentTools`, `TaskItemTools` (schemas), `TaskManager`, `TodoWriteExecutor` |
| `Tools/Planning/` | `PlanningInteractiveTools`, `PlanningInteractiveExecutors` (interactive questionnaires, plan mode, findings, skills/goals), `SkillTool` (schemas) |
| `Tools/Projects/` | `ArtifactWorktreeTools`, `ProjectDocTools`, `WorktreeExecutor` (git worktree isolation) |
| `Tools/Automation/` | `MonitoringNotificationTools`, `AutomationExecutors` (sleep, push notifications, config inspection, context telemetry) |

Three files outside `Tools/` matter: `State/AppModel+Tools.swift` (approval
flow and the prompt addendum), `Generation/ToolCallDiffFormatter.swift`, and
`Generation/ToolCallCardView.swift`. `AppModel.init` in `State/AppModel.swift`
is where executor callbacks are wired.

Tests, all under `Tests/TurboSparkAppTests/`:

- `FabricatedToolSuccessTests.swift`: the T5 pin. Enumerates implemented and
  excluded names, per agent type.
- `WorkspaceAndNetworkGateTests.swift`: the rooted and rootless split.
- `ToolPermissionsTests.swift`: categories, the four permission modes,
  session approval, sensitive paths.
- `TerminalRiskGateTests.swift`: the evasion corpus against the allowlist.
- `SubagentPermissionTests.swift`: the second execution path refuses rather
  than executes.
- `CoreFileToolsTests.swift`: directory listing, file reading/writing/editing,
  patch application, code searching.
- `PlanningAndInteractiveToolTests.swift`: user questions, plan modes,
  findings reports, skill/goal proposals, feedback, todo writes.
- `FileInspectionAndNotebookToolsTests.swift`: Jupyter notebook cell edits,
  code snipping, user file presentations.
- `TaskAndProcessToolsTests.swift`: background task manager and shell execution.
- `AutomationAndEnvironmentToolsTests.swift`: sleep, notifications, config,
  context inspection, git worktrees.
- `McpAndIntegrationToolsTests.swift`: MCP tools and resources, web search/fetch,
  skills, agent loop.
- `ExtendedToolExecutionTests.swift`: multi-tool integration matrix.
- `CustomToolsTests.swift`, `FileToolExecutionTests.swift`,
  `ToolCallParserTests.swift`, `TodoWriteExecutorTests.swift`,
  `ProcessExecutorTests.swift`, `ApplyPatchExecutorTests.swift`,
  `McpClientEngineTests.swift`: per component.

## 5. Three ways to add a tool, cheapest first

**A custom JSON tool, no code.** One JSON file per tool. Project scope is
`<project>/.turbospark/tools/`. User scope is `~/.turbospark/tools/` or the
`tools` directory under the app's Application Support root. The file
declares a name, description, `JSONSchema` parameters, a category, and an
execution block of type `command`, `script` or `http`. Read
`Tools/Custom/CustomToolDefinition.swift` for the fields rather than a copy
of them here.

A command runs through `/bin/zsh -c` with every argument value
single-quoted, substituted in one pass so a value cannot be substituted
into, and offered again as `TOOL_ARG_<KEY>` in the environment (state#70).
A custom tool is always workspace-rooted and is gated on the category it
declares (state#71).

**An MCP server, no code.** Configure it globally or in the project's
`.mcp.json`. Its tools arrive as `mcp__<server>__<tool>` and route
dynamically. A global server wins a name collision with a project one
(state#61). Always rooted: the server gets the project root as its working
directory. Stdio children get `PATH`, `HOME`, `LANG`, `TMPDIR` and the
config's own `env`, nothing else (Gotcha 32).

**A native Swift tool.** Section 6.

## 6. Adding a native tool

Do these in order. Step 8 is where the build tells you what you missed.

1. **Schema.** An `OpenAITool.function(name:description:parameters:)` in a
   `*Definitions` enum under `Tools/<Domain>/`, appended to that enum's
   `all`. Pick an existing domain directory before making a new one.
2. **Catalog** (`Tools/Registry/AppToolCatalog.swift`). Add the enum's `all`
   to a collection, add the collection to each `tools(for:)` profile that
   should see it, and add an explicit arm in `category(for:)`. The `default`
   arm is `.automation`. Landing there is a bug rather than a default.
3. **Vocabulary** (`Tools/Registry/AppToolRegistry+Vocabulary.swift`). The
   lowercase name and every alias into `supportedToolNames`. Into
   `workspaceRootedToolNames` as well if the handler resolves a path or
   spawns a process.
4. **Handler and `case`** (`Tools/Registry/AppToolRegistry.swift`). The
   file tools keep their handlers as statics in `+Handlers`. Web, patch, and
   todo use a separate `*Executor` enum with a static `execute`. Either
   shape is fine. The handler must throw on failure so `isError` follows,
   never return a success string for work not done, bound every read
   through `AppFileReadLimits` or `compactOutput`, go through
   `resolveSecurePath` and `AppToolSandbox.validateWritePath` for any path,
   and let `CancellationError` propagate.
5. **Aliases and risk.** Add the Claude Code spelling to
   `AppHookToolNameAliases.claudeCodeToApp` so hooks match it. If the tool
   is read-only, add its names to the always-safe arm of
   `ToolRiskClassifier.assessRisk`.
6. **State callback, if the tool changes app state.** Copy
   `TodoWriteExecutor.onTodosUpdated`: a static closure on the executor,
   assigned in `AppModel.init`, hopping to `@MainActor` and keying on the
   `chatID` the call carried rather than the selection. Execute takes a
   `chatID` for exactly this reason.
7. **Presentation, optional.** A branch in `ToolCallDiffFormatter.summarize`
   keyed on the name.
8. **Tests, not optional.** `FabricatedToolSuccessTests` enumerates the
   implemented set per agent type and goes red on a new advertised name
   until step 3 is done. `WorkspaceAndNetworkGateTests` pins the rooted
   split for step 3's second half. `ToolPermissionsTests` pins categories
   for step 2. Add an execution test for the handler's happy path, its
   missing-argument error, and its refusal without a project if rooted.

Run from the package directory. The `-L` linker flag in `Package.swift`
resolves against the working directory, so `--package-path` from the
repository root fails at link time (Gotcha 41):

```bash
cd swift/TurboSparkApp && swift test --filter FabricatedToolSuccessTests
```

```bash
cd swift/TurboSparkApp && swift test
```

Read the `Executed N tests, with M failures` line. The swift-testing summary
above it reports zero tests and is not the result (Gotcha 44).

## 7. Removing a tool

Reverse the order of section 6 and expect the same three test files to name
what is left. Remove the `case`, both vocabulary sets, the category arm, the
alias entry, the catalog registration, then the schema.

Half-removals fail in predictable ways. A schema left behind with no `case`
is filtered out by `isImplemented` and never reaches the model, which is the
intended failure. A `case` left behind with no schema is dead code the
compiler will not flag. A name left in `supportedToolNames` with no `case`
is advertised and then errors honestly on use.

## 8. Case study: TodoWrite

The one stateful, interactive tool, and the reference for anything that
touches a chat rather than the filesystem.

1. **Schema** (`Tools/Tasks/TaskItemTools.swift`): `TodoWrite` takes
   `todos`, each with `content`, `status` (`pending`, `in_progress`,
   `completed`, `cancelled`) and `activeForm`.
2. **Executor** (`Tools/Tasks/TodoWriteExecutor.swift`): `parseTodos`
   accepts four JSON shapes, `execute(arguments:chatID:)` renders a markdown
   checklist and fires `onTodosUpdated(chatID, todos)`.
3. **Wiring** (`State/AppModel.swift`, in `init`): the closure hops to the
   main actor and calls `updateTodos(for:todos:)` on the chat the call named,
   falling back to the selection only when the call carried none. The chat
   persists to `chats_archive.json` on the next `persistChats()`.
4. **Card** (`Generation/ToolCallDiffFormatter.swift`,
   `Generation/ToolCallCardView.swift`): the formatter's `todo` branch
   builds the "3/5 done" summary and the card view renders the checklist.
5. **Category**: `.fileWrite`, listed in `category(for:)`. Rootless: it is
   absent from `workspaceRootedToolNames` on purpose.

## 9. Known gaps

- `standardTools` is dead and `systemPromptAddendum(for:tools:)` ignores
  its `tools` parameter.
- `McpClientEngine`'s SSE transport throws by design until it is
  implemented (Gotcha 32).
- `Projects`, `Artifact`, `REPL`, and `Workflow` are schema definitions
  without local engines and are filtered out by `isImplemented` rather
  than stubbed (T5).
