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

Hook output contract, and where this client deliberately differs from
Claude Code (https://code.claude.com/docs/en/hooks):

- `continue: false` (with optional `stopReason`) ENDS the turn and shows
  the reason to the user, on every event. It is not the block path: a
  Stop hook's `decision: "block"` or exit 2 feeds its reason to the MODEL
  and re-enters the loop (capped at 8); a prevent-continuation verdict
  never consumes one of those re-entries.
- Plain-text stdout on exit 0 is advisory only and never reaches the
  model (Claude Code feeds it to the model on some events). This is also
  why `suppressOutput` has nothing to suppress here; the field is parsed
  and carried for a future surface that prints hook stdout.
- The legacy `decision` field maps onto the permission ladder on
  `PreToolUse`/`PermissionRequest` (`approve` allows, `block` denies)
  and blocks the action on events that carry no permission decision.
- `PermissionDenied` fires when the permission engine denies a call and
  when the user refuses one at the approval card. Its
  `hookSpecificOutput.retry` is parsed but NOT acted on: automatically
  re-running a call a human just refused is not a decision this app
  makes on a hook's word.
- `prompt` and `agent` hook types load but are never evaluated. Discovery
  says so in `discoveryDiagnostics`, the hooks pane badges the row, and a
  dispatch produces a visible non-blocking outcome. An `agent` entry maps
  to the unevaluated type rather than to `command`, which would run the
  prompt text as a shell command.
- `SubagentStart`/`SubagentStop` wrap every `SubagentRunner.run` exit;
  they are notification-grade and cannot block the run.

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
whole story (Gotcha 11). The one piece of shell state that persists is the
working directory: `ShellCwdTracker` records where each command ended via a
`pwd -P` capture appended to every command, starts the next call there, and
RESETS to the project root with a note when a command ends outside the root,
so a `cd /tmp` cannot silently widen every later call.

**Every read is bounded before it happens.** `AppFileReadLimits` caps a
single read at 16 MiB and checks the size before loading. `searchCode` stops
at 5,000 files, 64 MiB, or 40 matches. `compactOutput` keeps 25 head and 65
tail lines of anything over 120. `ProcessExecutor` (internal, not public)
carries a timeout and an output cap, and `ShellOutputFormatting.compact`
bounds what the MODEL sees from one shell command at 30,000 characters (head
20K and tail 8K around a marker, applied after ANSI stripping).

The app is unsandboxed. Nothing here is a sandbox, and `AppToolSandbox` is a
path and domain policy rather than one.

**A failure throws, so `isError` follows.** A non-zero exit from
`run_command` throws with the combined output (state#81), and a TIMEOUT
throws the same way with whatever output the command produced -- a timeout is
work that did not finish, never a success string. The exceptions are the
benign exit codes (`ShellOutputFormatting.benignExitNote`): grep/rg 1 is
"No matches found", diff 1 is "Files differ", test/[ 1 is "Condition
evaluated to false", and each returns as a non-error with the note appended,
because those are commands ANSWERING rather than failing. A stale file under
`edit_file` throws because `FileSnapshotStore` saw it change since the last
read. A
`CancellationError` is caught separately and reaches the model as
"Stopped by the user before it finished. Do not retry" rather than as
`Error: cancelled` (state#64).

**The shell runs merged, bounded, and with the hang-prevention environment.**
`ShellCommandRunner` passes `mergeStreams: true` to `ProcessExecutor`, so
stdout and stderr interleave in arrival order instead of arriving as a
stdout block followed by a stderr block; hooks keep the streams separate
because their verdict JSON parses per stream. The child environment inherits
the app's own plus `GIT_EDITOR=true`, `GIT_PAGER=cat`, `PAGER=cat`,
`TERM=dumb` and `NO_COLOR=1`, so a messageless `git commit` fails fast
instead of hanging on an invisible editor.

**Background execution is real.** `run_in_background: true` hands the
command to `BackgroundShellManager` and returns an id (`bg_N`) immediately:
no timeout applies, the output accumulates in the same 1 MB capped buffer
design, and `BashOutput` (`task_id`, optional `wait_seconds` up to 120)
polls or waits for a snapshot while `KillShell` terminates via the
SIGTERM-then-SIGKILL ladder. Ids are scoped to the launching conversation,
so a subagent or an unrelated chat can neither read another turn's output
nor kill another turn's process, and there is a ceiling of 20 concurrently
running shells.

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
| `Tools/Registry/AppToolRegistry+Handlers.swift` | the file handlers: `resolveSecurePath`, `listDirectory`, `readFile`, `writeFile`, `editFile`, `searchCode`, plus `AppFileReadLimits` and `compactOutput` (the shell handler lives in `Tools/Terminal/ShellCommandRunner.swift`) |
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
| `Tools/Terminal/` | `TerminalTools` (schemas: `Bash`, `BashOutput`, `KillShell`), `ShellCommandRunner` (the execution path: wrapping, timeout clamp, output shaping, background handoff), `BackgroundShellManager` (the background registry), `ShellCwdTracker`, `ShellOutputFormatting`, `TerminalCommandClassifier` (the auto-approve allowlist, and `isCollapsible` which is presentation only) |
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
`.mcp.json`, or install it from a catalog (section 10). Its tools arrive as
`mcp__<server>__<tool>` and route dynamically. A global server wins a name
collision with a project one (state#61), which is why both editors refuse a
duplicate name: the loser can never be dialled.

Always rooted, but a server may state WHERE. The caller passes the project
root and a server's own `cwd` beats it, because the caller's value is a
default applied to every server alike while a `cwd` is a choice
(`McpClientEngine.resolveWorkingDirectory`). An empty `cwd` falls through to
the default rather than meaning the filesystem root.

Stdio children get `PATH`, `HOME`, `LANG`, `TMPDIR`, then any host variable
the config NAMES in `envPassthrough`, then the config's own `env`, and
nothing else (Gotcha 32). The passthrough is an allowlist of names and must
stay one: the child environment is built from scratch precisely so a
third-party binary does not receive every credential the app was launched
with, and a wildcard would undo that.
`McpClientEngine.childEnvironment` is a pure function so that rule is
testable without spawning anything, and
`testAVariableThatWasNotNamedIsNotForwarded` is the case that pins it.

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
  implemented (Gotcha 32). Remote servers therefore have no OAuth, no
  reconnect, and no streamable HTTP.
- MCP `prompts/list` (prompts as slash commands), real
  `resources/list` / `resources/read` (the two resource tools answer from
  stubs), and server `instructions` injection are not implemented.
- Detection of project MCP config files scans the project ROOT only. The
  reference implementation also walks parent directories; importing
  servers declared OUTSIDE the workspace the user pointed at is the
  riskier behavior, so this port does not.
- `Projects`, `Artifact`, `REPL`, and `Workflow` are schema definitions
  without local engines and are filtered out by `isImplemented` rather
  than stubbed (T5).

## 10. Installing an MCP server from a catalog

A catalog is a JSON manifest in a Git repository or a local folder, listing
servers a user can install in one click. Settings, MCP Servers, Browse
Marketplace.

### The manifest

`mcp-marketplace.json` at the repository root, or at the path the source
names.

```json
{
  "name": "Example Catalog",
  "description": "Servers we run internally",
  "owner": "example",
  "servers": [
    {
      "name": "memory",
      "description": "Knowledge graph over the codebase",
      "category": "Storage",
      "version": "1.2.0",
      "transport": {
        "type": "stdio",
        "command": "npx",
        "args": ["-y", "@example/memory"],
        "cwd": "~/work",
        "envPassthrough": ["GITHUB_TOKEN"]
      }
    }
  ]
}
```

Every field except `name` and `transport` is optional. `transport` is
deliberately NOT tolerant-decoded, the same decision
`McpServerConfig.transport` makes: an entry with no readable transport cannot
be launched, so the row drops rather than the catalog failing. One unreadable
entry is one dropped row (`decodeLossyArray`); a `servers` key that is not an
array still throws, so a mangled file is quarantined rather than read as an
empty catalog.

### What install does, and what it refuses

`McpMarketplaceManager.makeServerConfig` validates BEFORE anything reaches
the store, which is state#107's rule: the skills equivalent copies files into
place and then parses them, so a bad entry is already installed by the time
the throw happens. It refuses an empty name, a name already taken at the
target scope, and a command that resolves nowhere
(`McpClientEngine.resolveExecutablePath`, shared with the spawn path so two
spellings of "can this be launched" cannot drift).

**An installed server arrives DISABLED and not auto-approved.** A catalog is
a file in somebody else's repository naming a binary this app will spawn. The
hand-add path defaults to enabled only because the user typed that command
themselves. For the same reason the import sheet shows the full command line
above the Install button: what is being approved is that command, not a name.

### Acquisition

`MarketplaceSource` (shared with the skills marketplace) models `github`,
`git`, `url` and `directory`. The import sheet builds the case from the
segment the user picked rather than sniffing the text, because a heuristic
routes `git@github.com:owner/repo.git` and a bare `owner/repo` differently
for a reason the user cannot see.

Git goes through `MarketplaceGit`, which routes to `ProcessExecutor` for a
timeout and an output cap, sets `GIT_TERMINAL_PROMPT=0` so a private
repository fails instead of waiting on a prompt nobody can answer, and checks
EVERY step's exit status. The skills version checked only the clone, so a
failed `pull` was silent and the caller read a stale cache as fresh.

Registered sources persist through `AppStorageRoot` and `AppJSONStore` like
every other store here, and the clone cache is a constructor parameter so
tests get a temp directory. That is the deliberate departure from
`SkillMarketplaceManager`, which hardcodes `~/.turbospark` and so is neither
quarantine-protected nor test-redirected (Gotcha 43).

## 11. How MCP tools reach the model: advertisement, approval, rules

Three layers sit between a discovered MCP tool and a model's use of it.
Each exists because the turn path has NO `tools` array: the model reads its
vocabulary from the system prompt and emits `<tool_call>` blocks, so
whatever the prompt does not name, the model cannot call reliably.

### Advertisement (`AppToolCatalogMcp`)

`AppToolCatalogMcp.toolDefinitions` turns the cache into one advertised
entry per discovered tool, named `mcp__<server>__<tool>` -- the exact
spelling `AppToolRegistry.execute` and `AppToolPermissionEngine` resolve.
`AppToolCatalog.systemPromptAddendum(for:mcpServers:project:)` appends the
lines, and `SubagentRunner.buildSystemPrompt` appends the same section so a
subagent sees what the parent sees. The `.coder` and `.general` agent types
now include the static MCP tool group.

- Tools come from `McpToolCatalogCache`, never inline: discovery spawns the
  server and pays a full JSON-RPC handshake (seconds), and a turn must not
  wait on that. Refreshes are event-driven (project selection, server
  CRUD, Test Connection, the project approval flow), never on a timer. A
  stale entry is safe: a tool advertised but since removed fails the call
  with the server's own "unknown tool" error.
- The description line carries the argument names and types, summarized
  from the tool's input schema (`path: string, force: boolean, optional`),
  because the description is the only channel that survives the prompt.
  The server's description text is capped at 160 characters; the argument
  summary is appended after the cap, since it is the part the model cannot
  guess.
- DENY rules strip tools here, before the model sees them, at server or
  tool level. Disabled servers are refused in `toolDefinitions` too, not
  only in `visibleServers`, so a caller that forgets the filter still
  cannot advertise a server the user switched off.

### Project approval lifecycle (`AppModel+Mcp`)

A repo-declared server (`mcp.json`, `opencode.json`, `.cursor/mcp.json`,
and the other formats `ProjectMcpDetector` reads) is never dialed or
advertised until the user has answered for it once. On project selection
`detectProjectMcpServers` scans the root and sorts every declared name into
one of four buckets: already imported, in `approvedMcpJsonServers`, in
`rejectedMcpJsonServers`, or PENDING. Pending names raise
`ProjectMcpApprovalSheet` over the root view, one server at a time, with
the reference implementation's three answers: Approve (import enabled,
never auto-approved), Approve All Future (sets
`approveAllProjectMcpServers`, which also imports the config's other
undeclared names), Reject (records the name; it never prompts again). The
registries live on `AppProject` and survive relaunch. Dismissing the sheet
DEFERS rather than rejects: an undecided name re-prompts on the next
selection, and a config that GAINS a name prompts only for the new one.

Imported servers are gated again at the permission layer: a server whose
`sourcePath` is set (repo provenance) ASKS on every call in `auto` mode
unless the user explicitly opted that server into `autoApprove` in-app.
Without that arm, the auto mode's trailing default let a cloned config's
servers run non-high-risk calls silently the moment they were imported.
Permissive keeps its documented contract (explicitly chosen, still gated
by deny rules and high risk).

### Permission rules (`McpPermissionRules.swift`)

`AppProjectPermissions.mcpAllowRules` / `mcpDenyRules` hold rules in the
reference syntax: `mcp__server` covers every tool on a server,
`mcp__server__tool` is exact, and `mcp__server__*` is accepted as a
wildcard spelling. Tool names may themselves contain `__`, so everything
after the second separator is the tool name -- the same parse
`McpPermissionRule.targetOfCall` applies to a call, the approval card uses
when writing a rule, and the engine uses when matching one, so a rule is
always matched with the parse that wrote it.

Evaluation order in `AppToolPermissionEngine.evaluate`, relative to the
existing spine: a deny rule sits immediately after the category deny (an
explicitly denied tool cannot be resurrected by a session approval, an
allow rule, or `autoApprove`); an allow rule sits AFTER the high-risk gate
and BEFORE session approvals -- a grant buys back the ask prompt, never
the risk ceiling, and this call's own arguments were assessed first.

Server-declared `annotations` (`readOnly`, `destructiveHint`, ...) are
parsed at discovery time and folded into risk at evaluation time via
`ToolRiskClassifier.adjusting`. They can only RAISE risk, never lower it:
`destructiveHint` on an otherwise-safe name forces the ask (even in
permissive mode, through the high-risk gate), while a `readOnly` claim
never overrides a heuristic verdict. The lookup goes through
`McpToolCatalogCache` because the parser that precomputes most assessments
cannot know which server a bare tool name belongs to.

Tests: `McpCatalogApprovalRulesTests.swift` covers the rule parser, the
engine ordering (deny over session approval, allow under high risk, the
repo-import ask, permissive unchanged), the annotation path, deny
stripping, the approval lifecycle against a temp project root, and archive
compatibility. Every guard above is mutation-checked: disabling the deny
arm, the allow arm, the repo-import gate, the annotation adjustment, the
catalog's deny strip, its `isEnabled` filter, or flipping a decode default
each reddens exactly its own case.

## 12. Subagents: the `agent` tool, progress, batches, and background

A subagent is a tool call whose handler runs a SECOND, isolated
conversation and returns its final message as the tool result. The loop is
`State/SubagentRunner` (fresh history, its own system-prompt assembler, its
own turn budget), the gate is `State/SubagentRunner+Gate` (`.ask` DENIES:
no approval UI exists to answer it), and the definitions live in
`State/AppAgentDefinition` + `State/AgentManager` (built-ins
explore/plan/general-purpose/reviewer, user, project, and plugin scopes).
Fresh context is free: the engine holds no history, so a new message array
just prefills from zero. This section is the SURFACE around that loop.

**Result contract (Claude Code's shape).** The tool result is the
subagent's final text plus a `<subagent_meta>` trailer (agent, id, status,
turns, tool_calls, duration_s). Empty output becomes the sentence
"(Subagent completed but returned no output.)". An error or cancellation
keeps whatever the run had already produced: the error path appends the
last completed turn's text as "Partial progress before the error", and a
cancelled run returns its partial answer with status `cancelled`. A run
that exhausted `maxTurns` or overflowed its window reports `max_turns` /
`context_overflow` with its last turn explicitly marked not-an-answer.
`SubagentRunner` is the only writer of these rules; the registry's `agent`
case formats the trailer.

**Progress.** `SubagentRunner.run` takes an optional `progress` sink
(`SubagentProgressEvent`: started, turnStarted, content, toolStarted,
toolFinished, finished), emitted in order from the run's own task. The app
installs `AppToolRegistry.subagentProgressSink` at startup; it routes by
run key into `AppModel.applySubagentEvent`, which maintains
`liveSubagentRuns` (foreground, keyed by tool-call UUID, removed on
`finished`) and `backgroundAgentRuns` (keyed `bga_N`, kept). The cards are
`Generation/SubagentLiveCardView.swift`; a foreground card lives in the
active turn's row, a background one in a strip under the transcript.

**Batches.** A reply whose calls are ALL agent-family (`agent`, `subagent`,
`task`) runs them CONCURRENTLY
(`AppModel+AgentLoop.runConcurrentAgentBatch`) and records them on ONE
assistant message; the batch costs one `maxAutonomousSteps` step, not one
per subagent. Each call still passes the single-call gate. Anything else
keeps the one-call-per-turn rule (state#37): a mixed batch runs only the
first call and records the rest refused, and a batch where ANY call would
raise the approval card falls back to that same single-call path, because
the card machinery parks exactly one call. Concurrent runs do not race the
engine: each turn queues on the one session, so runs interleave whole
turns. The cost is prefix-cache thrash (alternating conversations reset the
KV prefix, so each interleaved turn prefills fully) -- the wall-clock win
is one run's tool execution overlapping another's generation, never
parallel token generation.

**Background.** `run_in_background: true` launches through
`AppModel.launchBackgroundAgent` and returns the task id immediately. The
run's `Task` is unstructured ON PURPOSE: it inherits no cancellation, so
chat Stop cannot kill it, and `stopBackgroundAgent` (the `stop_agent`
tool, or the card's stop button) is the only kill path -- a stopped run
reports `killed`, not `cancelled`. The session is CAPTURED at launch, so
unloading the model in the UI does not kill a run already going.
Completion injects a USER-role `<task-notification>` turn (task_id, status
completed/failed/killed, agent, result, turns/tool_calls/duration) into
the originating chat and, if the chat is idle, answers it with a fresh
step-0 generation turn; if a turn is in flight or an approval card is up,
the notification parks in `pendingTaskNotifications` until a turn tail
drains it. Caps: 4 running, 20 finished kept, in-memory only, process
lifetime (nothing persists across relaunch, and no sidechain transcript is
written -- both deliberate deviations from Claude Code).

**`model` is refused, not ignored.** A non-inherit `model` argument fails
the call by name: honoring it needs a second open model session (the FFI
is structurally capable; it is untested and each open model pins
gigabytes), so until that lands the honest answer is the refusal. The seam
is `SubagentRunner.run(session:)`, which already takes whatever session
the caller resolves.

Names on this surface that must stay in sync:
`AppToolRegistry+Vocabulary` (supportedToolNames),
`AppToolCatalog.category`, and `AgentToolDefinitions`. `stop_agent` is
deliberately NOT `taskstop`, which is the task-list system's verb.
Tests: `SubagentProgressTests`, `SubagentBatchRoutingTests`,
`BackgroundAgentTests` (plus the pre-existing Subagent* suites, which pin
the gate and the hooks).
