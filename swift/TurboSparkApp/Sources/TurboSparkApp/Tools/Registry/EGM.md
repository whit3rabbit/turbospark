# Tools/Registry

## Overview
The tool vocabulary and the executor switch. `AppToolCatalog` decides which `OpenAITool` definitions each agent type, project, and tool profile is offered, how a tool name maps to a permission category, and what the system-prompt tool section says. `AppToolRegistry` is the single `execute` entry that turns a validated `AppToolCall` into an `AppToolResult`: it enforces the web and workspace-root refusals, dispatches by name to the file, shell, web, browser, computer-use, skill, agent, task, memory, MCP, and codemode handlers, and converts every outcome (including cancellation) into a result the model can read. `AppToolTypes` holds the persisted value types for calls and results. Gating decisions live in the sibling `Core` directory (see [../Core/EGM.md](../Core/EGM.md)).

## Interface
- Value types [AppToolTypes.swift:AppToolCategory, AppToolCallStatus, AppToolCall, AppToolResult, AppToolDefinition, AppToolMediaReference, AppToolMediaDisposition, AppToolBrowserCardMetadata, TransientTypedArguments, TOOL_REJECTED_MESSAGE]. `AppToolCall.shellCommand(in:)` is the one accessor for the command argument (`command`, `cmd`, `CommandLine`, `code`).
- Catalog [AppToolCatalog.swift:AppToolCatalog]
  - Definition lists: `fileTools`, `terminalTools`, `taskAgentTools`, `webTools`, `projectArtifactTools`, `mcpTools`, `toolSearchTools`, `planningInteractiveTools`, `automationTools`, plus computed `memoryTools`, `codemodeTools`, `browserTools`.
  - Assembly: `allTools(browserAvailability:)`, `tools(for:projectURL:contextTokens:webToolsEnabled:browserAvailability:)`, `captureTurnAvailableTools(...)` (returns the per-turn `TurnAvailableTools`), `openAIFormattedToolsJSON`.
  - Classification: `category(for:projectURL:)`, `builtInCategory(for:)`, `isBuiltInToolName`.
  - Prompt text: both `systemPromptAddendum` overloads.
- Profiles [AppToolCatalog+Profiles.swift:profileKey, profileAllows, callerToolFilter, applying, profileGroups, profileToolCount, ToolProfileGroup, ToolProfileEntry].
- MCP advertisement [AppToolCatalog+Mcp.swift:AppToolCatalogMcp] with `visibleServers`, `resolvedServers`, `catalogSnapshot`, `toolDefinitions`, `parameterSchema(from:)`, `advertisedName`, `advertisedDescription`, `argumentsSummary`, and `AppMcpToolCatalogSnapshot`.
- Executor [AppToolRegistry.swift:AppToolRegistry.execute(call:in:chatID:subagentDepth:webToolsEnabled:fallbackMode:)] and `executeMcpCall`.
- Vocabulary [AppToolRegistry+Vocabulary.swift]: `standardTools`, `supportedToolNames`, `workspaceRootedToolNames`, `readFileAliases`, `writeFileAliases`, `editFileAliases`, `isImplemented(_:projectURL:)`, and the host-installed provider closures (`activeSessionProvider`, `userSystemPromptProvider`, `subagentSamplingOptionsProvider`, `webToolsEnabledProvider`, `browserToolRuntimeProvider`, `browserToolAvailabilityProvider`, `browserActionApprovalProvider`, `computerUseToolRuntimeProvider`, `subagentProgressSink`, `backgroundAgentLauncher`, `backgroundAgentStopper`, `observationRecaller`).
- File handlers [AppToolRegistry+Handlers.swift]: `AppFileReadLimits`, `resolveSecurePath`, `listDirectory`, `readFile`, `writeFile`, `editFile`, `searchCode`, `syntextPathScope`, `compactOutput`.
- Glob [AppToolRegistry+Glob.swift:globRegex, globFiles].
- Deferred MCP bridge [AppToolRegistry+ToolSearch.swift:callerToolFilter, executeDeferredMcpCall, DeferredMcpContinuationStop].
- Subdirectories: none.

## Invariants
- Four lists describe one vocabulary and are kept in step by hand: `standardTools` (prompt), `supportedToolNames` (what `execute` implements), `workspaceRootedToolNames` (needs a root), and `AppToolCatalog.builtInCategory` (permission category). A name missing from the second is refused, from the third runs at the `/dev/null` root, from the fourth gates on the wrong switch [AppToolRegistry+Vocabulary.swift]. A name must have a real `case` in `execute` before it appears in `supportedToolNames`, and advertised lists are filtered through `isImplemented` so the model never sees a tool with no handler [AppToolCatalog.swift:allTools].
- Read, write, and edit alias sets drive both dispatch and category, so a name that reaches a write handler is always `.fileWrite` [AppToolRegistry+Vocabulary.swift:writeFileAliases]. Built-in names are classified first and a custom tool can never reclassify one [AppToolCatalog.swift:category].
- Projectless chats refuse rooted tools, every `mcp__` call, and every custom tool by name, with no home-directory or `/` fallback [AppToolRegistry.swift:execute]. Web tools are refused at execution when the composer switch or "Block all" is set, independent of prompt construction.
- Paths: `resolveSecurePath` rejects absolute and `~` paths and anything resolving outside the root. The one exception is the shell-output spill root, and symlinks are resolved on both sides [AppToolRegistry+Handlers.swift:resolveSecurePath]. Writes and edits also pass `AppToolSandbox.validateWritePath`. Search and glob skip symlinks, hidden entries, `node_modules`, `target`, `.build`, `.git`, files over 16 MB, and (for search) sensitive paths, within a 5000-file and 64 MB budget. Glob refuses `..` [AppToolRegistry+Glob.swift].
- Read-before-write: overwriting an existing file needs a tracked, non-stale `FileSnapshotStore` entry. `insert` and `pattern_replace` need a tracked file. Any edit refuses a stale file. `undo_edit` refuses when the file changed since the backup. The read snapshot hashes on-disk bytes (not decoded text) so a BOM file is not seen as stale [AppToolRegistry+Handlers.swift].
- Numeric arguments are clamped before arithmetic: `read_file` line math, `limit`, and `insert_line` (overflow traps previously crashed the app). `end_line` is absolute and `limit` is a count, read from separate keys.
- `edit_file` matches in a fixed fallback ladder: exact, line-prefix stripped, quote-normalized, trailing-whitespace tolerant, then normalized indentation (non-markdown only, unique candidate required). More than one match without `replace_all` is refused [AppToolRegistry+Handlers.swift:editFile].
- Argument conflicts are refused, not resolved: file tools throw `ToolArgumentConflict`, and the MCP bridge uses `ToolArgumentResolver.mcpTarget` [AppToolRegistry.swift].
- A tool with no handler throws "not implemented" and never fabricates success. A user-stopped call returns a "do not retry" message, not an error string. Hook and codemode stop signals are caught as dedicated error types and returned as results.
- MCP name resolution is global-first via `resolvedServers`, and the project is re-read from `AppProjectFileStore` at call time so a mid-turn change gates the next call [AppToolRegistry.swift:executeMcpCall]. Deferred MCP calls rebuild the inner call, run PreToolUse and PostToolUse hooks, and re-run `AppToolPermissionEngine.evaluate`, and any `ask` is returned as an error so the approval card can only come from the direct tool [AppToolRegistry+ToolSearch.swift].
- MCP schemas convert only the supported keyword subset. Any other keyword sets `validationIssue` so validation refuses the call instead of weakening the constraint. Denied tools are stripped from the catalog, and disabled servers are refused again inside `catalogSnapshot` [AppToolCatalog+Mcp.swift].
- Profiles apply to the captured turn set, not only to prompt text, so a removed tool is refused at execution. `recall_tool_output` is always on. MCP front tools and custom tools follow the MCP-and-custom switch [AppToolCatalog+Profiles.swift].
- Prompt hygiene: the agent roster is capped (32 entries, 8192 characters), names are restricted to alphanumerics and `-_.`, descriptions have structural characters stripped, and the block is wrapped in an untrusted-metadata tag. Computer-use tools join only the main turn's capture, never a subagent's [AppToolCatalog.swift].
- Subagent depth and background launch: background agents launch only at depth 0, and a `model` override is refused (one open model session). Browser origin for approval comes from the same parsed command the executor navigates to [AppToolRegistry.swift].
- Tool output persisted to the chat archive is truncated at encode time to 64 KB (head 3/4, tail 1/4), and `AppToolCall` and `AppToolResult` decode tolerantly so one old field cannot quarantine a chat [AppToolTypes.swift].
- `offeredKeysCache` and the provider closures are `nonisolated(unsafe)` statics. The cache is guarded by `NSLock`, and the providers are wired once at startup by the app [AppToolCatalog+Profiles.swift, AppToolRegistry+Vocabulary.swift].

## Side Effects
- Disk: reads and atomic writes of project files, directory creation, backups and snapshots through `FileSnapshotStore`, and `Data(contentsOf:)` on read for hashing [AppToolRegistry+Handlers.swift].
- Process spawn: shell commands via `ShellCommandRunner`, `git log` via `HardenedGit` for `time_machine` reads [AppToolRegistry+Handlers.swift, AppToolRegistry.swift].
- Network: web search, fetch, and HTTP request executors, and MCP calls through `McpClientEngine` [AppToolRegistry.swift:execute, executeMcpCall].
- Index and artifacts: `SyntextIndexManager` change and delete notifications, `ArtifactRegistrar.report` on success only, `SkillManager.notePathTouched` [AppToolRegistry.swift].
- Reads app state: `AppNetworkPolicyStore.current`, `AppProjectFileStore.load()`, `GlobalMcpFileStore.load()`, `CustomToolManager.shared`, `McpToolCatalogCache.shared`, `PluginManager.shared`, `SkillManager.shared`, `AgentManager.shared`.
- Calls the host-installed closures for subagent runs, background agents, browser and computer-use runtimes, and observation recall.
- Hook dispatch through `AppHookExecutionEngine.shared` for deferred MCP calls [AppToolRegistry+ToolSearch.swift].
- Writes a managed screenshot asset for browser screenshots [AppToolRegistry.swift].
- No Keychain access, NotificationCenter posts, or timers here, apart from what the delegated executors do.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
