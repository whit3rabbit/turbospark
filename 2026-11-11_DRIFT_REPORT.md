## crates/window-fit/EGM.md

- Overview is a single line ("Package: turbospark-window-fit.") and omits facts the code and Cargo.toml state: library name `turbospark_window_fit`, the `window_fit` alias used by `crates/cli` and `crates/ffi` (both depend on it), and that the crate has no dependencies.
- No Invariants, Side Effects or Verification sections. This is by design for the Intermediate Blueprint form, but the crate also has `tests/fit_rules.rs` and `tests/outcome_contract.rs`, which the contract never mentions.

## crates/window-fit/src/EGM.md

- Interface lists `retained_turns`, `measured_length`, `removed_turn_count` and `has_room_for_generation` as if they were fields. They are public accessor methods on `WindowFitOutcome<T>`. The struct fields are private, and the last field is named `room_for_generation`, not `has_room_for_generation` [outcome.rs].
- Interface omits the generic bounds `T: Clone` and `M: FnMut(&[T]) -> u64`. `T: Clone` is a real constraint on callers because the input is copied with `turns.to_vec()` [fit.rs:fit_conversation_window].
- Interface omits the derives on the outcome (`Debug`, `Clone`, `PartialEq`, `Eq`) and the crate-private constructor `WindowFitOutcome::new` [outcome.rs].
- Invariants omit the exact call pattern of `measure`: once before any removal, then once after each removal, so it runs `removed_turn_count + 1` times. `tests/fit_rules.rs` asserts this, and the contract does not state it.
- Invariants omit removal order: oldest eligible turn first, found by the private helper `next_removable_index` [fit.rs].
- Invariants omit the boundary rule. Trimming continues while `measured_length >= bound`, and room for generation is `measured_length < bound`. A measurement exactly equal to the bound means no room [fit.rs:fit_conversation_window].
- Invariants omit edge cases the code and tests cover: an empty conversation is returned unchanged, a single turn with no leading instruction is never removed, and `bound = 0` can never report room.
- Side Effects says "None" and "no state between calls". That holds for the function itself, but `measure` is `FnMut`, so a caller's callback can have side effects or state. The contract does not say this.
- Interface does not name the private helper `next_removable_index`, which holds the only eligibility rule (index 0 when a leading instruction exists, and the last index, are never eligible) [fit.rs].

## crates/window-fit/README.md

- "until the measured length satisfies a caller-supplied token bound" and "Evaluates token lengths" describe the bound and measurement as tokens. The code takes a unitless `u64` bound and a caller callback and never touches tokens [fit.rs]. `AGENTS.md` says the crate never tokenizes, so the README wording is the divergent one.
- "Drops oldest non-pinned turns until the conversation satisfies bounds" understates the end condition. It also stops when no eligible turn remains, and then returns an over-bound outcome [fit.rs:fit_conversation_window].
- `tests/fit_rules.rs` is described as validating "minimum turn constraints". No test uses that name. The tests cover empty input and single-turn input, which is a narrower claim [tests/fit_rules.rs].
- The README does not mention that `fit_conversation_window` requires `T: Clone`.

## crates/window-fit/AGENTS.md

No drift detected.

## swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Registry/EGM.md

- Web refusal: the doc says web tools are refused at execution when the composer switch or "Block all" is set. In code, "Block all" refuses only web-category calls whose arguments carry no http URL (`ToolApprovalPreviewSummary.httpURL(call.arguments) == nil`). Calls with a URL are left to `HttpRequestDestinationValidator`, which keeps localhost reachable [AppToolRegistry.swift:execute].
- Search and glob budget: the doc attributes "files over 16 MB, sensitive paths, within a 5000-file and 64 MB budget" to both. That budget belongs only to `searchCode` (`SearchBudget.maxFilesVisited` 5000, `maxBytesRead` 64 MB, 16 MB per-file skip). `globFiles` caps visited files at 50_000 and results at 200, does no size check, and prunes only heavy directories by name. `searchCode` prunes by path substring [AppToolRegistry+Handlers.swift:SearchBudget, searchCode] [AppToolRegistry+Glob.swift:globFiles].
- Provider closures: the doc says `offeredKeysCache` and the provider closures are `nonisolated(unsafe)` statics. Only `offeredKeysCache` is. The providers are plain `public static var`. `browserScreenshotAssetStoreProvider` is internal, exists for tests, and is not in the doc's list [AppToolCatalog+Profiles.swift:offeredKeysCache] [AppToolRegistry+Vocabulary.swift].
- The doc presents the four tool lists as kept in step, but they are not:
  - `memory_search` and `memory_explain` are in `supportedToolNames` and have `execute` cases, but are absent from `workspaceRootedToolNames` and `builtInCategory`, so they fall to `.automation` while `memory` and `remember` are `.fileWrite` and rooted.
  - `stopagent` has an `execute` case but is not in `supportedToolNames`, so `isImplemented` rejects it.
  - `codemode` has an `execute` case, is not in `supportedToolNames`, and is gated by `CodemodeSettings.isEnabled` in `isImplemented` and `execute`. The doc never mentions the flag.
  - `repl` is classified `.terminal` but has no `execute` case and is not in `supportedToolNames`, so it is never advertised or run [AppToolCatalog.swift:builtInCategory] [AppToolRegistry+Vocabulary.swift:supportedToolNames, isImplemented] [AppToolRegistry.swift:execute].
- `isBuiltInToolName` has a third source, `builtInDefinitionNames`, built from every advertised definition list. It decides which names a custom tool may not take. The doc omits it [AppToolCatalog.swift:isBuiltInToolName].
- `callerToolFilter` is listed under both Profiles and the Deferred MCP bridge without distinction. `AppToolCatalog.callerToolFilter(for:)` builds a closure from a profile. `AppToolRegistry.callerToolFilter` is a `@TaskLocal` read by `codemode`, `tool_search` and `tool_describe`. The task-local does not reach `Task.detached`, so the codemode case narrows its catalog itself before starting the worker. The doc omits this invariant [AppToolCatalog+Profiles.swift:callerToolFilter] [AppToolRegistry+ToolSearch.swift:callerToolFilter] [AppToolRegistry.swift:execute].
- `AppToolRegistry.systemPromptAddendum(for:tools:)` ignores `tools` and delegates to `AppToolCatalog.systemPromptAddendum(for:)`. `standardTools` is a five-entry legacy list, not the prompt vocabulary [AppToolRegistry+Vocabulary.swift:standardTools, systemPromptAddendum].
- `AppToolResult` persistence: the doc covers only the 64 KB output truncation. `promptProjection` is dropped entirely when it exceeds 64 KB. `archivalOutput` and `continuationStopReason` are never encoded and always decode to nil. `AppToolCall.typedArguments` is excluded from `CodingKeys` [AppToolTypes.swift:AppToolResult.encode, CodingKeys].
- `read_file` modes (`lines`, `stats`, `preview`, `diff`, `time_machine`, `search`) are never listed. Spill-root files are not snapshotted. `listDirectory` caps at 80 entries and hides dotfiles, `node_modules`, `target`, `.build`. Argument clamps include `context_lines` (0 to 10) and `num_revisions` (1 to 20) [AppToolRegistry+Handlers.swift:readFile, listDirectory].
- Agent roster hygiene: the doc gives only the 32-entry and 8192-character caps. Code also caps names at 64 characters and descriptions at 200 (truncated with "..."), skips names that fail the name rule, and stops at the character budget [AppToolCatalog.swift:safeAgentRosterName, safeAgentRosterDescription].

## swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Repl/EGM.md

- The doc says `repl.fs` "forwards bounded requests to an app-side broker" and lists disk operations "via broker". In the production serve path this never happens. `REPLWorkerMain.serveRequests` builds `REPLWorkerContext(limits:configuration:)` with no `fileChannel`, so the worker always binds `REPLNoAccessFileChannel` and every `repl.fs` call is denied. `REPLInProcessFileChannel` and `REPLFileAccessBroker` are not wired through process IPC (code comments say "until task 3.1") [REPLWorkerMain.swift:serveRequests] [REPLWorkerContext.swift:init] [REPLFileAccessBroker.swift:REPLInProcessFileChannel].
- The doc does not say the subsystem is unreachable from the tool executor. `AppToolRegistry.execute` has no `repl` case, `repl` is not in `supportedToolNames`, and `REPLWorkerContext`'s own comment says nothing constructs a REPL session from the tool path [REPLWorkerContext.swift:init doc comment] [Registry/AppToolRegistry+Vocabulary.swift:supportedToolNames].
- `REPLCallResult.Status` has `cancelled`, `memoryLimit` and `busy`, and `REPLLimits` has `defaultTimeout`, `maximumTimeout` and `footprintGrowthBudgetBytes`. Nothing in this directory produces or enforces them, and `sessionReset` is always false. The only `timedOut` producer is the lowered-path settlement deadline. The doc lists them as live [REPLSessionTypes.swift] [REPLWorkerContext.swift:makeResult, evaluateLowered].
- In `makeResult`, `warn` and `error` console events are removed from `consoleText` and merged into `errorText`, even when status is `completed`. Only `log`, `info` and `debug` go to `consoleText`. The doc omits this [REPLWorkerContext.swift:makeResult].
- The 4 MiB limit is checked only after a newline-terminated line is fully buffered. The REPL serve loop has no cap on an unterminated buffer, unlike the codemode loop, which checks `CodemodeWire.maximumLineBytes` against `buffer.count` before any newline. The doc's "enforces a 4 MiB transport limit per line" overstates the protection [REPLWorkerMain.swift:serveRequests vs serveCodemodeRequests].
- Lowering bounds not in the doc: awaits inside template substitutions are not detected and surface as `parseError`. Awaits inside shorthand class or object methods may be overcounted. After a `}` the lexer assumes division, not a regex. Only function declarations are hoisted; class declarations are installed in statement order [REPLScriptLowering.swift header, REPLBoundedLexer.slashIsRegex, REPLLoweredWrapperBuilder.build].
- File broker: reads must be valid UTF-8 or `REPLHostFacade.deliver` rejects with "not valid UTF-8 text", and writes take only string contents. Writes use `O_TRUNC` in place, not atomically [REPLHostFacade.swift:deliver] [REPLFileAccessBroker.swift:write].
- The worker context installs non-writable, non-enumerable globals `__turbosparkReplSettled` and `__turbosparkReplRender`, plus a completion renderer that captures `JSON.stringify` and `String` at install time. The doc omits both [REPLWorkerContext.swift:completionRendererScript, settleInstallationScript].

## swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Core/EGM.md

- The invariant "anything that is not a single simple invocation of a recognized command is `.high` and hard-gated" is half true. `assessTerminalCommand` hard-gates the not-a-single-simple-invocation case and the denylist case. The "not a recognized read-only or build command" branch (`isAutoApprovable` false) returns `.high` with `hardGated` nil, a soft ask that Agent mode can route to the classifier [ToolRiskClassifier.swift:assessTerminalCommand].
- `hardGated` for "sensitive path" guards holds for file read/write tools, not MCP. `assessMcpTool` returns `.high` with no `hardGated` even when an argument hits `isSensitivePath`, and `adjusting` likewise yields a soft `.high`. The doc does not say so [ToolRiskClassifier.swift:assessMcpTool, adjusting].
- "Mutable global state: the lazy `CommandGate.model`" is wrong. It is an immutable `static let model: Model? = loadModel()`, lazily initialised but not mutable [CommandGate.swift:model].
- `CommandGate.advisoryReason` is documented as decorating only an already-`.ask` verdict. In code it is applied to reasons of a `.high` assessment through `ToolRiskClassifier.withAdvisory`, and the engine later turns that into an ask. Its fixed 0.75 threshold, separate from `vetoThreshold` (0.90), is undocumented [CommandGate.swift:advisoryReason] [ToolRiskClassifier.swift:withAdvisory].
- `ToolCallDispatchGate.evaluate` behavior absent from the doc: `forgeGuardrailsEnabled` runs `ForgeGuardrailsEngine.inspect` over the text left after candidate markup is removed (`.accept`, `.retry(nudge)` or `.rescued(calls, sanitizedText)`); rescued calls get string arguments coerced to schema types and re-validated; the Forge step is skipped for native calls; `allowsParsing:false` short-circuits the gate; validated calls carry `typedArguments`; `executorArguments(from:)` flattens booleans to "1"/"0", null to "null" and containers to sorted JSON. `ToolCallDispatchResolution.resolve` drops all dispatchable calls and finishes as prose with the original content when a retry nudge exists but `completedAttempts >= maximumAttempts` [ToolCallDispatchGate.swift:evaluate] [ToolCallDispatchResolution.resolve].
- `AppToolSandbox` details not in the doc: `validateWritePath` and `validateDomain` throw `NSError` in domain `TurboSparkSandbox`, codes 1 to 5; `SandboxConfig.systemDeniedPaths` is the default deny list; `hostScope` also treats `metadata`, `instance-data`, the AWS IMDS IPv6 `fd00:ec2::254` and `.local` hosts as private network [AppToolSandbox.swift].
- `ProcessExecutor.run` has an undocumented `mergeStreams` mode (the shell tool interleaves stdout and stderr into one capped buffer and sets `Output.mergedOutput`). Also undocumented: `defaultTimeoutSeconds` 120, a 2 s wait for readers after the child exits, `exitCode` -1 while still running, and the `liveChildCount` test seam. The 1,000,000-byte output cap is correct but is per stream (a separate `CappedOutputBuffer` each) unless `mergeStreams` is used [ProcessExecutor.swift:ProcessExecutor.run].

## swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Core/AgentMode/EGM.md

- The projection description is overstated. The "2000 bytes per argument under a 16000-byte shared budget" rule (`boundedArguments`) applies only to `.mcp`, `.automation` and `.fileRead`. For `.fileWrite`, every argument other than `path`/`file_path` is cut at 300 bytes with no shared budget [ToolCallClassifier.swift:ToolCallProjection.project].
- The `.fileWrite` projection recognizes only `path` and `file_path`. A call using `TargetFile`, `filePath` or `AbsolutePath` is shown to the classifier as "write to path: (no path)". The doc omits this [ToolCallClassifier.swift:ToolCallProjection.project].
- `.computerUse` projects only the tool name and `app`/`apps`, not typed text or keys. `.browser` projects only `call.name`. Neither limit is documented [ToolCallClassifier.swift].
- `ToolCallProjection.agentPromptCharacters` (4,000) is declared and referenced nowhere else in `swift/` (checked with grep). The doc is silent on the dead constant [ToolCallClassifier.swift].
- "A reply is accepted only if `verdict` is the string 'allow' or 'block'" omits that `parseVerdict` lowercases the verdict, so "Allow" and "BLOCK" are accepted [LocalModelToolClassifier.swift:parseVerdict].
- "No model, timeout, or an unparsable reply after one corrective retry all return `.unavailable`" is inexact for timeout. A timeout makes `generate` return "", which fails to parse and triggers the corrective retry, which can take up to another 20 s. Only no model (immediate) or a failed second parse gives `.unavailable` [LocalModelToolClassifier.swift:classify].
- The doc says nothing about the system prompt policy (soft deny, hard deny, injected-instruction rule, hints appended, "Workspace root") or the `recentUserIntent` field of `ClassifierRequest` [LocalModelToolClassifier.swift:promptMessages].

## swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Memory/EGM.md

- "`save` and `forget` call `ledger.propose` with confidence `agent-proposed`" is wrong for forget. `execute` sets `proposal.confidence = "agent-proposed"` only for project and profile `save`. The project `forget` proposal (`kind: "forget"`, `requestedForgetID`) never sets confidence [MemoryTool.swift:execute].
- Profile scope has no forget. A profile `forget` hits `default` and throws code 6, and that error text says "Use save, read, or search" and omits explain [MemoryTool.swift:executeProfile].
- `memory_search` with `scope=project` is not project-only. It returns profile claims plus the matching project scope. `scope=profile` (or omitted) returns profile only [MemoryLedger.swift:search].
- Profile `explain` returns only the claim text and evidence, with no status line or supersession chain. Only `memory_explain` (`explainClaim`) walks the chain. The doc blurs the two in one invariant [MemoryTool.swift:executeProfile, explainClaim].
- Undocumented behavior: an unrecognized `type` silently falls back to `.project`; profile save always uses kind `preference` and ignores `name`, `description` and `type`; project `read` with a name throws code 2 but `forget` of an unknown name returns a message string; the description fallback appends "..." when truncated at 120 characters, and the stored claim text is `<description>\n<body>` [MemoryTool.swift].

## swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Browser/EGM.md

No drift detected.

## swift/TurboSparkApp/Sources/TurboSparkApp/Tools/ComputerUse/EGM.md

- "A successful grant calls `port.resumeAfterStop()`" is too narrow. `execute` calls it for any `.completed` outcome from `requestAccess`, including when every requested app was refused and `granted` is empty [ComputerUseToolExecutor.swift:execute].
- On a screenshot whose media storage fails, `output` returns `.completed(output: "The screenshot could not be attached.", media: ...)`, not `.failed`. No frame is recorded, so no pixel click can follow. The doc's "stored pixel size recorded" invariant does not describe this path [ComputerUseToolExecutor.swift:output].
- A click with only part of the point group (for example `x` alone, no reference) is rejected with the generic "needs reference, or x, y and screenshot_id" message. This is undocumented [ComputerUseToolExecutor.swift].

## swift/TurboSparkApp/Sources/TurboSparkApp/Tools/EGM.md

- All 18 child directories exist, are linked, and each link target exists. No missing or unlinked children.
- The Automation summary ("sleep, notification, cron and monitoring tools") is incomplete or overstated. Monitor has only a schema and payload structs and no executor, and the directory also holds Config, CtxInspect, Workflow and ScheduleWakeup, which the summary omits [Automation/EGM.md, MonitoringNotificationTools.swift:MonitoringNotificationDefinitions].
- The Web summary ("web fetch, HTTP request, code search") omits WebSearch, which the child EGM.md lists first.
- The File summary ("read, write, patch and diff tools") omits Snip, NotebookEdit, MultiEdit, SendUserFile and the Syntext `grep_search`. There is no "diff" tool in File/, only ApplyPatch and the edit/read/write tools.
- The Projects summary ("project doc, artifact and worktree tools") is loose. The child describes a "Projects knowledge-base tool" with schema and payloads only, and only `WorktreeExecutor` has an executor [Projects/ProjectDocTools.swift].

## swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Automation/EGM.md

- The error-code claim is imprecise. Config uses 61 (unknown action) and 62 (`set` unsupported). Cron uses 61 (missing prompt) and 62 (no chat), so 61 and 62 are shared, and 60 is also used by the cron "never fires within a year" error [AutomationExecutors.swift:ConfigToolExecutor, CronScheduler.swift].
- Config behavior is absent from the doc. `get` resolves only `interaction.mode`, `reasoning.effort`, `guardrails.mode` and `project.root`. `power.profile`, `project.name` and `project.permissions.mode` appear only in `list`. Any other key returns "(Not explicitly configured or unknown)" rather than an error [AutomationExecutors.swift:getConfigValue, listConfig].
- CtxInspect output is partly hard-coded. "Guardrails & Approvals: Active" and "Memory Limits: Single file read cap 16 MiB, max scan files 5,000" are literal strings, not measured values, and the schema describes "context statistics" and "system prompt details" that it does not produce [AutomationExecutors.swift:CtxInspectExecutor].
- Sleep parse fallback is absent. A non-numeric `seconds` becomes 1.0, a non-numeric `ms` becomes 1000 ms, and `duration` and `delay` are accepted as aliases [AutomationExecutors.swift:SleepExecutor].
- PushNotification schema and executor disagree. The schema has no `title`, but the executor reads `title` (default "TurboSpark Notice") and accepts `body` and `text` for the message [MonitoringNotificationTools.swift:pushNotification, AutomationExecutors.swift:PushNotificationExecutor].
- A non-recurring cron job is deleted whether delivery was `fired` or `missed`, so a missed one-shot is not retried [CronScheduler.swift:completeFire].
- `jobs = archive` is assigned outside `lock`, before the lock is taken [CronScheduler.swift:load].

## swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Tasks/EGM.md

- `TaskOutput` ignores its parameters. The schema requires `block` and `timeout`, but the executor ignores both and returns only the stored `TrackedTask.output`, so it cannot read a background shell or subagent. `TaskStopInput.shellId` is also unused [TaskManager.swift:executeOutput, TaskItemTools.swift:taskOutput].
- The `deleted` status is not applied. The schema and `TaskItemInfo` advertise it, but `TaskManager` stores the status string unvalidated and never removes the task. `TrackedTask.status` comments also list `"stopped"` and `"failed"` [TaskItemTools.swift:taskUpdate, TaskManager.swift:updateTask].
- `AgentInput.isolation` is not in the `Agent` schema, and the `model` description says only 'inherit' is supported [AgentTools.swift:AgentToolDefinitions.agent].
- "Task IDs use `UUID()`" is imprecise. Task IDs are the first 8 lowercase characters of a UUID, so collisions are possible. Default todo IDs from the parser are positional `todo-N`, and `UUID()` is used only for a single-item shorthand or when `TodoItem.init` is used directly [TaskManager.swift:TrackedTask.init].
- Hook behavior in `batch` is absent. PreToolUse `updatedInput` is merged into the child's arguments before the permission check, child PostToolUse and PostToolUseFailure feedback is appended as `<hook_feedback>` and `<hook_context>`, and the session id falls back to `"batch"` when `chatID` is nil [BatchToolExecutor.swift:execute].
- The `batch` schema text says "Executes all calls concurrently", but `.fileWrite` and `.terminal` children are serialized by `MutationGate` [BatchToolDefinitions.swift:batch].
- A `tool_calls` value that fails to decode falls through to `tool_calls_json`, and only then is code 35 thrown [BatchToolExecutor.swift:parseItems].

## swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Terminal/EGM.md

- The network sandbox profile is incomplete. The doc says "`allow default` plus `deny network*`". When `policy.allowLocalhost` is true the profile also appends `(allow network-outbound (remote ip "localhost:*"))` and `(allow network-bind network-inbound (local ip "localhost:*"))` [ShellNetworkSandbox.swift:profile].
- The 20-running limit is soft. The running count is checked and then the lock is released before the record is inserted, so concurrent launches can exceed 20. `nextID` is also consumed even when the sandbox refuses the launch [BackgroundShellManager.swift:launch].
- Classifier scope is absent. Only the head word, interpreter inline flags and the `git` subcommand (`parts[1]`) are checked. Arguments are not inspected, so `find ... -delete`, `awk 'system(...)'` and `git push --force` pass here; code comments say `ToolRiskClassifier` covers these. `git -C dir status` is refused, and `python`, `python3`, `node` and `mkdir` are on the auto-approve build list [TerminalCommandClassifier.swift:isAutoApprovable].
- Background stdin is undocumented. The stdin write end is closed before `process.run()`, so a background command gets EOF immediately [BackgroundShellManager.swift:launch].
- `interpretExitCode` is public but nothing in this directory calls it. `ShellCommandRunner` uses `benignExitNote` and codes 40/42 instead.

## swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Planning/EGM.md

- The cancellation guarantee is overstated. The cancel handler only calls `submitAnswer(dismissed: true)`, a no-op unless `waitForAnswer` has already registered a continuation, so a cancel landing before registration would not dismiss the question [PlanningInteractiveExecutors.swift:executeAwaitingAnswer, submitAnswer].
- `onQuestionsAsked` fires only from the non-blocking `execute`. The parked path calls `answerWaiter(chatID, items, toolCallID)` and does not call `onQuestionsAsked` [PlanningInteractiveExecutors.swift:executeAwaitingAnswer].
- `pendingQuestions` and `savedPlans` are written but never read in this directory [PlanningInteractiveExecutors.swift].
- The `ExitPlanMode` plan argument is undocumented. The schema is an empty object, but the executor reads `plan`, `summary` or `content`, and `ExitPlanModeOutput` fields (`isAgent`, `filePath`) are unused [PlanningInteractiveTools.swift:exitPlanMode, PlanningInteractiveExecutors.swift:PlanModeExecutor.exit].
- `ProposeSkills` writes immediately and treats every proposal as new, with no approval step and no frontmatter or size validation of `skillMd`. `kind: "improvement"` and `target` are accepted but ignored, so an improvement for an existing skill is skipped as "already exists", and when all are skipped the message reads "Successfully saved 0 proposed skill(s)". `SkillProposalItem` decoding requires `name`, `kind`, `description` and `skillMd` [PlanningInteractiveExecutors.swift:ProposeSkillsExecutor].
- The `ReportFindings` schema is narrower than the parser. It omits `category`, `verdict`, `short_summary` and `outcome`, which the parser and formatter use. `ProposeGoal` throws code 33 for a missing condition before it throws 34 [PlanningInteractiveTools.swift:reportFindings].

## swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Custom/EGM.md

- `effectiveCategory` for http is overstated. The code is `category == .terminal ? .terminal : .web`, so a declared `.fileWrite`, `.mcp` or `.automation` collapses to `.web`, not "at least .web" [CustomToolDefinition.swift:effectiveCategory].
- "Never offered or executed until trusted" is enforced only at offer time. `CustomToolExecutor.execute` does no trust check of its own [CustomToolManager.swift:resolveEffectiveTools, CustomToolExecutor.swift:execute].
- The trust fingerprint excludes `parameters`, `toolDescription` and `isEnabled`, so changing the declared parameters (which decide which model arguments reach the environment or substitution) does not invalidate trust [CustomToolTrustStore.swift:fingerprint].
- Script substitution is multi-pass. The `.script` branch calls `replacingOccurrences` once per argument over a dictionary, so a value containing another argument's `{{k}}` or `${k}` marker can be re-substituted in dictionary order. Script arguments are not passed as `TOOL_ARG_*` environment variables [CustomToolExecutor.swift:execute].
- HTTP responses are not treated as failures. A non-2xx response returns `"HTTP <status>:\n<body>"` as a success, code 33 does not apply, and the body is sent for every method including GET. The command branch joins stdout then stderr rather than interleaving [CustomToolExecutor.swift:execute].
- The YAML path has two undocumented effects. A missing `command:` becomes `echo 'Ran custom tool <name>'`, and the parameter schema is empty, so every model argument is filtered out [CustomToolParser.swift:parseYAMLContent].
- Tool names are not validated. `tool.name` goes unsanitized into `<name>.json` and the temp script path `custom_<name>_<id>.sh`, and `saveTool` ignores its `projectURL` parameter and uses the scope's embedded `projectPath` [CustomToolManager.swift:saveTool, CustomToolExecutor.swift:execute].
- Scans are recursive (`FileManager.enumerator`), `TOOL.json` or `TOOL.yaml` falls back to the parent folder name, parse failures are silently dropped (`try?`), and when both global directories are scanned the later one wins for the same lowercased name [CustomToolManager.swift].
- Trust-store persistence errors are swallowed because the write uses `try?` [CustomToolTrustStore.swift:persist].

## swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Codemode/EGM.md

- Hidden globals are undocumented. The prelude installs `__codemodeComplete`, `__codemodeFail` and `__codemodeReportStalled` as non-writable, non-enumerable globals that stay callable by the script. The doc says only that bridge globals are deleted. `ALL_TOOLS` is also installed non-writable but is missing from the doc's list of installed names [CodemodePrelude.swift:source].
- The worker-side settlement backstop is missing. `CodemodeWorkerContext.settlementBackstop` is 900 s, after which the worker ends the run with "did not settle within 900 seconds" [CodemodeWorkerContext.swift:pollUntilSettled].
- The 4096-byte stderr tail kept for the "worker exited" failure is not mentioned [CodemodeWorkerSupervisor.swift:maximumStderrTailBytes].
- The prompt listing is also capped by `max(600, contextTokens / 20)`, not only `listingCharacterLimit` (12,000) [CodemodeToolDefinitions.swift:CodemodeCatalog.promptListing].

## swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Codemode/README.md

- The layout omits `CodemodeWorkerSandbox.swift` (Seatbelt profile, `apply`, `runProbe`). Only the Invariants section mentions it.
- "The `// @options:` line is parsed leniently" is wrong. Unknown fields throw a source error, as do a non-JSON line, non-positive `timeout_ms` or `max_output_chars`, and an options line with no code after it. A stale comment on `CodemodeSandbox` says the same thing [CodemodeSandbox.swift:ParsedSource.parse].
- "Stopped on deadline or cancellation with SIGTERM, then SIGKILL after 2 s" skips a step. It first closes stdin and waits up to 2 s for the worker to exit, then calls `ProcessExecutor.terminateAndReap` (SIGTERM, 2 s, SIGKILL, then a SIGKILL sweep of descendants). A script stuck in a synchronous loop costs about 2 s extra before SIGTERM [CodemodeWorkerSupervisor.swift:terminateAndWait].
- Settings describes only the 12,000-character listing budget, not the `max(600, contextTokens / 20)` cap [CodemodeToolDefinitions.swift:promptListing].
- The README is correct that nested results over the limit are rejected, never truncated. The stale comment in `CodemodeSandbox` ("tail-truncated") is what disagrees.

## swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Hooks/EGM.md

- The `if` condition is fail-open and looser than described. A condition without `(` and a trailing `)` is ignored entirely and matches everything. The tool-prefix match falls back to `toolName.localizedCaseInsensitiveContains(prefix)`, and the glob argument is `command`, else `cmd`, else `path`, else `""`. The doc says only "glob only a trailing `*`" [AppHookMatcher.swift:matchesCondition].
- Non-sensitive `${user_config.KEY}` values are substituted raw into the `-c` string, unquoted and unescaped. Sensitive ones become env references. The doc covers only the sensitive path [AppHookCommandRunner.swift:executeCommandHook].
- Every option, sensitive or not, is also exported as `TURBOSPARK_OPTION_*` and `CLAUDE_PLUGIN_OPTION_*`. The hook env also carries `TURBOSPARK_HOOK_EVENT`, `TURBOSPARK_SESSION_ID`, `TURBOSPARK_TOOL_NAME`, `TURBOSPARK_PROJECT_DIR`, `CLAUDE_PROJECT_DIR`, and the plugin root and data variables. The doc says only "plus hook variables".
- `${CLAUDE_PLUGIN_ROOT}` and `${CLAUDE_PLUGIN_DATA}` are expanded only if `PluginManager.findPlugin` resolves the hook's plugin. Otherwise the placeholder stays literal.
- HTTP hooks differ from command hooks in ways the doc does not state. They do no placeholder or option expansion and set no environment. Their payload omits `duration_ms` and `tool_response.success`. A non-2xx status is parsed as exit 1, so it is non-blocking feedback and can never block [AppHookCommandRunner.swift:executeHttpHook].
- Discovery details are undocumented. `settings.local.json` files are read from `~/.turbospark` (and `~/.claude` under the opt-in) and from the project, as `.localConfig` with its own `local_config` group, which falls back to `user_config` when no project is open. The project file list includes `hooks/hooks.json`. Unknown event names are skipped silently, `agent` maps to `.prompt`, and any other unknown `type` maps to `.command` [AppHookStore+Discovery.swift].
- `transcript_path` in the hook payload points at `chats_archive.json`, not a per-session file, and `PreCompact` sends only `trigger` [AppHookStdinPayload.swift].
- `name`, `statusMessage` and options are not part of `contentHash`. The doc's field list is correct but does not say this.
- Saving an empty sensitive value deletes the Keychain item [HookOptionKeychain.swift:save].
- Atomicity is not uniform. The trusted and disabled hash files use `Data.write(.atomic)`, but the options and custom-hooks files go through `AppJSONStore.save`. The doc's "atomic writes" claim is not verified for those two.

## swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Projects/EGM.md

- `EnterWorktree` also accepts `branch` as an alias for `name`, and with neither given the branch is `turbospark-worktree-<6 hex>`. `EnterWorktreeInput` has no `branch` field [WorktreeExecutor.swift:enter].
- The `enterWorktree` schema describes `path` as "Path to an existing git worktree", but the executor uses `path` as the destination of a new worktree [ArtifactWorktreeTools.swift:enterWorktree, WorktreeExecutor.swift:enter].
- `EnterWorktreeOutput` and `ExitWorktreeOutput` are listed as payloads, but the executor returns plain `String`s and never constructs them.
- The `Artifact` schema advertises only `action`, `file_path`, `title`, `description`, `url` and `force`, while `ArtifactInput` also carries `favicon`, `limit`, `scope`, `label`, `note`, `prompt`, `out_dir`, `asset_id`, `after` and `contract` [ArtifactWorktreeTools.swift].
- `ExitWorktree` with no `action` defaults to `keep`, and `action` is lowercased first.
- `resolveSecurePath` runs before the branch-format check, and `createDirectory` runs before the first `worktree add`, so a path-escape error takes precedence over error 72.

## swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Guardrails/EGM.md

- Casing is handled inconsistently. Rescue accepts tool names case-insensitively and keeps the model's casing in `AppToolCall.name`, but `inspect` then looks up `specsByName[call.name]` case-sensitively. A rescued `Read` for a tool named `read` returns `.retry` with "Unknown tool". The doc says only "unknown tool name yields `.retry`" [ForgeGuardrailsEngine.swift:inspect, isToolNameAllowed].
- "Accepts booleans only as `true` or `false`" is slightly off. The check lowercases first, so `TRUE` and `False` pass, and type names `int`, `float`, `double` and `bool` are accepted as aliases [ForgeGuardrailsEngine.swift:validateArguments].
- Matches accumulate or return early differently per dialect. XML patterns, Markdown, GLM, Kimi and Gemma accumulate all matches, while Mistral and Longcat return at the first parsable block. The doc says only "first pattern that yields calls wins".
- Pattern 4 (bare JSON) does not call `isPrefixedByIncompleteToolCallLabel`, so the "incomplete label" invariant does not cover every path.
- Argument parsing has a fallback chain the doc does not describe: `<parameter name="k">v</parameter>` (MiniMax/Anthropic style), `<parameter=k>v</parameter>` (Qwen), JSON, a line-based `key: value` extractor, and Gemma's own pair regex tried after JSON [ForgeGuardrailsEngine.swift:parseArgumentsString, parseGemmaArguments, extractKeyValuePairs].
- JSON dict conversion accepts tool-name keys `name`, `tool` and `function` and argument keys `arguments`, `parameters` and `args`. A string `arguments` goes through `parseArgumentsString` [ForgeGuardrailsEngine.swift:convertJsonDictToToolCall].
- `sanitizeProse` also strips `<minimax:tool_call>`, `<tool_call>`, `<|tool_calls_section_*|>` and similar markers, and returns empty for a bare code fence, `{}` or `[]`, not just "an empty fence" [ForgeGuardrailsEngine.swift:sanitizeProse].

## swift/TurboSparkApp/Sources/TurboSparkApp/Tools/MCP/EGM.md

- "Relative PATH entries are dropped everywhere" is overstated. Filtering (`dir.hasPrefix("/")`) happens only inside the `if let executableDirectory` block. When `executableDirectory` is nil the inherited baseline `PATH` is copied unfiltered, and a `PATH` in `declared` is applied last with no filtering. `spawnStdioServer` always passes `executableDirectory`, so only the declared-PATH case matters in production [McpClientEngine.swift:childEnvironment].
- The PATH ordering is not documented: resolved executable dir, `/opt/homebrew/bin`, `/opt/homebrew/sbin`, `/usr/local/bin`, login-shell PATH, then inherited PATH [McpClientEngine.swift:childEnvironment].
- The `resolveExecutablePath` search order is absent: absolute or `~` path as-is, then relative to the working directory, then login-shell `searchPath`, then a fixed list (`/usr/local/bin`, `/opt/homebrew/bin`, `/usr/bin`, `/bin`, `~/.cargo/bin`, `~/.volta/bin`, `~/.bun/bin`, `~/.deno/bin`, `~/.local/bin`, mise shims, asdf shims, nvm bin), then the process `PATH`. The doc names only `nvmNodeBinDirectory` [McpClientEngine.swift:resolveExecutablePath].
- `discoverTools` is a one-shot spawn, not a session. It sends `initialize` with `tools.listChanged:true` and kills the child in a `defer`, while the persistent session sends `capabilities: {tools: {}}` with its own id counter. The Overview's "persistent per-server sessions" does not say discovery bypasses `McpStdioSession` [McpClientEngine.swift:discoverToolsViaStdio] [McpStdioSession.swift:performStart].
- "`McpSessionKey` (all config fields plus working directory)" is inaccurate. The key holds only serverName, command, args, env, envPassthrough and workingDirectory, and excludes id, isEnabled, autoApprove and the other config fields [McpStdioSession.swift:McpSessionKey].
- `McpToolCatalogCache.removeServer(named:)` also fires `McpClientEngine.shared.invalidateSessions`, the only path that closes a removed, renamed or disabled server's live session. This is absent [McpToolCatalogCache.swift:removeServer].
- `refreshEnabled` prunes entries and in-flight requests for names no longer enabled, and a failed discovery (`try?` yields nil) leaves no entry, so the server stays stale and is retried on the next refresh. This is absent [McpToolCatalogCache.swift:refreshTasks, complete].
- The importer's accepted shapes are absent. `parseConfigData` accepts root keys `mcpServers`, `mcp` and `servers`, `command` as an array (OpenCode) or string, `url` or `endpoint` for SSE, and `enabled` or `disabled`. It strips `//` comments but not trailing commas despite the code comment, never imports `cwd` or `envPassthrough`, and applies the optional `urlTransform` to the raw URL. Command, URL and headers are not variable-expanded, only args and env [ProjectMcpDetector.swift:parseConfigData, sanitizeJsonComments].
- The `ExternalAgentMcpReader` Codex parser is a restricted TOML parser handling string, string-array and inline-string-table values plus `[mcp_servers.x.env]` subtables and quoted dotted names. A table with `url` becomes SSE, and a server with uninterpretable values is skipped. This is absent [ExternalAgentMcpReader.swift:parseCodexToml, codexServerConfig].
- "Resource list or read executors must not return a success string" is overstated. `listResources` returns a plain success string ("No active MCP servers configured or matching ...") when no enabled server matches, and only the enabled-server path throws (code 81). `ReadMcpResourceDir` has no executor in this directory though its schema is in `McpToolDefinitions.all`. The doc also omits the argument aliases (`server_name`, `url`, `path`) and error codes 80 and 81 [McpResourceExecutor.swift:listResources, readResource].
- Name matching in `McpResourceExecutor` uses `lowercased()` only, with no trim, which differs from the "trimmed and lowercased" `normalizedName` rule in the Invariants [McpResourceExecutor.swift:readResource].
- `LoginShellPath.shellInvocation` returns nil for `nu` (no probe) and `defaultShell` falls back to `/bin/zsh` unconditionally. The doc says only that a shell is trusted if `/etc/shells` lists it [LoginShellPath.swift:shellInvocation, defaultShell].
- `McpApprovalSummary.text` is undocumented, as is the rule that passthrough names are always shown while loader-style passthrough names go into `riskyEnvNames`, and that SSE shows header names only [McpApprovalSummary.swift:init(transport:)].

## swift/TurboSparkApp/Sources/TurboSparkApp/Tools/File/EGM.md

- The FFI and threading claim is overstated. Only the open/build (`Task.detached(priority: .utility)`) and the stats of an inactive project run detached. `notifyChange`, `notifyDelete`, `commitBatch` and live-handle `stats` run inline on the actor, and `searchAsync` and `searchFreshAsync` are awaited directly [SyntextCodeSearchTool.swift:loadIndex, stats, fileDidChange, commitPending].
- "A failed load is not retried for 60 s" is true only for `searchableIndex` (search). `ensureIndex`, `buildIndex` and `loadIndex` ignore the cooldown [SyntextCodeSearchTool.swift:searchableIndex].
- "Lock conflicts retry three times" is three attempts total (two retries, 100 ms then 200 ms backoff) [SyntextCodeSearchTool.swift:retryingLockConflicts].
- "A search never waits on a build" omits that `searchableIndex` awaits opening an existing on-disk index (`loadIndex`), which can fall through to a build if the open reports code 2 [SyntextCodeSearchTool.swift:searchableIndex, loadIndex].
- In a git repo (`.git` present), `grep` runs `searchFreshAsync` and falls back to `searchAsync` on error. It appends a "[The code index is behind recent changes ...]" note for `.budgetExceeded`, `.tooManyFiles` and `.overlayFull`, and a "[Showing the first N matches ...]" truncation note. This is absent [SyntextCodeSearchTool.swift:grep, staleNote].
- `buffer()` drops change notifications unless an index exists on disk or a load is in flight, and the rebuild-on-overflow rule applies only after that [SyntextCodeSearchTool.swift:buffer].
- Not listed: `SyntextIndexManager.isIndexed(for:)`, and `FileSnapshotStore.backupCount` and `trackedFileCount` [SyntextCodeSearchTool.swift:SyntextIndexManager] [FileSnapshotStore.swift:FileSnapshotStore].
- NotebookEdit behavior is absent. `cell_type` defaults to "code", so a replace without `cell_type` converts a markdown cell to code and clears outputs and execution_count, while replacing with `markdown` leaves old `outputs` and `execution_count` keys in place. `insert` with `cell_index` inserts at that index (when `0 <= idx <= count`), and with neither id nor a valid index it silently appends. The notebook is re-serialized with `JSONSerialization` `.prettyPrinted` and `.sortedKeys`, which reorders keys and reformats existing JSON, with no staleness check (`isStale`) and no size limit [NotebookEditExecutor.swift:execute].
- ApplyPatch behavior is absent. A new file (`--- /dev/null`) is written as added lines joined with `\n` and no trailing newline, and a delete hunk for a nonexistent file is silently skipped. `ApplyPatchInput.validate_command` and `validate_timeout_ms` are decoded but never used [ApplyPatchExecutor.swift:flushCurrentFile] [ApplyPatchTool.swift:ApplyPatchInput].
- The `ApplyPatchInput` "tolerant decoder for `patch_text`, `patchText`, `patch`" claim is off. The keyed container reads only `patch_text`, and the other spellings work only through a `[String: String]` single-value decode. If any value in the object is not a string (for example `validate_timeout_ms` as an Int), that decode fails and `patchText` becomes "" [ApplyPatchTool.swift:ApplyPatchInput.init(from:)].
- The invariant "JSON coding keys and tool schemas must stay in step" is violated for Grep. `GrepInput` decodes `offset` and `multiline`, but the `grep` schema in `FileSearchToolDefinitions` omits both [FileSearchTools.swift:GrepInput, FileSearchToolDefinitions].
- MultiEdit phase 3 writes in `currentFileContents` dictionary order (unordered). The failed file itself is not in `successfullyWrittenURLs`, and backups recorded in phase 2 are not discarded if phase 3 fails [MultiEditExecutor.swift:execute].
- `FileSearchToolDefinitions.all` lists `grep_search` with defaults (`context_lines` 2, `max_results` 50). The doc covers only the executor-side caps [FileSearchTools.swift:grepSearch].

## swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Web/EGM.md

- "All outbound requests share one SSRF defense" and "Every outbound hop goes through `HttpRequestDestinationValidator.validate`" are false or overstated. Only WebFetch and HttpRequest use `performRequest`, which applies the validator, redirect revalidation and the connected-address check. The search providers (Exa, Parallel, Brave, SearXNG, Tavily, plus `extractTavily`) use `fetchLimited` on `URLSession.shared` (Tavily uses its own ephemeral session), with no destination, redirect or connected-address validation. SearXNG takes a user-supplied base URL, and search hits are only sanitized after the fact [WebSearchExecutor.swift:fetchLimited, searchSearXNG] [HttpRequestExecutor.swift:performRequest].
- `WebFetchInput.prompt` is declared as an extraction instruction, but it is not in the `WebFetch` schema and `WebFetchExecutor.fetch` has no corresponding parameter, so it is dead input [WebTools.swift:WebFetchInput] [WebFetchExecutor.swift:fetch].
- The 403 retry description is imprecise. The condition `cfHeader == "challenge" || statusCode == 403` is always true on a 403, the retry result is swallowed with `try?` and adopted only if 2xx, and any non-2xx final status throws (code 5) with no body. The Accept and Accept-Language headers per format are not documented [WebFetchExecutor.swift:fetch].
- The Tavily provider prepends a synthetic result titled "Tavily Direct Answer" with URL `https://tavily.com` (`include_answer: true`, `search_depth: basic`), which counts toward the result cap and passes through `sanitize` [WebSearchExecutor.swift:searchTavily].
- "Reads the `TAVILY_API_KEY` environment variable as a fallback" is slightly off. The env var is used only when `apiKey` is nil, and an empty or whitespace-only passed key does not fall back and throws code 30 [WebSearchExecutor.swift:searchTavily].
- `searchParallel` and `searchSearXNG` take no result-count parameter, so count truncation happens only in `search`'s sanitize loop. `search` also truncates the query to 300 chars and clamps `num_results` to 1 through 20, default 5. Tavily is never part of auto mode [WebSearchExecutor.swift:search].
- `codesearch` prefixes the `framework` hint to the query when absent and forwards `provider` (default "auto"). The schema advertises `tokens_num` max 20000, but the result count is `min(10, max(3, tokens_num/1000))`, so values at or above 10000 all give 10 [CodeSearchExecutor.swift:execute] [CodeSearchToolDefinitions.swift:codesearch].
- The `HttpRequest` schema and `HttpRequestInput` comments list GET, POST, PUT, DELETE, PATCH, HEAD, but the executor also accepts OPTIONS. The supported `auth_type` values (`bearer`/`token`, `basic`, `api_key`/`apikey`, `custom`, with an unknown value defaulting to Bearer) and the Content-Type inference (JSON vs text/plain when the caller supplies none) are not documented [HttpRequestExecutor.swift:execute].

# Crates drift check (haiku-reported)

Each section below was produced by a haiku agent. Coverage is stated per doc, and most large directories were sampled (claims verified by targeted reads), not read in full. Treat "No drift detected" on a sampled doc as "none found in what was checked". Not independently re-verified.

## crates/audio/EGM.md

Coverage: sampled (every named file, dependency, and sub-EGM link checked; Cargo.toml exclude and deps verified)
No drift detected.

## crates/audio/README.md

Coverage: sampled (all 96 family README links resolve; named DSP and ops symbols checked against source; quant bit widths checked against [crates/audio/src/quant.rs:scheme validation])
No drift detected.

## crates/audio/AGENTS.md

Coverage: sampled (all relative links resolve; numeric-contract claims checked for FFT scaling [fft.rs:RealFftPlan::inverse], Slaney width [mel.rs:MelScale], Whisper tail drop [whisper.rs:whisper_log_mel], NeMo padding [nemo_mel.rs:nemo_log_mel_spectrogram_with_padding], MegaTTS3 steps [megatts3/facade.rs:Quality::steps], OmniVoice gate [runtime/tests/omnivoice_reference_transcript.rs]; source not fully read)
No drift detected.

## crates/audio/examples/EGM.md

Coverage: sampled (all 13 example files present; main signatures, CLI argument source, file-writing set, and release-mode headers checked; example bodies not read)
No drift detected.

## crates/audio/scripts/EGM.md

Coverage: sampled (all named generator, benchmark, and NeMo scripts exist; subprocess calls and docstring invariants checked in benchmark_music3.py and ab_music3_flow.py; other generators not opened)
- [crates/audio/scripts/benchmark_music3.py:run_sample] The doc says the script "builds turbospark-runtime --example music3_bench with cargo". The script does not invoke cargo. Its subprocess calls are `/usr/bin/time -l <binary>` (run_sample) plus `pmset -g batt`, `git`, and `sysctl -n machdep.cpu.brand_string`. The cargo build command appears only in the module docstring as a manual step. The doc also omits the `sysctl` call. Side Effects section should read "runs a prebuilt music3_bench binary under /usr/bin/time -l".

## crates/audio/scripts/nemo_218d0b4/README.md

Coverage: sampled (pyproject pin to NeMo commit 218d0b4, darwin/arm64 environment gate, convert/check-diar/convert-diar subcommands, MultitalkerAsr::open and NemotronDiarization::load, and every listed script path checked; prepare_frozen_e_clips.py existence checked, its sha256 behavior not read)
No drift detected.

## crates/audio/src/EGM.md

Coverage: sampled (named functions and types in wav.rs, conversion.rs, resample.rs, fft.rs, stft.rs, mel.rs, nemo_mel.rs, whisper.rs, dsp.rs, ops.rs, nn.rs, quant.rs, error.rs, backend.rs checked; #![forbid(unsafe_code)], AudioModelError alias, in_pool rayon use verified; source not fully read)
- [crates/audio/src/lib.rs:pub use] Public items omitted from the Interface list: `StftPaddingMode`, `StftWindowPlacement`, `stft_with_modes`, `stft_into`, `stft_each` [stft.rs], `mel_filterbank_cached` [mel.rs], and `nemo_log_mel_spectrogram_with_padding` [nemo_mel.rs]. Re-exported ones are in lib.rs; the others are reachable public API.

## crates/audio/src/codec/EGM.md

Coverage: sampled (pub mod list in codec/mod.rs matches the child list; conv.rs, wnconv.rs, vq.rs, dac.rs visibility and Conv1d::forward signature checked; REFERENCE_COMMIT pins checked in the four codec children)
No drift detected.

## crates/audio/src/codec/README.md

Coverage: sampled (17 family links resolve and match the pub mod list in codec/mod.rs; MODELS.md#codec anchor exists)
No drift detected.

## crates/audio/src/codec/bigvgan/EGM.md

Coverage: sampled (REFERENCE_COMMIT [bigvgan/mod.rs:REFERENCE_COMMIT] matches the doc; from_json, open, load, forward present; snakebeta refusal [bigvgan/mod.rs:load] returns Unsupported; env access only in tests.rs; mod.rs source not fully read)
No drift detected.

## crates/audio/src/codec/bigvgan/README.md

Coverage: sampled (commit e1b19b9 matches REFERENCE_COMMIT; tests snake_activation_is_refused, tiny_end_to_end_matches_reference, real_checkpoint_gate, multi_kernel_blocks_load_from_flat_reference_indices exist in bigvgan/tests.rs; resblocks.{i * num_kernels + j} routing confirmed in [bigvgan/mod.rs:load]; source not fully read)
No drift detected.

## crates/audio/src/codec/dacvae/EGM.md

Coverage: sampled (REFERENCE_COMMIT, validate rejection, open/load/encode/decode/padded_len/hop_length present; source not fully read)
No drift detected.

## crates/audio/src/codec/dacvae/README.md

Coverage: sampled (default geometry, hop length 1920, 48 kHz, batch-of-one, watermark non-port, and the test names checked in dacvae/mod.rs and dacvae/tests.rs; source not fully read)
- [crates/audio/src/codec/dacvae/tests.rs] The Verification section says "(3 tests)" for `cargo test --lib codec::dacvae`. tests.rs defines 5 #[test] functions: config_defaults_match_reference, tiny_end_to_end_matches_reference, real_checkpoint_gate, config_rejects_degenerate_geometry, load_revalidates_hand_built_config. The doc names only the first three.

## crates/audio/src/codec/descript/EGM.md

Coverage: sampled (every listed method present in descript/mod.rs; decode_codes codebook-count and index checks return SpeechError::Input; no env access in mod.rs; source not fully read)
No drift detected.

## crates/audio/src/codec/descript/README.md

Coverage: sampled (test names, L2-normalized codebook rows [codec/vq.rs], 1024 x 8 x 32 codebook defaults, and strides checked against DacConfig::from_json defaults; source not fully read)
- [crates/audio/src/codec/descript/mod.rs:DacConfig::from_json] The Architecture table lists strides "[2, 4, 8, 8] (44.1 kHz)". The code default for `encoder_rates` is [2, 4, 5, 8] (hop 320), while the code default `sample_rate` is 44100. The README row and the code defaults disagree for the 44.1 kHz case. Checkpoint config.json values override the defaults, so this only matters for configs that omit encoder_rates.

## crates/audio/src/codec/ecapa_tdnn/EGM.md

Coverage: sampled (from_json, open, load, forward present; Res2NetBlock BadConfig on channel/scale mismatch [ecapa_tdnn/mod.rs:Res2NetBlock::new] confirmed; source not fully read)
No drift detected.

## crates/audio/src/codec/ecapa_tdnn/README.md

Coverage: sampled (stem k5 and block kernel/dilation defaults, Res2Net scale 8, SE bottleneck 128, channels 1024, and test names checked; source not fully read)
- [crates/audio/src/codec/ecapa_tdnn/mod.rs:EcapaTdnnConfig::from_json] The doc states the output embedding is a fixed 192 dimensions. The code default for `embed_dim` is 256. The value is config-driven, so 192 can only be a checkpoint value, not a property of the architecture.
- [crates/audio/src/codec/ecapa_tdnn/mod.rs:EcapaTdnn::load, EcapaTdnn::forward] The doc's order is "Dense Linear + BatchNorm -> embedding". The code applies BatchNorm (`asp_bn`) to the pooled statistics before the `fc` layer, and `fc` has no BatchNorm after it.

## crates/audio/src/codec/encodec/EGM.md

Coverage: sampled (claims verified ~12 of ~14, source not fully read; mod.rs is about 1400 lines)
No drift detected. Interface symbols, REFERENCE_COMMIT and the test names resolve [encodec/mod.rs:Encodec::open, Encodec::load, Encodec::encode, Encodec::decode_codes, Encodec::frame_rate, EncodecConfig::from_json, EncodecConfig::hop_length].

## crates/audio/src/codec/encodec/README.md

Coverage: sampled (claims verified ~10 of ~13, source not fully read)
- Omits the env gate for the real checkpoint test: real_checkpoint_gate reads TURBOSPARK_ENCODEC_MODEL [encodec/tests.rs:real_checkpoint_gate]. Low.
- Test list is incomplete: lstm_forward_matches_original_over_many_steps_bitwise exists [encodec/tests.rs] but is not listed. Low.
- Hop/fps check: README states hop 320 / 75 fps "for [8,5,4,2] strides", but the code default is upsampling_ratios [8,8,4,2] (hop 512) [encodec/mod.rs:EncodecConfig::from_json]. The README qualifies the strides, so this is not a false claim, but confirm whether the default is meant to match the 24 kHz checkpoint.
- Verified: bandwidths [1.5, 3, 6, 12, 24] [encodec/mod.rs:target_bandwidths default], codebook 1024 x 128 [encodec/mod.rs:codebook_size, codebook_dim], fixtures testdata/encodec and tools/gen_encodec_fixtures.py exist.
- Not verified: "48 kHz supported by architecture".

## crates/audio/src/codec/fish_s1_dac/EGM.md

Coverage: sampled (claims verified ~12 of ~16, source not fully read; mod.rs is about 2900 lines)
- Omits public API: FishS1Dac::stream_on [fish_s1_dac/stream.rs:stream_on] is not listed. Low.
- Otherwise no drift. Interface symbols resolve, including FishS1Dac::load [fish_s1_dac/mod.rs:FishS1Dac::load], encode_zq and decode_zq [fish_s1_dac/mod.rs:FishS1Dac::encode_zq, FishS1Dac::decode_zq], DacExec::host and device [fish_s1_dac/exec.rs:DacExec], FishS1DacStream::push and position [fish_s1_dac/stream.rs:FishS1DacStream].

## crates/audio/src/codec/fish_s1_dac/README.md

Coverage: sampled (claims verified ~10 of ~13, source not fully read)
- (a) False claim: "encode_zq / decode_zq / from_latents ... are not ported". encode_zq and decode_zq are implemented and public [fish_s1_dac/mod.rs:FishS1Dac::encode_zq, FishS1Dac::decode_zq] and exercised by [fish_s1_dac/tests.rs:encode_zq_decode_zq_roundtrip_matches_shapes, decode_zq_validates_lengths]. The EGM.md already lists them as interface. Remove or correct the "Scope and Divergences" bullet.
- (a) from_latents: no such symbol exists anywhere in the crate.
- Verified: build_ae geometry [fish_s1_dac/mod.rs:FishS1DacConfig::build_ae] (44.1 kHz, encoder 64 with rates 2,4,8,8, latent 1024, decoder 1536 with rates 8,8,4,2, 4096 semantic plus 9 x 1024 acoustic books, dim 8, 2x2 downsample, transformer layers on last encoder block). Frame length 2048 is consistent with hop 512 x downsample 4. Env gate TURBOSPARK_FISH_S1_DAC_MODEL matches [fish_s1_dac/tests.rs:real_checkpoint_gate]. config.json is optional [fish_s1_dac/mod.rs:open], matching the README.

## crates/audio/src/codec/higgs_audio/EGM.md

Coverage: sampled (claims verified ~12 of ~15, source not fully read; mod.rs is about 1750 lines)
- Omits an invariant: HiggsAudioTokenizer::encode_tokens returns SpeechError::Unsupported unless the config carries semantic_model_config [higgs_audio/mod.rs:HiggsAudioTokenizer::encode_tokens, HiggsAudioTokenizer::load_encode_path]. Medium, since the EGM Invariants section is the place for it.
- Otherwise no drift. Interface symbols resolve, including HiggsAudioConfig::defaults and from_pytorch_json, HiggsAudioTokenizer::load, encode_acoustic, rvq_encode, rvq_decode_rows, AcousticEncoder, AcousticDecoder, HiggsRvq. sinc_resample is pub(crate) as stated [higgs_audio/semantic.rs:sinc_resample].

## crates/audio/src/codec/higgs_audio/README.md

Coverage: sampled (claims verified ~10 of ~14, source not fully read)
- (a) False claim: "The sinc resampler (_sinc_resample, torchaudio-compatible) ... is not ported". It is implemented [higgs_audio/semantic.rs:sinc_resample] and called from the full encode path [higgs_audio/mod.rs:HiggsAudioTokenizer::encode_tokens].
- (a) False claim: the full encode "fuses a wav2vec2 (HuBERT) semantic tower ... that tower is an STT model outside the codec crate and is not ported", and "semantic fusion happens in a future STT integration". The semantic encoder is ported [higgs_audio/semantic.rs:SemanticEncoder] and wired to the STT tower [higgs_audio/mod.rs:HiggsAudioTokenizer::load_encode_path, crate::stt::wav2vec::SemanticTower]. encode_tokens runs when semantic_model_config is present. The README should say the encode path is wired but gated on that config field, and that the released-format config refuses encode [higgs_audio/tests.rs:released_format_config_parses_and_encode_refuses_without_the_semantic_path]. Whether the parity test encode_fixture_pipeline_matches_the_reference passes was not run.
- Verified: defaults 8 books x 1024, dim 64, quantizer_latent_dim 1024, decoder_input_dim 256 [higgs_audio/mod.rs:HiggsAudioConfig::defaults]; env gate TURBOSPARK_HIGGS_AUDIO_MODEL matches [higgs_audio/tests.rs:real_checkpoint_gate].
- Not verified: "Encoder channels 64 -> 2048", "no weight norm".

## crates/audio/src/codec/mimi/EGM.md

Coverage: sampled (claims verified ~8 of ~9, source not fully read; mod.rs is about 1770 lines)
No drift detected. Interface resolves [mimi/mod.rs:MimiConfig::mimi_202407, MimiConfig::from_json, Mimi::load, Mimi::encode, Mimi::decode]. Load is in-memory only, consistent with Side Effects.

## crates/audio/src/codec/mimi/README.md

Coverage: sampled (claims verified ~8 of ~10, source not fully read)
- (a) Test count is wrong: the README says "cargo test -p turbospark-audio --lib codec::mimi (4 tests)". mimi/tests.rs has 6 #[test] functions: mimi_202407_factory_defaults, tiny_end_to_end_matches_reference, isolated_transformer_matches_reference, real_checkpoint_gate, attention_matches_original_within_ulps_of_scale, decode_refuses_codes_outside_the_codebook.
- Verified: mimi_202407 defaults (512 dim, 64 filters, ratios [8,6,5,4], 8 transformer layers, 8 heads, FF 2048, context 250, 2048 bins x 256 dims, 24 kHz, 12.5 fps) [mimi/mod.rs:MimiConfig::mimi_202407]. Batch-only scope matches [mimi/mod.rs:1-20 module doc]. Env gate TURBOSPARK_MIMI_MODEL matches [mimi/tests.rs:real_checkpoint_gate].

## crates/audio/src/codec/mimo_audio_tokenizer/EGM.md

Coverage: sampled (claims verified ~9 of ~11, source not fully read; mod.rs is about 1450 lines)
- Omits public API: MimoConfig::defaults [mimo_audio_tokenizer/mod.rs:MimoConfig::defaults] and MimoConfig's encode path are not listed. Low.
- Otherwise no drift. Interface symbols resolve, including downsample_rate, log_mel_spectrogram, encode_mels, decode, and LnormType [mimo_audio_tokenizer/mod.rs].

## crates/audio/src/codec/mimo_audio_tokenizer/README.md

Coverage: sampled (claims verified ~9 of ~12, source not fully read)
- Verified: defaults nfft 960, hop 240, encoder 32 layers, dim 1280, 20 heads, skip layer 3, pooler 2; decoder 32 causal layers; vocoder 16 layers, dim 256; 20 books [mimo_audio_tokenizer/mod.rs:MimoConfig::defaults]. Env gate TURBOSPARK_MIMO_AUDIO_TOKENIZER_MODEL matches [mimo_audio_tokenizer/tests.rs:real_checkpoint_gate]. RMSNorm via ln_type is parsed [mimo_audio_tokenizer/mod.rs:LnormType].
- Unclear, not drift: README writes the Vocos band mask as [-40, +10]. The code stores vocoder_attn_window as (40, 10) [mimo_audio_tokenizer/mod.rs:MimoConfig::defaults]. The sign convention is not documented in code or doc; check the mask builder before the README is trusted on this point.
- Not verified: istft.window loading and the "segment_size not reproduced" claim.

## crates/audio/src/codec/moss_audio_tokenizer/EGM.md

Coverage: sampled (claims verified ~8 of ~10, source not fully read; mod.rs is about 1430 lines)
- (b) Overstated Overview: "whole-signal and per-frame streaming encode/decode". encode_frame and decode_frame [moss_audio_tokenizer/mod.rs:MossAudioTokenizer::encode_frame, MossAudioTokenizer::decode_frame] are per-call versions of the reference _encode_frame/_decode_frame with no streaming state. The streaming step path (AttentionStepCache) is not ported [moss_audio_tokenizer/mod.rs:1-20 module doc]. Reword to "per-frame (stateless) encode/decode".
- Side Effects is accurate: read_dir is in open_checkpoint [moss_audio_tokenizer/mod.rs:open_checkpoint], called from open.
- Interface symbols resolve.

## crates/audio/src/codec/moss_audio_tokenizer/README.md

Coverage: sampled (claims verified ~7 of ~9, source not fully read)
- (c) Streaming section says the streaming step path is not ported but does not mention that the public per-frame API MossAudioTokenizer::encode_frame / decode_frame exists [moss_audio_tokenizer/mod.rs]. A reader will miss the per-frame entry points.
- Verified: weight-norm tensors parametrizations.weight.original0/1 folded at load [moss_audio_tokenizer/mod.rs:288-289]; default sample rate 48 kHz [moss_audio_tokenizer/mod.rs:from_json]; env gate TURBOSPARK_MOSS_AUDIO_TOKENIZER_MODEL matches [moss_audio_tokenizer/tests.rs:real_checkpoint_gate].
- Not verified: 32 books x 1024 with codebook dim 32, downsample 3840, channel count 2 default.

## crates/audio/src/codec/nemotron_voicechat/EGM.md

Coverage: sampled (claims verified ~9 of ~10, source not fully read; mod.rs is about 900 lines)
No drift detected. Interface resolves [nemotron_voicechat/mod.rs:NemotronVoiceChatConfig::from_json, frame_rate, waveform_to_token_ratio, NemotronVoiceChatCodec::open, load, encode, encode_latents, decode]. Low: stft_channels is public and not listed [nemotron_voicechat/mod.rs:NemotronVoiceChatConfig::stft_channels].

## crates/audio/src/codec/nemotron_voicechat/README.md

Coverage: sampled (claims verified ~9 of ~10, source not fully read)
No drift detected. Verified: base 384, 3 blocks per stage, downsample 7/7/9, latent 512, n_fft 16, hop 4, 31 quantizers x 1024 [nemotron_voicechat/mod.rs:NemotronVoiceChatConfig::from_json defaults]. Token length is hop x product of downsample rates = 1764 [nemotron_voicechat/mod.rs:waveform_to_token_ratio]. prvq.variance_list loaded strictly [nemotron_voicechat/mod.rs:load]. Env gate TURBOSPARK_NEMOTRON_VOICECHAT_MODEL matches [nemotron_voicechat/tests.rs:real_checkpoint_gate].

## crates/audio/src/codec/s3/EGM.md

Coverage: sampled (claims verified ~12 of ~13, source not fully read; mod.rs is about 1320 lines)
No drift detected. All listed symbols resolve: S3Config::v1, v2, from_json_with_defaults [s3/mod.rs]; S3TokenizerV1::new, open, load, quantize, dequantize; S3TokenizerV2::open, load, quantize; log_mel_spectrogram; V1_25HZ. Side Effects match open_checkpoint and read_dir [s3/mod.rs:open_checkpoint_file].

## crates/audio/src/codec/s3/README.md

Coverage: sampled (claims verified ~9 of ~11, source not fully read)
No drift detected on checked claims. Verified: v1 128 mels, state 1280, 4096 codebook [s3/mod.rs:S3Config::v1]; v2 FSQ 3^8 = 6561 [s3/mod.rs:S3Config::v2]; v2 sliding window 3000 frames with 2600 stride, so 400 overlap [s3/mod.rs:MAX_FRAMES, FRAMES_PER_STRIDE]; codebook read from quantizer.embed with the two aliases [s3/mod.rs:quantizer.embed alias list]; real checkpoint test is ignored and reads TURBOSPARK_S3_MODEL_DIR [s3/tests.rs:real_checkpoint_gate].
Not verified: the 25 Hz versus 50 Hz profile names and strides.

## crates/audio/src/codec/snac/EGM.md

Coverage: sampled (claims verified ~9 of ~10, source not fully read; mod.rs is about 1120 lines)
No drift detected. Interface resolves: SnacConfig::from_json, hop_length, frames_per_level; Snac::open, load, padded_len, encode, decode, decode_with_noise. Side Effects match open reading config.json [snac/mod.rs:open].

## crates/audio/src/codec/snac/README.md

Coverage: sampled (claims verified ~8 of ~11, source not fully read)
No drift detected on checked claims. Verified: LocalMHA attention configs refused with SpeechError::Unsupported [snac/mod.rs:load, attn_window_size check]; decode refuses noise-enabled checkpoints and points to decode_with_noise [snac/mod.rs:Snac::decode]; noise defaults on [snac/mod.rs:SnacConfig::from_json]; 9 tests in codec::snac [snac/tests.rs]; env gate TURBOSPARK_SNAC_MODEL and HF cache fallback match [snac/tests.rs:real_checkpoint_matches_reference].
Not verified: the transpose-conv output_padding = 1 claim (needs the ConvTranspose call site), the real-checkpoint 12 k-sample encode claim.

## crates/audio/src/codec/spark_bicodec/EGM.md

Coverage: sampled (claims verified ~8 of ~9, source not fully read; mod.rs is about 1160 lines)
No drift detected. Interface resolves: BiCodecConfig::from_yaml, BiCodec::open, load, detokenize, QuantizerConfig, SpeakerEncoderConfig, FeatDecoderConfig, WaveGeneratorConfig. Side Effects match config.yaml read in open [spark_bicodec/mod.rs:BiCodec::open].

## crates/audio/src/codec/spark_bicodec/README.md

Coverage: sampled (claims verified ~8 of ~10, source not fully read)
No drift detected on checked claims. Verified: speaker out_dim must equal prenet out channels, rejecting the 256/512 defaults [spark_bicodec/mod.rs:BiCodecConfig::validate]; prenet sample_ratios above 1 rejected [spark_bicodec/mod.rs:BiCodecConfig::validate]; negative tests exist [spark_bicodec/tests.rs:empty_and_out_of_range_tokens_are_errors, config_validation_rejects_inconsistent_geometry]; no real-checkpoint gate exists, consistent with "not run"; ../vocos link resolves [crates/audio/src/codec/vocos/EGM.md exists].
Not verified: SamplingBlock identity x3 behavior, the family README reference.

## Summary

Docs covered: 20 (10 EGM.md and 10 README.md).
Directories not finished: none.
Divergences needing a fix: fish_s1_dac/README.md (encode_zq/decode_zq/from_latents "not ported" is false), higgs_audio/README.md (sinc resampler and semantic tower "not ported" is stale), mimi/README.md (test count 4 vs 6), moss_audio_tokenizer/EGM.md (overstated "per-frame streaming"), higgs_audio/EGM.md (missing semantic_model_config invariant).

## crates/audio/src/codec/stepaudio2/EGM.md

Coverage: sampled (claims verified about 25 of 30, source 3.7k lines not fully read)
- Side Effects says `open` reads the real weights when the tiny fixture is absent. False: [crates/audio/src/codec/stepaudio2/mod.rs:StepAudio2Token2Wav::open] returns `SpeechError::BadConfig` when `tiny_weights.safetensors` is missing and never loads a real checkpoint.
- Interface omits the public `StepAudio2Config::from_json_with_defaults` [crates/audio/src/codec/stepaudio2/mod.rs:from_json_with_defaults].

## crates/audio/src/codec/stepaudio2/README.md

Coverage: sampled (about 30 numeric and structural claims checked: DiT depth 16, hidden 512, 10-step CFM, CFG 0.7, HiFT upsample [8,5,3], iSTFT 16/4, clip 0.99, 192-dim embedding, 6561 vocab, 320 input channels, 50*600 noise buffer, final leaky slope 0.01, SEG_POOL_LEN 100, test names and env var; source not fully read)
No drift detected.

## crates/audio/src/codec/vocos/EGM.md

Coverage: sampled (claims verified about 9 of 10; mod.rs read by targeted reads, not in full)
No drift detected.

## crates/audio/src/codec/vocos/README.md

Coverage: sampled (architecture defaults dim 512, intermediate 1536, 8 layers, n_fft 1024, hop 256, magnitude cap 1e2, test names, fixture dir and tool paths checked; source not fully read)
No drift detected.

## crates/audio/src/codec/wavvae/EGM.md

Coverage: sampled (interface, has_encoder gate, and error variants checked; mod.rs read in targeted sections)
- Side Effects says `open` reads `config.json` and the weights. Divergence: [crates/audio/src/codec/wavvae/mod.rs:WavVae::open] reads only `dir/model.safetensors` and hard-codes `WavVaeConfig::megatts3()`; no config file is read.

## crates/audio/src/codec/wavvae/README.md

Coverage: sampled (geometry, ratios, 960-sample hop, 2e-5 tiny-fixture tolerance, `encode` Unsupported path, and env vars checked; source not fully read)
- Omits the optional env var `TURBOSPARK_WAVVAE_TOL` (default 5e-3) that the real-checkpoint gate reads [crates/audio/src/codec/wavvae/tests.rs:real_checkpoint_gate or the test fn at line ~427].

## crates/audio/src/lid/EGM.md

Coverage: full read (mod.rs is 3 lines, only `//!` docs)
No drift detected.

## crates/audio/src/lid/README.md

Coverage: full read (no code claims; the link to `../../MODELS.md#lid` resolves to the `## LID` heading)
No drift detected.

## crates/audio/src/lm/EGM.md

Coverage: sampled (about 45 interface, invariant and side-effect claims checked, including sliding-window refusal, GQA divisibility, MoE tie order, Rope mrope sections == 3, cohere2 global-layer rule, norm_head refusal, ssm pub(crate); engine.rs, falcon_h1.rs and execution.rs not read in full)
No drift detected.

## crates/audio/src/lm/README.md

Coverage: sampled (fixture case list of 16 and per-case names, 2e-4 gate, MoeGate and RotatingKvCache unit tests, gpt2 constructors, Input::Embeddings path, ssm update formula checked; source not fully read)
- Type names differ from code: the README uses `KVCache` and `RotatingKVCache`, but the code defines [crates/audio/src/lm/cache.rs:KvCache] and [crates/audio/src/lm/cache.rs:RotatingKvCache].

## crates/audio/src/models/EGM.md

Coverage: full read (mod.rs is 28 lines; re-exports and aliases checked against source)
No drift detected.
- Not an EGM claim, but the module doc at [crates/audio/src/models/mod.rs:module doc] says "MiniMax Music 0.5"; the port in `music/minimax_music3` is MiniMax Music 3.

## crates/audio/src/music/EGM.md

Coverage: full read (mod.rs is 6 lines)
No drift detected.

## crates/audio/src/music/README.md

Coverage: full read (no code claims; links to `minimax_music3/README.md` and `../../MODELS.md#music` resolve)
No drift detected.

## crates/audio/src/music/minimax_music3/EGM.md

Coverage: sampled (constants, flow step counts, guidance, chunk constants, env var, and internal stage types checked; mod.rs (2.1k lines) read in targeted sections)
- Side Effects cites `Model::load_official_tree_with_precision` for the `std::fs::read_dir` enumeration. The call is in the helper [crates/audio/src/music/minimax_music3/mod.rs:load_official_tensors] (line ~1921); the citation is wrong.
- Interface and constants omit `AR_SAMPLING_TOP_K`, `CROP_RIGHT_LATENT`, `LATENT_HOP_LENGTH` and `OVERLAP_LATENT_LENGTH` [crates/audio/src/music/minimax_music3/mod.rs:CROP_RIGHT_LATENT].

## crates/audio/src/music/minimax_music3/README.md

Coverage: sampled (about 60 identifiers, file paths, CLI flags, defaults, test names and kernel names checked against source; benchmark numbers, WAV hashes, Metal timings and the upstream MLX/audio.cpp comparisons are not verifiable from this code and were not checked; mutation-check claims were not re-run)
- Omits the test and gate env vars `TURBOSPARK_MUSIC3_BENCH_TREES` (only in the README's timing text by name, not listed), `TURBOSPARK_MUSIC3_INSTALL_DIR`, `TURBOSPARK_MUSIC3_TURBO_INSTALL_DIR` [crates/runtime/tests/music3_metal.rs], and `TURBOSPARK_MUSIC3_ORACLE_DEVICE`, `TURBOSPARK_MUSIC3_ORACLE_ANY_DEVICE` [crates/gpu/tests/music3_oracle.rs]. These gate the device tests the README cites.
- Text frontend says the default is "30 flow steps". The default follows the active profile: [crates/audio/src/music/minimax_music3/mod.rs:Model::set_flow_profile] changes `default_steps`. The README does not say so in that section.
- The `MM3_*` env vars (`MM3_AR_CFG_FRAMES`, `MM3_DEPTH_B1`, `MM3_DEPTH_GPU_*`) appear only as upstream references in the README. Correctly labeled, no code drift.

## crates/audio/src/ops/EGM.md

Coverage: sampled (module gate, reference function names, and test names checked; kernel_parity.rs (1.2k lines) read by listing its fn items, not in full)
- Interface lists five reference functions. The file also defines [crates/audio/src/ops/kernel_parity.rs:conv2d_reference], `rope_tables_reference`, `rope_neox_reference`, `rope_interleaved_reference`, `split_qkv_reference`, `layernorm_rows_reference`, `layernorm_affine_free_reference` and `layernorm_channels_reference`. These are omitted.

## crates/audio/src/sts/EGM.md

Coverage: full read (mod.rs is 15 lines; re-exports match exactly; sub-EGM links resolve)
No drift detected.

## crates/audio/src/sts/README.md

Coverage: sampled (directory links to the five family READMEs and `../../MODELS.md#sts` checked; the family architecture claims live in subdirectories assigned to other jobs and were not checked here)
No drift detected.

## crates/audio/src/sts/deepfilternet/EGM.md

Coverage: sampled (interface, open() reads, and invariants verified; mod.rs read in targeted sections, not in full)
- Omits that `enhance` expects 48 kHz input: there is no rate check, and the `sample_rate` field is dead code [crates/audio/src/sts/deepfilternet/mod.rs:DeepFilterNet::enhance]

## crates/audio/src/sts/deepfilternet/README.md

Coverage: sampled (FFT geometry, example path, and layer names verified; mod.rs read in targeted sections, not in full)
- "Supported Versions: DeepFilterNet v1, v2, and v3": the code has a v3 layout plus a v2 switch (`enc_concat`); no v1-specific path was found [crates/audio/src/sts/deepfilternet/mod.rs:DeepFilterNet::open]
- "square-root Hann (Vorbis) window": the code builds the Vorbis power-complementary window sin(pi/2 * sin^2(...)), which is not a square-root Hann window [crates/audio/src/sts/deepfilternet/mod.rs:vorbis_window]
- Witness numbers (waveform correlation 0.964, max abs error 0.021) appear only in this README; no fixture or assertion in the crate pins them [crates/audio/src/sts/deepfilternet/mod.rs:tests]

## crates/audio/src/sts/dialogue_sidon/EGM.md

Coverage: sampled (all listed interface names, error variants, and both side-effect file reads verified)
No drift detected.

## crates/audio/src/sts/dialogue_sidon/README.md

Coverage: sampled (frontend framing, schedule, defaults, and fixture file list verified; parity table not re-run)
- Overview says the model runs "at 24 kHz", but the frontend frames at 400/160 samples (16 kHz timing). The config `sample_rate` (default 24000) is parsed and never read; nothing resamples or checks the input rate [crates/audio/src/sts/dialogue_sidon/mod.rs:SidonConfig::from_json, crates/audio/src/sts/dialogue_sidon/mod.rs:DialogueSidon::extract_features]
- Fixture path `testdata/dialogue_sidon/` does not resolve from the repo root; the directory is `crates/audio/testdata/dialogue_sidon/` [crates/audio/src/sts/dialogue_sidon/tests.rs:testdata]

## crates/audio/src/sts/mel_roformer/EGM.md

Coverage: sampled (interface names, error variants, and read_dir scan verified; config and forward read in targeted sections)
No drift detected.

## crates/audio/src/sts/mel_roformer/README.md

Coverage: sampled (STFT, band count, tanh mask MLP, and fixture file list verified; parity table not re-run)
- Window naming is inconsistent with the code: the overview says "symmetric-Hann" and the architecture section says "periodic (symmetric) Hann". The code builds a periodic Hann (`dsp::hann_window`) and its comment says so [crates/audio/src/sts/mel_roformer/mod.rs:MelRoFormer::load]
- "The reference refuses to guess a config; MelRoFormer::open does the same" and "this port ships ... no default config": `open` does require config.json, but `MelRoFormerConfig::from_json` fills defaults for every missing key (dim 384, depth 6, mask_estimator_depth 2, n_fft 2048, hop_length 441, num_bands 60, num_overlap 2) [crates/audio/src/sts/mel_roformer/mod.rs:MelRoFormerConfig::from_json]
- The named presets (kim_vocal_2, viperx_vocals, zfturbo_bs_roformer, zfturbo_vocals_v1) do not exist in the crate; they describe the reference only, which the section does not say [crates/audio/src/sts/mel_roformer/mod.rs:MelRoFormerConfig::from_json]
- Fixture path `testdata/mel_roformer/` does not resolve from the repo root; the directory is `crates/audio/testdata/mel_roformer/` [crates/audio/src/sts/mel_roformer/tests.rs]

## crates/audio/src/sts/mossformer2_se/EGM.md

Coverage: sampled (interface names, enhance wrapper, fbank dither refusal, and quantization scheme verified)
- Side Effects says `open` reads config.json. The file is optional: when it is absent, `open` uses the reference defaults [crates/audio/src/sts/mossformer2_se/mod.rs:MossFormer2Se::open]

## crates/audio/src/sts/mossformer2_se/README.md

Coverage: sampled (fbank size, FFT rule, segment and chunk thresholds, and quantization verified; parity tables not re-run)
- Fixture path `testdata/mossformer2_se/` does not resolve from the repo root; the directory is `crates/audio/testdata/mossformer2_se/` [crates/audio/src/sts/mossformer2_se/tests.rs]

## crates/audio/src/sts/sam_audio/EGM.md

Coverage: sampled (interface names, shape check at load, and open() file handling verified; T5 and DiT read in targeted sections)
- Side Effects lists only config.json and the weights. `open` also requires a separate `tiny_codec.safetensors` (the codec file name is hard-coded) and falls back from `model.safetensors` to `tiny_weights.safetensors` [crates/audio/src/sts/sam_audio/mod.rs:SamAudio::open]

## crates/audio/src/sts/sam_audio/README.md

Coverage: sampled (ODE defaults, T5 attention scaling, gates, anchors, and head-dim layout verified; RoPE and parity numbers partly verified)
- RoPE theta: README says theta = max(10000, 2 * max_positions). The code uses `ROPE_THETA_BASE.max(2.0 * 1024.0)` and never reads the config's `max_positions` (parsed, default 10000, otherwise unused). For the default max_positions the README gives 20000 and the code uses 10000 [crates/audio/src/sts/sam_audio/mod.rs:DiT::rope_tables]
- Fixture path `testdata/sam_audio/` does not resolve from the repo root; the directory is `crates/audio/testdata/sam_audio/` [crates/audio/src/sts/sam_audio/tests.rs]

## crates/audio/src/vad/EGM.md

Coverage: full read (mod.rs is 14 lines; all re-exports checked against the listed names)
No drift detected.

## crates/audio/src/vad/README.md

Coverage: full read (index README; family links and the MODELS.md#vad anchor checked)
No drift detected.

## crates/audio/src/vad/nemotron_diarization/EGM.md

Coverage: sampled (interface names, StreamingProfile re-export, error variants, and coupled-mode checks verified; load and coupled read in targeted sections)
- Side Effects cites `[mod.rs:open]`, but there is no `open` function. The loader is `NemotronDiarization::load` [crates/audio/src/vad/nemotron_diarization/mod.rs:NemotronDiarization::load]; the doc comment above it says "Opens"
- "no writes, network, or env access" holds for non-test code only. `coupled_chunk_preds_match_nemo` reads the env var `TURBOSPEECH_NEMOTRON_MODEL` [crates/audio/src/vad/nemotron_diarization/mod.rs:coupled_chunk_preds_match_nemo]

## crates/audio/src/vad/nemotron_diarization/README.md

Coverage: sampled (preset windows, coupled geometry, Metal executor path, and listed files verified; parity and benchmark claims not re-run)
- Example usage is wrong. The example takes `<model-dir> <mode> <wav>` with mode `offline|stream|window`, not `<model-dir> <audio.wav>` [crates/audio/examples/nemotron_diarize.rs:main]
- `scripts/convert_multitalker_nemo.py` does not exist at the repo root. The script is at `crates/audio/scripts/convert_multitalker_nemo.py`
- The coupled parity test is skipped unless `TURBOSPEECH_NEMOTRON_MODEL` is set. The README does not name this variable [crates/audio/src/vad/nemotron_diarization/mod.rs:coupled_chunk_preds_match_nemo]

## crates/audio/src/vad/silero_vad/EGM.md

Coverage: sampled (interface names, per-caller state ownership, and load-time file read verified)
- Side Effects is accurate, but the doc comment on `SileroVad::load` says an optional `config.json` is read. `load` never opens config.json [crates/audio/src/vad/silero_vad/mod.rs:SileroVad::load]

## crates/audio/src/vad/silero_vad/README.md

Coverage: sampled (branch sizes, LSTM width, chunk sizes, conv layers, and hysteresis fields verified; parity and latency numbers not reproduced)
- Parity claim (188 streaming chunks, max probability error 4.8e-7, identical timestamps) is not pinned by any assertion or fixture reference in `silero_vad/mod.rs` [crates/audio/src/vad/silero_vad/mod.rs:tests]
- The hysteresis release threshold (`threshold - 0.15`, floor 0.01) is not documented; the README says only "dynamic hysteresis" [crates/audio/src/vad/silero_vad/mod.rs:SileroVad::timestamps]

## crates/audio/src/vad/sortformer/EGM.md

Coverage: sampled (interface names, StreamingProfile presets, and error variants verified; load read in targeted sections)
- Side Effects cites `[mod.rs:open]`, but there is no `open` function. The loader is `Sortformer::load` [crates/audio/src/vad/sortformer/mod.rs:Sortformer::load]; the doc comment above it says "Opens"

## crates/audio/src/vad/sortformer/README.md

Coverage: sampled (25 ms window, 10 ms hop, 4-speaker default, Slaney mel scale, and example files verified; RTF and DER claims are not verifiable from the source tree)
- "AOSC speaker-cache compression matches reference fixture byte-for-byte": the test `aosc_compression_matches_reference_fixture` compares with a 1e-4 tolerance, not byte equality [crates/audio/src/vad/sortformer/mod.rs:aosc_compression_matches_reference_fixture]
- "FastConformer subsampling, relative self-attention, and Bart encoder outputs match golden fixtures within 1e-5": there is no stage-level test for subsampling or the BART encoder. `offline_forward_matches_reference` checks only final predictions at 1e-5, and the relative-attention tests are known-answer checks [crates/audio/src/vad/sortformer/mod.rs:offline_forward_matches_reference]

## crates/audio/tools/EGM.md

Coverage: sampled (generator names, probe names, OUT path, --real flag, and the megatts3 test import verified; 71 gen_*.py files are listed, not all read)
- Overview limits scope to "codec and TTS families", but the directory also holds STS generators (`gen_dialogue_sidon_fixtures.py`, `gen_mel_roformer_fixtures.py`, `gen_mossformer2_se_fixtures.py`, `gen_sam_audio_fixtures.py`) and LM generators (`gen_llama_fixtures.py`, `gen_qwen3_fixtures.py`, `gen_lm_fixtures.py`) [crates/audio/tools/gen_dialogue_sidon_fixtures.py]

## crates/audio/src/stt/EGM.md

Coverage: full read. All 30 child family links resolve, every `pub mod` and `pub use` named in the doc matches mod.rs, and the cross-family reuse claims (qwen3_asr decoder/tokenizer, nemo alignment types, wav2vec helpers in mms) match imports.
No drift detected.

## crates/audio/src/stt/README.md

Coverage: sampled (family links, example binaries, MODELS.md and docs/COREML.md paths, runtime::native_audio, coreml parakeet files, server --stt-model and HTTP/WebSocket routes all verified; per-family prose claims such as bit widths and benchmark-free descriptions not verified).
No drift detected.

## crates/audio/src/stt/canary/EGM.md

Coverage: sampled (mod.rs load and transcribe_impl, config.rs from_json refusals, tokenizer build_prompt_tokens and decode, decoder load and API read; pins, env var, and max_tokens semantics checked; decoder forward bounds and Linear bias check not re-read).
No drift detected.

## crates/audio/src/stt/canary/README.md

Coverage: sampled (pinned repository and revision, 8-bit/group-64 statement, and env var checked against source; prose body not read in full).
No drift detected.

## crates/audio/src/stt/cohere_asr/EGM.md

Coverage: full read (mod.rs read in full; detect and load error variants, pins, and test names match).
No drift detected.

## crates/audio/src/stt/cohere_asr/README.md

Coverage: sampled (both pinned revisions and repository names match mod.rs; body prose not read in full).
No drift detected.

## crates/audio/src/stt/fireredasr2/EGM.md

Coverage: sampled (config.rs geometry checks, cmvn/dict parsing, frontend 32768 scaling and 25/10 ms framing, encoder conv remap, decoder defaults, beam init 0 vs -1e10, length/eos checks, output cleanup, pins, and #[ignore] parity tests verified; full encoder attention and KV cache internals not re-read).
No drift detected.

## crates/audio/src/stt/fireredasr2/README.md

Coverage: sampled (pinned revision and TURBOSPARK_FIREREDASR2_DIR checked against source; prose not read in full).
No drift detected.

## crates/audio/src/stt/fun_asr_nano/EGM.md

Coverage: sampled (profile pin, Config::from_json geometry, load quantization (4-bit, group 128) and tied 1024-wide decoder, stop tokens, ceil-div audio token count, LFR edge padding, reuse imports from sensevoice and qwen3_asr verified; tokenizer.json path verified).
No drift detected.

## crates/audio/src/stt/fun_asr_nano/README.md

Coverage: sampled (revision hashes present in the README that are absent from source are file SHA-256 digests of snapshot files, not pins; env var TURBOSPARK_FUN_ASR_NANO_DIR found in source).
No drift detected.

## crates/audio/src/stt/glmasr/EGM.md

Coverage: sampled (pin, model_type/adapter/RoPE checks, affine 4-bit group 64 check, hardcoded text geometry (59264, 2048, 28, 16, 4, 128), 30 s limit and 1 s zero-pad, 1500-row encoder bound, default 128 tokens, 400/160/128 frontend with peak-8 and (x+4)/4, 32-dim audio RoPE, stop tokens, env var verified).
No drift detected.

## crates/audio/src/stt/glmasr/README.md

Coverage: sampled (pinned revision and TURBOSPARK_GLMASR_DIR checked against source; prose not read in full).
No drift detected.

## crates/audio/src/stt/granite_speech/EGM.md

Coverage: sampled (pins for 1B and Plus, quantized refusal (.scales, .bias_packed, quantization_config, has_lora_adapter), projector/encoder-size invariant, rope_type and bias checks, tokenizer <|audio|> and <|end_of_text|> checks, frontend peak/floor/normalization, midpoint self-conditioning at num_layers/2, f32::MIN mask, embedding/logits multipliers, max_tokens 0 mapped to 256, pub(crate) visibility of GraniteLlm methods and frontend items verified).
No drift detected.

## crates/audio/src/stt/granite_speech/README.md

Coverage: sampled (1B pinned revision and env var verified; several revision-like hashes in the README do not appear in source and were not traced to a pinned claim).
No drift detected.

## crates/audio/src/stt/granite_speech5_ctc/EGM.md

Coverage: sampled (pin, model_type/architecture/tie_word_embeddings checks, num_mel_bins 80|128, subsample_layers [0,1], pad_token_id 0, odd kernel, head_dim check, frontend 2-frame rounding and 2^27 cap, LOGIT_ROW_BATCH 8 and batched collapse, relative table size, crate-private items verified).
No drift detected.

## crates/audio/src/stt/granite_speech5_ctc/README.md

Coverage: sampled (pinned revision 947f59af... matches source; the other revision-like hash is not a pin in source).
No drift detected.

## crates/audio/src/stt/granite_speech_nar/EGM.md

Coverage: sampled (pin 6acb7892..., GraniteSpeechNarForASR and scale_projected_embeddings checks, 348-wide self-conditioning vocabulary, 1e-6 pooling floor, frontend 512/160/80 with 2*(n/320) truncation and peak-minus-8 floor, ctc_collapse_decode and add_insertion_slots, rope_half, qwen3_asr::load_tokenizer, TURBOSPARK_GRANITE_SPEECH_NAR_MODEL_DIR verified; Editor embed range check seen only in part).
No drift detected.

## crates/audio/src/stt/granite_speech_nar/README.md

Coverage: sampled (pinned revision and env var verified; other revision-like hashes not traced).
No drift detected.

## crates/audio/src/stt/higgs_audio_3/EGM.md

Coverage: sampled (pin 2ffd1aa..., STOP_IDS 151645/151643, load_with_vad and tokenizer id checks for <|AUDIO|> and <|audio_eos|>, parse_output and remove_special_spans semantics (no DOTALL, unclosed think keeps tail), SpeechRanges, SileroRanges, split_long, chunk_ranges, chunk_samples, qk_norm: true, GELU and SiLU checks, log_mel_128 constants, d_model divisibility verified).
No drift detected.

## crates/audio/src/stt/higgs_audio_3/README.md

Coverage: sampled (pinned revision and TURBOSPARK_HIGGS_AUDIO_3_MODEL_DIR verified; many revision-like hashes in the README are not in source and were not traced).
No drift detected.

## crates/audio/src/stt/lasr_ctc/EGM.md

Coverage: sampled (pin 3b967580..., N_FFT 512 / hop 160 / 125-7500 Hz mel, MEL_CLAMP 1e-5, residual weights [1.5,0.5] and [2.0,1.0], BatchNorm eps 1e-5, attention_bias and convolution_bias checks, SiLU check, pad_token_id < vocab_size, FFT power-of-two assert, short-audio Input error, ctc_decode repeat/blank/special/U+2581 handling and leading-space strip verified).
No drift detected.

## crates/audio/src/stt/lasr_ctc/README.md

Coverage: sampled (the README's hashes are SHA-256 digests of snapshot files, which source does not contain; env var TURBOSPARK_LASR_CTC_MODEL_DIR found in source; prose not read in full).
No drift detected.

## crates/audio/src/stt/mega_asr/EGM.md

Coverage: sampled (approx. 30 claims checked; module is 2452 lines, not fully read)
No drift detected.

## crates/audio/src/stt/mega_asr/README.md

Coverage: sampled (pinned revisions, env var, script and fixture paths, profile rules checked; artifact sizes and hashes not checkable from source)
No drift detected.

## crates/audio/src/stt/mms/EGM.md

Coverage: sampled (approx. 13 claims checked; mod.rs is 839 lines, under the full-read threshold but not read in full)
No drift detected.

## crates/audio/src/stt/mms/README.md

Coverage: sampled (env vars, pinned revision, fixture and test paths, CTC decode and adapter claims checked; artifact sizes and hashes not checkable from source)
- Fixture-run snippet exports `TURBOSPARK_MMS_CACHE=/tmp/turbospark-mms-hf-cache`, but no Rust or Python file in the repo reads it; the snippet's `snapshot_download(..., cache_dir=...)` call hardcodes the path, so the export has no effect [crates/audio/src/stt/mms/mod.rs:tests::pinned_english_checkpoint_matches_mlx_reference] (the only env var read is `TURBOSPARK_MMS_DIR`).

## crates/audio/src/stt/moonshine/EGM.md

Coverage: sampled (approx. 14 claims checked; mod.rs is 1136 lines, not read in full)
- Overview calls the module a "pure CPU f32 port". That is true of this module, but the module's GPU sibling (`TURBOSPARK_MOONSHINE_DEVICE`, `TURBOSPARK_MOONSHINE_PROFILE`) lives in [crates/runtime/src/moonshine.rs] and is not mentioned, so the EGM gives no pointer to it (low).

## crates/audio/src/stt/moonshine/README.md

Coverage: sampled (env vars, example binaries, relative links, benchmark anchor checked; checkpoint hashes not checkable from source)
- The table presents revision `2ccd087d043f77a4a07bd412b6725266355b08f4` as the pinned checkpoint, but no constant or check in code holds it. [crates/audio/src/stt/moonshine/mod.rs:Moonshine::open] loads whatever directory it is given (low; doc-only pin).

## crates/audio/src/stt/moss_music/EGM.md

Coverage: sampled (approx. 14 claims checked; mod.rs is 1209 lines, not read in full)
No drift detected.

## crates/audio/src/stt/moss_music/README.md

Coverage: sampled (pinned revision, env var, ignored-test reason, fixture path checked; the 40 GiB and 36 GiB host figures are not verifiable from source)
No drift detected.

## crates/audio/src/stt/moss_transcribe_diarize/EGM.md

Coverage: sampled (approx. 15 claims checked; 2036 lines, not read in full)
No drift detected.

## crates/audio/src/stt/moss_transcribe_diarize/README.md

Coverage: sampled (pinned chat template digest, env var, fixture and module paths checked)
No drift detected. The `mlx_audio/...` paths are upstream files outside this repo, not broken links.

## crates/audio/src/stt/multitalker_asr/EGM.md

Coverage: sampled (approx. 18 claims checked; 2814 lines, not read in full)
- Interface omits the public method `MultitalkerAsr::pre_encode_cache_frames` [crates/audio/src/stt/multitalker_asr/mod.rs:MultitalkerAsr::pre_encode_cache_frames] (low).

## crates/audio/src/stt/multitalker_asr/README.md

Coverage: sampled (pinned revisions, env vars, the five named test IDs, relative links checked; not every benchmark statement verified)
No drift detected. `speaker_streaming.py` is an upstream NeMo file named for contrast, not a repo link.

## crates/audio/src/stt/nemo/EGM.md

Coverage: full read of non-test source (alignment.rs lines 1-328 and mod.rs); the test module (alignment.rs lines 329-504) was not read
No drift detected.

## crates/audio/src/stt/nemotron_asr/EGM.md

Coverage: sampled (approx. 14 claims checked; 2880 lines, not read in full; the blank-id invariant in rnnt.rs was not re-read)
No drift detected.

## crates/audio/src/stt/nemotron_asr/README.md

Coverage: sampled (pinned revision, example binary, method names, cross-link to multitalker checked)
No drift detected.

## crates/audio/src/stt/parakeet/EGM.md

Coverage: sampled (approx. 14 claims checked; 2124 lines, not read in full)
- Calls `normalized_config` and `convert_checkpoint` "crate-private", but both are `pub(super)` [crates/audio/src/stt/parakeet/redux.rs:convert_checkpoint], so they are visible only to the parakeet module (wording, low).

## crates/audio/src/stt/parakeet/README.md

Coverage: sampled (three pinned revisions, default checkpoint path, example binary, fixture script and testdata directory checked)
No drift detected.

## crates/audio/src/stt/phonon/EGM.md

Coverage: sampled (approx. 14 claims checked; 2295 lines, not read in full)
- Calls `materialize::{convert_checkpoint, open_shard, write_safetensors, temporary_path, load_models}` "crate-private", but they are `pub(super)` [crates/audio/src/stt/phonon/materialize.rs:convert_checkpoint], and the module is `pub mod materialize` [crates/audio/src/stt/phonon/mod.rs:materialize] (wording, low).

## crates/audio/src/stt/phonon/README.md

Coverage: sampled (pinned revision, shard size and SHA-256, manifest format strings, env var checked)
No drift detected.

## crates/audio/src/stt/qwen2_audio/EGM.md

Coverage: sampled (approx. 16 claims checked; 2525 lines, not read in full)
No drift detected.

## crates/audio/src/stt/qwen2_audio/README.md

Coverage: sampled (pinned revision, env var, fixture paths checked)
No drift detected. The README's mention of `qwen2_audio_reference.wav` is corrected in the same paragraph (the file used is `qwen3_forced_aligner_reference.wav`), so it is not reported.

## crates/audio/src/stt/qwen3_asr/EGM.md

Coverage: sampled (claims verified about 20 of 27 bullets; decoder.rs, encoder.rs and checkpoint.rs not fully read)
- Interface omits the public default on `Qwen3Asr::transcribe`, which passes `max_tokens` 512 [mod.rs:Qwen3Asr::transcribe]

## crates/audio/src/stt/qwen3_asr/README.md

Coverage: sampled (pinned revision, profile constant, env vars, test names, fixture and script paths, ignored-test count and Metal engine tests checked; timing, memory and checksum figures are measurements and were not verifiable from source)
No drift detected.

## crates/audio/src/stt/qwen3_forced_aligner/EGM.md

Coverage: sampled (claims verified about 20 of 22 bullets; mod.rs read in large part, 584 lines)
No drift detected.

## crates/audio/src/stt/qwen3_forced_aligner/README.md

Coverage: sampled (pinned profile, fixture word count of 9, ignored test and env var, language gate and fixture scripts checked; artifact sizes and digests not recomputed)
No drift detected.

## crates/audio/src/stt/sensevoice/EGM.md

Coverage: sampled (claims verified about 18 of 20 bullets; frontend.rs and mod.rs read in large part, sanm.rs sampled)
- Verification section omits the ignored checkpoint gate and its env var `TURBOSPARK_SENSEVOICE_MODEL_DIR` [mod.rs:pinned_checkpoint_matches_mlx_transcript_and_tags]
- Interface calls `frontend::extract_features` crate-visible, but it is `pub(super)` [frontend.rs:extract_features]

## crates/audio/src/stt/sensevoice/README.md

Coverage: sampled (pinned revision, re-export, language list, `use_itn` token, load signature, test name and fixture and script paths checked; file size and tolerance claims not verified)
No drift detected.

## crates/audio/src/stt/vibevoice_asr/EGM.md

Coverage: sampled (claims verified about 20 of 25 bullets; mod.rs, language_model.rs and resample_poly.rs sampled, tokenizer_encoder.rs partly)
No drift detected.

## crates/audio/src/stt/vibevoice_asr/README.md

Coverage: sampled (pinned revision, prompt keys, decoder geometry, EOS and 151665 ids, refusals, `transcribe` signature, resampler constants, fixture and script paths checked; stage parity numbers and file digests not recomputed)
No drift detected.

## crates/audio/src/stt/voxtral/EGM.md

Coverage: sampled (claims verified about 18 of 22 bullets; mod.rs read in relevant parts, 1050 lines)
- Frontend claim is false: `compute_features` does `padded.resize(TARGET_SAMPLES, 0.0)`, which truncates input longer than 480000 samples, so frames are capped at 3000 and "pads (never truncates)" / "yields more than 3000 frames" is wrong [mod.rs:compute_features]

## crates/audio/src/stt/voxtral/README.md

Coverage: sampled (pinned revision, profile constant, Tekken loader, 480000-sample limit and 128-token cap checked; the 32/1280/20 and 24/3072 tower and decoder geometry are not hardcoded in source and were not verified)
- "pinned English transcription prompt tokens" overstates the code: the crate pins no prompt, and `Voxtral::transcribe` takes `prompt_ids` from the caller [mod.rs:Voxtral::transcribe]
- Input "up to 30 seconds" omits that longer audio is silently truncated to 480000 samples [mod.rs:compute_features]

## crates/audio/src/stt/voxtral_realtime/EGM.md

Coverage: sampled (claims verified about 22 of 25 bullets; mod.rs and encoder.rs/decoder.rs spot-read, tokenizer.rs sampled)
- Generation cap is off by one: the token is pushed before the `generated.len() > MAX_NEW_TOKENS` check, so up to 4097 entries (including EOS) are recorded, not "max 4096" [mod.rs:VoxtralRealtime::transcribe]

## crates/audio/src/stt/voxtral_realtime/README.md

Coverage: sampled (profile pin, 1280-sample token unit, 1.5 mel clamp, encoder and decoder geometry, 750 window, interleaved RoPE, `encode_full` 750 limit, StreamingSession refusal, script and fixture paths checked)
No drift detected.

## crates/audio/src/stt/wav2vec/EGM.md

Coverage: sampled (pinned revision, config refusals, decode_ctc blank and pipe handling, weight-norm gain rule, lm_head, env var checked; forward_hidden_mean and the short-input error not traced)
No drift detected.

## crates/audio/src/stt/wav2vec/README.md

Coverage: sampled (pinned revision, config refusals, epsilon placement vs mms, greedy CTC, fixture script and testdata paths checked; 32-piece vocabulary and layer and conv counts not verified from source)
No drift detected.

## crates/audio/src/stt/whisper/EGM.md

Coverage: sampled (config constructors and validate, 3000-frame and 1500-step encode, 50259 language token, eps 1e-5, special token ids checked; decode_step logits-only and prompt exclusion not traced; mod.rs is 779 lines, read in part)
No drift detected.

## crates/audio/src/stt/whisper/README.md

Coverage: sampled (module path, profile table rows for tiny, base, small, large_v3 and distil_large_v3 checked against constructors)
- Profile table lists `whisper-medium` (24 layers, d_model 1024, 16 heads), but `WhisperConfig` has no medium constructor [mod.rs:WhisperConfig::tiny]
- "Greedy or beam search decoding": only greedy decoding exists, with no beam search code [mod.rs:WhisperModel::transcribe]
- Decoding lists a language token as a standard prompt token, but the language is hardcoded to English (50259) with no selection [mod.rs:WhisperModel::transcribe]

## crates/audio/src/stt/whistle/EGM.md

Coverage: sampled (container tag, count and size limits, 64-byte alignment, dtypes, PCM bounds, mel bank shape, 318-token and 8192-logit limits, LANGUAGES, engram layers 3 and 7, sqrt(DIM) embedding scale, permutation tensors 226 and 227 checked; decoder and cact.rs internals partly read)
No drift detected.

## crates/audio/src/stt/whistle/README.md

Coverage: sampled (example path, 160 to 480000 sample bounds, module path `models::stt::whistle`, embedding width 512, KV and query geometry, stem kernels, CQ quantization, fixture file set, env var, token and language facts checked; container tensor count, checkpoint byte size and test tolerances not verifiable offline)
No drift detected.

# Drift report: crates/audio/src/tts (11 directories, 22 doc files)

## crates/audio/src/tts/EGM.md

Coverage: sampled (claims verified: module and alias lists in mod.rs, pocket_tts no-alias invariant, all 40 child links resolve; sampling.rs not read)
No drift detected.

## crates/audio/src/tts/README.md

Coverage: sampled (claims verified: 19 relative links resolve, ../../MODELS.md resolves to crates/audio/MODELS.md, every family dir has a README.md; family blurbs not checked against code)
- Family list omits 20 of the 39 family modules declared in mod.rs: arktts, bark, chatterbox, chatterbox_turbo, confucius4, dia, dramabox, echo_tts, fish_drama3, fish_qwen3_omni, irodori_tts, longcat_audiodit, melotts, omnivoice, pocket_tts, tada, vibevoice, voxcpm, voxcpm2, voxtral_tts [mod.rs:pub mod]

## crates/audio/src/tts/arktts/EGM.md

Coverage: sampled (claims verified: all listed types and fns, SamplingParams defaults, ModelConfig defaults, 15 tests, empty-generation error, no env/thread/log use; source 3065 lines, targeted reads)
No drift detected.

## crates/audio/src/tts/arktts/README.md

Coverage: sampled (claims verified: falcon_h1 seam symbols, codec widths and strides, 15 test count, testdata/tts_arktts cases, tolerances; targeted reads)
- Fixture generator path does not resolve. README says `../../../../tools/gen_arktts_fixtures.py` (relative to arktts/, that is crates/tools/), but the script is at crates/audio/tools/gen_arktts_fixtures.py [README.md:Fixtures]

## crates/audio/src/tts/bailingmm/EGM.md

Coverage: sampled (claims verified: listed symbols in mod.rs, dit.rs, audio_vae.rs, ming_qwen2.rs; patch-geometry and max_decode_steps checks; stop rule step > 3 with scores[1] > 0.5; no env/thread/log use; source 3354 lines, targeted reads)
No drift detected.

## crates/audio/src/tts/bailingmm/README.md

Coverage: sampled (claims verified: frozen tolerance table against tests.rs constants, fixture cases, deviations on dense lm_head, speed no-op, lm README link; targeted reads)
- Voice-clone waveform gate is misstated. README table says gate 2e-4 (measured 1.2e-4), but the test asserts WAVEFORM_TOLERANCE = 2e-5 [tests.rs:voice-clone waveform assert, tests.rs:WAVEFORM_TOLERANCE]. At 2e-5 the measured 1.2e-4 would fail, so the README or the gate is stale.
- Fixture script is cited as `tools/gen_bailingmm_fixtures.py` with no directory; it lives at crates/audio/tools/gen_bailingmm_fixtures.py (minor).

## crates/audio/src/tts/bark/EGM.md

Coverage: sampled (claims verified: Model/Gpt/FineGpt/GenerateParams symbols, listed consts, forced coarse params in Model::generate, sliding_window_len check, generate_fine empty check, no RNG/env/thread use; source 1440 lines, targeted reads)
- Constant list is partial. Omits TEXT_PAD_TOKEN, SEMANTIC_RATE_HZ, SEMANTIC_VOCAB_SIZE, CODEBOOK_SIZE, COARSE_RATE_HZ, COARSE_SEMANTIC_PAD_TOKEN, COARSE_INFER_TOKEN [mod.rs:SEMANTIC_VOCAB_SIZE] (minor; the doc says "Token and rate consts")
- Voice-prompt history is accepted but dropped in the coarse stage. generate_coarse takes semantic_history and coarse_history but discards both with `let _ = (semantic_history, coarse_history);` [mod.rs:Model::generate_coarse]. generate_text_semantic does use semantic_history. The EGM says history is caller-owned and does not mention the silent drop.

## crates/audio/src/tts/bark/README.md

Coverage: sampled (claims verified: REFERENCE_COMMIT, waveform gate 1e-5 and prefill gate 1e-4 against tests.rs, 10 tests, fixture dirs and tools/gen_bark_fixtures.py exist, pad-to-256 history; targeted reads)
- Says history arrays are "accepted as optional slices" in the scope section, implying they take effect. generate_coarse ignores both history slices (mod.rs:Model::generate_coarse, `let _ = (semantic_history, coarse_history);`). Only the text-semantic stage uses semantic history.

## crates/audio/src/tts/breeze_tts/EGM.md

Coverage: sampled (claims verified: GenerateParams::validate messages and bounds, use_cfg, combine_cfg width check, EOS id rule, KeySequence seeding, symbol list; source 2381 lines, targeted reads)
- CFG citation is off. Invariant "CFG runs only when scale is set and != 1.0 and an uncond prompt is given" is attributed to [mod.rs:GenerateParams::use_cfg], but use_cfg (mod.rs:568) checks only the scale. The uncond-prompt condition is applied in Model::generate (mod.rs:1739, `params.use_cfg() && uncond_ids.is_some()`).

## crates/audio/src/tts/breeze_tts/README.md

Coverage: sampled (claims verified: GenerateParams defaults 750/0.9/1.0/50/1.0, EMBED and LOGIT tolerances against tests.rs, fixture dirs and tools/gen_breeze_tts_fixtures.py exist, 14 tests; text-encoder and cfg gates not traced)
No drift detected.

## crates/audio/src/tts/chatterbox/EGM.md

Coverage: sampled (claims verified: symbol lists across mod.rs, t3.rs, flow.rs, hift.rs, campplus.rs, voice_encoder.rs, s3gen.rs; checkpoint and prefix invariants; speech token span and 6561 filter; trim_fade 480/480; UPSAMPLE_RATES, ISTFT_N_FFT, UPSCALE; TURBOSPARK_DEBUG_CONV at s3gen.rs:133; source 6289 lines, targeted reads)
- Flow prompt check is described as a row-major check, but flow.rs:Flow infer (line 1013) only tests `prompt_feat.len() % mel != 0` (divisibility). Minor wording mismatch.

## crates/audio/src/tts/chatterbox/README.md

Coverage: sampled (claims verified: tools/gen_chatterbox_fixtures.py and tools/dbg_chatterbox_traces.py exist, real_checkpoint_gate is #[ignore] and reads TURBOSPARK_CHATTERBOX_MODEL_DIR and TURBOSPARK_S3_MODEL_DIR, debug_flow_stages and debug_hift_stages exist, flow guidance rate 0.7, kaldi/voice-encoder constants; tolerance table only partly traced)
No drift detected.

## crates/audio/src/tts/chatterbox_turbo/EGM.md

Coverage: sampled (claims verified: symbol list, S3GEN_SIL 4299 and 6561 cut and 3-token tail, inference_turbo empty check and trailing stop pop, encoder_type check, n_timesteps default 2, prompt_mel reuse at mod.rs:380, no env/print/log use; flow.rs infer length rule not traced; source 5212 lines, targeted reads)
No drift detected.

## crates/audio/src/tts/chatterbox_turbo/README.md

Coverage: sampled (claims verified: enc_cond_len 15 s and dec_cond_len, vocab 50276, with_position_table_at_root, conformer depth 4 and 0.01 leaky slope, HiFT 0.1 upsample slope, CAMPPlus 1e-5 epsilon split, voice encoder 1e-8, pop of trailing stop, tools/gen_chatterbox_turbo_fixtures.py; tolerance table not traced)
No drift detected.

## crates/audio/src/tts/confucius4/EGM.md

Coverage: sampled (claims verified: Model method list, BOS/EOS 8192/8193, check_stats_len, temperature validation, 400-sample refusal, 26-point linspace and cfg 0.7 in s2a_solve, seed+1 S2A noise, safetensors copies replacing npz; source 4198 lines, targeted reads)
No drift detected.

## crates/audio/src/tts/confucius4/README.md

Coverage: sampled (claims verified: S2A depth 13, W2VBert/CampPlus/S2A types, safetensors checkpoint names, 22.05 kHz, f_min 20, testdata/tts_confucius4 and tools/gen_confucius4_fixtures.py exist, bigvgan path; gate values not traced)
- Test count is stale. README says `--lib confucius4` runs "6 tests", but the directory has 8 #[test] functions.
- Not a doc, but related: the mod.rs module doc says the npz archives are read "through the model_io npz reader" [mod.rs:module doc, lines 16-17]. Code reads same-key safetensors copies [mod.rs:Model::open_with_config]. README and EGM are correct; the code comment is stale.

## crates/audio/src/tts/dia/EGM.md

Coverage: sampled (claims verified: REFERENCE_COMMIT, EXTRA_STEPS_AFTER_EOS 30, KEEP_VOCAB 1025, MASK_DENIED -1e9, self_cache_capacity, dtype refusal, delay_pattern check, kv_heads check, first-shard weight load, RNG seed use, no env/print use; source 2801 lines, targeted reads)
- Out-of-range rule is incomplete and not what the code does. EGM says out-of-range embedding ids "yield a zero row". embed_lookup returns an empty slice for id >= rows and wraps negative ids from the end [model.rs:embed_lookup]. Negative wrap is not documented. Decoder call sites (model.rs:384, 400) add nothing for an empty row, which is zero-like, but the encoder call site does `copy_from_slice(row)` into a width embed_dim slice [model.rs:161], which panics on an empty row. The encoder table has src_vocab_size rows [model.rs:136], default 128 [config.rs:src_vocab_size], while text is raw bytes up to 255. Read from source, not executed.

## crates/audio/src/tts/dia/README.md

Coverage: sampled (claims verified: GenerateParams defaults 1.3/0.95/35, round_128 rule, delay pattern default [0, 8..15], codec/descript README link resolves, testdata/tts_dia and tools/gen_dia_fixtures.py exist, Greedy/Sampled structure; cache sizing and fixture numbers not traced)
- Same embedding-id issue as EGM: README says out-of-vocabulary ids "produce zero rows" and negative ids wrap. The encoder path panics on an out-of-range id instead of producing a zero row [model.rs:161, model.rs:embed_lookup].

## crates/audio/src/tts/dramabox/EGM.md

Coverage: sampled (claims verified: GenerateParams defaults 2.5/1.5/29/30, sigma-0 refusal in to_velocity, decode_waveform_refusal, first-shard load with exact shape check, split_freqs_cis max_pos check, DRAMABOX_MEASURE only in tests; source 2657 lines, targeted reads)
- Channel invariant is stated as an upfront check that does not exist. EGM says "audio_in_channels must equal vae_channels * mel_bins [mod.rs:Model::generate]". The only check is on the generated width at unpatchify: `width != channels * mel_bins` with channels = audio.vae_channels [mod.rs:Model::generate, lines 1015-1017]. audio_in_channels is never compared to that product.

## crates/audio/src/tts/dramabox/README.md

Coverage: sampled (claims verified: spec alias dramabox-tts appears in MODELS.md and fixture config, tools/gen_dramabox_fixtures.py and testdata/tts_dramabox exist, DRAMABOX_MEASURE in tests.rs, waveform refusal test exists, STG all-perturbed path in dit.rs; numeric gates and duration heuristics not traced)
No drift detected.

## crates/audio/src/tts/echo_tts/EGM.md

Coverage: sampled (claims verified: symbol list, blockwise refusal when delete_blockwise_modules is set [mod.rs:Model::generate_latents], EchoDiTConfig::valid divisibility checks, DEFAULT_TRUNCATION_FACTOR 0.96, concat_kv, MASK_VALUE -1e9, pca_filename existence check, model. prefix, no env/print use; source 3357 lines, targeted reads)
No drift detected.

## crates/audio/src/tts/echo_tts/README.md

Coverage: sampled (claims verified: config defaults (latent 80, model 2048, 24 layers, 16 heads, 1280 stacks, adaln 256, 40 steps, sequence length 640), t = 0.999 to 0 schedule, [S1] prefix rule, 2048-sample crop with pass-through, 640 * downsample speaker chunk default, codec/fish_s1_dac README and encode_zq/decode_zq exist, testdata fixture dirs and tools/gen_echo_tts_fixtures.py exist)
- Test count is stale. README says `--lib echo_tts` runs "9 tests", but the directory has 10 #[test] functions.

## crates/audio/src/tts/fish_drama3/EGM.md

Coverage: sampled (claims verified approx. 28 of 30; generate.rs and tower.rs read by targeted ranges, not in full)
- Omitted behavior: the residual tower heads are cut to the first `acoustic_codebook_size()` entries (1024 from `FishS1DacConfig::build_ae()`) before top-k and top-p, so the nucleus mass runs only over codes the codec can decode. The doc lists neither the method nor the cut among its Interface or Invariants. [crates/audio/src/tts/fish_drama3/config.rs:Drama3Config::acoustic_codebook_size] [crates/audio/src/tts/fish_drama3/generate.rs:Drama3::generate]

## crates/audio/src/tts/fish_drama3/README.md

Coverage: sampled (claims verified approx. 8 of 10; the 1024 acoustic width was not re-derived from build_ae, and timing and footprint figures were not re-measured)
No drift detected.

## crates/audio/src/tts/fish_qwen3_omni/EGM.md

Coverage: sampled (claims verified approx. 10 of 12; mod.rs read by targeted ranges)
No drift detected.

## crates/audio/src/tts/fish_qwen3_omni/README.md

Coverage: sampled (claims verified approx. 4 of 6; parity gates and fixture generation not re-run)
No drift detected.

## crates/audio/src/tts/higgs_audio/EGM.md

Coverage: sampled (claims verified approx. 12 of 13; mod.rs read by targeted ranges)
No drift detected.

## crates/audio/src/tts/higgs_audio/README.md

Coverage: sampled (claims verified approx. 4 of 8; parity figures and fixture contents not re-run)
No drift detected.

## crates/audio/src/tts/higgs_audio_v3/EGM.md

Coverage: sampled (claims verified approx. 12 of 13; mod.rs read by targeted ranges)
No drift detected.

## crates/audio/src/tts/higgs_audio_v3/README.md

Coverage: sampled (claims verified approx. 4 of 7; parity figures and fixture contents not re-run)
No drift detected.

## crates/audio/src/tts/indextts/EGM.md

Coverage: sampled (claims verified approx. 12 of 13; mod.rs and conformer.rs read by targeted ranges)
No drift detected.

## crates/audio/src/tts/indextts/README.md

Coverage: sampled (claims verified approx. 5 of 8; parity figures not re-run)
No drift detected.

## crates/audio/src/tts/irodori_tts/EGM.md

Coverage: sampled (claims verified approx. 10 of 12; mod.rs (about 136 KB) read by targeted ranges)
No drift detected.

## crates/audio/src/tts/irodori_tts/README.md

Coverage: sampled (claims verified approx. 4 of 8; mod.rs read by targeted ranges, parity figures not re-run)
No drift detected.

## crates/audio/src/tts/kitten_tts/EGM.md

Coverage: sampled (claims verified approx. 12 of 14; mod.rs (about 105 KB) read by targeted ranges)
No drift detected.

## crates/audio/src/tts/kitten_tts/README.md

Coverage: sampled (claims verified approx. 4 of 7; parity figures not re-run)
No drift detected.

## crates/audio/src/tts/kokoro/EGM.md

Coverage: sampled (claims verified approx. 14 of 15; mod.rs (about 82 KB) read by targeted ranges)
No drift detected.

## crates/audio/src/tts/kokoro/README.md

Coverage: sampled (claims verified approx. 4 of 9; the Metal split-K, tile and convolution constants were not re-derived, and the checkpoint tensor count was not checked)
No drift detected.

## crates/audio/src/tts/kokoro/frontend/EGM.md

Coverage: sampled (claims verified approx. 12 of 13; normalize.rs and mod.rs read by targeted ranges)
No drift detected.

## crates/audio/src/tts/kugelaudio/EGM.md

Coverage: sampled (claims verified approx. 12 of 14; mod.rs and scheduler.rs read by targeted ranges)
- Scope mismatch: the doc says the scheduler "accepts only cosine-family betas". `SDEDpmSolver::new` also accepts `scaled_linear`, which is not a cosine-family schedule. The prediction-type check (epsilon, v_prediction, sample) matches the doc. [crates/audio/src/tts/kugelaudio/scheduler.rs:SDEDpmSolver::new]

## crates/audio/src/tts/kugelaudio/README.md

Coverage: sampled (claims verified approx. 4 of 7; decoder-only tensor load and the speech-end loop path not traced end to end)
No drift detected.

## crates/audio/src/tts/llama/EGM.md

Coverage: sampled (claims verified approx. 14 of 15; mod.rs read by targeted ranges)
No drift detected.

## crates/audio/src/tts/llama/README.md

Coverage: sampled (claims verified approx. 5 of 7; parity figures not re-run)
No drift detected.

## crates/audio/src/tts/longcat_audiodit/EGM.md

Coverage: sampled (constants, defaults, invariants, env/logging scan, and all [file:symbol] cites resolve; source not fully read)
- Divergence: states "No env vars, threads, or logging found" under Side Effects, but the code reads `LC_DEBUG` and writes debug traces to stderr with eprintln when it is set [vae.rs:EncoderStage::forward] (env read at vae.rs lines 309, 372 and 715; eprintln traces gated by `if debug`).

## crates/audio/src/tts/longcat_audiodit/README.md

Coverage: sampled (constants, APG/CFG values, prompt padding, duration constants and fixture paths checked; source not fully read)
- Overstated: the reparameterization draw is documented as `normal * (softplus(scale) + 1e-4) / 0.71`, but the code divides by the config field `scale` (default 0.71) [mod.rs:VaeConfig] at vae.rs lines 659-672. The 0.71 is a config value, not a constant.

## crates/audio/src/tts/megatts3/EGM.md

Coverage: sampled (constants MAX_PROMPT_SECONDS, MAX_LATENT_FRAMES, MAX_TOKENS, FRAMES_MULTIPLE, layer bounds, sampler step bound, facade file names and env/logging scan checked; source not fully read)
No drift detected.

## crates/audio/src/tts/megatts3/README.md

Coverage: sampled (Quality presets 8/16/32, crossfade 0.16 s, 0.95 peak guard, 130 budget (byte-based), steps 1..100 in the clone CLI, model_type/voice_clone checks, and fixture/tool paths checked; source not fully read. Runtime, FFI, server, Swift and Metal wiring claims are outside this directory and were not checked)
No drift detected.

## crates/audio/src/tts/melotts/EGM.md

Coverage: sampled (SAMPLE_RATE, REFERENCE_BERT_DIM, LAYER_NORM_EPS, speaker resolution order, UNK and language fallback, sdp_ratio refusal, four coupling pairs and env/logging scan checked; source not fully read)
No drift detected.

## crates/audio/src/tts/melotts/README.md

Coverage: sampled (12-layer BERT, LayerNorm eps split (1e-12 embedding, 1e-5 layers), sdp_ratio refusal, fixture tool and testdata paths checked; source not fully read)
No drift detected.

## crates/audio/src/tts/moss_tts/EGM.md

Coverage: sampled (INT64_MAX sentinel, legacy and v1.5 refusals in open/load, sampler order (penalty, top-k, top-p), GPT-2 head divisibility, private sampling module and tokenizer repo constants checked; source not fully read)
No drift detected.

## crates/audio/src/tts/moss_tts/README.md

Coverage: sampled (variant aliases, legacy refusal, v1.5 load refusal, tokenizer repos, top-k before top-p, and tool paths checked; source not fully read)
No drift detected.

## crates/audio/src/tts/moss_tts_nano/EGM.md

Coverage: sampled (resolve_nq, StopReason, PromptMode, TemplateIds, non-positive temperature refusal, GPT-2 head check and env/logging scan checked; source not fully read)
No drift detected.

## crates/audio/src/tts/moss_tts_nano/README.md

Coverage: sampled (GenerateParams defaults 375 frames, 1.0/50/1.0 text, 0.8/25/0.95 audio, penalty 1.2, 48 kHz default, 1024 audio pad, tokenizer repo; source not fully read)
No drift detected.

## crates/audio/src/tts/omnivoice/EGM.md

Coverage: sampled (num_steps and target_len refusals, Rc backend, seed + step RNG keying, 960-sample frame rate, quantization format gating, pipeline visibility and env/logging scan checked; source not fully read)
- Visibility wrong: lists `time_steps` as pub(crate) [mod.rs:time_steps], but it is `pub fn` (mod.rs line 811).
- Doc-to-doc conflict: says the `duration` module is "private and not documented here", while omnivoice/duration/EGM.md says the production duration logic "is documented in ../EGM.md". One of the two is wrong, and neither describes the duration logic.

## crates/audio/src/tts/omnivoice/README.md

Coverage: sampled (module layout, 1025 vocab with mask 1024, 960-sample frame rate, 8-bit gating, 4-bit refusal, RNG and fixture paths checked; source not fully read)
- Broken link: `[codec::higgs_audio](../../../codec/higgs_audio)` resolves to crates/audio/src/codec/higgs_audio only if the path is `../../codec/higgs_audio` (from crates/audio/src/tts/omnivoice the three-level path lands in crates/audio/). The target directory exists; the link is wrong by one level.

## crates/audio/src/tts/omnivoice/duration/EGM.md

Coverage: full read (the directory holds only tests.rs, 84 lines, and this doc; the parent declaration `mod duration;` was confirmed)
- Points to ../EGM.md for the production duration logic, but ../EGM.md says the duration module is not documented there. Fix one or the other.

## crates/audio/src/tts/outetts/EGM.md

Coverage: sampled (DEFAULT_SAMPLE_RATE, CODEBOOK_TOKENS, -18 LUFS / -1 dB / 0.4 s block, 4096-code chunks, empty-prompt and logit-length refusals, speaker separator mutation and env/logging scan checked; source not fully read)
No drift detected.

## crates/audio/src/tts/outetts/README.md

Coverage: sampled (GenerateParams defaults 0.4/0.9/0.05/40/1.1/64-token context/1200, DAC repo and loudness block checked; source not fully read)
No drift detected.

## crates/audio/src/tts/pocket_tts/EGM.md

Coverage: sampled (DEFAULT_EOS_THRESHOLD -4.0, LSD steps, bos substitution for NaN seed, config reads limited to model.safetensors shards via lm::Weights::open, env/logging scan checked; source not fully read)
No drift detected.

## crates/audio/src/tts/pocket_tts/README.md

Coverage: sampled (frames_after_eos default 3+2 and prepare_text_prompt branch, EOS threshold, Mimi not read by Model::open, fixture tool path checked; source not fully read)
No drift detected.

## crates/audio/src/tts/qwen3/EGM.md

Coverage: sampled (TOKENIZER_LENGTH and all special-token offsets, codes_to_layers, 7-code groups, zeroprompt refusal and env/logging scan checked; source not fully read)
No drift detected.

## crates/audio/src/tts/qwen3/README.md

Coverage: sampled (special-token table ids 151669 to 151679, defaults 0.6/0.8/off/1.3/20-token context/1200, shared sampler path checked; source not fully read)
No drift detected.

## crates/audio/src/tts/rumik_oss/EGM.md

Coverage: sampled (quantizer resync and non-unit clear in push_audio_token, min_tokens gating, max_tokens 0, no-frame error, two stop-predictor tensor layouts, frames_to_codes/decode_frames and env/logging scan checked; source not fully read)
No drift detected.

## crates/audio/src/tts/rumik_oss/README.md

Coverage: sampled (Mimi 24 kHz / 12.5 Hz / 8 quantizers, stop-predictor head shape, min_tokens, no-frame error, speaker list and fixture tool path checked; source not fully read)
No drift detected.

## crates/audio/src/tts/sesame/EGM.md

Coverage: sampled (claims verified about 18 of 20; mod.rs 1112 lines, read targeted ranges, not fully read)
No drift detected.

## crates/audio/src/tts/sesame/README.md

Coverage: sampled (claims verified about 12 of 14; mod.rs read targeted ranges, not fully read; MODELS.md alias row confirmed)
No drift detected.

## crates/audio/src/tts/soprano/EGM.md

Coverage: sampled (claims verified about 16 of 18; mod.rs and decoder.rs read targeted ranges, text.rs clean_text read, not fully read)
No drift detected.

## crates/audio/src/tts/soprano/README.md

Coverage: sampled (claims verified about 12 of 14; decoder defaults and clean_text ASCII handling confirmed; not fully read)
No drift detected.

## crates/audio/src/tts/spark/EGM.md

Coverage: sampled (claims verified about 12 of 13; mod.rs 558 lines read in targeted ranges)
- Invariant "Token map tables must hold exactly 8192 semantic and 4096 global entries" is overstated. [mod.rs:AudioTokenMaps::from_ids] takes the counts as caller arguments (`semantic_codes`, `global_codes`) and only checks the slice lengths against them. The constants SEMANTIC_TOKENS / GLOBAL_TOKENS are not enforced there. The only in-crate caller is tests.rs, which passes `semantic.len()` and `global.len()`.

## crates/audio/src/tts/spark/README.md

Coverage: sampled (claims verified about 8 of 10; sampler defaults, generate loop and stop ids confirmed; wav2vec2 and BiCodec claims are cross-codec, not verified here)
No drift detected.

## crates/audio/src/tts/tada/EGM.md

Coverage: sampled (claims verified about 14 of 16; mod.rs 911 lines, gray_code.rs and diffusion_head.rs read in targeted ranges; mask-id bound not traced)
- Side effects says "Env vars, threads, logging: None". [tests.rs:TADA_MEASURE] reads `TADA_MEASURE` (test-only knob, tests.rs:50). Minor, test scope only.

## crates/audio/src/tts/tada/README.md

Coverage: sampled (claims verified about 12 of 15; grid endpoints, skip math, gray code bit order, CFG split and waveform refusal confirmed; real-checkpoint status is an open item, not checkable from source)
No drift detected.

## crates/audio/src/tts/vibevoice/EGM.md

Coverage: sampled (claims verified about 16 of 18; mod.rs 944 lines and scheduler.rs 393 lines read in targeted ranges)
- Invariant "Scheduler accepts only cosine-family betas and epsilon/v_prediction/sample" is wrong for the beta name. [scheduler.rs:DpmSolver::new] also accepts `scaled_linear`, and `DpmSolver::new` always builds betas with `betas_for_alpha_bar(num_train_timesteps, 0.999)` regardless of `beta_schedule`. A checkpoint that declares `scaled_linear` passes the gate and silently runs cosine betas. The doc describes a gate the code does not enforce, and the code itself has the bug.
- Side effects says "Env vars, threads, logging: None". [tests.rs:VIBEVOICE_MEASURE] reads `VIBEVOICE_MEASURE` (test-only, tests.rs:58). Minor, test scope only.

## crates/audio/src/tts/vibevoice/README.md

Coverage: sampled (claims verified about 12 of 14; window sizes, voice layer check in generate, EOS threshold 0.5 and speech scaling confirmed; vibevoice_streaming alias confirmed in crates/audio/MODELS.md)
No drift detected.

## crates/audio/src/tts/voxcpm/EGM.md

Coverage: sampled (claims verified about 18 of 20; mod.rs 688 lines, dit.rs, encoder.rs, audio_vae.rs, minicpm.rs read in targeted ranges)
- Side effects says "Env vars, threads, logging: None". [tests.rs:VOXCPM_MEASURE] reads `VOXCPM_MEASURE` (test-only, tests.rs:71 and 90). Minor, test scope only.

## crates/audio/src/tts/voxcpm/README.md

Coverage: sampled (claims verified about 14 of 16; rope per-head-block, MiniCPM cache layout, conv weight layouts, stop-head class handling and noise-block refusal confirmed; alias voxcpm1.5 confirmed in mod.rs)
No drift detected.

## crates/audio/src/tts/voxcpm2/EGM.md

Coverage: sampled (claims verified about 15 of 17; mod.rs 759 lines, audio_vae.rs and dit.rs read in targeted ranges)
- Side effects says "No env vars, threads, or logging found." [tests.rs:VOXCPM2_MEASURE] reads `VOXCPM2_MEASURE` (test-only, tests.rs:63 and 77). Minor, test scope only.

## crates/audio/src/tts/voxcpm2/README.md

Coverage: sampled (claims verified about 14 of 16; mean_mode switch, use_noise_block refusal, continuation trim, scipy resample and encode rate confirmed)
No drift detected.

## crates/audio/src/tts/voxtral_tts/EGM.md

Coverage: sampled (claims verified about 16 of 18; mod.rs 1709 lines read in targeted ranges; voice lookup, guard checks, end-token rule, +2 offset, CFG blend and tie-to-even rounding confirmed)
No drift detected.

## crates/audio/src/tts/voxtral_tts/README.md

Coverage: sampled (claims verified about 12 of 14; unknown-voice refusal, ignored sampler params, decoder stage validation and name remap confirmed; real-checkpoint gate is an open item, not checkable from source)
No drift detected.

## crates/audio/src/tts/zonos2/EGM.md

Coverage: sampled (claims verified about 16 of 18; mod.rs 1825 lines read in targeted ranges; sampler defaults, EOS countdown, byte ids, step seed, &self generate confirmed)
No drift detected.

## crates/audio/src/tts/zonos2/README.md

Coverage: sampled (claims verified about 13 of 15; mod.rs and tests.rs test names checked)
- Layout section names `Zonos2Config.from_dict`, which does not exist. The config entry point is `ModelConfig::from_json` [mod.rs:ModelConfig::from_json]. The same stale name appears in the doc comment on [mod.rs:ModelConfig] (line 353).

## crates/bench/EGM.md

Coverage: sampled (all 20 named claims located; lib.rs, main.rs, protocol.rs, memory.rs, startup_probe.rs, model_mode.rs seams, real_model.rs, Cargo.toml read in part; source not fully read)
- Invariant "Memory, real-model and startup-probe modules are gated on target_os = macos" omits tree_replay_record, which is also cfg-gated [src/lib.rs:tree_replay_record]
- Side Effects says the startup probe reads TURBOSPARK_METAL_* vars; it also reads TURBOSPARK_QWEN_SHARED_READ_OVERLAP and TURBOSPARK_MAPPED_DEMAND_PREP [src/startup_probe.rs:run]
- Omits the --model measurement guards and flags: --case, --expert-cache-slots, --prefill-chunk, --power-profile, --max-tokens-per-sec, --kv-bits. KV quantization and prefill chunking default OFF on purpose, so an unlabelled arm cannot be inherited [src/main.rs:main]
- Interface protocol-constant list omits PROTOCOL_EXPERT_CACHE_SLOTS and swift_reason_name, which src/EGM.md lists [src/protocol.rs:PROTOCOL_EXPERT_CACHE_SLOTS]

## crates/bench/AGENTS.md

Coverage: sampled (all named symbols and paths located; directory map, lib.rs contents, tests/ and prompts/ and testdata/frozen_e/ existence, repo-root scripts, memory and quality gate commands read; source not fully read)
- Says lib.rs `meeting` (cpWER) is shared with crates/lab (tslab). No file under crates/lab imports turbospark_bench::meeting; lab uses logit_compare, quality_corpus, tree_replay and the real-model modules only [src/lib.rs:meeting, crates/lab/src/worker.rs]

## crates/bench/README.md

Coverage: sampled (test file names, env vars, ignore markers, binary modes, CLI args, meeting_eval flags, linked docs checked; crate gotcha 3 (assistant-slot perplexity) not checked against the quality gates)
- "This crate contains 81 test files in tests/" is stale. tests/ holds 91 top-level .rs files (99 .rs including helpers in 8 helper directories). The enumerated list omits 10 files: lfm3_memory_oracle, lfm3_q8_memory_oracle, lfm3_quality_gate, lfm3_q8_quality_gate, lfm3_prefill_arm_probe, dflash_generation_gate, dflash_tree_pin, moe_route_capture, turn_snapshot_gate, turn_snapshot_ttft_probe [tests/]
- Key Modules says lib.rs `meeting/` is shared with turbospark-lab (tslab). No crates/lab file imports the meeting module [src/lib.rs:meeting]

## crates/bench/scripts/EGM.md

Coverage: sampled (main(), sha256, cpwer and output writes checked by targeted reads; scripts not read in full)
- prepare_frozen_e_clips.py also checks each source Mix-Headset WAV against MEETING_SHA256 before cutting and refuses on mismatch. The doc mentions only the output clip check [prepare_frozen_e_clips.py:main]

## crates/bench/src/EGM.md

Coverage: sampled (lib.rs, main.rs guard, meeting_eval_main.rs read; other named symbols located, not read in full)
- Invariant "--model mode refuses to run when TURBOSPARK_PREFILL_CHUNK is set" cites "the crate worker report", a session artifact that is not in the repo. The reference does not resolve [src/main.rs:main]
- Invariant gating list omits tree_replay_record, as in crates/bench/EGM.md [src/lib.rs:tree_replay_record]

## crates/bench/src/meeting/EGM.md

Coverage: sampled (mod.rs score_panel and read_frozen_counts, cpwer guard, seglst group_by_session read; tokens/edit internals and seglst parsers not read; all named symbols located)
- Invariant list omits the group_by_session guard: a hypothesis segment whose session_id is not a reference clip fails the whole scoring with FormatError, it is not skipped [src/meeting/seglst.rs:group_by_session]
- meeting_eval reads frozen_counts.json with unwrap_or_default(), so a missing or unreadable file silently blanks the frozen_errors column to "-" instead of failing. Side Effects does not mention this [src/meeting_eval_main.rs:score]
- Verification says cargo test checks Rust scorer parity. The test cpwer_reproduces_frozen_e_per_clip_counts compares against the committed frozen_counts.json, not against a live run of scripts/frozen_e_score.py, so "agree count for count with the Python scorer" is not enforced by any test [src/meeting/mod.rs:cpwer_reproduces_frozen_e_per_clip_counts]

Docs covered: 6 of 6. Directories not finished: none.

## crates/catalog/EGM.md

Coverage: sampled (claims verified 8 of 9; module links and Cargo.toml read, source not fully read)
- [crates/catalog/Cargo.toml:turbospark-coreml] Global Invariants says dependencies are "core, model-io, repack, tokenizer, and audio only". The manifest also has `turbospark-coreml` (crates/coreml) as a normal dependency, plus direct third-party crates reqwest, serde, serde_json, sha2. The invariant is false as written.

## crates/catalog/AGENTS.md

Coverage: sampled (claims verified 10 of 11; links, test commands, status variants, shard rules, directory-wins rule, and flint arms checked; source not fully read)
- [crates/repack/src/flint_config.rs:parse_flint_quantization] Says the flint INT8 embedding comes from `repack::parse_flint_quantization`. The code admits an embedding at 4 or 8 bits, so "INT8" overstates it.
- [crates/repack/src/flint_config.rs:parse_flint_quantization] Says the function "lives in `mlx_export`". It is defined in flint_config.rs and re-exported from crates/repack/src/lib.rs. It reads the `mlx_export` section of the config, so the location is misleading. Minor.

## crates/catalog/README.md

Coverage: sampled (claims verified about 25 of 30; crate deps, module list, CLI subcommands, probe exit code, chunk retention, sidecar pruning checked; the "~26s" runtime and the "tokenizer_config.json is optional" rule not verified; source not fully read)
- [crates/catalog/src/lib.rs:library] Key Modules omits the `library/` subsystem, a public module (`pub mod library`) with move, delete, import, and the libraries.json registry.
- [crates/catalog/src/check.rs:check_repo] Key Modules omits `check.rs`, which exports `check_repo`, `arch_hint`, and `RepoCheck`.
- [crates/catalog/src/stream.rs:write_flint_install_streamed] Says stream.rs holds the flint intake helper `write_flint_install_streamed`. The function is defined in crates/repack/src/gemma4_checkpoint/mod.rs:write_flint_install_streamed. stream.rs only calls it (line 438).
- [crates/catalog/tests/] Tests list omits check_repo.rs, hub_search_network.rs, library_install.rs, and store_relocation_libraries.rs.

## crates/catalog/examples/EGM.md

Coverage: sampled (claims verified 4 of 6; USAGE string, exit status and error prefix, SHA-256 use, and the verify command checked by grep; examples/audio_pin.rs not fully read)
No drift detected.

## crates/catalog/src/EGM.md

Coverage: sampled (claims verified about 45 of 50; interface names, embedded JSON list, SCHEMA_VERSION, INSTALL_CANCELLED string, Store root, hub validation with no HTTP, token order, and directory-wins rule checked; source (about 55k lines incl. data) not fully read)
- [crates/catalog/src/store.rs:default_root] Says the store root is `$TURBOSPARK_HOME`, else `$HOME/.turbospark`, and lists only those two env vars as inputs. The function first checks a process-global `DEFAULT_ROOT_OVERRIDE` (set by `relocate_default_store`), which wins over both. This is an omitted side effect and root source.

## crates/catalog/src/library/EGM.md

Coverage: sampled (claims verified about 25 of 28; all listed interface names, registry and marker methods, HOME_ONLY_FOLDERS, MAX_MARKER_BYTES, tombstone in .staging, no-openat race note checked; move and delete bodies not fully read)
No drift detected.

## crates/catalog/src/probe/EGM.md

Coverage: sampled (claims verified 12 of 13; probe signature, four gates in cheapest-first order, F32/F16/BF16 transcode note, SLOT_COUNTS, KNOWN_SIDECARS count checked; gguf.rs and safetensors.rs bodies not fully read)
No drift detected.

## crates/catalog/src/recommend/EGM.md

Coverage: sampled (claims verified about 15 of 17; interface names, TIGHT_FRACTION, LADDER_WINDOWS, QUANT_LADDER, evidence-first and working-set module docs checked; the "no network in recommend_catalog" side effect and fit/rank bodies not fully read)
No drift detected.

# Drift report: crates/cli (and its src subdirectories)

Method: each doc read in full. Symbol, env var, file and invariant claims checked by grep and targeted reads of the cited source. The directories hold about 15k lines of Rust, so the source was not read in full. Where a claim was not checked it is not reported as verified.

## crates/cli/EGM.md

Coverage: sampled (binary list, Cargo.toml [[bin]] targets, ui and front-door module-doc citations and the test command checked; about 15k lines of source not fully read)
No drift detected.

## crates/cli/src/EGM.md

Coverage: sampled (daemon.rs and agent.rs symbols, API_KEY_ENV, require_root, acquire_daemon_lock, write_private_file and create_private_run_dir checked by grep and targeted reads; log file privacy not checked; about 15k lines of source not fully read)
No drift detected.

## crates/cli/src/bin/EGM.md

Coverage: sampled (#[path] includes, exit-code mapping in steer.rs, model.rs delegation and steer_data/ checked; per-modality binaries not read)
- Omission: [src/bin/turbospark.rs:server] is also included by `#[path = "../front/server.rs"]`, but the Interface list names only agent.rs, daemon.rs, ui/mod.rs and front/models.rs.

## crates/cli/src/bin/model_cmd/EGM.md

Coverage: sampled (listed mod.rs, audio.rs, auth.rs, drama3.rs, library.rs, transfer.rs, views.rs, render.rs, progress.rs, destination.rs and image_source.rs symbols, DF, KINDS, INTERRUPTED, the SIGINT handler, and the image_source overwrite refusal all checked; not every function body read)
- Overstated invariant: "Alias validity is checked before network work [mod.rs:check_alias]". check_alias runs in the --repo branch of [mod.rs:pull_source] (mod.rs:867) and in [mod.rs:pull_image] and the vision pull path (mod.rs:514, mod.rs:794). The plain positional-alias path in [mod.rs:pull] (mod.rs:419) resolves the destination before the network but does not call check_alias. Either the claim needs scoping or the alias check is done somewhere this pass did not find.

## crates/cli/src/bin/steer_cmd/EGM.md

Coverage: sampled (ExtractOptions, CheckOptions, run, TraitPack::parse file-name rule, pack-count minimum, write_atomically .partial rename, CAPTURE_ENV, MAX_CAPTURE_CONTEXT, resolve_model family refusal and read_capture/read_positions checked; not every function read)
No drift detected.

## crates/cli/src/front/EGM.md

Coverage: sampled (models.rs resolve and looks_like_path, check_command setting TURBOSPARK_HIDE_RESOLVED_REQUEST, check_args, RunPlan, confirm, and server.rs snapshot and its foreground-serve note checked; pull and rm bodies not read)
No drift detected.

## crates/cli/src/generate/EGM.md

Coverage: sampled (Outcome semantics, Session fields, reject_image_install, resolve_chunk_tokens, attach_* family list and env var reads checked; stream_turn and vision preprocessing bodies not read)
- Invariant not borne out: "Images are decoded and validated before the model opens [vision.rs module doc]". In [generate/mod.rs:run_messages_file] the session is opened (mod.rs:284) before [generate/mod.rs:run_one_turn] calls vision::prepare_images (mod.rs:390). The vision.rs module doc gives the order inside the vision pipeline only and does not say it runs before the model opens. Either the invariant is stated too strongly or an earlier decode exists that this pass did not find.
- Omitted env vars read by this module: TURBOSPARK_DEBUG_PROMPT_IDS [generate/mod.rs:run_one_turn] (prints prompt ids to stderr) and TURBOSPARK_EXPERT_DISK_IO [generate/format.rs:print_phases] (changes the expert-bytes footer). The Side Effects line lists only TURBOSPARK_PHASES, TURBOSPARK_DISPATCH_PROFILE and TURBOSPARK_PREFILL_CHUNK.

## crates/cli/src/ui/EGM.md

Coverage: sampled (color_enabled env handling, Paint, ok/note/warn/error, columns, clip, Table, Cell, Align, section and the ASCII-only claim checked; table internals not read)
- Mark list is stale or unimplemented: the EGM Invariants line and the module doc in [ui/mod.rs:module doc] say marks are `+`, `>`, `!` and `*`. The helpers in ui/mod.rs emit only `+` ([ui/mod.rs:ok]), `>` ([ui/mod.rs:note]) and `!` ([ui/mod.rs:warn]); no `*` is emitted anywhere in the file.

## crates/cli/AGENTS.md

Coverage: sampled (--chat image refusal, run/check_args routing through the chat template, invocation::OPTIONS use, view verbs ls/search/show/doctor/home, src/ui having no unsafe code, referenced doc paths checked; the "archived guide" checklist and the --help/--version rule not checked)
No drift detected.

## crates/cli/README.md

Coverage: sampled (binary list, exit-status mapping, --chat image refusal, view verbs, relative doc links and scripts/make-app-bundle.sh existence checked; the flag checklist and model-tag rules not checked)
- Incomplete description: "turbospark-ocr: Document OCR via Core ML (macOS 15+)". [src/bin/ocr.rs:module doc] says the command runs Unlimited-OCR-2 on Metal or the Core ML export (macOS 15+), and the model directory picks which path runs.

## crates/compute/EGM.md

Coverage: full read
No drift detected.

## crates/compute/README.md

Coverage: sampled (symbol presence checked for all named public items, behavior checked for about 10 claims; source not fully read)
- rms_norm.rs entry says "optional learned weight scaling". Code requires the weight: [crates/compute/src/rms_norm.rs:rms_norm] asserts `x.len() == weight.len()` and has no weightless form. Overstated.
- Key Modules omits `short_conv.rs` (LFM3 gated short convolution, PORT-LOCAL, `short_conv_reference`): [crates/compute/src/short_conv.rs:short_conv_reference]
- Key Modules omits `steering_extract.rs` (`difference_of_means`, `CaptureSet`, per-layer steering diagnostics): [crates/compute/src/steering_extract.rs:difference_of_means]
- Key Modules for rms_norm.rs omits the centered variants `rms_norm_centered` (x * (1 + w)) and `rms_norm_grouped_centered`, which the README does not mention: [crates/compute/src/rms_norm.rs:rms_norm_centered]
- Tests list omits `tests/dash_oracle.rs` (which `include!`s `generated/dash_oracle.rs`) and `tests/gemma4_vision_reference.rs`. The `tests/generated/` bullet says the fixtures are included by "the quantization tests above", which is incomplete: [crates/compute/tests/dash_oracle.rs]
- steering.rs bullet lists `Ablate`, `Add`, `Clamp`, `Renorm` as if defined in this crate. They are variants of `SteeringMode`, which comes from turbospark-core (`foundation`) and is only imported here: [crates/compute/src/steering.rs:steer_in_place]

## crates/compute/AGENTS.md

Coverage: sampled (manifest and code checked for every rule; links resolved; source not fully read)
- Rule "FP16 uses the `half` crate" is false. [crates/compute/Cargo.toml] has no `half` dependency. FP16 is hand-rolled in [crates/compute/src/quant_1bit.rs:f16_to_f32] and [crates/compute/src/quant_1bit.rs:f32_to_f16] (re-exported to `quant_2bit`), while [crates/compute/src/gdn.rs:h] rounds through `foundation::LogitValue` from turbospark-core, and the comment above it says "never hand-roll binary16". So the code has two FP16 paths, and the rule "Do not introduce a second representation" describes neither. AGENTS.md is human-written, so the user has to decide whether the rule or the code is stale.

## crates/compute/src/EGM.md

Coverage: sampled (all ~60 named symbols checked for presence; ~10 invariants and constants checked for behavior; source not fully read, src is about 8.6k lines excluding the 4.4k-line generated table file)
- Interface omits public API exported from lib.rs, for example `cls_pool_and_normalize`, `cosine_similarity` [crates/compute/src/lib.rs:cosine_similarity], `rms_norm_centered` [crates/compute/src/rms_norm.rs:rms_norm_centered], and `sigmoid_head_gate_mul` [crates/compute/src/gating.rs:sigmoid_head_gate_mul]. Most of the list is a selection, so this is minor.
- "IQ codebook tables are generated from scripts/ggml_tables.c" uses a path that exists only at the repo root (`scripts/ggml_tables.c`), not under crates/compute. The doc does not say the path is repo-relative. [crates/compute/src/quant_gguf_iq_tables.rs:IQ4NL_VALUES] header says the same thing.
- Adjacent code comments, not this doc: the crate doc in [crates/compute/src/lib.rs] lists `quant_gguf` as "Q4_K, Q6_K, and Q8_0" only, although the module has nine formats, and the `/// Mixture-of-Experts` doc comment sits above `pub mod mla;` while `moe` is declared a few lines later.

## crates/compute/src/quant_gguf/EGM.md

Coverage: sampled (all format re-exports checked in mod.rs, which was read in full; asserts checked in q4_0, q5_k, q6_k and pearson; the codec bodies were not read individually)
No drift detected.

## crates/core/README.md

Coverage: full read (all crate source and tests/ read; the ALLOWED_SPECULATION_BLOCKS consumer claim was checked only for crates/ffi, not the turbospark-check or turbospark-server parsers)
- Omits the engagement module and the TURBOSPARK_ENGAGEMENT env var, and does not mention the public `engaged!` macro. [crates/core/src/engagement.rs:engaged]
- "Enforces ascending order and valid defaults at compile time via const assertions" is overstated. The const assertions cover only ALLOWED_CACHE_SLOTS and ALLOWED_CHUNK_SIZES (non-empty, sorted, defaults in set). ALLOWED_SPECULATION_BLOCKS has no assertion, and DEFAULT_ADAPTIVE_CAP / MIN_ADAPTIVE_CAP are not checked against any set. [crates/core/src/runtime_config.rs:ALLOWED_SPECULATION_BLOCKS] [crates/core/src/runtime_config.rs:DEFAULT_ADAPTIVE_CAP]
- Safety section says "Pure, deterministic logic", but engagement.rs:enabled reads the TURBOSPARK_ENGAGEMENT environment variable once and caches it in a OnceLock. No filesystem or network access, so only the word "pure" is off. [crates/core/src/engagement.rs:enabled]

## crates/core/EGM.md

Coverage: full read
- Subdirectory note for src lists "Primitives, runtime_config, prefill, steering, humanizer" and omits the chunk_sizing and engagement modules, both declared in lib.rs. [crates/core/src/lib.rs:chunk_sizing] [crates/core/src/lib.rs:engagement]
- Link ./src/EGM.md resolves.

## crates/core/AGENTS.md

Coverage: full read (links verified to resolve; the `cargo check --target x86_64-unknown-linux-gnu` command was not executed)
No drift detected.

## crates/core/src/EGM.md

Coverage: full read
- Interface lists only engagement.rs:enabled. Omits the public `engaged!` macro (#[macro_export], the call sites use `foundation::engaged!`, e.g. crates/gpu/src/attention_decode.rs) and the public `line` helper. [crates/core/src/engagement.rs:engaged] [crates/core/src/engagement.rs:line]
- Interface omits humanizer public items: HUMANIZER_CONTEXT, default_max_new (output budget, 256..=2048, saturating), REUSE_FLAG_THRESHOLD (0.22), and RewriteAudit::warnings (the stderr report lines). [crates/core/src/humanizer.rs:default_max_new] [crates/core/src/humanizer.rs:RewriteAudit] [crates/core/src/humanizer.rs:REUSE_FLAG_THRESHOLD] [crates/core/src/humanizer.rs:HUMANIZER_CONTEXT]
- Interface omits the public const fn `max_of`, which MAX_CHUNK_TOKENS derives from. [crates/core/src/runtime_config.rs:max_of]
- Invariant "Allowed-value sets are non-empty and defaults stay inside them, enforced at compile time by const assertions" is overstated. Only ALLOWED_CACHE_SLOTS and ALLOWED_CHUNK_SIZES (and their defaults) are asserted. ALLOWED_SPECULATION_BLOCKS and the adaptive caps have no compile-time check. [crates/core/src/runtime_config.rs:ALLOWED_SPECULATION_BLOCKS] [crates/core/src/runtime_config.rs:MIN_ADAPTIVE_CAP]
- Invariant "A prefill chunk that errors or panics mid-chunk must not be reused until reset()" is overstated as an enforced guarantee. PrefillChunkCommitState does not detect errors or panics. It refuses reuse only if the caller has called mark_dirty() before the chunk. [crates/core/src/prefill.rs:PrefillChunkCommitState] [crates/core/src/prefill.rs:mark_dirty]
- Omits prefill_chunk_spans behavior: chunk_tokens is clamped to [1, MAX_CHUNK_TOKENS], zero tokens yields an empty plan, and start_position + offset overflow panics. [crates/core/src/prefill.rs:prefill_chunk_spans]
- Omits that SteeringMode derives Default and the default is Ablate. [crates/core/src/steering.rs:SteeringMode]

## crates/coreml/EGM.md

Coverage: sampled (the three subdirectory links resolve; the listed subdirectory set was compared with the directory listing; the Overview claims were checked against Cargo.toml and src).
- Subdirectory list omits `tests/` (Core ML round-trip tests, file-level `#![cfg(target_os = "macos")]`) and `testdata/` (tiny fixture packages). [crates/coreml/tests/fixtures.rs:fixture]

## crates/coreml/AGENTS.md

Coverage: sampled (about 25 rule claims checked by targeted reads and grep across lib.rs, cache.rs, tensor.rs, stub.rs, error.rs and sys.rs; sys.rs was read in part, see the src section for the unread ranges).
- The `ane-prefill` bullet says generated programs are "validated against `manifest.json`". Nothing in crates/coreml/src reads a manifest. The only manifest check described in the repo is runtime-side: the generator docstring says "The runtime refuses a manifest whose shapes do not match the install" [crates/coreml/tools/gen_ane_prefill_programs.py:__doc__], which points at crates/runtime/src/families/qwen/ane_split.rs (not read here). Name the validating crate, or drop the claim from this file.
- Omitted rule: cold compiles and their publication are serialized by a process-wide `static` Mutex, because Core ML may hand the same temporary directory to simultaneous compilers of one package. Prediction stays per-model. [crates/coreml/src/sys.rs:serialized_cold_compile]
- Tests bullet implies `tests/fixtures.rs` runs the packages under `cargo test -p turbospark-coreml`. The file is compiled only on macOS (`#![cfg(target_os = "macos")]`), so off macOS only the unit tests run. [crates/coreml/tests/fixtures.rs:fixture]
- Floor claim: "Any API newer than macOS 13 (the Swift package's floor)". swift/TurboSpark/Package.swift declares `.macOS(.v13)`, but swift/TurboSparkApp/Package.swift declares `.macOS(.v14)`. This review did not confirm which package links this crate. Name the package.
- Not a doc divergence, noticed in passing: the comment in crates/coreml/Cargo.toml says macOS 15 APIs are gated "by a respondsToSelector check in `sys::availability`". No such module exists. The gate is the `available!` macro inside [crates/coreml/src/sys.rs:supports].

## crates/coreml/examples/EGM.md

Coverage: full read (both example files read in full).
- Minor omission: `coreml_describe` treats an empty or `-` function argument as no function. [crates/coreml/examples/coreml_describe.rs:main]
- Checked and matching: CPU-only load for describe, `--units all` default for plan, flag set (`--units`, `--function`, `--cache`, `--ops`), default `.compiled` cache directory, and the first-ANE-compile side effect. Verification command (`cargo build -p turbospark-coreml --examples`) was not run.

## crates/coreml/src/EGM.md

Coverage: sampled (lib.rs, cache.rs, tensor.rs, error.rs and stub.rs read in full; sys.rs read at lines 1-110, 405-630 and 800-1000, plus grep for pub items, cfg gates, unsafe markers and autoreleasepool calls. Not read: sys.rs 110-405, 630-800, 1000-1130 and the ane-prefill block 1130-1553, which was grep-checked only).
- Exported names, error variants, `cache_key` contents, strided bounds checks, the `ane-prefill` surface (`SharedF16Plane` with new, base_ptr, alloc_size, rows, cols, stride, and `predict_shared`) match the code. [crates/coreml/src/sys.rs:SharedF16Plane], [crates/coreml/src/sys.rs:CoreMlModel]
- Omitted invariant: cold compile and publish are serialized by a process-wide static Mutex, with a second cache check inside the lock. [crates/coreml/src/sys.rs:serialized_cold_compile]
- Omitted contract: `CoreMlState::copy_from` requires every buffer to match in name, shape, strides and dtype. The source state may come from a different model of the same package (for example prefill and decode functions). [crates/coreml/src/sys.rs:CoreMlState]
- Omitted behavior: on an OS without MLState (before macOS 15), `CoreMlModel::load` returns an empty `states()` list with no error. Callers of `new_state` get `OsTooOld`. [crates/coreml/src/sys.rs:CoreMlModel]
- Public surface not listed: `CoreMlModel::path` and `function`; `Tensor::shape`, `dtype`, `data`, `into_data`, `as_f32`, `as_f16`, `as_i32`, `to_f32_vec`; `LoadOptions::with_function`; `ComputeUnits::as_str` and `parse`. The Tensor entry lists constructors only. [crates/coreml/src/tensor.rs:Tensor], [crates/coreml/src/lib.rs:LoadOptions]
- Overstated in-code claim (not repeated in this EGM, but the stub is documented here): the lib.rs module doc says every non-macOS entry point returns `UnsupportedPlatform`. The stub returns `false` from `supports`, `(0, 0, 0)` from `os_version`, empty slices from `inputs`, `outputs` and `states`, and `""` from `path`. [crates/coreml/src/stub.rs:supports], [crates/coreml/src/stub.rs:CoreMlModel]

## crates/coreml/tools/EGM.md

Coverage: sampled (argparse flags of all five scripts checked; gen_ane_prefill_programs.py `build` read; bake_ane_prefill_programs.py `Install`, `fp16_mapping` and `baked_program` read in part; gen_ane_sweep_programs.py docstring and output checked; gen_coreml_fixtures.py docstring and manifest write checked; bake_ane_vision_programs.py checked by argparse only).
- Verification command fails as written. `python3 -I crates/coreml/tools/gen_ane_prefill_programs.py --help` with system python3 raises `ModuleNotFoundError: No module named 'coremltools'`, because `import coremltools` runs at module top, before argparse. Either document the `uv run --with coremltools==9.0` form for `--help`, or move the import below argument parsing. [crates/coreml/tools/gen_ane_prefill_programs.py:module]
- Interface omits most of the bake modes. `bake_ane_prefill_programs.py` has `--tensors` (sweep.json mode) versus `--layers` and `--classes` (manifest.json class mode), `--lut` (4-bit indices into a 16-entry fp16 table, about half the ANE memory), `--shared-f16`, `--with-f16`, `--share`, and `--rows`. [crates/coreml/tools/bake_ane_prefill_programs.py:main]
- Omitted refusal rule: a tensor is refused unless the fp16 scale and offset reproduce the BF16 values within 1e-5 absolute or 0.05 of the group scale. Scales below 6.1e-5 are subnormal and can fail the second test. [crates/coreml/tools/bake_ane_prefill_programs.py:fp16_mapping]
- Omitted constraint: `gen_ane_sweep_programs.py` requires shape widths in multiples of 32 halfs and rounds MA down to a multiple of 32 rows. [crates/coreml/tools/gen_ane_sweep_programs.py:__doc__]
- Checked and matching: K sliced at `CHUNK = 2048` with FLOAT16 compute precision [crates/coreml/tools/gen_ane_prefill_programs.py:build]; INT4 group size `GROUP = 64` [crates/coreml/tools/bake_ane_prefill_programs.py:GROUP]; `sweep.json` written [crates/coreml/tools/gen_ane_sweep_programs.py:main]; `manifest.json` with per-fixture `sha256_tree` digests [crates/coreml/tools/gen_coreml_fixtures.py:main]; fixture set in testdata/ (add_f32, f16_scale, i32_argmax, multifunction, accumulator).

## crates/embeddings/AGENTS.md

Coverage: sampled (claims verified about 12 of 15: kernel imports from turbospark_audio ops/nn, TURBOSPARK_*_DIR env var names, ~/models/embeddings fallback, pins file, linked docs and paths; test and cargo commands not run)
No drift detected.

## crates/embeddings/MODELS.md

Coverage: full read for the pin tables (all 24 unique pinned revisions and their repos diffed against src/pins.rs; the only revision in MODELS.md not in pins.rs is CLAP upstream main 8fa0f1c6..., which the doc says is deliberately not used); sampled for the verification numbers (not re-measured)
No drift detected.

## crates/embeddings/EGM.md

Coverage: full read (Cargo.toml read fully; lib.rs read fully)
No drift detected.

## crates/embeddings/src/EGM.md

Coverage: sampled (claims verified about 30 of 32: re-exports in lib.rs, Weights/EmbeddingTable methods, pooling and normalize constants, similarity and st_config items, pins.rs functions, EmbeddingError variants and helpers, affine-only quantization gate; source files under ~400 lines read in part, not every file read)
No drift detected.

## crates/embeddings/src/bert/EGM.md

Coverage: sampled (claims verified about 15 of 16: BertEmbedder and Encoder API, Layout/PoolingSource/known_conversion, BertConfig/Arch/xlmr_position_ids, VERIFIED_QUANT_BITS (4, 6, 8) and group 64, refusal of unimplemented config values; mod.rs and encoder.rs not read in full)
No drift detected.

## crates/embeddings/src/bert/README.md

Coverage: sampled (claims verified about 25 of 30: public API list, XLM-R position formula, refusal list (model_type, position_embedding_type, is_decoder, add_cross_attention, hidden_act, prefixed tensor names, do_lower_case, include_prompt, per-layer quant overrides), Table::Mapped word table, pooling policy, testdata file list; measured timings and parity numbers not re-run)
No drift detected.

## crates/embeddings/src/clap/EGM.md

Coverage: sampled (claims verified about 20 of 21: Clap, ClapModel, feature and encoder types, SAMPLE_RATE and PIPELINE_TEMPLATE, LongAudioCrop default Head, validate() on ClapConfig and FeatureConfig, f64 spectrum with complex64 rounding, file reads)
No drift detected.

## crates/embeddings/src/clap/README.md

Coverage: sampled (claims verified about 20 of 24: API signatures of Clap and ClapModel, LongAudioCrop variants, 512-token text limit (config default), ReLU-only projection and top_db refusals, f64 frontend and MEL_FLOOR, is_longer field, Weights::linear refusal of quantized layers without a scheme, named test functions and the ignored real-checkpoint tests; the parity and timing numbers were not re-run)
No drift detected.

## crates/embeddings/src/embeddinggemma/EGM.md

Coverage: sampled (claims verified about 14 of 15: EmbeddingGemma, Backbone, Layout, GemmaConfig/LayerType, VERIFIED_QUANT_BITS (4, 8) and group 64, GemmaTokenizer truncation that keeps specials, file reads at load; config.rs and mod.rs read in large part)
No drift detected.

## crates/embeddings/src/embeddinggemma/README.md

Coverage: sampled (claims verified about 20 of 24: sliding band as sliding_window / 2 + 1 (257), layout detection by tensor name, MLX fallback to mean pooling when 1_Pooling is absent, Dense count checked against modules.json, load-time refusals (use_bidirectional_attention, softcapping, rope_scaling, activation, attention_bias, include_prompt, do_lower_case), Matryoshka renormalize; measured numbers not re-run)
No drift detected.

## crates/embeddings/src/gte/EGM.md

Coverage: sampled (claims verified about 8 of 8: Gte and Encoder API, inv_freq and rope_tables, rope tables built per forward (encoder.rs:243), BadConfig on malformed keys, GteConfig)
No drift detected.

## crates/embeddings/src/gte/README.md

Coverage: sampled (claims verified about 18 of 20: public API, optional tokenizer.json, embed_ids 8192 limit, mapped F32 word table, 8192 max_seq_length rule, refusals (position_embedding_type, rope_scaling, hidden_act, layer_norm_type, pack_qkv, logn flags, type_vocab_size 0), unpad/memory-efficient flags accepted; remote-code numerics not re-run)
No drift detected.

## crates/embeddings/src/nomic_bert/EGM.md

Coverage: sampled (claims verified about 9 of 10: Nomic API, Encoder items, ntk_base formula, MLP_MULTIPLE_OF, NomicBertConfig)
- [crates/embeddings/src/nomic_bert/config.rs:padded_vocab_size] Invariant says "MLP hidden size and padded vocab are rounded to MLP_MULTIPLE_OF". Only the MLP hidden size uses MLP_MULTIPLE_OF (config.rs:mlp_hidden, 256). Padded vocab is rounded to the checkpoint field pad_vocab_size_multiple (default 1; 64 in the pinned checkpoint), per padded_vocab_size(). The sibling README states this correctly.

## crates/embeddings/src/nomic_bert/README.md

Coverage: sampled (claims verified about 20 of 22: NTK base formula and its n > 2048 boundary (encoder.rs:ntk_base), base 1000 default, n_inner rounding to 256, padded vocab to pad_vocab_size_multiple, swiglu-only and refusal list (rotary fraction, interleaved, scale base, prenorm, bias flags, causal, parallel_block, use_rms_norm), rotary_scaling_factor absent = 1.0 and null/0 disable NTK, 8192 n_positions limit, word table copied to heap; numeric parity not re-run)
No drift detected.

## crates/embeddings/src/siglip2/EGM.md

Coverage: sampled (claims verified about 12 of 13: Siglip2 API incl. logit_params, probabilities, sigmoid; VisionTower/TextTower/NaflexInput/processing items; HIDDEN_ACT, TEXT_MAX_LENGTH, FAST_PROCESSOR; refusals of non-gelu_pytorch_tanh activation)
- [crates/embeddings/src/siglip2/mod.rs:Siglip2::tokenize] Invariant says "loading fails if the tower holds fewer positions". The check (TEXT_MAX_LENGTH > text.max_position_embeddings) runs inside tokenize(), and Siglip2::load does not reference TEXT_MAX_LENGTH. The error is raised on the first tokenize or embed_text call, not at load.

## crates/embeddings/src/siglip2/README.md

Coverage: sampled (claims verified about 16 of 18: Siglip2 API (embed_image, embed_image_file, embed_text, logit, probabilities, vision/text forward_trace), Fast processor refusal and resample=2 check, vision_use_head, RGB-only and quantization refusals, rescale/normalize must-be-true checks, target-size eps 1e-5, encode_padded (content to len-1, eos, pad), embed_text passes text unchanged, logit_scale and logit_bias loaded from weights; parity numbers and the Fast-vs-PIL measurement not re-run)
No drift detected.

## crates/embeddings/tools/EGM.md

Coverage: sampled (claims verified 6 of 6 generator files for their argparse and docstring claims; module-level imports checked by reading the heads of all six scripts; fixtures not regenerated)
- [crates/embeddings/tools/gen_gte_fixtures.py:module docstring] [crates/embeddings/tools/gen_nomic_bert_fixtures.py:module docstring] Invariant says "Reference is Hugging Face transformers 4.57.1 in f32 on CPU, no trust_remote_code", citing only gen_bert_fixtures.py. The gte and nomic_bert generators load their pinned local remote code with trust_remote_code=True (with HF_HUB_OFFLINE forced). The statement is true for the bert generator only, not for the directory.
- [crates/embeddings/tools/gen_bert_fixtures.py:argparse] Interface lists "[--tiny, --real TAG MODEL_DIR]" only. The script also takes "--mlx TAG MLX_DIR" and "--model-dir" (for --mlx without --real l_v2). gen_embeddinggemma_fixtures.py also takes --mlx, --qat and --model-dir, which the entry omits.
- [crates/embeddings/tools/gen_bert_fixtures.py:module imports] Verification says "Run: python3 crates/embeddings/tools/gen_bert_fixtures.py --help". The script imports torch at module level (line 69), so this command fails on a Python without torch. Observed here with the default python3: ModuleNotFoundError: No module named 'torch'. The regeneration commands in the same doc (and the sibling READMEs) use the pinned venv (emb-venv/bin/python -I), which is the command that works.

## crates/ffi/EGM.md

Coverage: full read (doc is 8 lines; child links resolve, package name matches Cargo.toml).
No drift detected.

## crates/ffi/AGENTS.md

Coverage: sampled (claims verified about 9 of 10; Cargo.toml, the referenced docs and the Make targets checked by existence and grep, not fully read).
- [crates/ffi/src/api/models.rs:ts_chip_supports_ane_prefill] The rule says the hardware detection symbol is mirrored across turbospark.h, wire.rs, open.rs and testing.rs. The symbol is defined in api/models.rs and does not appear in wire.rs. Only the wire options (`anePrefill`, `expertSlotsWiring`) live in wire.rs. Wording only; the option mirrors themselves check out in all four files.

## crates/ffi/README.md

Coverage: sampled (claims verified about 25 of 26; Makefile targets and test files checked by name, tests not run).
- [crates/ffi/src/embedding_probe.rs] Not in the "Key Modules" list. The module is compiled (lib.rs) and is the source of `embedding_probe::probe`.
- [crates/ffi/tests] The "Tests" section names 7 files. Omitted: daemon_pid_identity.rs, gemma4_vision_real.rs, libraries_surface.rs, and the library_* suites (deletes, installs, lifecycle, moves, reports).

## crates/ffi/include/EGM.md

Coverage: sampled (claims verified 14 of 14 listed; turbospark.h is 2281 lines and was checked by targeted reads and grep, not fully read).
No drift detected.

## crates/ffi/src/EGM.md

Coverage: sampled (claims verified about 35 of 36 named items and invariants; source not fully read).
- [crates/ffi/src/session.rs:Session] Doc says `Session` is a "cloneable handle over `Arc<SessionCore>`". `Session` (session.rs:192) has no `Clone` derive or impl. It is a move-only wrapper around an Arc.
- [crates/ffi/src/server_idle.rs:IdlePolicy] Side effects say the "Idle sweep unloads models every 30 s with a 30 min default TTL." The sweep does run every 30 s (server_idle.rs:100), but the default policy has `enabled: false` (server_idle.rs:29-37). By default it releases only on-demand (request-loaded) models after `DEFAULT_IDLE_TTL` (server_idle.rs:72-80). Host-attached models stay loaded until the host opts in with a nonzero value (`IdlePolicy::from_seconds`, server_idle.rs:17-25).
- [crates/ffi/src/api:ts_server_set_idle_unload] This is the public switch that changes the idle behavior above. The doc does not name it.

## crates/ffi/src/api/EGM.md

Coverage: sampled (all named entry points located in the cited files; invariants checked by grep and targeted reads; source not fully read).
- [crates/ffi/src/api/models.rs:ts_hf_token_set] Side effects cover "disk" only in general terms. They omit that `ts_hf_token_set` saves the Hugging Face token to `~/.turbospark/hf_token` as a plaintext file with mode 0600 (catalog/src/store.rs:set_hf_token). AGENTS.md says Keychain use is opt-in per credential, so this deserves a deliberate decision and a doc line.
- Omitted public entry points (c): ts_server_set_idle_unload, ts_server_attach_image_model, ts_server_attach_embedding_model, ts_store_relocate, ts_store_import_moved_root, ts_library_remove, ts_library_set_default, ts_model_delete, ts_model_forget, ts_install_pause, ts_install_resume, ts_install_cancel, ts_heavy_work_try_acquire, ts_heavy_work_release, ts_hf_token_get, ts_hf_token_info_json, ts_hf_token_clear, ts_hf_token_validate_json, ts_session_release_vision, ts_audio_job_* and ts_ocr_add_page_* families.

## crates/ffi/src/generate/EGM.md

Coverage: sampled (claims verified 19 of 20; generate/ is about 1,500 lines and was not read in full).
- [crates/ffi/src/generate/prompt.rs:decode_messages] [prompt.rs:role_of] [prompt.rs:historical_call] are labeled "crate-private", but they are private `fn` items (prompt.rs:13, 32, 79). Only `ToolOffer` (prompt.rs:98) is `pub(crate)`. Wording only.

## crates/ffi/src/models/EGM.md

Coverage: sampled (claims verified about 20 of 21 named items; install.rs, hub.rs and catalog.rs checked by targeted grep and reads, not fully read).
- Interface and Invariants omit the install pause/resume and active-move cancel surface (c): `pause_active_installs`, `resume_active_installs`, `cancel_active_move` (models/mod.rs re-exports from install.rs) and the matching `ts_install_pause` / `ts_install_resume` entry points in api/.

## crates/gpu/EGM.md

Coverage: full read of doc; build.rs sampled (9 of 9 claims located by targeted reads)
No drift detected.

## crates/gpu/AGENTS.md

Coverage: full read of doc; code sampled (about 30 claims located, GQA/MMA/GDN/MoE constants and tests checked; CI history, perf and bit-exactness rationale not re-run)
- Kernel names do not exist: doc names `dequant_int4_gemv_fused_gate_up_act` and `dequant_int4_gemv_fused_down_epilogue`; the shaders define `dequant_int4_gate_up_act_gemv_simd` and `dequant_int4_down_epi_gemv_simd` [src/shaders/dequant_int4_fused.metal:dequant_int4_gate_up_act_gemv_simd] [src/shaders/dequant_int4_fused.metal:dequant_int4_down_epi_gemv_simd]. The Rust wrappers are `encode_gate_up_act_gemv` and `encode_down_epi_gemv` [src/dequant_int4_fused.rs].
- Type does not exist: doc cites `ane_split.rs` (`GpuAneSplit`) as the bridge type. No `GpuAneSplit` exists in the crate or repo. The file exports free functions (`encode_ane_dequant_rows`, `encode_ane_scale_rows`, `encode_ane_convert_out`) and `HostEvent` [src/ane_split.rs:HostEvent].

## crates/gpu/README.md

Coverage: full read of doc; code sampled (module list, every named test file, counts, env vars and cfg gating checked)
- Stale count: doc says "97 integration test suites in tests/", but tests/ holds 112 .rs files [tests/].
- Stale count: doc says "50 .metal files", but src/shaders/ holds 51 [src/shaders/].
- Name not found: "Unlimited-OCR-2" does not appear in the vision code [src/vision.rs:encode_vision_*]. Vision.rs names the Gemma 4 tower and the DeepSeek-OCR SAM partition. Confirm the name or where it is wired.
- Omitted public modules from Key Modules: `mla.rs` (encode_mla_attention_decode), `prefill_scratch.rs`, `warmup.rs` (KernelWarmupPlan, opt-in JIT warmup), `dispatch_profile.rs`, `device_memory.rs`, `power_state.rs`, `precompiled.rs`, `bytes.rs` [src/lib.rs].
- Omitted env vars that select kernels or pipelines: `TURBOSPARK_ATTN_DECODE_KERNEL=barrier|gqa` [src/attention_decode.rs] and `TURBOSPARK_GDN_DK128=0` [src/gdn.rs:BAKED_KEY_DIM]. Doc covers only INT4_MMA8 and the vision switches.

## crates/gpu/examples/EGM.md

Coverage: full read of doc; both example sources checked by targeted reads (modes, MLX version, generator script)
No drift detected.

## crates/gpu/src/EGM.md

Coverage: sampled (about 60 named symbols located by name, 5 invariants spot-checked, source not fully read)
- Wrong symbol: `moe_decode.rs:RoutedBlobBuffer` does not exist. The type is `RoutedBlobsBuffer` [src/moe_decode.rs:RoutedBlobsBuffer].
- Wrong location for the opt-in: Invariants say "Metal JIT warmup is opt-in [warmup.rs:KernelWarmupPlan]". warmup.rs reads no env var. The gate is `TURBOSPARK_METAL_KERNEL_WARMUP=1` in the runtime [crates/runtime/src/real_forward_open.rs]. Cite the runtime or say the gate lives outside this crate.
- Omitted env vars in Side Effects, which change kernel selection: `TURBOSPARK_ATTN_DECODE_KERNEL` [src/attention_decode.rs], `TURBOSPARK_GDN_DK128` [src/gdn.rs], `TURBOSPARK_VISION_MMA` and `TURBOSPARK_VISION_ATTN_MMA` [src/vision.rs]. Side Effects lists only TURBOSPARK_DISPATCH_PROFILE.

## crates/gpu/src/context/EGM.md

Coverage: sampled (every named function, struct and env var located; device.rs and pass.rs read by targeted greps, not in full)
- Wrong citation: `TURBOSPARK_DISPATCH_PROFILE` is attributed to [src/context/pass.rs]. pass.rs mentions it only in a doc comment. The env read is [src/dispatch_profile.rs:enabled].
- Omitted public API from MetalContext: `begin_pass_labeled`, `begin_pass_concurrent_labeled`, `new_buffer_with_data`, `new_output_buffer` [src/context/device.rs:MetalContext].
- Omitted public API from CommittedPass: `waiting_on_drop`, `wait`, `wait_ref`, `wait_with_gpu_time`, `wait_gpu_start_end` [src/context/pass.rs:CommittedPass].
- Omitted opt-in and security caveat for pipeline archives: archives are used only when `TURBOSPARK_METAL_PIPELINE_CACHE=1`, need macOS 15 or later, and load without content verification, so `TURBOSPARK_METAL_CACHE_DIR` must not be writable by other users [src/context/pipeline_cache.rs:supports_specialized_archives]. The doc says only "may write/read pipeline archives on disk".

## crates/gpu/src/moe_gguf/EGM.md

Coverage: sampled (every named encode_* function, const and type located; kernel bodies not read)
No drift detected.

## crates/gpu/src/music3/EGM.md

Coverage: sampled (types, Music3Buffer methods and constructors located; bit-identity claim not tested)
- Omitted public surface: `Music3Sequence` exposes the operation methods (`upload`, `bind`, `copy`, `rope`, `view`, `linear`, `convolution`, `snake`, `rms_norm`, `layer_norm`, `attention`, `add_round_bf16`, `swiglu_bf16`, `partial_rope_bf16`, `gated_silu_bf16`, `add_f32`, `partial_rope_f32`, `gated_silu_f32`) [src/music3/sequence.rs:Music3Sequence]. Interface lists only the types and constructors.

## crates/gpu/src/shaders/EGM.md

Coverage: sampled (kernel and function-constant names located, file list checked against the 51 .metal files on disk; kernel bodies not read)
- Incomplete function-constant list: the doc lists the attention constants as FC_ATTN_HEAD_DIM, FC_ATTN_NUM_Q_HEADS, FC_ATTN_NUM_KV_HEADS, FC_ATTN_USE_FC, FC_ATTN_SCALE. attention.metal also declares FC_ATTN_NUM_CHUNKS (65), FC_ATTN_RING_CAP (69), FC_ATTN_HAS_SINKS (70), FC_ATTN_GQA_SIMDGROUPS (72) and FC_ATTN_TREE (77) [src/shaders/attention.metal:FC_ATTN_NUM_CHUNKS]. The Invariants section says every function constant must be in the Rust cache key, so the full list matters.

## crates/image/EGM.md

Coverage: sampled (claims checked by grep and targeted reads; crate source of about 13.6k lines not fully read)
- Doc says "All code lives under src/; the crate root has no direct source files." Code also has Rust test sources outside src/ (eight files in crates/image/tests/, e.g. [crates/image/tests/metal_parity.rs]). Doc omits tests/.
- "Requests use a fixed 9-step, batch 1, guidance 0.0 envelope [crates/image/src/runtime.rs:ImageRequest::validate]" is true for validate(), but the crate defines three more per-family envelopes (Qwen-Image-2.1 40 steps, guidance 1.0; Qwen-Image-2.1-Turbo and Krea 2 Turbo 8 steps, guidance 1.0) in [crates/image/src/install.rs:ImageFamily::supported], which ImageManifest::validate admits. The request path rejects those step counts, so generate() cannot run those families. The doc does not say so.

## crates/image/AGENTS.md

Coverage: full read (short file; link targets checked, check commands not executed)
No drift detected.

## crates/image/README.md

Coverage: sampled (tests list, env var, Cargo targets, constants, and packed format checked; the claim that the vendored Swift ZImage and QwenImage packages back the app was not verified)
- Top paragraph lists Z-Image-Turbo, Qwen-Image-2.1, Qwen-Image-2.1-Turbo, and Krea 2 Turbo as families this pipeline handles. In this crate the generation path ([crates/image/src/runtime.rs:ImageRequest::validate]) admits only the 9-step, guidance 0.0 Z-Image envelope, so the other three families cannot generate through [crates/image/src/runtime.rs:generate]. Install-level support exists in [crates/image/src/install.rs:ImageFamily::supported]. The README overstates runtime support for those three.
- Crate Gotchas and Key Modules omit the per-family envelope mechanism ([crates/image/src/install.rs:ImageFamily::supported], [crates/image/src/install.rs:image_family_for_model_id], which falls back to the Z-Image envelope for unknown ids).

## crates/image/src/EGM.md

Coverage: sampled (about 60 symbols and invariants checked by grep and targeted reads; src/ is about 13k lines and was not fully read)
- Install: [crates/image/src/install.rs:image_family_for_model_id] is listed without its behavior. Unknown model ids fall back to the Z-Image envelope, and the three other families ([crates/image/src/install.rs:ImageFamily]) are not named anywhere in this doc.
- Interface omits public items re-exported from [crates/image/src/lib.rs]: ImageMetadata, image_quantization_label, ImageStage, ImageBudgetError, the IMAGE_* constants, the Metal trace types ImageFirstStepBoundary, ImageFirstStepTrace, ImageIntraBlockTrace ([crates/image/src/metal.rs]), encode_tokens_with_cancel ([crates/image/src/text_encoder.rs]), and the helpers build_unified_sequence, create_coordinate_grid, pad_with_ids ([crates/image/src/patchify.rs]), timestep_embedding ([crates/image/src/pipeline.rs]), conv2d, group_norm, upsample_nearest_2x, decoded_to_rgb8 ([crates/image/src/vae.rs]).
- Invariants: [crates/image/src/install.rs:ImageFamily::supported] sets per-family envelopes that ImageManifest::validate enforces. The doc's single fixed-envelope invariant does not cover them.
- Side effects, builder: [crates/image/src/builder.rs:build_image_install_with_progress] also writes the install receipt (verified-install.json, [crates/image/src/builder_files.rs:write_receipt]) and verifies files via verify_files_for_publication before the rename. The doc lists manifest.json only.
- Side effects, GPU anchor: the anchor [crates/image/src/metal_ops.rs:autorelease_pool] does not resolve. The file has no such symbol. The only use is the call gpu::autorelease_pool at [crates/image/src/metal_ops.rs:459], which wraps pass.commit().

## crates/image/src/shaders/EGM.md

Coverage: full read (crates/image/src/shaders/image.metal, 842 lines, read in full; kernel list and attributes checked against all 20 kernels)
- Kernel list is complete and every name matches a kernel in [crates/image/src/shaders/image.metal].
- Invariant "every kernel is declared with [[kernel, max_total_threads_per_threadgroup(256)]] or a similar attribute": [crates/image/src/shaders/image.metal:image_attention] is declared with 512, not 256.
- Invariant "kernels take element counts as constant uint & and guard index < count": only five kernels take a count argument ([image.metal:image_round_bf16], [image.metal:image_add], [image.metal:image_silu_mul], [image.metal:image_scheduler_step], [image.metal:image_silu]). The others take a params struct (LinearParams, NormParams, RopeParams, and so on) and guard on rows, dims, or totals.
- Missing dispatch contract: several kernels cooperate across the threadgroup and need an exact threadgroup size, not just a cap. [image.metal:image_rms_norm], [image.metal:image_rope_adjacent] (partials[256], stride loops of 256), [image.metal:image_group_norm] (partial[8], 256-thread loop step) and [image.metal:image_vae_attention] (8 SIMD groups) require 256 threads. [image.metal:image_attention] requires 4 * head_dim threads (query_slot = lane / head_dim) and has key_shared[128] and partials[4][4], so it only supports head_dim <= 128 with head_dim a multiple of 32.
- Missing limit: [image.metal:image_vae_attention] writes numerator[16] per lane across 32 lanes, so it covers at most 512 channels. Channels above 512 are silently dropped, not rejected.
- Missing storage contract: the kernels decode storage codes 1 (F32), 2 (BF16), 3 (local INT4), 4 (MLX affine) and companion codes 2 (BF16) and 4 (F16) ([image.metal:LinearParams], [image.metal:affine_mlx_value]). The doc does not describe these codes.

## crates/invocation/EGM.md

Coverage: full read (doc is short; link `./src/EGM.md` resolves; overview matches the crate).
No drift detected.

## crates/invocation/README.md

Coverage: full read of src (all 7 files, about 1,611 lines), Cargo.toml, and the tests by fn name plus the option-count block; tests otherwise sampled.
- Overstated: "Downstream workspace crates import this package" and "peer tools" imply several consumers. Only crates/cli depends on turbospark-invocation [crates/cli/Cargo.toml:dependencies]. No other workspace crate or peer tool does.
- Omission: the Five-Place gotcha says the compiler does not catch a flag in OPTIONS with no match arm, but does not mention the test that catches it at test time [crates/invocation/tests/usage_and_status.rs:every_declared_option_reaches_a_handled_parser_arm].
- Omission: failure diagnostics are the Rust Debug text of the ParseFailure value followed by usage, not a formatted message [crates/invocation/src/diagnostics.rs:describe]. The README does not say so, so a reader would not expect the Debug form on the diagnostic stream.
- Verified and matching: no filesystem, env, or process I/O in src; no unsafe code and no forbid attribute in lib.rs; the 5 integration test files exist; `cargo test -p turbospark-invocation` is the test command; turbospark-check is a bin in crates/cli.

## crates/invocation/AGENTS.md

Coverage: full read; all four links resolve (.claude/docs/modules/invocation.md, docs/CLI.md, docs/TOOL_CALLING.md, docs/TESTING.md); the server does not import this crate, matching "restates these rules rather than importing them" [crates/cli/Cargo.toml:dependencies].
No drift detected.

## crates/invocation/src/EGM.md

Coverage: full read (all 7 src files, about 1,611 lines; parser, request, options, diagnostics, failure, usage, lib). Tests sampled by fn name only.
- Overstated invariant: "Defaults are re-exported from foundation (the turbospark-core crate) rather than restated" is false for most defaults. Only DEFAULT_MAX_CONTEXT, DEFAULT_ADAPTIVE_CAP, MIN_ADAPTIVE_CAP, ALLOWED_SPECULATION_BLOCKS and SteeringMode come from foundation [crates/invocation/src/request.rs:DEFAULT_MAX_CONTEXT]. DEFAULT_MAX_NEW (1024), DEFAULT_TEMPERATURE (0.2), DEFAULT_TOP_K (64), MAX_TOP_K (256), DEFAULT_TOP_P (0.95) and DEFAULT_REPETITION_PENALTY (1.0) are literals in this crate [crates/invocation/src/request.rs:DEFAULT_MAX_NEW]. The Interface line in the same doc lists these as this crate's defaults, so the invariant contradicts the Interface line.
- Misattributed interface item: SteeringMode is listed as defined in request.rs, but request.rs only re-exports it (`pub use foundation::SteeringMode;`) [crates/invocation/src/request.rs:SteeringMode]. Same for ALLOWED_SPECULATION_BLOCKS, which the doc does not list.
- Omitted behavior: stream_routing puts the failure's Debug text plus usage on the diagnostic stream [crates/invocation/src/diagnostics.rs:stream_routing]. The doc lists stream_routing but not this text form.
- Omitted invariants: the whole-invocation checks run in a fixed order (model, the mutual-exclusion pairs, humanize, system-outside-chat, then steering orphans). The steering rules are refused rather than ignored: a steering knob with no --steering, more knob values than paths, and an inverted layer range [crates/invocation/src/parser.rs:parse]. The doc says only "one fixed order", so these refusals are not recorded anywhere in the crate docs.
- Verified and matching: ExitStatus codes 0 and 2 [diagnostics.rs:ExitStatus]; Help and Version are separate outcomes both routed to stdout with exit 0; ParseOutcome has four variants; the value-taking option consumes the next token; the six ParseFailure variants; render_version and render_usage exist; no side effects or unsafe code in src.

## crates/lab/EGM.md

Coverage: sampled (claims verified about 26 of 26 symbols, cfg gates, and commands; source not fully read)
No drift detected.

## crates/lab/README.md

Coverage: sampled (claims verified about 36 of 40, source not fully read)
- Verdict rule omits the sign-test requirement. WIN or LOSS also needs every clean pair to agree when there are fewer than 6 pairs, and a one-sided exact sign test at p <= 0.05 from 6 pairs up [crates/lab/src/stats.rs:decide] [crates/lab/src/stats.rs:UNANIMITY_BELOW] [crates/lab/src/stats.rs:SIGN_TEST_ALPHA]. The doc gives only the median floor and the bootstrap CI. program.md ("Reading a verdict") states the sign-test rule correctly, so this README is the outlier.
- Exit code "2 usage" is overstated. Only an empty argv exits 2 [crates/lab/src/cli.rs:main]. Subcommand flag and usage errors return Err and exit 1 [crates/lab/src/cli.rs:finish].

## crates/lab/AGENTS.md

Coverage: sampled (claims verified about 45 of 50, source not fully read)
- Imprecise feature gate for ane_matmul. The doc says "kernel/ane_matmul.rs (feature ane-prefill)", but the module is compiled unconditionally [crates/lab/src/kernel/mod.rs:registry] and only specific items are cfg-gated [crates/lab/src/kernel/ane_matmul.rs:MAX_PLANE_COPIES] [crates/lab/src/kernel/ane_matmul.rs:ANE_PLACEMENT_MIN_SHARE]. Without the feature the module still registers and takes a non-feature branch.
- All other paths, test files, symbols (DASH_JS, run_sweep, VisionTiming, encode_vision_matmul, ENERGY_MODEL) and the dashboard guards resolve.

## crates/lab/program.md

Coverage: sampled (claims verified about 33 of 35, source not fully read; ANE and chunk flags not traced)
No drift detected.

## crates/lab/families/README.md

Coverage: sampled (field rules checked against family_profile.rs constants and validators, about 14 of 15)
No drift detected.

## crates/lab/pinned/README.md

Coverage: sampled (claims verified about 12 of 12, source not fully read)
No drift detected.

## crates/lab/skill/EGM.md

Coverage: sampled (claims verified 3 of 3, source not fully read)
No drift detected.

## crates/lab/skill/SKILL.md

Coverage: sampled (claims verified about 22 of 25, source not fully read)
- Exit code line "0 ok or keep, 1 error, 2 usage, 3 discard, 4 inconclusive" overstates 2, for the same reason as the README: only an empty argv exits 2 [crates/lab/src/cli.rs:main]; other usage errors exit 1 [crates/lab/src/cli.rs:finish].
- Otherwise the flags and defaults checked (ab metric default prefill_tok_s, metric Debug line, workflow --dry-run, report --html, --audio-job, pinned, family --findings, serve --open) match the code.

## crates/lab/skill/scripts/EGM.md

Coverage: sampled (claims verified about 6 of 8, source not fully read)
- Threshold claim is only partly true. "Thresholds copy the guard's defaults in crates/lab/src/host.rs" holds for the 10 GiB free-disk check [crates/lab/src/host.rs:default_min_free_gb]. preflight.sh has no CPU-load check, but the guard rejects on sustained load above 150% total or 90% single-process with a 5 s window [crates/lab/src/host.rs:HostRules] (defaults around line 365). So "quiet enough that a run would not be ledgered dirty" overstates what preflight covers.
- The busy-process rule cites a list that does not exist. preflight.sh's comment names "host.rs MODEL_PROCESS_NAMES"; the guard's list is MODEL_MARKERS [crates/lab/src/host.rs:MODEL_MARKERS], and it matches by name fragment, including test binaries (_probe-, _gate-, _oracle-, _bench-). preflight.sh (busy) matches exact first-word basenames, so test binaries such as quality_gate-<hash> are not flagged. The doc citation [skill/scripts/preflight.sh:fail] does not cover this.
- "None beyond reading process and power state" is inaccurate wording. preflight.sh also reads free disk (df), swap (sysctl), and the lock holder (lsof). All read-only, so only the wording is off.

## crates/lab/src/EGM.md

Coverage: sampled (claims verified about 45 of 50, source not fully read)
- Invariants omit the existing-test rule. An experiment may not change an existing test file (any status other than A under crates/*/tests/) [crates/lab/src/integrity.rs:is_test_file] [crates/lab/src/integrity.rs:violations]. program.md states this rule; the EGM does not.
- Exit-code citation is wrong and "2 usage" is overstated. The run-experiment and confirm codes come from [crates/lab/src/experiment.rs:exit_code_for] (EXIT_DISCARD, EXIT_INCONCLUSIVE), not [crates/lab/src/cli.rs:USAGE]. Exit 2 is returned only for an empty argv [crates/lab/src/cli.rs:main].
- Worker environment is incomplete. Each worker also receives TURBOSPARK_ENGAGEMENT=1 [crates/lab/src/orchestrate.rs:ORCHESTRATOR_ENV] and the TURBOSPARK_HOME and TURBOSPARK_LAB_DIR passthrough [crates/lab/src/orchestrate.rs:PASSTHROUGH_TURBOSPARK]. The doc lists only PASSTHROUGH plus the arm's variables.

## crates/lab/src/bin/EGM.md

Coverage: sampled (claims verified 4 of 4, source not fully read)
No drift detected.

## crates/lab/src/dash/EGM.md

Coverage: sampled (claims verified about 12 of 12, source not fully read)
No drift detected. Routes, serve flags and defaults, GET-only registration, loopback Host check, nosniff and CSP headers, and the (mtime, len) index refresh were all checked.

## crates/lab/src/kernel/EGM.md

Coverage: sampled (claims verified about 11 of 12, source not fully read)
- Stale claim that benches load no weights. The Overview says "Benches run in the orchestrator process (no worker, no model weights)", and the module doc at [crates/lab/src/kernel/mod.rs:kernel_cmd] repeats that. The vision_encode bench opens the full runner with weights [crates/lab/src/kernel/vision_encode.rs:RealForwardRunner::open_with_max_context]. crates/lab/AGENTS.md already calls it "the one bench that loads weights", so the EGM and the module doc need the same exception.
- The registry (13 benches), bench names, MachineLock, kernel_curve ledger record and the feature gating of ane_matmul and ane_handoff all match.

## crates/lab/tools/EGM.md

Coverage: sampled (claims verified 5 of 5, source not fully read)
No drift detected.

## crates/lab/workloads/audio/README.md

Coverage: sampled (claims verified about 5 of 6, source not fully read)
No drift detected. Not traced: the claim that each job runs "once warm and once measured", and that the lab does not resample WAV input.

## crates/model-io/EGM.md

Coverage: sampled (claims verified about 6 of 6, source not fully read; crate root has no .rs files of its own)
- [Cargo.toml:dependencies] omits `serde_json`, which the crate depends on. `turbospark-core` is imported under the alias `foundation`, not by its package name in code.

## crates/model-io/AGENTS.md

Coverage: sampled (claims verified about 6 of 7, source not fully read)
- Unsafe rule lists only the `mmap` calls and the `statvfs` call. A second unsafe call exists: `libc::sysconf(libc::_SC_PAGESIZE)` in the page-size helper of [crates/model-io/src/resident_buffer.rs:sysconf] (line ~109, `#[allow(unsafe_code)]`). The src EGM already lists `sysconf`, so the rule and the child doc disagree.
- Read-first links to `../../docs/GTURBO.md`, `../../docs/MODEL_FAMILY.md`, `../../docs/FLINT.md` and `../../.claude/docs/modules/model-io.md` resolve.

## crates/model-io/README.md

Coverage: sampled (claims verified about 40 of 45, source not fully read)
- `llama.rs` is described as "Llama 3, Mixtral, and related Dense/MoE variants". The file defines one constructor, `mixtral_8x7b` [crates/model-io/src/arch_baselines/llama.rs:mixtral_8x7b]. No Llama 3 baseline exists in the crate.
- Flint router: README says "top-6 INT8 sigmoid router". The source says `p = sigmoid(h @ router^T)  # BF16 router, no bias` and sets `router_scoring_func: "sigmoid"`; the file has no INT8 router [crates/model-io/src/arch_baselines/flint.rs:flint_40b_a2_5b]. Expert weight dtype may differ, but the router claim is not supported by this file.
- `gpt_oss.rs` is described as "(Harmony chat template) architecture". `gpt_oss.rs` has no "harmony", chat, or template reference [crates/model-io/src/arch_baselines/gpt_oss.rs:gpt_oss_20b]. Not found in this crate.
- Safety section and Key Modules list only `mmap` and `statvfs` as unsafe. `sysconf` is also called in [crates/model-io/src/resident_buffer.rs:sysconf] (same omission as AGENTS.md).
- Key Modules list omits `cgroup.rs` (`probe_self`, `parse_limit`) and `bounded_read.rs` (`read_capped`), both public or crate-level pieces that the src EGM documents.
- `deepseek2.rs` is described without the `deepseek_vl2_tiny` constructor, which `all_known_architectures` registers [crates/model-io/src/arch_baselines/deepseek2.rs:deepseek_vl2_tiny].
- Tests section omits `tests/metadata_read_bounds.rs` and `tests/metadata_size_caps.rs` [crates/model-io/tests/].
- Crate Gotcha 1 (`newBufferWithBytesNoCopy` pins mapped pages and charges phys_footprint) is not implemented or stated in this crate; it appears only in comments in crates/bench and crates/catalog. Not verified here.

## crates/model-io/src/EGM.md

Coverage: sampled (claims verified about 40 of 42, source not fully read; src holds about 12k lines of .rs)
- Invariants line says unsafe is limited to `mmap` plus the single `statvfs` call, but the Overview and Side Effects sections name `sysconf` too, and [crates/model-io/src/resident_buffer.rs:sysconf] calls it. The two lines disagree.
- "Reads cgroup memory files on Linux [cgroup.rs:probe_self]": `probe_self` has no `target_os` gate. It reads `/sys/fs/cgroup/memory.max` and `memory.high` on every platform and returns None when they are absent. "On Linux" describes the typical case, not a gate.
- Public API omitted: `validate_install_receipt_manifest_binding` is re-exported from the crate root [crates/model-io/src/install_receipt.rs:validate_manifest_binding] (exported as `validate_install_receipt_manifest_binding` in lib.rs). Not listed under Integrity.
- Interface names `load_manifest`, `validate_manifest`, `load_resident_index`, `load_install_receipt`, `load_vision_sidecar` and `load_packed_experts_layout` are crate-root aliases of `manifest::load`, `manifest::validate`, `resident_index::load`, `install_receipt::load`, `vision_sidecar::load` and `packed_experts_layout::load` [crates/model-io/src/lib.rs:pub use]. They resolve, but the doc's citations point to lib.rs, not to the defining files. Low priority.
- All other named symbols, constants (`CONTEXT_*`, `HEADROOM_*`, `INDEX_MAGIC`, `HEADER_BYTES`, `ENTRY_BYTES`, `TRAILER_BYTES`, `INDEX_FORMAT_VERSION`), error variants (`UnsupportedIndexFormat`, `ExpertStrideNotPageAligned`, `PartialInstall`, `IndexCorrupt`), and file-name claims (`speech-receipt.json`, `unlimited_ocr2_dash.json`, `trunk_role.json`) match the source.

## crates/model-io/src/arch_baselines/EGM.md

Coverage: sampled (claims verified 25 of 25 named constructors and both invariants, source not fully read; directory is about 2000 lines)
- Overview says "one constructor per supported model family". `gemma4_12b_dense` is a public constructor [crates/model-io/src/arch_baselines/gemma.rs:gemma4_12b_dense] with no `ModelFamily` variant, so `known_architecture` and `all_known_architectures` never return it. It is used only by a repack test. The doc lists it without saying it sits outside the registry.
- Otherwise no drift detected: `known_architecture` is an exhaustive match over all 18 `ModelFamily` variants, `all_known_architectures` lists the 18 plus `deepseek_vl2_tiny`, and the constructor names in the list exist.

## crates/model-io/src/arch_config/EGM.md

Coverage: sampled (claims verified about 20 of 20 named items, source not fully read; directory is about 1440 lines)
No drift detected. Verified: `ModelFamily::as_str` and `parse` exist and have 18 variants each mapped, `ArchConfig` predicates (`layer_is_full`, `layer_is_linear`, `layer_is_csa`, `layer_is_mla`, `layer_is_hca`, `layer_is_compressed`, `layer_is_short_conv`, `layer_is_hash_routed`, `has_*_layers`, `decode_int4_gemv_shapes`, `decode_int8_gemv_shapes`) exist, and `MAX_VISION_DEEPSTACK_MERGERS` is 3 [crates/model-io/src/arch_config/sub_configs.rs:MAX_VISION_DEEPSTACK_MERGERS]. The predicates' use of the layer mask was not traced line by line.

## crates/model-io/src/manifest/EGM.md

Coverage: sampled (claims verified about 15 of 15 named items, source not fully read; directory is about 1120 lines)
- Interface lists `validate_quant` as a public item. It is `pub(crate)` [crates/model-io/src/manifest/quant.rs:validate_quant]; it is not reachable from outside the crate.
- Omitted invariant: manifest validation also rejects a non-page-aligned expert stride, returning `ModelError::ExpertStrideNotPageAligned` [crates/model-io/src/manifest/mod.rs:validate]. The doc lists no such check here.
- Otherwise verified: `load(dir, expecting, max_bytes)`, `validate`, `peek_family`, `known_flags` (three flags), `MANIFEST_VERSION_MAJOR` 1, `MANIFEST_VERSION_MINOR` 1, `REQUIRED_FILES` (`model_weights.bin`, `packed_experts/layout.json`), `DEFAULT_MAX_BYTES` 4 MiB, a missing file returns `PartialInstall`, an oversize file returns `IndexCorrupt`, and `EXECUTABLE_GGUF_TYPES` has 17 entries.

## crates/repack/EGM.md

Coverage: full read (short overview; the two subdirectory links resolve to crates/repack/src/EGM.md and crates/repack/examples/EGM.md).
No drift detected.

## crates/repack/README.md

Coverage: sampled (claims verified against named symbols, constants, test files and commands; crate source not fully read).
- Tests section says "This crate contains 83 integration test files in `tests/`"; crates/repack/tests holds 91 .rs files [crates/repack/tests:directory].
- Key Modules describes `ranged_download/` as the "HTTP range-request client (`RangeSource`)"; `RangeSource` is a trait, not the HTTP client [crates/repack/src/ranged_download/mod.rs:RangeSource].

## crates/repack/AGENTS.md

Coverage: sampled (flint canonicalization, rebalance, mxfp4 and GGUF rules, checks commands and linked docs verified; gemma4_checkpoint and gguf_checkpoint bodies not read in full).
- Flint rule "the `softmax_offset` sinks are resident": `softmax_offset` is not referenced anywhere in crates/repack/src/gemma4_checkpoint/ or flint_config.rs; the only occurrence is the fixture [crates/repack/src/synthetic_flint.rs:tiny_flint_arch]. flint_config.rs only checks an `attention_sink` flag [crates/repack/src/flint_config.rs:parse_flint_config]. Residency could not be confirmed by name; verify against classify.rs before keeping the rule.

## crates/repack/examples/EGM.md

Coverage: sampled (all four example files searched for each named symbol, side effect and refusal; bodies not read line by line).
- Side effects / invariants say "the catalog and runtime refuse to attach" the MiniCPM tower output [crates/repack/examples/minicpm_tower_local.rs:run]. run() contains no such refusal; the claim concerns other crates and is not established by the cited symbol.

## crates/repack/src/EGM.md

Coverage: sampled (named symbols, constants, side effects and invariants checked by targeted reads; roughly a third of the crate root source read in part, no full read).
- Invariant "Layer types map to mask codes (conv 6, full 1, sliding 0, linear attention 2); unknown types are refused [lfm3_config.rs:layer_mask]": layer_mask accepts only "conv", "full_attention" and "sliding_attention"; there is no mapping to 2, and linear attention is refused as an unknown type.
- Side effect "mutates the passed header in place on canonicalize [unlimited_ocr2_tower.rs:write_unlimited_ocr2_tower]": write_unlimited_ocr2_tower takes `&[SafetensorsHeader]` and does not mutate. The in-place mutation is in [crates/repack/src/unlimited_ocr2_config.rs:canonicalize_unlimited_ocr2_header], which takes `&mut SafetensorsHeader`.
- Invariant "Header and manifest reads are size-capped before allocation (cap+1 bytes) [safetensors_header.rs:parse_header]": parse_header compares the header length to max_bytes and slices bytes the caller already holds. The cap+1 read happens only in [manifest_peek.rs:read_manifest_capped].
- Invariant "text_config beats root [trained_context.rs:sane]": the text_config precedence lives in [trained_context.rs:from_config_json]; `sane` only bounds the value (0 or above u32::MAX reads as unknown).

## crates/repack/src/gemma4_checkpoint/EGM.md

Coverage: sampled (every named Interface symbol located; quant tables, expert stride, staging names, shard duplicate refusal, Hadamard refusal, vision block counts checked; mod.rs and vision_*.rs (about 12k lines) not read in full).
- Interface lists `write_gemma4_install_streamed_in_place` [mod.rs:write_gemma4_install_streamed_in_place] as an entry point, but it is a private `fn` (crates/repack/src/gemma4_checkpoint/mod.rs:write_gemma4_install_streamed_in_place). Same for `sibling_work_dir` and `publish_staged_install` (mod.rs:sibling_work_dir, mod.rs:publish_staged_install), both private.
- Interface lists `routed_expert_bits` [expert_blobs.rs:routed_expert_bits] alongside public items; it is a private `fn` (expert_blobs.rs:routed_expert_bits).

## crates/repack/src/gguf_checkpoint/EGM.md

Coverage: sampled (named symbols checked; resume adoption and provenance, DeepseekV4Flash refusal, layer-position rule, rope_freqs thresholds, logical_shape panic, Qwen4Exp QSA and IQ4_NL checks, router transcode and n-gram fsync read from source; plan.rs and transcode.rs bodies not read in full).
- Interface names `lfm3_rope_freqs_are_zero_tail` [rope_freqs.rs:zero_tail_pattern]; no symbol with that name exists. The module exposes `verify_plan` [crates/repack/src/gguf_checkpoint/rope_freqs.rs:verify_plan] and `zero_tail_pattern` [crates/repack/src/gguf_checkpoint/rope_freqs.rs:zero_tail_pattern]. Rename the entry to one of these.

## crates/repack/src/gguf_config/EGM.md

Coverage: sampled (named symbols, epsilon pins, block_count and nextn bounds, unported family list, rope base swap, Gemma and Qwen mask builders read; attention.rs and most of mod.rs not read in full).
- Invariant "lfm3 ... kind 6 (short conv) where per-layer KV heads are 0, else Gemma mask [masks.rs:lfm3_layer_mask]": a zero KV-head entry becomes kind 6 only when the Gemma kind is full (1). Any other kind with zero KV heads is refused with BadValue, which the doc does not say.
- Invariant "`partial_rotary_factor` is never read from `rope.dimension_count` except for Qwen4Exp and consistency checks on Spark25 and LFM3 [mod.rs:arch_from_gguf]": the Deepseek2 MLA branch also reads `rope.dimension_count` [crates/repack/src/gguf_config/mod.rs:arch_from_gguf, Deepseek2 block]. It sets the MLA rope dimension, not the partial factor, so this is an omission rather than a contradiction. Low confidence on whether the doc meant to cover it.

## crates/repack/src/gguf_header/EGM.md

Coverage: sampled (error variant list, constants, pre-allocation count floor, bounds checks and ggml block table verified; parser.rs body not read in full).
- Interface error list omits `InvalidUtf8` [crates/repack/src/gguf_header/types.rs:GgufHeaderError]. Minor.

## crates/repack/src/gguf_names/EGM.md

Coverage: sampled (named functions, GgufMapping and GgufNameError variants, rope_freqs and pre-merge refusals, Qwen A_log/dt_bias suffix rule, Gemma fused gate/up, architecture strings checked; per-family tables not read in full).
No drift detected.

## crates/repack/src/gturbo_writer/EGM.md

Coverage: sampled (streaming.rs read in full; layers.rs and manifest.rs read in targeted ranges; types.rs and mod.rs checked by symbol search; claims verified 16 of 17, source not fully read)
- Omits that `StreamingGturboWriter::set_quant` must be called or the loader rejects production-shape manifests [streaming.rs:StreamingGturboWriter::set_quant]
- Overstated: `write_packed_vision` is described as a tower of `depth` experts, but it takes the block count from `blocks.experts.len()` and never checks it against `arch.vision.depth` [layers.rs:write_packed_vision]

## crates/repack/src/ranged_download/EGM.md

Coverage: sampled (mod.rs header-fetch logic, chunks.rs chunk_ranges and fill_chunks, and http.rs retry, range, cache, CancelFlag and download paths read; about 1,660 lines including tests, source not fully read; claims verified 28 of 30)
- Omits that `fill_chunks` does not stop at the first failure: every remaining chunk is still fetched before the lowest-indexed error is returned [chunks.rs:fill_chunks]
- Omits public `CancelFlag` methods `is_paused` and `is_cancelled` [http.rs:CancelFlag]

## crates/repack/src/synthetic_gguf/EGM.md

Coverage: sampled (gemma4_shape.rs presets and QuantMix, gemma4.rs fused gate/up and attn_v placement, gptoss.rs default and metadata, minimax.rs, qwen2.rs read by targeted lines; all named symbols exist; the two-layer default count was not verified)
No drift detected.

## crates/repack/src/synthetic_gguf/builder/EGM.md

Coverage: sampled (mod.rs with_alignment, array homogeneity and build path, and tensors.rs iq_tensor, mxfp4_tensor and q*_tensor read by targeted lines; ggml type ids 8, 11, 12, 14, 18, 20, 23 and 39 and MXFP4 exponent 124 checked; claims verified in full for the listed symbols)
No drift detected.

## crates/repack/src/synthetic_model/EGM.md

Coverage: sampled (arch.rs constants, names and xorshift helper, dense.rs SWA layer mask, and moe.rs streamed builder stride, sub-tensor count and seeds read; about 490 lines, not fully read)
- Overstated: "All dims are multiples of 64" does not hold for every dim, since `FULL_HEAD_DIM` is 32 and `NUM_HEADS` is 2; only the contraction dims (HIDDEN_SIZE, INTERMEDIATE_SIZE) are 64 [arch.rs:HIDDEN_SIZE, arch.rs:FULL_HEAD_DIM]
- Conflated: the MoE variants omit the dense FFN as a deliberate deviation from real Gemma 4 (the source comment points to DEVIATIONS.md). The doc attributes that to `attention_k_eq_v`, which only explains the missing V weights [moe.rs:build_synthetic_gemma4_moe_install]
- Omits that routed-only MoE is a known fidelity gap versus real Gemma 4 (`RealForwardRunner` runs routed-only when `num_experts > 0`), which the Overview's "honestly small, not invented" framing does not mention [moe.rs:build_synthetic_gemma4_moe_install]

## crates/repack/src/synthetic_qwen/EGM.md

Coverage: sampled (every named pub builder checked against source; invariants checked in dense_tensors.rs (bit widths), dense_arch.rs (group_for), vision.rs (V_HIDDEN, preprocessor_config.json, Result return), drama3.rs (fixture layers, bit overrides), qwen4.rs (num_experts 0), moe_mxfp4.rs (MiniCPM decoys, affine arch reuse); about 2,660 lines, not fully read)
No drift detected.

## crates/repack/src/synthetic_qwen/qwen4_decode/EGM.md

Coverage: sampled (arch.rs shape constants, layer mask [2, 2, 1, 2], indexer budget assert and index_top_k, attention_scale 0.125, PLE layer id; tensors.rs A_log and dt_bias naming, plain gated norm, INT8 router overrides and raw-router path; ngram.rs group size, PLE_LAYER and build_layer_multipliers use; mod.rs writer dispatch; about 830 lines, read by targeted ranges)
No drift detected.

Directories covered: 7 of 7 (every doc file read; 7 doc files total).
Directories not finished: none.

## crates/runtime/src/families/EGM.md

Coverage: full read (EGM.md and families/mod.rs read in full; the 12 linked child EGM.md paths confirmed to exist; the child one-line descriptions were not checked against child source)
No drift detected.

## crates/runtime/src/families/deepseek2/EGM.md

Coverage: sampled (claims verified 15 of 19, source not fully read; mod.rs produce path read in full, state.rs build guards and moe.rs/attn.rs key functions read)
- Omits the prefill `skip_head` mode: when set, the final norm, head GEMV and logits readback are skipped, KV still advances and residual capture records the skip [deepseek2/mod.rs:produce_real_deepseek2_inner]. The Interface line says the inner flow always runs "final norm and raw-logit head".
- Omits the per-layer steering dispatch (`encode_steering`) on both dense-lead and MoE layers [deepseek2/mod.rs:produce_real_deepseek2_inner]; Side Effects does not mention it.

## crates/runtime/src/families/flint/EGM.md

Coverage: sampled (claims verified 15 of 19, source not fully read; state.rs build guards, prefill.rs env gate and refusals, attn.rs window/RoPE logic, moe.rs sigmoid router read directly)
- Overstated diagnostic: the doc says `TURBOSPARK_FLINT_DEBUG` gives "readback and stderr per layer". The readback runs per layer, but stderr is written only for a value that is non-finite or above `DEBUG_REPORT_ABOVE` (2000.0) [flint/mod.rs:debug_report, flint/mod.rs:DEBUG_REPORT_ABOVE].
- Overview names the dense lead as "layers 0 and 1". The code takes the lead from `arch.num_dense_leading_layers` and only requires `1..num_layers` [flint/state.rs:RealFlintState::build]. The statement holds only for the published checkpoint.

## crates/runtime/src/families/gemma4/EGM.md

Coverage: sampled (claims verified 18 of 21, source not fully read; state.rs build, attn.rs router offsets, moe_batch.rs and attn_batch.rs refusals, vision_prefill.rs span refusal and embed-scale note read directly)
- Interface names do not match the code. The doc lists `encode_gemma4_layer_attn_and_router` / `_qkv` / `_attend_and_router` [attn.rs]. The real functions are `encode_gemma4_layer_attn_and_router` [gemma4/attn.rs:encode_gemma4_layer_attn_and_router], `encode_gemma4_layer_qkv` [gemma4/attn.rs:encode_gemma4_layer_qkv] and `encode_gemma4_layer_attend_and_router` [gemma4/attn.rs:encode_gemma4_layer_attend_and_router]. The shorthand reads as an `_attn_and_router_qkv` name that does not exist.

## crates/runtime/src/families/gptoss/EGM.md

Coverage: sampled (claims verified 17 of 19, source not fully read; state.rs build guards, attn.rs bias/sink ordering, moe.rs router bias placement, moe_batch.rs and prefill.rs refusals read directly)
- Omits the expert weight rule: top-k is taken on the bias-added logits, then softmax runs over the selected k experts with no renormalization of the full set. The doc states the bias placement but not the weight rule [gptoss/moe.rs:encode_gpt_oss_layer_moe, real_forward_dispatch.rs:router_topk_gemma4].

## crates/runtime/src/families/lfm3/EGM.md

Coverage: sampled (claims verified 20 of 22, source not fully read; state.rs build guards and constants, conv.rs, prefill.rs refusals, width cap and env gates read directly)
- Citation: the `TURBOSPARK_FFN_HIST` refusal lives in `prefill_chunk_real_lfm3` [lfm3/prefill.rs:prefill_chunk_real_lfm3], not in `lfm3_chunked_refusal` [lfm3/prefill.rs:lfm3_chunked_refusal], which the Invariant's citation pair implies. Minor.

## crates/runtime/src/families/llama/EGM.md

Coverage: sampled (claims verified 18 of 21, source not fully read; state.rs build guards and MiniMax block, router.rs sigmoid_topk, dense.rs width, moe_prefill.rs live-image refusal read directly)
- Omits MiniMax requirements: the build refuses MiniMax unless the install is MoE, uses sigmoid scoring, and sets `rope_neox_subdim` [llama/state.rs:RealLlamaState::build]. The MiniMax invariant lists sigmoid and the untied head only.
- Omits the dense-install refusal when `intermediate_size <= 0` [llama/state.rs:RealLlamaState::build].

## crates/runtime/src/families/museglimmer/EGM.md

Coverage: sampled (state.rs build read in full; mod.rs, prefill.rs, attn.rs by targeted reads; dir is 1261 lines, under the 1500 threshold but not fully read; roughly 20 of 26 claims verified)
- Build-time checks omitted from the doc: head_dim must equal full_head_dim, and rotated_pairs must be positive and at most half the head (partial_rotary_factor range) [state.rs:RealMuseState::build]

## crates/runtime/src/families/qwen/EGM.md

Coverage: sampled (13,351-line dir, not read in full; all 48 unique [file:symbol] cites resolve; env vars and about 20 invariants checked by targeted reads)
- Env vars read in code and absent from the Side Effects env list: TURBOSPARK_EXPERT_RESIDENCY [moe_batch.rs:encode_qwen_layer_moe_batched] (the "mapped and slot-cache arms" invariant never names this switch; mapped residency and batched verify are refused together at moe_batch.rs:112)
- TURBOSPARK_PILOT_PROBE [state.rs:RealQwenState::build] (router pilot probe; refused with batched verify at moe_batch.rs:130)
- TURBOSPARK_MAPPED_DEMAND_PREP [state.rs:RealQwenState::build] (off, advice, or touch; invalid value is an error)
- TURBOSPARK_QWEN_SHARED_READ_OVERLAP [moe_prefill.rs:prefill_chunk_real_qwen_moe] (refused for the MoE chunked driver at moe_prefill.rs:113)
- TURBOSPARK_ROUTED_BATCH and TURBOSPARK_ROUTED_BATCH_OVERLAP [moe_batch.rs:encode_qwen_layer_moe_batched]
- TURBOSPARK_INT4_MMA8 [dense.rs, attn.rs, moe.rs, dflash_state.rs] (switches the GEMV arithmetic path)
- TURBOSPARK_DFLASH_CHUNKED_PREFILL [dflash.rs] (on unless set to 0)
- TURBOSPARK_ROUTER_LOGITS_LAYER [moe.rs:encode_qwen_router] (router-logit trace)
- TURBOSPARK_STANDIN [produce.rs:produce_real_qwen_inner] (lab hook that skips a block or the FFN)
- TURBOSPARK_PREFILL_STEEL, TURBOSPARK_MTP_BATCHED_PRIME [prefill.rs], TURBOSPARK_ATTN_DECODE_KERNEL [batched_layers.rs], TURBOSPARK_PREFILL_ROWS_ATTN [batched_scratch.rs], TURBOSPARK_PHASES [moe.rs, moe_batch.rs]
- Note: the doc uses shorthand (for example "_DFLASH_TREE", "_GDN_LAZY_TAPE"); those match code and are not drift.

## crates/runtime/src/families/qwen/dflash_draft/EGM.md

Coverage: sampled (1,382-line dir, under the 1500 threshold but not read in full; targeted checks on constants, selectors, cursor and vocab guards; roughly 12 of 16 claims verified)
No drift detected.

## crates/runtime/src/families/qwen4/EGM.md

Coverage: sampled (5,833-line dir, not read in full; build refusals, hyper-connection divisions, prefill env parsing and cites checked; roughly 18 of 24 claims verified)
- Env vars read in code and absent from the Side Effects list: TURBOSPARK_ROUTED_BATCH_OVERLAP [moe_batch.rs:encode_qwen4_layer_routed_prefill], TURBOSPARK_INT4_MMA8 [attn.rs:encode_linear_block], TURBOSPARK_PHASES [cb1_trace.rs], TURBOSPARK_PILOT_PROBE, TURBOSPARK_QWEN4_TAP_LAYERS [state.rs:GdnNormCapture]. The doc's "_POSITION/_LAYERS/_LAYER companions" wording covers some of the capture ones, but TAP_LAYERS is not clearly one of them.

## crates/runtime/src/families/spark/EGM.md

Coverage: sampled (1,101-line dir, not read in full; checked exact-erf GELU, hiddenActivation gate, softcap and lm_head refusals, even full_rotary_dim, qkv buffer sizing, rope_params_for, advance_by; roughly 10 of 14 claims verified)
No drift detected.

## crates/runtime/src/families/synthetic/EGM.md

Coverage: sampled (620-line dir, under the 1500 threshold but targeted reads only; note_eager_advance placement, --kv-bits refusal, attention_k_eq_v check, softcap sign, zero-padded routing weights, and no-env claim verified; the cargo test command was not run)
No drift detected.

## crates/runtime/src/families/unlimited_ocr2/EGM.md

Coverage: sampled (1,100-line dir, not read in full; build refusals, top_k bound, set_prompt row check, DASH config load from the install dir, softmax-no-renorm, and two commits per layer verified; roughly 8 of 10 claims)
- Doc says build refuses "no dense lead". Code refuses `num_dense_leading_layers >= num_layers` (no MoE layers) and `dense_lead_intermediate_size <= 0`, but a lead count of 0 with a positive intermediate size passes build [state.rs:RealUnlimitedOcr2State::build]. Either the doc overstates the check or the check is missing; intent needs confirming.

## crates/runtime/EGM.md

Coverage: sampled (verified the 4 subdirectory links and the boundary claim; source not read)
- Overview says everything outside `src/` is "examples and helper scripts", but the crate also has a `tests/` directory (105 top-level .rs files) that the EGM does not list [crates/runtime/tests:tests]

## crates/runtime/src/EGM.md

Coverage: sampled (claims verified about 75 of about 100 by grep and targeted reads: named symbols, env vars, enum variants, thread names, admission, speculation refusals, steering build checks, heavy guard, sleep site; families/ subtree out of scope; some invariants such as "Steering Off is byte-identical" and "Unknown dtypes return Unsupported" not traced)
- Pins for "measuring open entry points" (speculation off, steering off, one session slot, KvQuant::Off, Streamed residency) are attributed to [crates/runtime/src/real_forward_api.rs:open_with_kv_quant]. That function takes speculation, steering, session count and KvQuant from the caller and pins none of them. The pins are in [crates/runtime/src/real_forward_api.rs:open_with_options], which passes DraftPolicies::off(), SteeringPolicy::off(), 1, KvQuant::Off and ExpertResidency::Streamed to open_inner.
- Env var TURBOSPARK_METAL_KERNEL_WARMUP is read at line 766 but is absent from the Side Effects env list [crates/runtime/src/real_forward_open.rs:TURBOSPARK_METAL_KERNEL_WARMUP]. AGENTS.md says it is opt-in, so the omission matters.
- Overview says only "the generic loop, power, speculation and turn-stream modules build elsewhere" (off macOS). lib.rs leaves more modules ungated: config, dflash_tree, draft_lattice, error, expert_layout_validation, lab_hooks, pacing, producer, raw_completion_chunked, speculative, speculative_adaptive, token_sink. speculation_policy is macOS-only, so "speculation" is ambiguous [crates/runtime/src/lib.rs:crate]
- Admission invariant omits the zero-budget refusal: max_new_tokens == 0 returns RuntimeError::ZeroBudget [crates/runtime/src/raw_completion.rs:check_admission]

## crates/runtime/src/coreml/EGM.md

Coverage: sampled (claims verified about 40 symbols, the 5 *_UNITS / quantized-GPU env vars, the macOS 15 gates in parakeet, parakeet_unified, diarization, embedding_gemma2 (MultiFunction capability) and unlimited_ocr, the OCR encoder GPU pin and ANE decoder refusal, the 1 MiB config cap, the 32000-sample append limit, cache_root; the streaming state machine and parse functions not read)
No drift detected.

## crates/runtime/src/encoder/EGM.md

Coverage: sampled (verified SUPPORTED_MODEL_TYPES, model_type refusal, CLS-only pooling with missing 1_Pooling passing, 8-bit group size check, tokenizer load swallowed to None, erf GELU flag, cosine re-export; forward math and weight loading not read)
No drift detected.

## crates/runtime/src/native_audio/EGM.md

Coverage: sampled (verified busy and cancelled guards, the "stream was cancelled" and "cancelled" strings, Unified admission at 3x weights plus 256 MiB, Multitalker admission at weights plus a quarter plus 512 MiB, MAX_PCM_CHUNK gate, the multitalker-stream thread name; Unified finish restart-at-zero and worker panic behavior not traced)
No drift detected.

## crates/runtime/src/router_hist/EGM.md

Coverage: sampled (verified BatchCapture method set, truncation and completion logic, TURBOSPARK_ROUTER_BATCHES and TURBOSPARK_ROUTER_BATCH_LIMIT with the 1,000,000 default, the Drop serialization in the parent; batches.rs read by grep and excerpts only)
No drift detected.

## crates/runtime/src/vision/EGM.md

Coverage: sampled (verified VISION_SLOTS = 2, TURBOSPARK_VISION_RESIDENCY and TURBOSPARK_VISION_OVERFLOW, Qwen2.5 block%8==7 window rule, tanh vs erf GELU split, SigLIP final-norm eps 1e-5, FP16 narrowing bound, Uocr2 1 MiB / 2 GiB caps, LFM3-VL 65,536-patch cap, head_dim checks, pub mod unlimited_ocr2; tower forward passes not read)
No drift detected.

## crates/runtime/src/whisper/EGM.md

Coverage: sampled (verified config and tokenizer read caps, 1 MiB header cap, 24 GiB hard cap, TURBOSPARK_WHISPER_DEVICE and TURBOSPARK_WHISPER_PROFILE, the 480000-sample window and stride, 440-token budget, 8-repeat stop, English-only language gate, static open lock, Metal autorelease pools, head-divisibility check; receipt matching and the 2*seq mel-frame check not read)
No drift detected.

## crates/runtime/examples/EGM.md

Coverage: sampled (verified all 19 example files and every file named in the doc, the --gate parity mode, router env setup in fish_drama3_parity, music3 AR-only and trace env vars, OCR_MAX_NEW_TOKENS, the music3 !Send note, the 16-bit stereo WAV output, the wer --smoke flag; probe warmup behavior and fish_drama3 internals not read)
- Says the Fish Drama 3 examples are gated `#![cfg(target_os = "macos")]`. Only fish_drama3_guard_sweep.rs, fish_drama3_silence_probe.rs, fish_drama3_parity.rs and fish_drama3_wer.rs carry that gate. fish_drama3_say.rs, fish_drama3_runner.rs and fish_drama3_step_cost.rs have none [crates/runtime/examples/fish_drama3_say.rs:main]

## crates/runtime/scripts/EGM.md

Coverage: sampled (verified both drivers' flags, BINARY paths, quiet_host and digests_agree logic, the pmset and /usr/bin/time calls, the ids-once tokenization, and the --help entry point; full file reads not done)
No drift detected.

## crates/runtime/tools/EGM.md

Coverage: sampled (verified the SHA-256 pin refusal, 16 MiB-style download cap, ffmpeg requirement, OUT_DIR default chain, converted.json output, the Arial path, the testdata and fixture paths, and the no-argparse status of the two generators; full reads not done)
- Output list for ref_embeddinggemma2_coreml.py omits pattern.png, which the script also writes and records in manifest.json [crates/runtime/tools/ref_embeddinggemma2_coreml.py:OUT]. The doc lists "reference.json, manifest.json, wav".

## crates/runtime/README.md

Coverage: sampled (verified module file names, the families/ directories (except lfm3), the Cargo alias pattern used by crates/bench and crates/lab, the test file names listed, and the qwen4 and gemma4 moe_batch / vision_prefill paths; the GDN reset gotcha was not traced to code)
- "This crate contains 81 integration test suites in tests/" is stale: tests/ holds 105 top-level .rs files [crates/runtime/tests:tests]
- The families/ list omits lfm3/, which exists and has its own EGM [crates/runtime/src/families/lfm3:lfm3]

## crates/runtime/AGENTS.md

Coverage: sampled (verified all Read-first links and referenced paths, the Flint refuse-by-name for BATCHED_GEMV and ROUTED_BATCH, the Flint mapped-residency refusal, the ANE M4/A18 hardware gate and 2048-token ANE width, the MegaTTS3 Metal-first device list with CPU only as opt-in fallback, the Music 3 Apple M4 Max oracle gate, the 1e-2 KL bound, and the existence of named env vars and functions; benchmark numbers, Steel default claims, and the Qwen MoE batching claims not traced)
No drift detected.

## crates/selection/EGM.md

Coverage: sampled (package name, src link, test-file count, and cargo command checked against the crate; test files checked by name only)
No drift detected.

## crates/selection/README.md

Coverage: sampled (symbol, path, alias, and gotcha claims checked against src, which was read in full; tests/ checked by name, #[ignore] attributes, and two bodies, not fully read)
- [src/derive.rs:derive_step_value] reads `SystemTime::now()` when `seed` is `None`, so the Safety section's "zero external side effects" and "pure" wording is overstated. Output with no seed is not reproducible. [src/choose.rs:SCRATCH] also keeps per-thread buffers alive.
- [src/choose.rs:select] does not dispatch to rejection sampling. It branches only between argmax (deterministic temperature) and reweight plus seeded draw. Callers build rejection sampling from [src/choose.rs:shaped_distribution] and [src/distribution.rs:residual]. The Key Modules line for choose.rs is wrong.
- Gotcha 1 says pre-normalized input "collapses the distribution toward uniform when temperatures differ from 1.0". The flattening does not depend on temperature. [src/choose.rs:shape_candidates] applies softmax to whatever it is given, and at T=1 [src/choose.rs:reweight] is the identity, so probabilities passed as logits flatten at every temperature.
- Gotcha 2 says the seeded generator "guarantees bit-exact reproducible token sequences across identical runs". That holds only when the config has `seed: Some(_)` ([src/derive.rs:derive_step_value]). With `None`, draws are clock-derived.
- Gotcha 3 credits "truncation.rs" with the sequential kth-largest cut. The hot path is [src/truncation.rs:rank_top_k_u32_into]. The public [src/truncation.rs:rank_top_k] uses `select_nth_unstable_by`, which is a different algorithm. Name the function.
- The intro lists "rejection sampling, and distribution guards" as parts of the shaping configuration. [src/shaping.rs:ShapingConfig] has no rejection-sampling field. Distribution guards are errors raised by [src/choose.rs:select] and [src/shaping.rs:ShapingConfig::new], not settings.
- The penalty description ("historical token frequency and occurrence") is loose and omits the key convention. [src/penalty.rs:apply_repetition_penalty] penalizes each distinct history token once over the full history, prompt included, and ignores frequency. [src/penalty.rs:apply_presence_and_frequency_penalty] counts only the generated suffix, which [src/choose.rs:shape_candidates] passes in. The generated-only rule is a deliberate deviation from OpenAI (the function doc says so) and is not mentioned here.
- The Tests section says `cargo test -p turbospark-selection` runs all tests. Four are `#[ignore]` and skipped: [tests/host_sampler_cost.rs:where_the_host_sampler_time_goes], [tests/host_sampler_cost.rs:the_cut_path_against_the_partition_it_replaced], [tests/rank_top_k.rs:a_tied_cut_stays_cheap_on_the_collected_order] (a timing assertion), and [tests/rank_top_k.rs:cost_at_the_real_vocabulary]. They run only with `--release -- --ignored`.
- [tests/domain_guards.rs:a_rank_based_working_set_larger_than_the_candidate_domain_still_returns_a_valid_choice] covers a top_k larger than the candidate domain (250 over 3 candidates), not an "oversized top-k" rejection. Rejection of top_k above MAX_TOP_K is in [tests/shaping_validation.rs:top_k_outside_bounded_range_is_rejected].
- [tests/rank_top_k.rs] checks its results against a full-sort reference. "Without full array sorting" describes the code under test, not the test. Minor wording.
- Verified with no drift: the `selection` alias snippet (six workspace crates use the same path dependency), no `unsafe` in src, the absence of `#![forbid(unsafe_code)]` in lib.rs, the module list, the test-file count (11), and the test-file descriptions except as noted above.

## crates/selection/AGENTS.md

Coverage: sampled (rules, links, and check command verified; the rest of the crate was not reviewed for this file)
No drift detected.

## crates/selection/src/EGM.md

Coverage: full read (all seven files in src/, 1477 lines, read in full)
- Side Effects says "Otherwise pure". [src/derive.rs:derive_step_value] reads `SystemTime::now()` when `seed` is `None`, so this is false.
- Invariant "Draws are deterministic for a given seed and position" holds only for `Some(seed)` ([src/derive.rs:derive_step_value]). The `None` branch is clock-derived. The invariant should say so.
- Interface omits `pub fn residual` [src/distribution.rs:residual], which lib.rs re-exports. It is the rejection-sampling entry point.
- Interface presents `softmax`, `truncate_by_probability_mass`, `truncate_by_rank`, and `truncate_by_min_p` as the truncation pipeline. [src/choose.rs:select] calls none of them. The same steps run inlined in [src/choose.rs:shape_candidates], which uses only `rank_top_k_u32_into` and `rank_indices_u32_into` from truncation.rs. Say the helpers are reference implementations (tests/min_p.rs cites `truncate_by_min_p` that way) and that choose.rs is the live path, or the two can drift.
- Invariants omit the min-p check [src/shaping.rs:ShapingConfig::with_min_p] (finite, in [0, 1)). The presence and frequency range checks live only in [src/shaping.rs:ShapingConfig::with_presence_penalty] and [src/shaping.rs:ShapingConfig::with_frequency_penalty], not in `new`. The current wording implies `new` validates them.
- Side Effects omits that [src/choose.rs:SCRATCH] keeps buffers sized to the vocabulary (multi-megabyte at V=262,144) alive for the life of each thread. Reuse is the point, but a reader sizing memory should know it.
- Verified with no drift: Interface symbols `select`, `shaped_distribution`, `ShapingConfig` builders, `SelectionError`, `DEFAULT_REPETITION_PENALTY`, `MAX_TOP_K`, the `ShapedDistribution` methods, the penalty and truncation function names, `derive_step_value`, `to_unit_interval`, the thread-local SCRATCH, the total-order ranking invariant, the `ShapedDistribution::new` assertions, and the verification command.

## crates/server/AGENTS.md

Coverage: sampled (claims verified about 8 of 10; TLS absence not checked; st index stale, claims confirmed by Read)
- The FFI default contradicts the doc. The doc says prefix reuse is off by default and that the in-process FFI host "reaches the same posture through ServerOptions.prefixReuse". The binary default is off ([crates/server/src/args.rs:parse] `prefix_reuse: false`, line 178), but an FFI session opens with `options.prefix_reuse.unwrap_or(true)` ([crates/ffi/src/open.rs:open], line 442). [crates/ffi/src/server.rs:attach] only overrides that when ServerOptions.prefixReuse is Some, so an FFI host that omits the option runs with reuse ON.

## crates/server/README.md

Coverage: sampled (roughly 30 of 35 claims checked; source not fully read; st index stale, absence claims re-read)
- Key Modules, queue.rs: "no queue timeout or depth cap" is contradicted. [crates/server/src/queue.rs:DEFAULT_MAX_QUEUED] is 64, `--max-queued` ([crates/server/src/args.rs:parse]) sets it, 0 means no limit, and a request past the cap gets 429 with Retry-After.
- Intro: "zero-allocation translations between Anthropic and OpenAI protocols" is not supported by the crate. [crates/server/src/messages.rs] translates through anyllm_translate types into owned values. Overstated; drop or qualify.
- Key Modules, guardrails.rs: "schema repair" does not exist. [crates/server/src/guardrails.rs:inspect] rescues calls, validates arguments against the request schema, and [crates/server/src/guardrails.rs:run_guarded] re-asks once. Validation and one retry only.
- Key Modules, registry.rs: "least-busy dispatch (round-robin across idle members)" is loose. [crates/server/src/registry.rs:PoolRegistry] takes the first idle member scanning from the round-robin cursor, else the least-busy member; round-robin applies on ties.
- Omitted: the API key may come from the TURBOSPARK_API_KEY environment variable, for `--api-key` and for `--bind tailnet` ([crates/server/src/main.rs:resolve_api_key]).
- Omitted: `--prefix-reuse` (default off; the shared KV cache is not partitioned by API key or client, so prefix matches can be seen in timing) and `--guardrails` (default on; tool-bearing requests are buffered) are not in the README CLI notes. The caveat lives only in [crates/server/src/args.rs:USAGE]. This is the security boundary AGENTS.md calls out.
- Omitted routes: `/v1/models/{model}` ([crates/server/src/lib.rs:build_router_with_options]), `/v1/images/edits` (stub that refuses, [crates/server/src/images.rs:edits_unsupported]), and `/v1/audio/translations` (refused, [crates/server/src/audio/stt.rs:translations_unsupported]).
- Verified, no drift: the 22 suites named in Tests match the 22 .rs files in tests/; the tailnet range 100.64.0.0/10 and the mandatory key ([crates/server/src/bind.rs:is_tailscale_ipv4], [crates/server/src/main.rs:resolve_api_key]); speech installs refused on --model ([crates/server/src/main.rs] speech receipt check); `--embedding-model` alone and audio-only modes ([crates/server/src/main.rs] model match); Anthropic `message_stop` terminator ([crates/server/src/messages.rs] StreamEvent::MessageStop).

## crates/server/EGM.md

Coverage: sampled (3 of 4 claims checked by targeted reads; source not fully read)
- Global Invariants: "Auth and loopback-host rules are applied in one place ([src/lib.rs:build_router_with_options])" is only partly true. The Host check and the auth layer live in lib.rs, but the rule that a Tailnet bind requires a key, and the TURBOSPARK_API_KEY fallback, are enforced in [crates/server/src/main.rs:resolve_api_key]. That is a second place.
- Verified: /health sits outside the auth layer (lib.rs), and the lib/bin names and Run command match Cargo.toml and the crate.

## crates/server/src/EGM.md

Coverage: full read of lib.rs (510 of 525 lines); other named symbols checked by st lookup and targeted reads; sampled overall
- Interface omits the bounded resolution executor: [crates/server/src/lib.rs:ServerState::with_max_concurrent_resolutions] and [crates/server/src/lib.rs:bounded_resolve::BoundedResolveExecutor] cap blocking model resolutions (default 4) and return a retryable 503 when saturated.
- Interface and Invariants omit the queue depth cap: [crates/server/src/queue.rs:DEFAULT_MAX_QUEUED] (64, `--max-queued`) and the 429 with Retry-After for queued waiters.
- Invariant "Request bodies are size-limited ([lib.rs:MAX_REQUEST_BODY_BYTES])" holds for the chat router (25 MiB) but omits the per-route caps: [crates/server/src/images.rs:MAX_IMAGE_REQUEST_BODY_BYTES] on `/v1/images/generations` and [crates/server/src/audio/mod.rs:MAX_AUDIO_UPLOAD_BYTES] on the audio router ([crates/server/src/lib.rs:build_router_with_options]).
- Routes line omits `/v1/models/{model}` and the audio routes (present only with a provider, which the Invariants do state).
- Verified: every named symbol exists; auth applies to every route except /health; no-key Host check; no audio route without a provider; no event built without an observer.

## crates/server/src/audio/EGM.md

Coverage: sampled (routes, symbols and limits checked; stt.rs and ws.rs audio decode not fully read)
- Invariant "Model loading accepts installed catalog aliases only, never filesystem paths ([audio/models.rs:load_model])" cites the wrong file. [crates/server/src/audio/models.rs:load_model] trims the string and passes it to `provider.load`. The alias and path checks live in the provider: [crates/server/src/audio_real.rs:valid_alias] and the catalog lookup near [crates/server/src/audio_real.rs] line 494. A non-real AudioProvider carries no such guarantee from the route.
- Speech consent is described only for MegaTTS3. [crates/server/src/audio/speech.rs:speech] also requires `consent: true` for OmniVoice when a reference is given (about line 194) and in a further clone branch (about line 236). Clone families are set by [crates/server/src/audio/mod.rs:clones_voices].
- Omitted: `DELETE /v1/audio/models/{id}` refuses with 409 while that model has running jobs, and 404 for an unattached id ([crates/server/src/audio/models.rs:delete_model]).
- Omitted security check: the realtime WebSocket refuses a browser Origin that is not loopback ([crates/server/src/audio/ws.rs:origin_allowed]). The doc only covers header-only auth.
- Citation: "deny_unknown_fields in [audio/generate.rs:generate]" points at the function. The attribute is on the `GenerateInput` struct ([crates/server/src/audio/generate.rs:GenerateInput]).
- Verified: AUDIO_ROUTE_PATHS, AudioProvider and the request/response types, MAX_AUDIO_UPLOAD_BYTES (128 MiB), MAX_TRANSCRIBE_SECONDS (30 min), STT_SAMPLE_RATE (16 kHz), decode_wav_16k, the JobStore limits (MAX_ACTIVE_JOBS 8, FINISHED_TTL 1 h, MAX_RESULT_BYTES 256 MiB), the running-job abort limit, RealtimeQuery (no key field), AudioMetrics, and metrics_endpoint.

## crates/server/src/guardrails/EGM.md

Coverage: sampled (parent guardrails.rs config, inspect and run_guarded read; extra_formats.rs module doc read; parser bodies not fully read)
- Omitted invariant: rescue runs before validation, so a rescued call is schema-checked like a parsed one ([crates/server/src/guardrails.rs:inspect]). The doc does not say this.
- Verified: retry budget is one (`GuardrailConfig::default()` retries 1, `OFF` retries 0, [crates/server/src/guardrails.rs:GuardrailConfig]); `mod extra_formats` declaration; `GuardrailConfig` re-export from lib.rs; the Longcat exclusion and the tests/fixture that pins it; extra formats are tried before forge's strategies.

## crates/server/src/handler/EGM.md

Coverage: sampled (all named symbols checked; exec.rs cancellation path and mod.rs lease/keep-alive paths read; plan.rs not fully read)
- No drift detected.

## crates/server/src/responses/EGM.md

Coverage: sampled (all named symbols checked; map.rs warnings path and mod.rs streaming decision read; sse.rs builders not fully read)
- No drift detected.

## crates/streaming/EGM.md

Coverage: sampled (claims verified 3 of 3; overview and subdirectory link only)
No drift detected.

## crates/streaming/AGENTS.md

Coverage: sampled (claims verified 6 of 6; the cited doc and script paths resolve, the rules were checked against source)
No drift detected.

## crates/streaming/README.md

Coverage: sampled (claims verified 27 of 30, source not fully read)
- Key modules, aligned_slot.rs: "Page-aligned buffer allocations required for unbuffered direct disk I/O" overstates. [crates/streaming/src/aligned_slot.rs:SLOT_ALIGNMENT] is 2 MiB, and the stated purpose in [crates/streaming/src/pread_streamer.rs] header is zero-copy Metal wrapping. The O_DIRECT alignment constant is [crates/streaming/src/linux_uring.rs:DIO_ALIGNMENT] (512 bytes), and that path is not wired to any streamer.
- Crate Gotchas 2, "even a single expert cache miss executes with wide I/O parallelism," is true only when an expert stride exceeds the 840 KiB chunk. [crates/streaming/src/pread_streamer.rs:execute_expert_cache_plan] splits each miss into `MISS_READ_CHUNK_BYTES` pieces. A stride at or below that gives one chunk, and [crates/streaming/src/read_pool.rs:run_batch] runs a single chunk inline on the calling thread, with no pool.
- Tests list omits `tests/offset_overflow.rs` (135 lines, checked-offset tests). [crates/streaming/tests/offset_overflow.rs] exists and is not documented.
- Key modules, expert_cache.rs does not state the default policy. [crates/streaming/src/expert_cache.rs:ExpertCachePolicy] `DEFAULT` is `Lfu`, and `from_env` returns it when `TURBOSPARK_EXPERT_CACHE_DECAY` is unset. The doc only says "deterministic LFU and LRU" and "opt-in decaying LFU".

## crates/streaming/src/EGM.md

Coverage: sampled (claims verified 42 of 46, source not fully read; about 2,400 non-test lines in src, read by targeted lookups and reads of expert_cache.rs plan path, pread_streamer.rs open/prefetch/wire/plan paths, mapped_experts.rs open path, read_pool.rs run_batch, and linux_uring.rs gating)
- Side Effects, env var list: `TURBOSPARK_LINUX_IO` is documented as read at runtime. [crates/streaming/src/linux_uring.rs:Mode] has `from_env` and `from_env_default`, but the only caller of `from_env_default` is a unit test (linux_uring.rs test `the_linux_io_mode_defaults_to_pread`). No production code in the workspace parses this variable, and the module header says nothing selects `Mode::Uring`. The seam is parse-only.
- Side Effects: "Spawns a process-wide read thread pool" is lazy. [crates/streaming/src/read_pool.rs:pool] builds the pool on first multi-chunk batch through a `OnceLock`, and single-chunk batches never reach it.
- Interface, Pread backend list omits public items that the doc's own invariants depend on or that callers use: [crates/streaming/src/pread_streamer.rs:prefetch_expert] (refused under F_NOCACHE, named in the Invariants section), [crates/streaming/src/pread_streamer.rs:nocache_active] (the caller-facing truth the source comment says to print), [crates/streaming/src/pread_streamer.rs:read_experts_into_bank], [crates/streaming/src/pread_streamer.rs:execute_speculative_reservation], [crates/streaming/src/pread_streamer.rs:advise_expert_cache_plan_misses], [crates/streaming/src/pread_streamer.rs:slot_data], [crates/streaming/src/pread_streamer.rs:slot_allocation], [crates/streaming/src/pread_streamer.rs:load_expert_into_slot], [crates/streaming/src/pread_streamer.rs:plan_experts_cached_if_possible], [crates/streaming/src/pread_streamer.rs:resident_experts_snapshot], [crates/streaming/src/pread_streamer.rs:non_resident_experts].
- Interface, Mapped backend list omits [crates/streaming/src/mapped_experts.rs:page_aligned_bytes] (the pointer a caller wraps, per the field comment on `MappedExpertLayer.shift`), plus [crates/streaming/src/mapped_experts.rs:expert_offset], [crates/streaming/src/mapped_experts.rs:expert_stride], [crates/streaming/src/mapped_experts.rs:experts_per_layer], and [crates/streaming/src/mapped_experts.rs:layout].
- Interface, Cache list omits [crates/streaming/src/expert_cache.rs:ExpertCache] `plan_if_possible`, `commit_plan`, `publish_speculative_reservation`, `resident_experts_snapshot`, and `cache_policy`, which are the non-panicking plan entry and the commit path the streamer uses.
- Overview and Interface do not state the default policy. [crates/streaming/src/expert_cache.rs:ExpertCachePolicy] `DEFAULT` is `Lfu`.
- Invariant "Every ReadChunk in a batch addresses a byte range no other chunk touches" is consistent with the code (chunks are built from disjoint `start` offsets in [crates/streaming/src/pread_streamer.rs:execute_expert_cache_plan], and distinct slots are asserted there). Not verified exhaustively, so not reported as drift.

## crates/vision-io/EGM.md

Coverage: full read (doc), with Cargo.toml, lib.rs and decode.rs read in full and the dependency and side-effect claims checked against them.
- Global Side Effects says "None at crate level", which is overstated: `decode_image_file` reads a file from disk, as crates/vision-io/src/EGM.md itself records. [decode.rs:decode_image_file]
- Global Invariants names the dependency `turbospark-core`. The package name matches, but Cargo.toml declares it as `foundation = { package = "turbospark-core" }`, so the crate imports it as `foundation` (mrope.rs uses `foundation::TokenId`). The alias is not stated. [Cargo.toml:dependencies]
- System Overview lists decode, resize, patchify and position tables. It omits the nine per-family pipelines (CLIP, MiniCpm, DeepSeek-OCR, DeepSeek-VL2, Pixtral, Muse, OneVision, Unlimited-OCR-2, Gemma 4) and the mRoPE splice, which the src doc covers. [lib.rs:pub mod]

## crates/vision-io/AGENTS.md

Coverage: full read of the doc. Link targets exist (.claude/docs/modules/vision-io.md, docs/VISION.md, docs/VISION_PHASE0.md, docs/TESTING.md). The cargo commands match the package name. The commands themselves were not run.
- Code comments cite "AGENTS.md Gotcha 8" (Cargo.toml) and "AGENTS.md Gotcha 39" (params.rs:from_preprocessor_config_json). This file has no numbered gotchas. The numbered list is in .claude/docs/engineering-gotchas.md (item 8 is the platform-claim rule, item 39 is the optional-key rule), so those citations do not resolve here. [Cargo.toml:dependencies] [params.rs:from_preprocessor_config_json]

## crates/vision-io/README.md

Coverage: full read of the doc. Claims checked against lib.rs, decode.rs, params.rs, preprocess.rs, patchify.rs, resize.rs, smart_resize.rs, pos_embed.rs, rope.rs, mrope.rs, gemma4.rs, unlimited_ocr2.rs, the tests/ directory and the tests/generated/ headers. No build or test was run. The non-macOS build claim was not checked.
- Fixture provenance: the doc says every golden fixture under tests/generated/ is probed by scripts/qwen3vl_vision_oracle.py. tests/generated/gemma4_preprocess_oracle.rs is headed "@generated by scripts/gemma4_vision_oracle.py" (that script exists). [tests/generated/gemma4_preprocess_oracle.rs:header]
- Tests section lists 4 files. Five more integration tests exist and are not listed: gemma4_preprocess_parity.rs, muse_params.rs, muse_wrap.rs, params_geometry.rs, rope_theta.rs. [tests/]
- Muse-Glimmer is absent from the doc. preprocess() dispatches `ResizeRule::MuseTokens` to `muse::muse_preprocess` instead of the Qwen chain. The doc describes preprocess.rs as a fixed four-step chain. [preprocess.rs:preprocess] [params.rs:ResizeRule]
- resize.rs is described as bicubic only. It also exports the Lanczos path (`resize_lanczos_pil`, `resample_pil_with`, `PilFilter::Lanczos`). [resize.rs:resize_lanczos_pil]
- unlimited_ocr2.rs is listed under Key Modules, but lib.rs declares `mod unlimited_ocr2;` (private). Callers reach it only through the `uocr2_*` re-exports. [lib.rs:unlimited_ocr2]
- Gotcha 4 is accurate on the mandatory pixel budget but omits the config bounds that apply to it. `min_pixels` above MAX_CONFIG_PIXELS is refused, config-declared edges are capped at MAX_IMAGE_DIM, and `rescale_factor` defaults to 1/255 while the budget does not. [params.rs:MAX_CONFIG_PIXELS] [params.rs:check_config_edge] [params.rs:from_preprocessor_config_json]
- Omitted ordering invariant: normalization runs on the resized uint8 samples, after resize. Normalizing first would push negative values through the 0..255 clamp. [preprocess.rs:preprocess]
- Omitted position-table behavior: pos_embed uses an endpoint-preserving linspace, not the half-pixel convention, and an axis with one patch reads table row 0. [pos_embed.rs:pos_embed_weights]
- Omitted: the vision RoPE base is 10,000 by design, not the trunk's 1e7. [rope.rs:VISION_ROPE_THETA]
- Omitted: placeholders are counted from vision_start markers but located from image_pad tokens, deliberately matching the reference on malformed prompts. `splice_and_walk` is the documented single entry point for front ends. [mrope.rs:mrope_position_triples] [mrope.rs:splice_and_walk]

## crates/vision-io/src/EGM.md

Coverage: sampled (claims verified: every symbol named in the Interface section located by pub-item listing, names only; invariants and side effects checked by reading or grep; 15 of 22 source files read in full: lib, error, decode, preprocess, smart_resize, patchify, normalize, rounding, params, resize, pos_embed, rope, gemma4, unlimited_ocr2, mrope; clip, deepseek_ocr, deepseek_vl2, minicpm, muse, onevision, pixtral checked only by pub-item listing and targeted reads).
- Interface names `best_grid` for OneVision, but the crate root exports it as `ov_best_grid`. The bare name exists only at module path onevision::best_grid. [lib.rs:ov_best_grid]
- Interface omits public types a downstream caller needs: ClipPreprocessParams, ClipPatches, Anyres, Gemma4Patches, Gemma4ImageIds, MiniCpmParams, MiniCpmIds, DeepseekOcrParams, DeepseekVl2Params, DeepseekVl2Tiles, OvPlan, Uocr2Page, Uocr2ImageIds. It also omits the public `vision_rope_freq_rows_default` wrapper. [lib.rs:pub use]
- Invariant "Aspect ratio is checked before any rounding" cites params.rs:MAX_ASPECT_RATIO. The check runs in smart_resize.rs:resized_dims (the ratio test precedes the rounding); params.rs only holds the constant. [smart_resize.rs:resized_dims]
- Omitted invariant: normalization runs after resize on the resized uint8 samples. [preprocess.rs:preprocess]
- Omitted: preprocess() routes `ResizeRule::MuseTokens` to muse_preprocess, so the Qwen chain is not the only entry. [preprocess.rs:preprocess]
- Omitted config rules: the pixel budget has no default and `size.shortest_edge` / `size.longest_edge` are pixel counts. Config edges are capped at MAX_IMAGE_DIM, and `min_pixels` above MAX_CONFIG_PIXELS is refused. [params.rs:read_pixel_budget] [params.rs:MAX_CONFIG_PIXELS]
- Omitted: VISION_ROPE_THETA is 1e4 by design and must not be the trunk's 1e7. [rope.rs:VISION_ROPE_THETA]
- Omitted: `uocr2_expand_prompt` returns `Result<_, String>`, not VisionIoError, so the Errors line does not cover it. [unlimited_ocr2.rs:uocr2_expand_prompt]
- Side Effects "None otherwise (no network, no GPU)" checked by grep: the only filesystem access in src is `std::fs::read` in decode_image_file. No drift on this point.
- Invariant "Decoded images are capped at 4096 per side and the decoder allocation is bounded" and the Verification command: no drift detected.

## crates/tokenizer/EGM.md

Coverage: full read (crate index doc; Cargo.toml read in full; child link ./src/EGM.md resolves).
No drift detected.

## crates/tokenizer/README.md

Coverage: sampled (claims verified 17 of 20; source not fully read; checked lib.rs, Cargo.toml, tests/ listing, dialect/mod.rs, reasoning.rs, config.rs, json_value.rs depth check, downstream Cargo aliases).
- [tests:directory] says "This crate includes 18 comprehensive test suites" and lists 18. tests/ holds 25 .rs files. Unlisted: content_marker_spelling.rs, deepseek_vl2_image_render.rs, chatml_no_tool_call.rs, regex_backend_hostile_pattern.rs, sidecar_size_cap.rs, stop_matcher_long_stop.rs, vision_markers_muse.rs.
- [src/dialect/config.rs] is described as "Generation configuration dialect mapping". The file holds TokenizerConfig, chat-template sidecar parsing and the bounded EOS list. It has no ChatDialect mapping (zero references to ChatDialect). Dialect detection lives in [src/dialect/resolve.rs:detect_dialect].
- [src/reasoning.rs] is described as handling "reasoning token formatting". The file only parses and names levels [src/reasoning.rs:ReasoningEffort] and probes template support [src/reasoning.rs:MfTokenizer::reasoning_support, MfTokenizer::accepted_reasoning_levels]. No token formatting code is in that file; rendering is in [src/chat_template/mod.rs:apply_chat_template_with_reasoning].

## crates/tokenizer/AGENTS.md

Coverage: sampled (claims verified 14 of 16; same source reads as README.md; the "chat template reads" gotcha and detokenizer gotcha checked against detokenizer.rs and tests/ fixtures).
- Same test-count drift as README.md: "This crate includes 18 comprehensive test suites" under Tests. tests/ holds 25 .rs files [tests:directory].
- Same config.rs drift: "`config.rs`: Generation configuration dialect mapping." [src/dialect/config.rs] has no dialect mapping.
- Same reasoning.rs drift: "`reasoning.rs`: ... and reasoning token formatting" [src/reasoning.rs].
- Not verified: Gotcha 1 states the tokenizers loader renumbers added tokens sequentially after the base vocabulary. This is external crate behavior and was not checked against the tokenizers 0.20 source.

## crates/tokenizer/src/EGM.md

Coverage: sampled (claims verified 30 of 34; src/lib.rs, error.rs, reasoning.rs (lines 20-233), stop_matcher.rs and detokenizer.rs (first halves) and jinja_date.rs (first 85 lines) read; json_value.rs, jinja_compat.rs and jinja_chat_template.rs checked by symbol grep only).
- Public re-exports list omits the tool-call parser re-exports: DeepseekToolCallParser, GemmaToolCallParser, GlmToolCallParser, KimiToolCallParser, MistralToolCallParser, ParsedToolCall, QwenToolCallParser. [src/lib.rs:pub use tool_call]
- Interface omits the public MfTokenizer methods that expose the reasoning level probe: reasoning_support and accepted_reasoning_levels. [src/reasoning.rs:MfTokenizer::accepted_reasoning_levels]
- Overstated invariant: "A pinned chat date must be YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS, else it is rejected." [src/jinja_date.rs:parse_pinned] also accepts YYYY-MM-DDTHH:MM (seconds default to 0), and it range-checks the day only as 1..=31, so 2026-02-31 is accepted.
- Citation: [src/lib.rs:TokenId] is only a re-export. The definition is `pub type TokenId = i32` at crates/core/src/primitives.rs:11 (turbospark-core). Cite the definition.

## crates/tokenizer/src/chat_template/EGM.md

Coverage: sampled (claims verified 12 of 13; mod.rs read by targeted ranges (apply_chat_template, apply_dialect_chat_template, verify_image_markers); fallback renderer bodies not read, module docs checked).
- Side Effects says "None (pure string rendering)". apply_chat_template prefers the checkpoint's Jinja template, and that environment registers strftime_now [src/jinja_chat_template.rs:add_function("strftime_now")]. strftime_now reads TURBOSPARK_CHAT_DATE or the system clock [src/jinja_date.rs:strftime_now]. Any installed template that calls strftime_now is therefore not pure.
- Minor citation: "DeepSeek tool-chat encoders [mod.rs]". Only encode_deepseek_tool_chat is in mod.rs. The helpers deepseek_invoke and deepseek_tools_section are in [src/chat_template/deepseek.rs].

## crates/tokenizer/src/dialect/EGM.md

Coverage: sampled (claims verified 10 of 12; MfTokenizer struct and load_from_dir read in full; minimax.rs read in full; resolve.rs, resolvers.rs and config.rs checked by symbol grep).
- Side Effects says load_from_dir "reads tokenizer and generation config files". It also reads tokenizer_config.json and the chat-template sidecars, with precedence chat_template.jinja, then chat_template.json, then the config key [src/dialect/mod.rs:MfTokenizer::load_from_dir].
- Invariant omitted: generation_config EOS lists are capped at MAX_EOS_TOKEN_IDS = 256 and filtered to the vocabulary range before they join stop_token_ids [src/dialect/config.rs:MAX_EOS_TOKEN_IDS], [src/dialect/mod.rs:MfTokenizer::new].
- Interface lists MfTokenizer methods only. It omits the public fields that consumers read directly: dialect, bos_id, eos_id, pad_id, end_of_turn_id, tool_call_start_id, tool_call_end_id, tool_response_id, tool_response_end_id, tool_call_stop_id, channel_start_id, channel_end_id, message_start_id, message_end_id, think_start_id, think_end_id, stop_token_ids, vocab_size [src/dialect/mod.rs:MfTokenizer].

## crates/tokenizer/src/structured_decoder/EGM.md

Coverage: sampled (claims verified 9 of 9 checkable; mod.rs API signatures and finish() body read; glm.rs, kimi.rs, mistral.rs and muse.rs module docs read; chatml.rs, deepseek.rs and harmony.rs checked by name only).
No drift detected.

## crates/tokenizer/src/tool_call/EGM.md

Coverage: sampled (claims verified 11 of 12; mod.rs read in full; parser bodies checked by symbol and size-cap grep, not read; the Result return type was confirmed for KimiToolCallParser only).
No drift detected.

# Crates drift check, second pass after doc fixes (haiku-reported)

Rerun of the same 34 jobs after sonnet agents corrected the EGM.md files. Same caveats: mostly sampled, haiku-reported, not independently verified. README.md and AGENTS.md findings are unchanged because those files are human-written and were not edited.

## crates/audio/AGENTS.md

Coverage: sampled (claims verified 14 of about 25; source not fully read)
No drift detected.

## crates/audio/README.md

Coverage: sampled (module list, quant bit widths, WAV/FFT/mel/whisper claims, and the OmniVoice gate test file checked; evidence prose not rerun)
No drift detected.

## crates/audio/MODELS.md

Coverage: sampled (101-directory count, reference commit, layout tree, and all relative links checked; about 4 of the evidence rows spot-checked, the rest not re-verified)
- [crates/audio/MODELS.md:Family and profile layout] The example `tts/qwen3_tts` names an upstream-only directory. The crate has `src/tts/qwen3` and no `qwen3_tts`. Illustrative only, low severity.

## crates/audio/EGM.md

Coverage: full read (file is short); Cargo.toml exclude and dependency claims and child links checked
- [crates/audio/EGM.md:Subdirectories & Boundaries] Omits the `tests/` directory (`dsp_pipeline.rs`, `mel_roformer_boundary.rs`, `qwen3_metadata.rs`, `wav_roundtrip.rs`, plus `fixtures/` and `reference/`). Low severity.

## crates/audio/examples/EGM.md

Coverage: sampled (all 13 example files and run commands checked, `--release` headers checked, file-write side effects checked by grep; example bodies not read in full)
No drift detected.

## crates/audio/scripts/EGM.md

Coverage: sampled (generator, benchmark and NeMo file lists checked against disk; `benchmark_music3.py` subprocess calls checked; generator bodies not read)
- [crates/audio/scripts/spectral_distance.py:__main__] The Verification command `python3 -I crates/audio/scripts/spectral_distance.py --help` does not print help. The script has no argparse. `__main__` calls `main(*sys.argv[1:3])`, so `--help` is passed as `ref` and `main` would raise TypeError for the missing `cand` argument (inferred from reading the code, not executed). Usage per the module docstring is `REFERENCE.wav CANDIDATE.wav`.

## crates/audio/scripts/nemo_218d0b4/README.md

Coverage: full read (README is short); pyproject.toml, uv.lock, pinned commit, darwin arm64 marker, subcommands, script files and referenced paths checked
No drift detected.

## crates/audio/src/EGM.md

Coverage: sampled (module lists, about 50 public signatures, `#![forbid(unsafe_code)]`, `extern crate self`, `AudioModelError` alias, `in_pool` and quant/nn exports checked; not every source file read)
- [crates/audio/src/fft.rs:RealFftPlan::cached], [crates/audio/src/mel.rs:mel_projector_cached], [crates/audio/src/resample.rs] Side Effects lists only `ops::in_pool`, but these functions use process-wide caches (`static PLANS: TableCache`, `static BANKS: TableCache`, `static KERNELS: TableCache`). The caches persist across calls and are not documented.
- [crates/audio/src/wav.rs:MAX_WAV_DATA_BYTES] Omits the 1 GiB WAV data cap from the WAV read invariants.
- Interface omits the public Whisper constants and builders `whisper_mel_filterbank` and `whisper_mel_projector` [crates/audio/src/whisper.rs], and ops helpers such as `silu`, `gelu_erf`, `sdpa_strided`, `mha`, `rope_tables` [crates/audio/src/ops.rs]. Low severity.

## crates/audio/src/codec/EGM.md

Coverage: sampled (`pub mod` list, `REFERENCE_COMMIT` pattern, shared-block signatures, visibility of `CodebookIndex`/`ResidualUnit`/`EncoderBlock`/`DecoderBlock`, and child EGM links checked; family internals not read)
No drift detected.

## crates/audio/src/codec/README.md

Coverage: sampled (all 17 family links resolve and the directory list matches the disk; family claims checked only through their own READMEs)
- Scope overstated: the intro says the directory holds "neural audio codecs, vocoders, and discrete audio tokenizers", but it lists ECAPA-TDNN, a speaker-embedding network (its own EGM calls it a "speaker encoder") [crates/audio/src/codec/ecapa_tdnn/EGM.md], and DACVAE, which its README says has no discrete codes [crates/audio/src/codec/dacvae/README.md:DACVAE]. Minor.

## crates/audio/src/codec/bigvgan/EGM.md

Coverage: sampled (source is 775 lines across `mod.rs` and `tests.rs`, not read in full; targeted reads of `open`/`load`, activation check, config errors, and env use)
No drift detected.

## crates/audio/src/codec/bigvgan/README.md

Coverage: sampled (test names, fixture tool and testdata path, `TURBOSPARK_BIGVGAN_MODEL`, snakebeta refusal, and upsample product checked; numeric claims not rerun)
No drift detected.

## crates/audio/src/codec/dacvae/EGM.md

Coverage: sampled (interface, `validate` checks, and `open` inputs verified; `BadConfig` confirmed, `Input` and `Tensor` variants not checked; source not read in full)
No drift detected.

## crates/audio/src/codec/dacvae/README.md

Coverage: sampled (default geometry, hop length, test list, env var, and watermark omission checked; padding and chunking claims not rerun)
- [crates/audio/src/codec/dacvae/tests.rs] The README says `codec::dacvae` runs "3 tests", but tests.rs has 5 `#[test]` functions: config_defaults_match_reference, tiny_end_to_end_matches_reference, real_checkpoint_gate, config_rejects_degenerate_geometry, load_revalidates_hand_built_config.

## crates/audio/src/codec/descript/EGM.md

Coverage: sampled (all listed `Dac` methods exist, `decode_codes` count and index checks verified; source not read in full)
- [crates/audio/src/codec/descript/tests.rs:real_checkpoint_gate] Omits the env var `TURBOSPARK_DAC_MODEL` used by the real-checkpoint gate. Test-only, low severity.

## crates/audio/src/codec/descript/README.md

Coverage: sampled (test names, dilations, codebook geometry, and normalization checked against code; stride defaults and consumer list checked against code)
- [crates/audio/src/codec/descript/mod.rs:DacConfig::from_json] The table gives 44.1 kHz strides as [2, 4, 8, 8], but when a config omits them the code defaults `encoder_rates` to [2, 4, 5, 8] (hop 320) and `decoder_rates` to [8, 5, 4, 2] while defaulting `sample_rate` to 44100. Either the default is wrong for 44.1 kHz or the table describes only checkpoints. Resolve against the mlx-audio reference.
- The README says DAC serves "Zonos, VoxCPM, Chatterbox". In this crate the `codec::descript` consumers are dia, outetts, zonos2 and dialogue_sidon. VoxCPM and Chatterbox do not reference it (crates/audio/AGENTS.md says Chatterbox uses `s3`). May be upstream-only; not verified against mlx-audio.

## crates/audio/src/codec/ecapa_tdnn/EGM.md

Coverage: sampled (interface, Res2Net divisibility check with BadConfig, and reference commit verified; source not read in full)
No drift detected.

## crates/audio/src/codec/ecapa_tdnn/README.md

Coverage: sampled (layer specs, Res2Net scale, SE width, and test names checked against defaults; checkpoint values not verified)
- [crates/audio/src/codec/ecapa_tdnn/mod.rs:EcapaTdnnConfig::from_json] The architecture table states a fixed "192 dimensions" speaker embedding, but the code defaults `embed_dim` to 256. If the checkpoint sets 192, the table should say the value is checkpoint-specific.
- Scope: this README sits under codec, which the codec README describes as codecs and vocoders (see the codec README item above).

## crates/audio/src/codec/encodec/EGM.md

Coverage: sampled (claims verified 10 of 11, source not fully read)
No drift detected.

## crates/audio/src/codec/encodec/README.md

Coverage: sampled (claims verified 7 of 9, source not fully read)
- Architecture table says latent hop 320 "(for [8, 5, 4, 2] strides)" and 75 fps as the deployed value. The code fallback is `upsampling_ratios` [8, 8, 4, 2] [crates/audio/src/codec/encodec/mod.rs:EncodecConfig::from_json], and `hop_length` is the product of those ratios [mod.rs:EncodecConfig::hop_length], i.e. 512 (about 46.9 fps). 320 only holds if the checkpoint config.json supplies [8, 5, 4, 2] (not checked here). Label the row as checkpoint config or fix the fallback doc.
- Verification says `real_checkpoint_gate` "runs when the full checkpoint is available locally". It actually runs only when env `TURBOSPARK_ENCODEC_MODEL` is set [crates/audio/src/codec/encodec/tests.rs:real_checkpoint_gate]. The variable is not named in the doc.

## crates/audio/src/codec/fish_s1_dac/EGM.md

Coverage: sampled (claims verified 20 of 23, source not fully read)
- Invariants and Side Effects omit a silent fallback. [crates/audio/src/codec/fish_s1_dac/mod.rs:FishS1Dac::open] uses `FishS1DacConfig::build_ae()` when `config.json` is missing or does not parse (both steps are `if let Ok`), so a malformed config.json does not surface as `SpeechError` as the Invariants bullet implies. Document the fallback.

## crates/audio/src/codec/fish_s1_dac/README.md

Coverage: sampled (claims verified 12 of 13, source not fully read)
- Scope and Divergences says `encode_zq` / `decode_zq` / `from_latents` are not ported. `encode_zq` and `decode_zq` exist [crates/audio/src/codec/fish_s1_dac/mod.rs:FishS1Dac::encode_zq] [mod.rs:FishS1Dac::decode_zq], with tests [tests.rs:encode_zq_decode_zq_roundtrip_matches_shapes] and [tests.rs:decode_zq_validates_lengths]. `from_latents` has no match in the crate, so only that part of the claim holds. EGM.md already lists `encode_zq` / `decode_zq` as interface, so the README is the stale doc.

## crates/audio/src/codec/higgs_audio/EGM.md

Coverage: sampled (claims verified 12 of 12 checked, source not fully read)
No drift detected.

## crates/audio/src/codec/higgs_audio/README.md

Coverage: sampled (claims verified 9 of 11, source not fully read)
- Scope and Divergences says the wav2vec2 semantic tower and the sinc resampler are not ported, and that semantic fusion is future STT work. The code implements the encode path: [crates/audio/src/codec/higgs_audio/mod.rs:HiggsAudioTokenizer::load_encode_path] loads `crate::stt::wav2vec::SemanticTower`, `SemanticEncoder`, and the `fc` fusion when `semantic_model_config` is present, and [mod.rs:HiggsAudioTokenizer::encode_tokens] runs the full reference encode through [semantic.rs:sinc_resample]. Only checkpoints without `semantic_model_config` return `Unsupported`. The "Encode" overview and the "Scope" bullets are stale. Whether an encode_tokens parity fixture exists was not checked; the Verification section names none.

## crates/audio/src/codec/mimi/EGM.md

Coverage: sampled (claims verified 9 of 9 checked, source not fully read)
No drift detected.

## crates/audio/src/codec/mimi/README.md

Coverage: sampled (claims verified 8 of 9, source not fully read)
- Verification says "cargo test -p turbospark-audio --lib codec::mimi (4 tests)". [crates/audio/src/codec/mimi/tests.rs] has 6 `#[test]` functions (factory defaults, tiny end-to-end, isolated transformer, real-checkpoint gate, attention parity, decode refusal). The count is stale.

## crates/audio/src/codec/mimo_audio_tokenizer/EGM.md

Coverage: sampled (claims verified 9 of 9 checked, source not fully read)
No drift detected.

## crates/audio/src/codec/mimo_audio_tokenizer/README.md

Coverage: sampled (claims verified 10 of 10 checked, source not fully read)
No drift detected.

## crates/audio/src/codec/moss_audio_tokenizer/EGM.md

Coverage: sampled (claims verified 9 of 9 checked, source not fully read)
No drift detected.

## crates/audio/src/codec/moss_audio_tokenizer/README.md

Coverage: sampled (claims verified 7 of 8, source not fully read)
- Parameters table, headed "(default config)", gives RLFQ codebook dim 32. The code fallback is `codebook_dim` 8 [crates/audio/src/codec/moss_audio_tokenizer/mod.rs:MossConfig::from_json] (`qu("codebook_dim", 8)`). The 32 comes only from the checkpoint config (not checked). Either relabel the table as checkpoint values or correct the default doc.

## crates/audio/src/codec/nemotron_voicechat/EGM.md

Coverage: sampled (claims verified 8 of 8 checked, source not fully read)
No drift detected.

## crates/audio/src/codec/nemotron_voicechat/README.md

Coverage: sampled (claims verified 9 of 10, source not fully read)
No drift detected.

## crates/audio/src/codec/s3/EGM.md

Coverage: sampled (claims verified 10 of 10 checked, source not fully read)
No drift detected.

## crates/audio/src/codec/s3/README.md

Coverage: sampled (claims verified 9 of 9 checked, source not fully read)
No drift detected.

## crates/audio/src/codec/snac/EGM.md

Coverage: sampled (claims verified 9 of 9 checked, source not fully read)
No drift detected.

## crates/audio/src/codec/snac/README.md

Coverage: sampled (claims verified 10 of 10 checked, source not fully read)
No drift detected.

## crates/audio/src/codec/spark_bicodec/EGM.md

Coverage: sampled (claims verified 9 of 9 checked, source not fully read)
No drift detected.

## crates/audio/src/codec/spark_bicodec/README.md

Coverage: sampled (claims verified 10 of 10 checked, source not fully read)
No drift detected.

## crates/audio/src/codec/stepaudio2/EGM.md

Coverage: sampled (every named symbol in the EGM checked, open/load/decode/prepare_prompt bodies read; mod.rs and the other submodules not fully read, about 3.7k lines)
- No drift detected.
- Note (code, not doc): the doc comment on [mod.rs:StepAudio2Token2Wav::open] describes a converted per-submodel layout (flow.safetensors, hift.safetensors, campplus.safetensors) that the same function rejects with BadConfig. The EGM correctly says converted loading is not implemented.

## crates/audio/src/codec/stepaudio2/README.md

Coverage: sampled (checked the commit pin, the 0.99 clip [hift.rs:AUDIO_LIMIT], leaky slope 0.01 [hift.rs], upsample rates [hift.rs:UPSAMPLE_RATES], dit_depth 16 [mod.rs:Config defaults], inference_cfg_rate 0.7 [mod.rs], the file references; 10-step Euler, n_fft/hop values and batch-of-one scope not verified)
- [tests.rs:real_checkpoint_smoke] The README says the real checkpoint smoke gate is "skipped unless the directory exists". The test is gated on the environment variable TURBOSPARK_STEPAUDIO2_MODEL being set, not on a directory check, and it then opens flow.safetensors and hift.safetensors directly, the split layout that [mod.rs:StepAudio2Token2Wav::open] refuses.

## crates/audio/src/codec/vocos/EGM.md

Coverage: sampled (all named interface symbols and the open/config.yaml/model.safetensors read path checked; mod.rs (1118 lines) and tests.rs not fully read)
- No drift detected.

## crates/audio/src/codec/vocos/README.md

Coverage: sampled (architecture constants 512/1536/8 layers, FFT 1024 / hop 256, exp magnitude clamp at 1e2 [mod.rs:decode] and the three test names checked; the ConvNeXt block order not verified)
- [tests.rs:real_checkpoint_gate] The real checkpoint gate is opt-in through the environment variable TURBOSPARK_VOCOS_MODEL, which the README does not name. It says only "when present".

## crates/audio/src/codec/wavvae/EGM.md

Coverage: sampled (all named interface symbols, the open() file reads, the has_encoder/Unsupported encode guard, and the [-30, 20] logvar clamp checked; mod.rs (895 lines) not fully read)
- No drift detected.

## crates/audio/src/codec/wavvae/README.md

Coverage: sampled (960-sample hop, [-30, 20] clamp, decoder-only Unsupported path, env var names, Rust API signatures checked; SEANet/upsample geometry and the test tolerances not verified)
- [tests.rs] The tolerance override environment variable TURBOSPARK_WAVVAE_TOL is read by the env-gated real-checkpoint test but is not documented in the Verification section.

## crates/audio/src/lid/EGM.md

Coverage: full read (lid/mod.rs is 3 lines and was read in full)
- [lid/mod.rs] The Verification command `cargo test -p turbospark-audio lid` runs no tests of its own: the module has no test code, so the claim is only that a name filter matches nothing. Overstated as verification.

## crates/audio/src/lid/README.md

Coverage: full read of the README; the link target ../../MODELS.md#lid resolves to crates/audio/MODELS.md ("## LID" heading) and the family list is future work described as targets
- No drift detected.

## crates/audio/src/lm/EGM.md

Coverage: sampled (every named interface symbol and type checked; invariants checked for sliding_window refusal, num_key_value_heads, cohere2 global layer index, bailing expert_bias and routed_scaling_factor, switch_layers negative index, Execution RefCell keyed by pointer, device attention expect, falcon 5 multipliers and d_ssm check, rope linear factor and mrope sections; the proportional odd-count rope check and the MoE slot-order sort not read; about 6.6k lines, not fully read)
- No drift detected.

## crates/audio/src/lm/README.md

Coverage: sampled (module file table, the 16 testdata cases [testdata/lm], gpt2 additive constructors, falcon/cohere2 and window behaviour cross-checked against the EGM; the "34 of 40 families" claim and the measured parity numbers were not verified)
- No drift detected.

## crates/audio/src/models/EGM.md

Coverage: full read (models/mod.rs, 28 lines)
- No drift detected.
- Note (code, not doc): the module doc in [models/mod.rs] lists "MiniMax Music 0.5" while the music family is MiniMax Music 3 ([music/mod.rs], [music/minimax_music3]).

## crates/audio/src/music/EGM.md

Coverage: full read (music/mod.rs, 6 lines)
- No drift detected.

## crates/audio/src/music/README.md

Coverage: full read of the README; the link ../../MODELS.md#music resolves to crates/audio/MODELS.md ("## MUSIC" heading) and the family link resolves
- No drift detected.

## crates/audio/src/music/minimax_music3/EGM.md

Coverage: sampled (every named interface symbol, the constants (SAMPLING_RATE 44100, FRAME_RATE 25, MAX_PROMPT_TOKENS 5000, MAX_AUDIO_FRAMES 9000, AR_CFG_SCALE 1.5, AR_CFG_TOP_K 50, DIT_CFG_SCALE 1.7, step counts 30/21/8, chunk and crop constants), the internal stage types and the env var TURBOSPARK_MUSIC3_BENCH_TREES checked; about 13.9k lines, not fully read)
- [mod.rs:Progress] The Interface lists Progress but not its variants DepthStep, FlowStep and VocoderStage, which are public (minor omission).

## crates/audio/src/music/minimax_music3/README.md

Coverage: sampled (checked the module map and file paths, the text frontend defaults (60 s, 30 steps, seed 0; duration cap 360 s; steps 1..=30), the tokenizer path, the Fast/AB2/21-step default, Model opening in Exact, CLI flags, music3_bench default Exact and its --flow-profile values, the tiny_long bench tree, and about 25 named test functions; numeric benchmark and parity claims not re-measured, as they are measured results)
- [rng.rs:normal] The Divergences section says rng::normal evaluates erfinv through "an f64 Acklam inverse-normal approximation". The code calls ppnd16, the Wichura AS241 inverse normal CDF in f64, and the rng.rs module doc comment says AS241. Acklam is a different approximation (used by a separate helper in rng.rs).
- [mod.rs:Progress] The Rust API section describes only Tokenized, ArFrame and FlowChunk. The enum also has DepthStep, FlowStep and VocoderStage (minor omission).

## crates/audio/src/ops/EGM.md

Coverage: sampled (every named reference function and assert_bits_eq confirmed present in kernel_parity.rs; the #[cfg(test)] mod declaration confirmed at ops.rs; kernel_parity.rs (1187 lines) not read)
- No drift detected.

## crates/audio/src/sts/EGM.md

Coverage: full read (sts/mod.rs, 15 lines; the re-export list and the five subdirectory EGM links checked)
- No drift detected.

## crates/audio/src/sts/README.md

Coverage: full read of the README and sts/mod.rs; the five family README links and the family list match the module list (family claims themselves not verified)
- No drift detected.

## crates/audio/src/sts/deepfilternet/EGM.md

Coverage: sampled (claims verified 5 of 5 checkable; mod.rs is about 1200 lines, targeted reads only)
No drift detected.

## crates/audio/src/sts/deepfilternet/README.md

Coverage: sampled (claims verified 6 of 9; benchmark and witness numbers are not checkable in the tree)
- [crates/audio/src/sts/deepfilternet/README.md] line 51 cites a "DeepFilterNet v3 local witness" (waveform correlation 0.964, max abs error 0.021). No test or fixture in the tree backs these numbers: there is no testdata/deepfilternet directory and the module has one #[test] (mod.rs, cfg(test) at line 1204).
- [crates/audio/src/sts/deepfilternet/mod.rs:open] README line 40 says "Supported Versions: DeepFilterNet v1, v2, and v3". The code documents the v3 conversion with v2 loading through the same loader (mod.rs:727-728). Nothing in the code handles v1. The v1 claim is overstated.
- README lines 59-61 PESQ/POLQA quality claims have no in-tree evidence (benchmark claim, not a code divergence).

## crates/audio/src/sts/dialogue_sidon/EGM.md

Coverage: sampled (interface names, error variants, open() file reads and the test command checked; mod.rs is about 1200 lines, not fully read)
No drift detected.

## crates/audio/src/sts/dialogue_sidon/README.md

Coverage: sampled (fixture file list checked against testdata/dialogue_sidon fully; frontend framing, 0.9 peak normalization, mask[1::2] pairing and batch-1 checked; parity table is a run result and not re-checked)
No drift detected.

## crates/audio/src/sts/mel_roformer/EGM.md

Coverage: sampled (interface names, read_dir in open(), config requirement checked; panic sites checked by targeted grep)
- [crates/audio/src/sts/mel_roformer/mod.rs:BandGeometry::build] The invariant says bad config surfaces as SpeechError "not as panics [mod.rs:load]". MelRoFormer::load calls BandGeometry::build (mod.rs:589), which calls `.expect("slaney filterbank")` (mod.rs:228). A filterbank rejection therefore panics inside load instead of returning SpeechError. Reachability depends on parameters that config validation may not reject; not exercised.

## crates/audio/src/sts/mel_roformer/README.md

Coverage: sampled (fixture list, to_qkv refusal, STFT defaults checked; stage-level numbers are run results)
- [crates/audio/src/sts/mel_roformer/mod.rs:load] README "Fixture files" lists `stft_real/imag.npy`. testdata/mel_roformer/ holds flat `stft_real.npy` and `stft_imag.npy` with no subdirectory, so the reference does not resolve.
- The `to_qkv` refusal (mod.rs:648-653), n_fft 2048 and hop 441 defaults (mod.rs:78-79) match the README.

## crates/audio/src/sts/mossformer2_se/EGM.md

Coverage: sampled (interface names, open() config fallback and model.safetensors read checked; mod.rs is about 1400 lines, targeted reads only)
No drift detected.

## crates/audio/src/sts/mossformer2_se/README.md

Coverage: sampled (fixture list, 8-bit group-64 quantization, dither refusal, TURBOSPARK_MOSSFORMER2_SE_MODEL gate, enhance thresholds checked; real-checkpoint numbers are run results)
- [crates/audio/testdata/mossformer2_se] README "Fixture files" lists `stft_real/imag.npy`. The directory holds flat `stft_real.npy` and `stft_imag.npy`, so the reference does not resolve. Also `masked_real.npy` and `masked_imag.npy` exist but are not listed (minor omission).
- [crates/audio/src/sts/mossformer2_se/mod.rs:kaldi_fbank] dither refusal, quantization acceptance and the 60 s auto-chunk threshold (mod.rs:83) match the README.

## crates/audio/src/sts/sam_audio/EGM.md

Coverage: sampled (all listed interface names, config and weight-file selection in open(), OdeOptions default checked; mod.rs is about 1500 lines, targeted reads only)
No drift detected.

## crates/audio/src/sts/sam_audio/README.md

Coverage: sampled (fixture file list checked against testdata/sam_audio fully; ODE defaults and T5 bucket defaults checked; parity table is a run result)
No drift detected.

## crates/audio/src/vad/EGM.md

Coverage: full read (mod.rs is 409 bytes; re-exports checked against the file)
No drift detected.

## crates/audio/src/vad/README.md

Coverage: sampled (link targets, speaker counts, Silero branch layout checked)
No drift detected.

## crates/audio/src/vad/nemotron_diarization/EGM.md

Coverage: sampled (every listed interface name checked; coupled and streaming bodies not fully read)
- [crates/audio/src/vad/nemotron_diarization/mod.rs:SpeakerConfig::multitalker_coupled] The Interface list says `NemotronDiarization::multitalker_coupled`. The method is on `SpeakerConfig` (mod.rs:1305-1311), not on `NemotronDiarization`.

## crates/audio/src/vad/nemotron_diarization/README.md

Coverage: sampled (layer count, subsampling, speaker count, coupled geometry, stage-fixture test names and tolerances checked; the "mean of each 8-frame group" step was not traced)
- [crates/audio/src/vad/nemotron_diarization/mod.rs:coupled_chunk_preds_match_nemo] The README says coupled predictions match NeMo "within 2.3e-6". The test asserts with tolerance 1e-4 (mod.rs:2026), so 2.3e-6 is an observed value, not an enforced gate. Overstated as a gate.
- README "Use weights converted from the pinned .nemo ... (scripts/convert_multitalker_nemo.py convert-diar)". The path does not resolve from the repo root; the file is crates/audio/scripts/convert_multitalker_nemo.py.
- The "no 0.5 decision flips" and 1e-4 Metal tolerances match crates/runtime/tests/nemotron_diar_metal.rs (lines 99, 135-153).

## crates/audio/src/vad/silero_vad/EGM.md

Coverage: sampled (all listed method names checked; feed() takes &mut state; load() reads only model.safetensors)
No drift detected.

## crates/audio/src/vad/silero_vad/README.md

Coverage: sampled (branch layout, 128-wide LSTM, 512/256 chunk sizes checked; benchmark and parameter counts not checkable in-tree)
- README "Parity validated against Python MLX reference across 188 streaming audio chunks; max probability abs error 4.8e-7; timestamps 100% identical" has no in-tree backing: there is no testdata/silero_vad directory, and the in-module parity test (mod.rs, around lines 903-915) compares against an inline reference branch, not these numbers.
- CPU throughput (>5,000x real time) and memory (<15 MiB) claims have no in-tree measurement.

## crates/audio/src/vad/sortformer/EGM.md

Coverage: sampled (all listed names checked; streaming bodies not fully read)
- [crates/audio/src/vad/sortformer/mod.rs:Sortformer] The Interface list omits public streaming API that exists on `impl Sortformer` (mod.rs:2282 onward): `feed` (mod.rs:2431), `generate_stream` (mod.rs:2497), `streaming_step` (mod.rs:2323), `set_streaming_profile` (mod.rs:2665), `config` (mod.rs:1027), `frame_duration` and `streaming_latency` (mod.rs:2652, 2659).

## crates/audio/src/vad/sortformer/README.md

Coverage: sampled (AOSC and golden-fixture tests, their tolerances, and the example targets checked; RTF is not checkable in-tree)
- [crates/audio/src/vad/sortformer/mod.rs:aosc_compression_matches_reference_fixture] The README says AOSC compression "matches reference fixture byte-for-byte". The test compares with a tolerance of 1e-4 (mod.rs:3190-3193), not byte equality.
- README "FastConformer subsampling, relative self-attention, and Bart encoder outputs match golden fixtures within 1e-5". No test checks those stages against golden fixtures. The golden-fixture tests are mel_frontend_matches_reference (1e-3, mod.rs:3089), aosc_compression_matches_reference_fixture (1e-4) and offline_forward_matches_reference (1e-5 on final predictions only, mod.rs:3201-3222). The relative-attention tests (rel_shift_known_answer, direct_relative_scores_match_dense_shift) are known-answer checks, not fixture comparisons. Overstated.
- The RTF ~0.05 on M4 Max claim has no in-tree measurement.

## crates/audio/tools/EGM.md

Coverage: sampled (every named generator, converter and probe file exists; OUT path and --real/local_files_only behavior checked in gen_snac_fixtures.py; generator bodies not read; test command not run)
No drift detected.

## crates/audio/src/stt/EGM.md

Coverage: sampled (module list and re-exports checked against stt/mod.rs, which was read in full; child EGM links all resolve)
No drift detected.

## crates/audio/src/stt/README.md

Coverage: sampled (all relative links to family READMEs, MODELS.md, docs/COREML.md, nemo/mod.rs and the five example files resolve; server route POST /v1/audio/transcriptions, WebSocket GET /v1/audio/transcriptions/realtime and --stt-model checked; family prose not verified)
- Parakeet Unified streaming is attributed to `ts_audio_stream_*` in crates/runtime/src/coreml/parakeet_unified.rs, but that file defines no `ts_audio_stream_` symbol. The C ABI is in crates/ffi/src/api/audio_stream.rs and the stream dispatch is in crates/runtime/src/native_audio/stream.rs and crates/runtime/src/coreml/mod.rs. [crates/runtime/src/coreml/parakeet_unified.rs] citation drift

## crates/audio/src/stt/canary/EGM.md

Coverage: sampled (interface symbols, config refusals in config.rs:from_json, tokenizer prompt/eos/decode, transcribe_impl empty-input guard, env var in tests, fixture files; decoder internals not read in full)
No drift detected.

## crates/audio/src/stt/canary/README.md

Coverage: sampled (profile constant, max_tokens default 200, load_canary reuse, prompt tokens, fixture and script paths, catalog row; checkpoint sizes, timings and parity numbers not re-run)
- "Runtime, catalog, FFI, Swift. No SpeechFamily entry, catalog row, install probe, ts_stt_* ABI dispatch, or Swift selection exists for this family yet." The catalog row exists: crates/catalog/src/audio_families.json has family "stt/canary" with source crates/audio/src/stt/canary/mod.rs. The ts_stt_* and Swift halves were not checked. [crates/catalog/src/audio_families.json:stt/canary] false claim

## crates/audio/src/stt/cohere_asr/EGM.md

Coverage: full read (EGM and mod.rs, 110 lines)
No drift detected.

## crates/audio/src/stt/cohere_asr/README.md

Coverage: full read of the README and source; the Hub 403 and mirror-output investigation record is external evidence and was not re-verified
No drift detected.

## crates/audio/src/stt/fireredasr2/EGM.md

Coverage: sampled (config invariants odim/d_model/kernel/position limits, decoder option defaults, beam and max_len rules, <space> and U+2581 mapping in decoder.rs:decode, fixture test name; encoder and beam internals not read in full)
No drift detected.

## crates/audio/src/stt/fireredasr2/README.md

Coverage: sampled (pinned profile dims, decoder option defaults, test name, output normalization; revision string and fixture digests not compared byte-for-byte)
No drift detected.

## crates/audio/src/stt/fun_asr_nano/EGM.md

Coverage: sampled (FUN_ASR_NANO_2512 revision, config geometry checks, 4-bit/group-128 decoder check, stop ids, fake_token_length, max_tokens guard, env var in tests only; stack_lfr internals not read)
No drift detected.

## crates/audio/src/stt/fun_asr_nano/README.md

Coverage: sampled (revision, geometry, tied 1024-wide decoder check, script and fixture paths, ignored test name; parity tolerances and artifact digests not re-run)
No drift detected.

## crates/audio/src/stt/glmasr/EGM.md

Coverage: sampled (model_type and adapter checks, 4-bit affine group-64 check, merge and 1500-frame limits, default 128 tokens, 30 s limit and 1 s zero-pad, stop and prompt token names, RoPE 32 dims, Conv1d layout, text config constants; decoder internals not read)
- "The Llama text config is hardcoded (...) geometry is validated [mod.rs:Config::from_json]": the hardcoded TextConfig is in Config::from_json (mod.rs:118-126), but the check that the loaded decoder matches hidden size and vocab is in GlmAsr::load ("Llama checkpoint geometry mismatch"), not in from_json. [crates/audio/src/stt/glmasr/mod.rs:GlmAsr::load] citation drift

## crates/audio/src/stt/glmasr/README.md

Coverage: sampled (revision, text decoder geometry, script and fixture paths, ignored test name, 30 s limit; encoder widths and parity tolerances not re-run)
No drift detected.

## crates/audio/src/stt/granite_speech5_ctc/EGM.md

Coverage: full read of the EGM; source sampled (from_json checks, feature bounds and 2^27 cap, HTK f64 filterbank, LOGIT_ROW_BATCH, midpoint and CTC collapse, u32 token conversion, trimmed decode; EncoderBlock internals not read in full)
No drift detected.

## crates/audio/src/stt/granite_speech5_ctc/README.md

Coverage: sampled (pinned repository and revision, periodic Hann via dsp::hann_window, 8 dB floor, pair stacking, subsample layers, tied CTC head, fixture script path; transcript and checkpoint sizes not re-run; replicate-padded delta claim not verified)
No drift detected.

## crates/audio/src/stt/granite_speech/EGM.md

Coverage: sampled (profile constants, prompt constants and LANGUAGE_CODES, lora and bias_packed refusals, max_tokens 0 mapping to 256 and zero-token error, embedding multiplier application, midpoint index, env var in tests only; config refusal list not fully traced)
No drift detected.

## crates/audio/src/stt/granite_speech/README.md

Coverage: sampled (pinned repository, six LANGUAGE_CODES, default ASR prompt text, Plus profile constant, EOS and audio token ids; multiplier values are the ones in config.rs test fixtures, not the pinned config.json, which is not in the repo; file digests not compared)
No drift detected.

## crates/audio/src/stt/granite_speech_nar/EGM.md

Coverage: sampled (from_json architecture and field checks, self_conditioning_layer error, insertion slot length, pooling denominator floor 1e-6, frontend truncation to 2*(n/320), 1e-10 clamp, 8 dB floor, editor token range check, load_tokenizer use; ShawAttention and editor internals not read)
- "encoder.out output_dim must be 348 [mod.rs:ConformerEncoder::forward]": no check enforces output_dim == 348. from_json only requires output_dim to be positive (mod.rs:140), and the self-conditioning softmax and blank-probability code slices a hardcoded 348-wide row (mod.rs:940, 945). A config with another output_dim would slice the wrong width or panic. [crates/audio/src/stt/granite_speech_nar/mod.rs:ConformerEncoder::forward] unenforced invariant

## crates/audio/src/stt/granite_speech_nar/README.md

Coverage: sampled (pinned repository and revision, frontend steps and clamps, periodic Hann via crate::dsp::hann_window, pooling window and insertion rules, self-conditioning and editor scaling; the 1025-entry Shaw table depends on config max_pos_emb and was not checked against the checkpoint)
No drift detected.

## crates/audio/src/stt/higgs_audio_3/EGM.md

Coverage: sampled (STOP_IDS, pinned revision, max_tokens guard, log_mel_128 drop and clamp, parse_output and remove_special_spans, load_with_vad and VAD fallback path, env var in tests only, text_lm_head candidate in qwen3_asr/decoder.rs; quantization refusal and tokenizer id checks not traced line by line)
No drift detected.

## crates/audio/src/stt/higgs_audio_3/README.md

Coverage: sampled (pinned revision, default prompt text, stop and audio token ids 151672/151670, untied LM head candidate in qwen3_asr/decoder.rs:46, max_tokens guard, uniform VAD fallback; VAD thresholds are delegated to the shared silero_vad module and were not traced; file digests and parity numbers not re-run)
No drift detected.

## crates/audio/src/stt/lasr_ctc/EGM.md

Coverage: sampled (config refusals, frontend guards and 1e-5 clamp, FFT power-of-two assert, conv steps assert, ctc_decode rules, argmax tie rule, batch-norm eps, pinned revision, env var in tests; residual default pairs not traced)
- The layer norm epsilon is read from config.json (`layer_norm_eps`, mod.rs:LasrCtcConfig::from_json) and the Layer norm section is absent from the EGM invariants. Omission only. [crates/audio/src/stt/lasr_ctc/mod.rs:LasrCtcConfig::from_json] omitted config input

## crates/audio/src/stt/lasr_ctc/README.md

Coverage: sampled (pinned revision, config refusals, ctc_decode rules, batch-norm eps, subsampler length arithmetic (277 to 67 frames), kaldi-mel and float64 frontend; layer norm eps 1e-6 is a checkpoint value read from config and was not verifiable offline; fixture transcript not re-run)
No drift detected.

Docs covered: 22 of 22. No directory left unfinished.

## crates/audio/src/stt/mega_asr/EGM.md

Coverage: sampled (claims verified about 12 of 17, source not fully read)
No drift detected.

## crates/audio/src/stt/mega_asr/README.md

Coverage: sampled (claims verified about 6 of 10, source not fully read)
- Test count claim "11 always-on tests" does not match source: mega_asr has 13 `#[test]` attributes, 1 of them `#[ignore]`, so 12 always-on [crates/audio/src/stt/mega_asr/config.rs, crates/audio/src/stt/mega_asr/lora.rs, crates/audio/src/stt/mega_asr/mod.rs, crates/audio/src/stt/mega_asr/router.rs]. Low severity, count is likely stale.

## crates/audio/src/stt/mms/EGM.md

Coverage: sampled (claims verified about 8 of 10, source not fully read)
- Invariant says input is "mean/variance normalized with a 1e-7 epsilon". Code divides by (standard deviation + 1e-7), so the epsilon is added to the standard deviation, not the variance [crates/audio/src/stt/mms/mod.rs:Mms::logits, around line 595].

## crates/audio/src/stt/mms/README.md

Coverage: sampled (claims verified about 5 of 8, source not fully read)
- Gated-test command filter `models::stt::mms::tests::pinned_english_checkpoint_matches_mlx_reference` does not match the test. The test's module path is `stt::mms::tests::...` (`crates/audio/src/lib.rs` declares `pub mod stt`; `crates/audio/src/models/mod.rs` only re-exports it), and libtest matches the filter as a substring of the full path, so the command runs 0 tests. Not executed; derived from the module tree [crates/audio/src/stt/mms/mod.rs:pinned_english_checkpoint_matches_mlx_reference].
- Same normalization wording as the EGM: README says "zero mean and unit variance with an epsilon of 1e-7"; the epsilon is added to the standard deviation [crates/audio/src/stt/mms/mod.rs:Mms::logits, around line 595].

## crates/audio/src/stt/moonshine/EGM.md

Coverage: sampled (claims verified about 9 of 10, source not fully read)
No drift detected.

## crates/audio/src/stt/moonshine/README.md

Coverage: sampled (claims verified about 6 of 8, source not fully read)
No drift detected.

## crates/audio/src/stt/moss_music/EGM.md

Coverage: sampled (claims verified about 9 of 12, source not fully read; encoder conv and deepstack layer numbers not checked against source)
No drift detected.

## crates/audio/src/stt/moss_music/README.md

Coverage: sampled (claims verified about 3 of 8, source not fully read; encoder geometry numbers in the overview not checked)
No drift detected.

## crates/audio/src/stt/moss_transcribe_diarize/EGM.md

Coverage: sampled (claims verified about 9 of 11, source not fully read)
No drift detected.

## crates/audio/src/stt/moss_transcribe_diarize/README.md

Coverage: sampled (claims verified about 6 of 9, source not fully read; encoder block count and width not checked)
No drift detected.

## crates/audio/src/stt/multitalker_asr/EGM.md

Coverage: sampled (claims verified about 10 of 14, source not fully read)
No drift detected.

## crates/audio/src/stt/multitalker_asr/README.md

Coverage: sampled (claims verified about 10 of 16, source not fully read; latency and admission-estimate claims not checked)
No drift detected.

## crates/audio/src/stt/nemo/EGM.md

Coverage: sampled (claims verified about 6 of 8, source not fully read)
No drift detected.

## crates/audio/src/stt/nemotron_asr/EGM.md

Coverage: sampled (claims verified about 6 of 9, source not fully read)
No drift detected.

## crates/audio/src/stt/nemotron_asr/README.md

Coverage: sampled (claims verified about 3 of 4, source not fully read)
No drift detected.

## crates/audio/src/stt/parakeet/EGM.md

Coverage: sampled (claims verified about 8 of 10, source not fully read)
No drift detected.

## crates/audio/src/stt/parakeet/README.md

Coverage: sampled (claims verified about 4 of 5, source not fully read)
- Fixture regeneration command `python3 scripts/generate_parakeet_fixtures.py` does not resolve from the repo root. The script is at `crates/audio/scripts/generate_parakeet_fixtures.py` and there is no top-level `scripts/` directory. It resolves only if the base is `crates/audio`, which the README does not say [crates/audio/scripts/generate_parakeet_fixtures.py].

## crates/audio/src/stt/phonon/EGM.md

Coverage: sampled (claims verified about 9 of 11, source not fully read)
No drift detected.

## crates/audio/src/stt/phonon/README.md

Coverage: sampled (claims verified about 6 of 9, source not fully read; materialized file sizes and digests not checked)
No drift detected.

## crates/audio/src/stt/qwen2_audio/EGM.md

Coverage: sampled (claims verified about 9 of 11, source not fully read)
No drift detected.

## crates/audio/src/stt/qwen2_audio/README.md

Coverage: sampled (claims verified about 6 of 8, source not fully read)
No drift detected.

## crates/audio/src/stt/qwen3_asr/EGM.md

Coverage: sampled (claims verified 18 of 22, source not fully read; decoder.rs and encoder.rs read by targeted greps and short ranges)
No drift detected.

## crates/audio/src/stt/qwen3_asr/README.md

Coverage: sampled (claims verified 6 of 25; pinned digests, benchmark numbers and the Metal/runtime figures are not checkable from this directory and were not re-run)
No drift detected in the checkable claims. Referenced scripts, testdata files and the docs/BENCHMARKING.md anchor (`#qwen3-asr-stt-backend-probe`) all resolve.

## crates/audio/src/stt/qwen3_forced_aligner/EGM.md

Coverage: sampled (claims verified 14 of 16, source not fully read)
- ForcedAlignResult is documented as "with `text`", but the struct holds `items: Vec<ForcedAlignItem>` and `text` is a method [mod.rs:ForcedAlignResult, mod.rs:ForcedAlignResult::text]. Minor wording drift.

## crates/audio/src/stt/qwen3_forced_aligner/README.md

Coverage: sampled (claims verified 6 of 10; the fixture digests were recomputed and match, the pinned artifact table was not checked)
No drift detected. Both fixture SHA-256 values (qwen3_forced_aligner.json and qwen3_forced_aligner_reference.wav) match the files on disk. The `models::stt::qwen3_forced_aligner` test path and the generator script resolve. The 9-word and 1 ms match claims were not re-run.

## crates/audio/src/stt/sensevoice/EGM.md

Coverage: sampled (claims verified 17 of 20, source not fully read)
No drift detected. Checked: pinned revision, dimension constants (VOCAB_SIZE 25055, INPUT_SIZE 560, OUTPUT_SIZE 512, 4 heads, LINEAR_UNITS 2048, sanm_shift 0) [mod.rs:Config::load], language ids and refusal [mod.rs:transcribe_impl], normalize_before and hamming gates [mod.rs:Config::load], FBANK constants 0.97 and 1e-8 [frontend.rs:compute_fbank], per-call FFT and mel build [frontend.rs:compute_fbank], SentencePiece kinds 3/5/2/6 [mod.rs:decode_sentencepiece], CTC blank 0 [mod.rs:greedy_ctc], FSMN padding bound [sanm.rs:SanmAttention::load, mod.rs:validate_sanm], MVN width check [mod.rs:parse_mvn], and the pinned test name and env var [mod.rs:pinned_checkpoint_matches_mlx_transcript_and_tags].

## crates/audio/src/stt/sensevoice/README.md

Coverage: sampled (claims verified 5 of 9; the model file size and digest were not checked)
No drift detected. The usage snippet matches the API (`transcribe` returns `Result<String>` [mod.rs:SenseVoiceSmall::transcribe]). The `use_itn` option, the accepted language list, the fixture generator, testdata/sensevoice_reference.json, and the test command all resolve. The reference WAV digest (ea38d350...) matches the file on disk.

## crates/audio/src/stt/vibevoice_asr/EGM.md

Coverage: sampled (claims verified 14 of 16, source not fully read)
No drift detected. Checked: EOS ids 151643/151645 and the 8192-token cap [mod.rs:EOS_TOKEN_IDS, mod.rs:MAX_NEW_TOKENS], model_type, architecture and disable_last_norm refusals [mod.rs:VibeVoiceAsrConfig::from_json], normalize_audio=false gate [mod.rs:from_preprocessor], .scales/.biases refusal [mod.rs:VibeVoiceAsr::load], embed bounds [language_model.rs:Qwen2::embed], lm_head never read and tied logits [language_model.rs:Qwen2::tied_logits], resampler constants [resample_poly.rs:NUM_ZEROS, ROLLOFF, BETA], SConv1d and the other listed helpers, and env var use only in tests [mod.rs:tests].

## crates/audio/src/stt/vibevoice_asr/README.md

Coverage: sampled (claims verified 6 of 12; transcripts, timings and stage-error figures depend on the reference run and were not re-run)
- Regenerate section hard-codes `/Users/whit3rabbit/Documents/GitHub/.venv-mlxaudio/bin/python`, a machine-local absolute path that does not resolve elsewhere. Use a relative or documented venv path. [README.md, Reference and parity section]
Checked and consistent: the 3200-sample stride product from the reversed ratios [tokenizer_encoder.rs:TokenizerEncoder::load], the JSON-shaped default prompt keys [mod.rs:TRANSCRIBE_KEYS], the fixture script and testdata/vibevoice_asr_reference.json paths.

## crates/audio/src/stt/voxtral/EGM.md

Coverage: sampled (claims verified 12 of 14, source not fully read)
No drift detected. Checked: pinned repository and revision [mod.rs:VOXTRAL_MINI_3B_BF16], N_FFT 400, hop 160, 128 mels, 480000-sample target [mod.rs:N_FFT, mod.rs:HOP_LENGTH, mod.rs:TARGET_SAMPLES], silent truncation via Vec::resize [mod.rs:compute_features], Hann window, Slaney mel and the peak-8 floor and (x+4)/4 scaling [mod.rs:compute_features], TOWER_POSITION_ROWS = 1500 [mod.rs:TOWER_POSITION_ROWS, mod.rs:AudioTower::validate_positions], FRAMES_PER_MERGE 4 [mod.rs:FRAMES_PER_MERGE], 128-token greedy cap and stop at id <= 2 [mod.rs:Voxtral::transcribe], the two fixed shard names [mod.rs:Voxtral::load], and the decoder reuse from qwen3_asr [mod.rs:imports].

## crates/audio/src/stt/voxtral/README.md

Coverage: sampled (claims verified 4 of 8; the encoder and decoder dimensions come from config.json and were not checked against a checkpoint)
- "Prompt: ... with pinned English transcription prompt tokens" is wrong for this code. `Voxtral::transcribe` takes caller-supplied `prompt_ids` and the module pins no prompt [mod.rs:Voxtral::transcribe]. The EGM says this correctly.
- "Input: 16 kHz mono f32 PCM up to 30 seconds" omits that longer input is silently truncated to 480000 samples with no error [mod.rs:compute_features]. Also, `transcribe` does not reject empty or non-finite audio [mod.rs:compute_features].
- Verification claims a feature match "within 9e-3 relative tolerance" and byte-identical transcripts. No 9e-3 tolerance appears in voxtral/mod.rs. If this rests on an external run, say so; otherwise the in-tree gate is missing.

## crates/audio/src/stt/voxtral_realtime/EGM.md

Coverage: sampled (claims verified 14 of 17, source not fully read)
No drift detected. Checked: model_type, affine-only and 4-bit group-64 refusals [mod.rs:VoxtralRealtimeConfig::from_json], delay 480 ms, left pad 32, BOS 1, EOS 2, pad 32, and the 4096 cap checked after the push [mod.rs:TRANSCRIPTION_DELAY_MS, mod.rs:N_LEFT_PAD_TOKENS, mod.rs:transcribe], bias layout (wq, wv, wo and w2 biased; wk, w1 and w3 not; decoder none) [encoder.rs:EncoderAttention::load, encoder.rs:AudioEncoder::load, decoder.rs:DecoderAttention::load], precompute_ada_scales called from load [mod.rs:VoxtralRealtime::load], checkpoint shard resolution order [mod.rs:open_checkpoint], non-finite rejection [mod.rs:VoxtralRealtime::prepare], and encode_full's sliding-window error [encoder.rs:AudioEncoder::encode_full].

## crates/audio/src/stt/voxtral_realtime/README.md

Coverage: sampled (claims verified 4 of 9; the stage-parity numbers and the window-boundary test claims were not re-run)
No drift detected in the checkable claims. The 1280-sample token alignment [mod.rs:RAW_AUDIO_LENGTH_PER_TOK], the 1.5 log-mel clamp constant [mod.rs:GLOBAL_LOG_MEL_MAX], the pinned repository and revision [mod.rs:VOXTRAL_MINI_4B_REALTIME_4BIT], and the ring-cache trim ordering [encoder.rs:EncoderAttention::forward_ring or the equivalent trim call at encoder.rs:313] are consistent.

## crates/audio/src/stt/wav2vec/EGM.md

Coverage: sampled (claims verified 12 of 14, source not fully read)
No drift detected. Checked: pinned revision [mod.rs:WAV2VEC2_BASE_960H], from_json model_type, architecture, group norm, do_stable_layer_norm, gelu and adapter_attn_dim refusals [mod.rs:Wav2VecConfig::from_json], from_semantic_dict accepting wav2vec2 or hubert [mod.rs:Wav2VecConfig::from_semantic_dict], normalization with epsilon inside the sqrt [mod.rs:Wav2Vec::transcribe], lm_head loaded with bias [mod.rs:Wav2Vec::load], CtcCollapse advancing `previous` on every frame [ctc.rs:CtcCollapse::push], SemanticTower pub(crate) surface [mod.rs:SemanticTower].

## crates/audio/src/stt/wav2vec/README.md

Coverage: sampled (claims verified 5 of 8; parity figures and the upstream sanitize claims were not re-run)
No drift detected. The epsilon placement claim holds: the mms port divides by (std + 1e-7) outside the sqrt [stt/mms/mod.rs:normalize], which is the contrast the README draws.

## crates/audio/src/stt/whisper/EGM.md

Coverage: sampled (claims verified 10 of 12, source not fully read)
- WHISPER_WINDOW_SAMPLES is cited to [mod.rs:WhisperModel::transcribe], but the constant is defined in crates/audio/src/whisper.rs:26, not in the stt/whisper module. Fix the citation.
No other drift detected. Checked: special token ids [mod.rs:WhisperSpecialTokens], the five named constructors tiny, base, small, large_v3 and distil_large_v3 [mod.rs:WhisperConfig], validate rules and from_json calling validate [mod.rs:WhisperConfig::from_json, mod.rs:WhisperConfig::validate], encode's 3000-frame requirement [mod.rs:WhisperModel::encode], and the WhisperModel surface (new, encode, decode_step, transcribe) [mod.rs:WhisperModel].

## crates/audio/src/stt/whisper/README.md

Coverage: sampled (claims verified 5 of 9; the profile parameter counts were not checked)
- "Decoding: Greedy or beam search decoding" is wrong. Only greedy decoding exists, and "beam" appears nowhere in stt/whisper [mod.rs:WhisperModel::transcribe].
- "Audio Frontend ... Provided by `turbospark_audio::models::whisper`" is wrong. `models::whisper` is a re-export of `stt::whisper` [models/mod.rs:pub use crate::stt::whisper], and that module has no mel code. The Slaney log-mel frontend is `whisper_log_mel` and `whisper_log_mel_window` in crates/audio/src/whisper.rs, exposed as `turbospark_audio::whisper`.
- Profile table lists `whisper-medium` (24 layers, 1024 width, 16 heads). There is no medium constructor in WhisperConfig [mod.rs:WhisperConfig], so the row describes a config that is not built in.
- Decoding bullet implies language and task tokens are selectable. The code hard-codes the English language token (50259) and exposes no language or task option [mod.rs:WhisperModel::transcribe].

## crates/audio/src/stt/whistle/EGM.md

Coverage: sampled (claims verified 16 of 19, source not fully read)
No drift detected. Checked: container tag 0x05e12a84, 28 codebook entries and the 1..=4096 tensor bound [cact.rs:Archive::parse], 64 MiB cap enforced with take(MAX_BYTES + 1) [cact.rs:Archive::read], rope_theta == 100000.0 and the 13-value manifest at tensor 237 [mod.rs:Whistle::load], Hadamard permutations at tensors 226 and 227 [mod.rs:Whistle::load], mel bank [257,80] [frontend.rs:Frontend::new], PCM bounds of 160 to 480000 samples in [-1,1] [frontend.rs:validate_pcm], 99th-percentile gain over 20 ms RMS windows [frontend.rs:level_gain], max_tokens 1..=318 [mod.rs:Whistle::transcribe], greedy argmax over the first 8192 logits [mod.rs:Whistle::transcribe], sqrt(DIM) input embedding scale [decoder.rs:Decoder::step], and engram layers 3 and 7 [decoder.rs:Decoder::step].

## crates/audio/src/stt/whistle/README.md

Coverage: sampled (claims verified 8 of 12; the checkpoint-derived architecture numbers and the oracle and fixture claims were not re-run)
No drift detected in the checkable claims. The example exists (crates/audio/examples/whistle_transcribe.rs), `read_wav_f32` is re-exported at the crate root [lib.rs:read_wav_f32], the `transcribe(pcm, language, max_tokens)` signature matches [mod.rs:Whistle::transcribe], the 681-tensor container and 8199-piece tokenizer match the header and tokenizer parse [mod.rs:Whistle::load], the 8192 text vocabulary matches the logits slice [mod.rs:Whistle::transcribe], the language list matches LANGUAGES, and the fixture manifest, LICENSE-APACHE, NOTICE and generate_reference.py all exist.
- Quickstart and the reference-test instructions write to /tmp/turbospark-whistle-reference. Not drift, but the local CLAUDE.local.md says not to keep irreplaceable test artifacts in /tmp, so consider a repo-local or ~/models path.

## crates/audio/src/tts/EGM.md

Coverage: full read of the doc and tts/mod.rs; every child link, alias, and alias-less module claim checked against mod.rs.
No drift detected.

## crates/audio/src/tts/README.md

Coverage: sampled (family list compared against all child directories; family-level claims checked only against mod.rs; child READMEs not re-verified here).
- Family list omits 20 family directories that exist and are declared in [crates/audio/src/tts/mod.rs:pub mod]: arktts, bark, chatterbox, chatterbox_turbo, confucius4, dia, dramabox, echo_tts, fish_drama3, fish_qwen3_omni, irodori_tts, longcat_audiodit, melotts, omnivoice, pocket_tts, tada, vibevoice, voxcpm, voxcpm2, voxtral_tts. The list shows only 19 of the 39 families.
- Link check: all family README links and the ../../MODELS.md link resolve (anchor #tts not checked). [README.md:links]

## crates/audio/src/tts/arktts/EGM.md

Coverage: sampled (about 20 of 25 claims verified: symbols, defaults in [mod.rs:SamplingParams], exponential race, empty-generation error in [mod.rs:Model::generate], decode-only codec and 1 + residual_books row check in [codec.rs:ArkttsCodec::decode]; codec and mod.rs read in targeted ranges).
No drift detected.

## crates/audio/src/tts/arktts/README.md

Coverage: sampled (about 15 claims verified: sampler defaults and RAS, falcon_h1 seam, Snake strides (8, 8, 4, 2), fixture case directories, 15 arktts tests; measured tolerances not re-run).
- Broken link: the fixture generator link [README.md:tools/gen_arktts_fixtures.py] uses ../../../../, which resolves to crates/tools/ (does not exist). The script is at crates/audio/tools/gen_arktts_fixtures.py, so the relative path should be ../../../tools/gen_arktts_fixtures.py.

## crates/audio/src/tts/bailingmm/EGM.md

Coverage: sampled (about 20 claims verified: exported types in [dit.rs:DiT], [audio_vae.rs:IstftHead], [mod.rs:GenerateParams], empty-input and max_decode_steps rejection in [mod.rs:Model::generate], patch-multiple checks in [mod.rs:check_patch_geometry], stop rule in [mod.rs:Model::generate]).
No drift detected.

## crates/audio/src/tts/bailingmm/README.md

Coverage: sampled (fixture case directories, tools/gen_bailingmm_fixtures.py, stop rule after step 3, module layout; the measured parity tables were not re-run).
No drift detected.

## crates/audio/src/tts/bark/EGM.md

Coverage: sampled (all listed consts present in [mod.rs:SAMPLE_RATE]; generate_coarse discards both histories at [mod.rs:Model::generate_coarse]; semantic stop index [mod.rs:generate_text_semantic]; coarse wiring override in [mod.rs:Model::generate]).
No drift detected.

## crates/audio/src/tts/bark/README.md

Coverage: sampled (fixture generator and testdata path resolve; 1024-row fine windows; ALLOWED_PROMPTS absent from code, consistent with the stated open item; numeric gates not re-run).
No drift detected.

## crates/audio/src/tts/breeze_tts/EGM.md

Coverage: sampled (about 20 claims verified: GenerateParams validate and use_cfg, combine_cfg, mask_reserved_codec_logits, DepthDecoder next_logits, DEFAULT_SPEAKER and DEFAULT_SAMPLE_RATE).
No drift detected.

## crates/audio/src/tts/breeze_tts/README.md

Coverage: sampled (GenerateParams defaults, fixture tool and testdata paths, config alias handling; numeric gates not re-run).
- Names a constructor that does not exist: README says ModelConfig.from_dict, but the parser is [breeze_tts/mod.rs:ModelConfig::from_json]. Only a doc comment still says from_dict.

## crates/audio/src/tts/chatterbox/EGM.md

Coverage: sampled (about 25 claims verified across T3, flow, HiFT, CAMPPlus, voice encoder; prompt_mel reuse in [chatterbox_turbo/mod.rs:prompt_mel]; TURBOSPARK_DEBUG_CONV in [s3gen.rs:SaConv1d::forward]; checkpoint file rule in [mod.rs:checkpoint_file]).
No drift detected.

## crates/audio/src/tts/chatterbox/README.md

Coverage: sampled (fixture tools, testdata path, real_checkpoint_gate and debug_*_stages tests, env vars, CAMPPlus 12/24/16 layers and dilations 1/2/2, voice-encoder constants; numeric gates not re-run).
- Stale scope statement: "chatterbox_turbo is a separate spec task (67) and is not included." The directory exists with its own README and EGM, and reuses this family's prompt_mel [chatterbox_turbo/mod.rs:prompt_mel] (the chatterbox EGM says so too).

## crates/audio/src/tts/chatterbox_turbo/EGM.md

Coverage: sampled (about 20 claims verified: Kaldi fbank front ends [fbank.rs:kaldi_fbank], [fbank.rs:mean_normalized_fbank], GPT-2 medium config [t3.rs:gpt2_medium_config], top-k/top-p helpers [t3.rs:apply_top_k_threshold], S3GEN_SIL 4299 in [mod.rs:S3GEN_SIL], flow frame check and 2-step default in [flow.rs:S3GenFlow::infer] and [flow.rs:FlowConfig], encoder_type check in [t3.rs:T3::load]).
No drift detected.

## crates/audio/src/tts/chatterbox_turbo/README.md

Coverage: sampled (4 post-upsample conformer layers, eight HiFT harmonic phases, no epsilon in the stats pool, GPT-2 medium dimensions, fixture tools and testdata paths; leaky-ReLU slope and the one-zero left pad in the last upsample stage were not confirmed; numeric gates not re-run).
No drift detected.

## crates/audio/src/tts/confucius4/EGM.md

Coverage: sampled (about 20 claims verified: Model method list in [mod.rs:Model], BOS 8192 and EOS 8193 in [t2s.rs:BOS], reference audio shorter than 400 samples refused in [dsp.rs:fbank_160], fixed 26-point solve schedule with cfg 0.7 in [mod.rs:Model::s2a_solve], ReferenceCondition alias, language_token).
No drift detected.

## crates/audio/src/tts/confucius4/README.md

Coverage: sampled (22050 output rate in [mod.rs:ModelConfig] sample_rate, 13-block DiT in [s2a.rs:S2AConfig] depth, 400-sample frame, resampler and codec links, test count; numeric gates not re-run).
- Stale test count: the README says "cargo test --lib confucius4 (6 tests)", but the family directory has 8 #[test] functions [confucius4/tests.rs].

## crates/audio/src/tts/dia/EGM.md

Coverage: sampled (about 20 claims verified: Model methods in [mod.rs:Model], Sampled::new signature, KEEP_VOCAB and EXTRA_STEPS_AFTER_EOS, self_cache_capacity, dtype refusal in [config.rs:DiaConfig::from_json], KvStore asserts in [layers.rs:KvStore::prefill] and [layers.rs:KvStore::update], validate checks in [mod.rs:Model::validate], pub(crate) visibility of Encoder/Decoder and audio helpers).
No drift detected.

## crates/audio/src/tts/dia/README.md

Coverage: sampled (sampler defaults 1.3 / 0.95 / top-k 35, delay pattern [0, 8, ..., 15] in [config.rs:DataConfig], round-to-128 rule in [config.rs:round_128], first-frame and final-30-frame trim, fixture tools and testdata paths; the fixture numbers were not re-run).
No drift detected.

## crates/audio/src/tts/dramabox/EGM.md

Coverage: sampled (about 20 claims verified: GenerateParams defaults 2.5 / 1.5 / 29 / 30 in [mod.rs:GenerateParams::default], decode_waveform_refusal, Model methods, sigma 0 refusal in [diffusion.rs:to_velocity], pub(crate) items, DRAMABOX_MEASURE in tests.rs).
No drift detected.

## crates/audio/src/tts/dramabox/README.md

Coverage: sampled (fixture tool and testdata, DRAMABOX_MEASURE, STG and AdaLN gotchas, causal patch-grid bound in [diffusion.rs:ltx2_sigmas] comment, decode refusal; the duration-rounding gotchas and the sigma shift source were not traced in full).
No drift detected. The dramabox-tts alias is listed in crates/audio/MODELS.md, not in the code; this is registry text, not a code claim.

## crates/audio/src/tts/echo_tts/EGM.md

Coverage: sampled (about 20 claims verified: config defaults in [config.rs:EchoDiTConfig], Model methods and GenerateControls::from_config, PCA and flattening helpers in [audio.rs], text helpers in [text.rs], DEFAULT_TRUNCATION_FACTOR, 2048-sample crop in [audio.rs:crop_audio_to_flattening_point], no env vars).
No drift detected.

## crates/audio/src/tts/echo_tts/README.md

Coverage: sampled (config defaults including 40 steps, 640 sequence length, 2048 crop, speaker chunk default 640 * downsample in [mod.rs:GenerateControls], fixture tool and testdata paths, codec and PCA gating; fish_codec_repo name not checked).
- Stale test count: the README says "cargo test --lib echo_tts (9 tests)", but the family directory has 10 #[test] functions [echo_tts/tests.rs].
- Overstated: "Streaming is refused (the reference raises NotImplementedError too)." No runtime refusal exists; the module only documents non-streaming behaviour [echo_tts/mod.rs:module doc]. Streaming is absent, not refused.

## crates/audio/src/tts/fish_drama3/EGM.md

Coverage: sampled (claims verified 15 of 17, source not fully read; directory is 4168 lines; measured figures are not checkable from source)
No drift detected.

## crates/audio/src/tts/fish_drama3/README.md

Coverage: sampled (claims verified 9 of 12, source not fully read; directory is 4168 lines; measured figures such as 58 ms/frame, 5.57 GB and 1.25x real time are not checkable from source and were not checked)
- Install layout: README lists `codec.safetensors` and `trunk/` as part of the install layout that `generate.rs` loads, but `Drama3::open_with` reads only `config.json`, `tokenizer.json` and `tts_side.safetensors` [generate.rs:Drama3::open_with]. `CODEC_FILE` and `TRUNK_DIR` are defined [generate.rs:CODEC_FILE] [generate.rs:TRUNK_DIR] but not referenced anywhere in the directory. Low.

## crates/audio/src/tts/fish_qwen3_omni/EGM.md

Coverage: sampled (claims verified 10 of 11, source not fully read; directory is 1892 lines)
No drift detected.

## crates/audio/src/tts/fish_qwen3_omni/README.md

Coverage: sampled (claims verified 6 of 8, source not fully read; directory is 1892 lines; reference gate tolerances in the README were not re-run)
No drift detected.

## crates/audio/src/tts/higgs_audio/EGM.md

Coverage: sampled (claims verified 10 of 12, source not fully read; directory is 1865 lines)
No drift detected.

## crates/audio/src/tts/higgs_audio/README.md

Coverage: sampled (claims verified 8 of 12, source not fully read; directory is 1865 lines)
- Overview names "a two-head projector (`HiggsAudioDecoderProjector`)". No Rust symbol of that name exists in the family directory, so the name reads as the reference class name rather than a port symbol. Low.

## crates/audio/src/tts/higgs_audio_v3/EGM.md

Coverage: sampled (claims verified 10 of 12, source not fully read; directory is 1568 lines)
No drift detected.

## crates/audio/src/tts/higgs_audio_v3/README.md

Coverage: sampled (claims verified 7 of 10, source not fully read; directory is 1568 lines)
No drift detected. (Reference-only names such as `BatchKVCache`, `batch_generate`, `genders`/`speeds`/`pitches` appear in the Open items as unported reference surface, not as port symbols; not counted as drift.)

## crates/audio/src/tts/indextts/EGM.md

Coverage: sampled (claims verified 11 of 12, source not fully read; directory is 3251 lines)
No drift detected.

## crates/audio/src/tts/indextts/README.md

Coverage: sampled (claims verified 7 of 10, source not fully read; directory is 3251 lines)
No drift detected. (`tokenize_by_CJK_char` and `normalize.py` are named as reference-side steps that the port does not carry; not counted as drift.)

## crates/audio/src/tts/irodori_tts/EGM.md

Coverage: sampled (claims verified 8 of 10, source not fully read; directory is 4204 lines)
No drift detected.

## crates/audio/src/tts/irodori_tts/README.md

Coverage: sampled (claims verified 9 of 12, source not fully read; directory is 4204 lines)
- Fixture description names the duration stage `predict_duration_log_frames`. No Rust symbol of that name exists in the directory; the port exposes [mod.rs:build_duration_features] and the duration path under the Model. Low.

## crates/audio/src/tts/kitten_tts/EGM.md

Coverage: sampled (claims verified 8 of 10, source not fully read; directory is 5690 lines)
No drift detected.

## crates/audio/src/tts/kitten_tts/README.md

Coverage: sampled (claims verified 6 of 10, source not fully read; directory is 5690 lines)
- "SineGen with 8 harmonics" is stated as a fixed family property. The code has no 8-harmonic constant: the harmonic width comes from the checkpoint's `l_linear` weight length [mod.rs:HarmonicSource::load], and the struct is named `HarmonicSource` (SineGen appears only in comments). Low.

## crates/audio/src/tts/kokoro/EGM.md

Coverage: sampled (claims verified 10 of 12, source not fully read; directory is 5396 lines including frontend)
No drift detected.

## crates/audio/src/tts/kokoro/README.md

Coverage: sampled (claims verified 6 of 10, source not fully read; directory is 5396 lines including frontend; Metal numeric claims such as the 2048-tile split-K threshold and 548-tensor count were not checked)
No drift detected. (`KokoroRunner` is in crates/runtime/src/kokoro.rs, outside this directory; `TURBOSPARK_KOKORO_DIR`, `TURBOSPARK_KOKORO_REFERENCE` and the three gpu/runtime test files exist.)

## crates/audio/src/tts/kokoro/frontend/EGM.md

Coverage: sampled (claims verified 8 of 12, source not fully read; directory is 2514 lines)
No drift detected.

## crates/audio/src/tts/kugelaudio/EGM.md

Coverage: sampled (claims verified 11 of 12, source not fully read; directory is 2416 lines)
No drift detected.

## crates/audio/src/tts/kugelaudio/README.md

Coverage: sampled (claims verified 7 of 10, source not fully read; directory is 2416 lines)
No drift detected.

## crates/audio/src/tts/llama/EGM.md

Coverage: sampled (claims verified 11 of 12, source not fully read; directory is 838 lines, below the full-read threshold but not read in full)
No drift detected.

## crates/audio/src/tts/llama/README.md

Coverage: sampled (claims verified 9 of 12, source not fully read; directory is 838 lines, below the full-read threshold but not read in full; the token table matches constants in mod.rs)
No drift detected.

## crates/audio/src/tts/longcat_audiodit/EGM.md

Coverage: sampled (about 20 of 25 claims verified; source of 3162 lines not fully read)
- Side Effects says LC_DEBUG traces come from the VAE encoder and decoder but cites only [vae.rs:EncoderStage::forward]; the same env check also gates prints in [vae.rs:DecoderStage::forward] and in an `AudioDiTVae` method near vae.rs:715. Omission, minor.

## crates/audio/src/tts/longcat_audiodit/README.md

Coverage: sampled (layout, REFERENCE_COMMIT, GenerateParams defaults, APG momentum and eta, fixture and tool paths checked; parity and timing numbers not re-run)
No drift detected.

## crates/audio/src/tts/megatts3/EGM.md

Coverage: sampled (about 25 of 30 claims verified: facade constants, Quality, sampler bound, tone reset, RNG type, model subdirs, chunk cap, duration cap; source of 6389 lines not fully read)
- Interface list omits public items that exist: [postprocess.rs:CROSSFADE_SECONDS] (pub const), and [aligner.rs:pad_prompt_audio], [aligner.rs:resample_for_aligner], [aligner.rs:prompt_mel] (pub fns). Omission, minor.

## crates/audio/src/tts/megatts3/README.md

Coverage: sampled (about 25 claims verified: Quality steps, MAX_PROMPT_SECONDS, 130-byte chunk budget, 0.16 s crossfade, CLI --steps 1 through 100, ffi install symbol, server /v1/audio/speech route, Metal default, referenced test file; benchmark, WER and memory numbers not re-run)
- Path references `tools/convert_megatts3.py`, `tools/gen_megatts3_*_fixtures.py` and `testdata/megatts3_*` resolve only from the crate root crates/audio (they live at crates/audio/tools and crates/audio/testdata), not relative to this README. Link-style ambiguity, minor (d).

## crates/audio/src/tts/melotts/EGM.md

Coverage: sampled (about 15 of 20 claims verified: sdp_ratio refusal, speaker resolution order, SAMPLE_RATE 44100, BERT third-to-last hidden state, public symbols; source of 3134 lines not fully read)
- Invariant overstates the LayerNorm epsilon: "LayerNorm eps 1e-5 [bert.rs:LAYER_NORM_EPS]" holds only for the per-layer norms. The embedding LayerNorm uses the config `layer_norm_eps`, default 1e-12 [bert.rs:embedding_eps] [mod.rs:BertDims].

## crates/audio/src/tts/melotts/README.md

Coverage: sampled (sdp_ratio refusal, LayerNorm eps split between embedding and per-layer norms, 44.1 kHz output, fixture and tool paths checked; parity numbers not re-run)
No drift detected.

## crates/audio/src/tts/moss_tts/EGM.md

Coverage: sampled (about 15 of 20 claims verified: v1.5 refusal in load, legacy refusal in open, INT64_MAX sentinel, sampling order, GPT-2 head divisibility, public entry points; source of 3166 lines not fully read)
No drift detected.

## crates/audio/src/tts/moss_tts/README.md

Coverage: sampled (variant aliases and default model_type, legacy refusal, Model::open layout probes, sampling order, fixture and tool paths checked; parity numbers not re-run)
No drift detected.

## crates/audio/src/tts/moss_tts_nano/EGM.md

Coverage: sampled (about 10 of 14 claims verified: GenerateParams defaults, temperature refusal, public entry points, sampling and prompt helpers; source of 1887 lines not fully read)
No drift detected.

## crates/audio/src/tts/moss_tts_nano/README.md

Coverage: sampled (GenerateParams defaults, lm::engine::Attention link target, sampling module split, fixture and tool paths checked; parity numbers not re-run)
No drift detected.

## crates/audio/src/tts/omnivoice/EGM.md

Coverage: sampled (about 15 of 18 claims verified: num_steps refusal, config hidden_size check, NONVERBAL_TAGS use, Rc backend, seed + step keying, pipeline helper at pipeline.rs:282 confirmed to be test-only; source of about 2900 lines not fully read)
No drift detected.

## crates/audio/src/tts/omnivoice/README.md

Coverage: sampled (time_steps schedule formula, tokens_per_sec = sample_rate / 960, vocab 1025 with mask 1024, bidirectional backbone setter, Rc backend, fixture, tool and test paths; checkpoint timings and A/B numbers not re-run)
No drift detected.

## crates/audio/src/tts/omnivoice/duration/EGM.md

Coverage: sampled (directory contents and the private `mod duration;` declaration in omnivoice/mod.rs checked; tests.rs not read)
No drift detected.

## crates/audio/src/tts/outetts/EGM.md

Coverage: sampled (about 14 of 16 claims verified: 4096-code decode chunks, -18 LUFS and -1 dB peak targets, silence as -inf LUFS, speaker word mutation in get_completion_prompt, public entry points; source of 1647 lines not fully read)
No drift detected.

## crates/audio/src/tts/outetts/README.md

Coverage: sampled (GenerateParams defaults, min-p default, 4096-code chunking, speaker-with-no-words refusal, fixture and tool paths checked; parity numbers not re-run)
- The deployed DAC repo id `mlx-community/dac-speech-24khz-1.5kbps` appears in no Rust source under crates/audio/src; it appears only in a comment in crates/audio/tools/gen_outetts_fixtures.py. Not drift in the code, but the deployed-profile claim cannot be checked from the crate.

## crates/audio/src/tts/pocket_tts/EGM.md

Coverage: sampled (about 14 of 16 claims verified: DEFAULT_* constants, frames_after_eos fallback, conditioning multiple-of-d_model check, empty text refusal, emb_std and emb_mean use; source of 1705 lines not fully read)
No drift detected.

## crates/audio/src/tts/pocket_tts/README.md

Coverage: sampled (EOS threshold default -4.0, frames_after_eos fallback of 3 + 2 and the prepare_text_prompt guess, end-of-run rule, de-normalization, fixture and tool paths checked; parity numbers not re-run)
No drift detected.

## crates/audio/src/tts/qwen3/EGM.md

Coverage: sampled (special token constants, 7-code group refusal, last-token handling in generate_tokens_from_prompt, public entry points; source of 848 lines partly read)
No drift detected.

## crates/audio/src/tts/qwen3/README.md

Coverage: sampled (full special token table, GenerateParams defaults, parse_output last-occurrence crop, group-count refusal, shared sampler link, fixture and tool paths; parity numbers not re-run)
No drift detected.

## crates/audio/src/tts/rumik_oss/EGM.md

Coverage: sampled (about 10 of 12 claims verified: quantizer index formula, out-of-order resync, stop predictor tensor layouts, min_tokens gate, empty prompt and max_tokens 0 handling, public entry points; source of 1103 lines not fully read)
No drift detected.

## crates/audio/src/tts/rumik_oss/README.md

Coverage: sampled (stop predictor head layout, both tensor name spellings, no-frame error path, 80 ms frame and 12.5 Hz consistency, fixture and tool paths; parity numbers not re-run)
No drift detected.

## crates/audio/src/tts/sesame/EGM.md

Coverage: sampled (claims verified 19 of 24, source not fully read)
No drift detected.

## crates/audio/src/tts/sesame/README.md

Coverage: sampled (claims verified 14 of 17, source not fully read)
No drift detected.

## crates/audio/src/tts/soprano/EGM.md

Coverage: sampled (claims verified 15 of 19, source not fully read)
No drift detected.

## crates/audio/src/tts/soprano/README.md

Coverage: sampled (claims verified 10 of 12, source not fully read)
No drift detected.

## crates/audio/src/tts/spark/EGM.md

Coverage: sampled (claims verified 14 of 16, source not fully read)
No drift detected.

## crates/audio/src/tts/spark/README.md

Coverage: sampled (claims verified 8 of 9, source not fully read)
No drift detected.

## crates/audio/src/tts/tada/EGM.md

Coverage: sampled (claims verified 15 of 18, source not fully read)
No drift detected.

## crates/audio/src/tts/tada/README.md

Coverage: sampled (claims verified 9 of 10, source not fully read)
No drift detected.

## crates/audio/src/tts/vibevoice/EGM.md

Coverage: sampled (claims verified 13 of 15, source not fully read)
No drift detected.

## crates/audio/src/tts/vibevoice/README.md

Coverage: sampled (claims verified 10 of 11, source not fully read)
No drift detected.

## crates/audio/src/tts/voxcpm/EGM.md

Coverage: sampled (claims verified 14 of 17, source not fully read)
No drift detected.

## crates/audio/src/tts/voxcpm/README.md

Coverage: sampled (claims verified 9 of 10, source not fully read)
No drift detected.

## crates/audio/src/tts/voxcpm2/EGM.md

Coverage: sampled (claims verified 16 of 18, source not fully read)
No drift detected.

## crates/audio/src/tts/voxcpm2/README.md

Coverage: sampled (claims verified 9 of 11, source not fully read)
- Open items say the `mean_mode` checkpoint variant is refused, but the code accepts it: `mean_mode` is normalized to `dit_mean_mode` [crates/audio/src/tts/voxcpm2/mod.rs:ModelConfig::from_json], and the mean-mode switch is implemented in the DiT [crates/audio/src/tts/voxcpm2/dit.rs:VoxCpmLocDiTv2]. The only refused variant is `cond_out_layer` [crates/audio/src/tts/voxcpm2/audio_vae.rs:SampleRateConditionLayer::load]. The README also describes `mean_mode` as a working switch earlier in the same file, so the open-items line contradicts it.

## crates/audio/src/tts/voxtral_tts/EGM.md

Coverage: sampled (claims verified 14 of 16, source not fully read)
No drift detected.

## crates/audio/src/tts/voxtral_tts/README.md

Coverage: sampled (claims verified 9 of 11, source not fully read)
- Verification says "gates 2e-5 unless noted", but the prompt-embedding gate is `EMBEDS_TOLERANCE` = 1e-6 (absolute), and the README does not note it as an exception [crates/audio/src/tts/voxtral_tts/tests.rs:EMBEDS_TOLERANCE].

## crates/audio/src/tts/zonos2/EGM.md

Coverage: sampled (claims verified 15 of 17, source not fully read)
No drift detected.

## crates/audio/src/tts/zonos2/README.md

Coverage: sampled (claims verified 11 of 13, source not fully read)
- Layout section names `Zonos2Config.from_dict` as the config loader. No such Rust symbol exists; the loader is `ModelConfig::from_json` [crates/audio/src/tts/zonos2/mod.rs:ModelConfig::from_json]. The `Zonos2Config.from_dict` name appears only in a doc comment that refers to the Python reference.

## crates/bench/AGENTS.md

Coverage: sampled (claims verified 20 of 20: doc links, directory map paths, root scripts, lab consumers, rule text; src/ not fully read)
- Rules: "The `meeting` scorer must ... agree count for count with `scripts/frozen_e_score.py`" implies a live check. No test runs the Python scorer. [crates/bench/src/meeting/mod.rs:cpwer_reproduces_frozen_e_per_clip_counts] compares only to the committed frozen_counts.json. Python also silently skips hypothesis segments with an unknown session_id, while Rust rejects them, so count-for-count agreement holds only for valid input. [crates/bench/src/meeting/seglst.rs:group_by_session, crates/bench/scripts/frozen_e_score.py:main]

## crates/bench/EGM.md

Coverage: sampled (claims verified 17 of 17; main.rs and Cargo.toml read in full, other src files by targeted reads)
- Invariant on `--model` flags lists six flags. It omits `--speculative`, `--speculative-drafter` and `--shaping`, which change the measured arm and default off. [crates/bench/src/main.rs:main]
- Interface omits public run paths that the protocol arms use: `open_model_runner_speculative`, `open_model_runner_steered`, `run_protocol_case_speculating`, `ProtocolShaping`. [crates/bench/src/real_model_open.rs:open_model_runner_speculative, crates/bench/src/real_model.rs:run_protocol_case_speculating]

## crates/bench/README.md

Coverage: sampled (claims verified 25 of 25; tests/ directory listing reconciled against disk; src/ not fully read)
- "This crate contains 81 test files in `tests/`" is wrong. `tests/` holds 91 `.rs` files. The README list omits ten: dflash_generation_gate.rs, dflash_tree_pin.rs, lfm3_memory_oracle.rs, lfm3_prefill_arm_probe.rs, lfm3_q8_memory_oracle.rs, lfm3_q8_quality_gate.rs, lfm3_quality_gate.rs, moe_route_capture.rs, turn_snapshot_gate.rs, turn_snapshot_ttft_probe.rs. [crates/bench/tests]
- "Binary Execution Modes" omits the synthetic `--real` scripted mode and most `--model` flags (`--speculative`, `--speculative-drafter`, `--shaping`, `--expert-cache-slots`, `--power-profile`, `--max-tokens-per-sec`, `--prefill-chunk`, `--kv-bits`). Mode 2 implies `--model` takes only `--case`. [crates/bench/src/main.rs:main, crates/bench/src/scripted.rs:run_real_mode]

## crates/bench/scripts/EGM.md

Coverage: full read (all three scripts, about 300 lines, read in full)
- Interface says `import_frozen_e_evidence.py:main` derives `frozen_counts.json` from evidence files. The per-clip counts and the matched-six list are hard-coded literal tables (FROZEN_E, MATCHED_SIX) in the script. Only references.json and E_seglst.json are read from the evidence directory. [crates/bench/scripts/import_frozen_e_evidence.py:FROZEN_E]
- Invariant says output clips "must match the recorded sha256, or the script refuses to continue". A source WAV mismatch stops immediately. An output clip mismatch prints BAD, keeps cutting the remaining clips, and exits non-zero only at the end. [crates/bench/scripts/prepare_frozen_e_clips.py:main]
- Dependencies are not documented. frozen_e_score.py needs numpy and scipy (its docstring says so). prepare_frozen_e_clips.py needs soundfile, and its header says to run it in the pinned venv `crates/audio/scripts/nemo_218d0b4`. Only import_frozen_e_evidence.py is stdlib-only. [crates/bench/scripts/frozen_e_score.py:main, crates/bench/scripts/prepare_frozen_e_clips.py:main]

## crates/bench/src/EGM.md

Coverage: sampled (claims verified 19 of 19; src/ excluding meeting/ is about 3,800 lines and was not fully read)
- Interface omits public items that exist in code: the `edge_*` tree replay API (`EdgeNode`, `edge_chain_nodes`, `edge_best_first_nodes`, `edge_accepted`, `edge_oracle_accepted`) and the `FIXED_PROMPTS`, `FIXED_SEED`, `FIXED_MAX_NEW_TOKENS` constants and `RunStats`. [crates/bench/src/tree_replay.rs:edge_accepted, crates/bench/src/scripted.rs:FIXED_SEED]
- Interface omits `ProtocolShaping`, `run_protocol_case_with_context`, `run_protocol_case_with_budget` (real_model.rs), and the vision-sidecar and KV-quant runner variants (real_model_open.rs). [crates/bench/src/real_model.rs:ProtocolShaping, crates/bench/src/real_model_open.rs:open_model_runner_with_context_and_vision_sidecar]

## crates/bench/src/meeting/EGM.md

Coverage: full read (mod.rs, cpwer.rs, seglst.rs and meeting_eval_main.rs read in full, about 850 lines)
- Invariant says the scorer must "agree count for count" with the Python scorer, and that a hypothesis segment with an unknown session_id fails scoring (not skipped). The Rust side does that. The Python scorer (`main`) silently drops such segments because it only reads sessions it was asked to score. The doc does not say the two differ on this input. [crates/bench/src/meeting/seglst.rs:group_by_session, crates/bench/scripts/frozen_e_score.py:main]
- The doc-comment on `group_by_session` calls the guard "the frozen scorer's guard against cross-clip attribution mistakes". The Python frozen scorer has no such guard, so the comment misattributes it. Fix the comment or the doc. [crates/bench/src/meeting/seglst.rs:group_by_session]

## crates/catalog/EGM.md

Coverage: sampled (claims verified 9 of 9 checked, source not fully read; crate is about 50k lines including tests and JSON)
No drift detected.

## crates/catalog/AGENTS.md

Coverage: sampled (claims verified 14 of 16 checked, source not fully read)
- AGENTS says the flint quantization parser "lives in `mlx_export`, not `quantization`". The function is defined in [crates/repack/src/flint_config.rs:parse_flint_quantization] and re-exported from [crates/repack/src/lib.rs]. `mlx_export` is the JSON block name in the flint config (flint_config.rs doc comments around lines 261-275), not a module. Reword to point at `repack::flint_config`.
- Verified without drift: `#![forbid(unsafe_code)]` [crates/catalog/src/lib.rs], the four `cargo test` network targets (all test files exist under tests/), the five doc links (docs/MODELS.md, docs/GTURBO.md, docs/NEW_MODEL.md, docs/FLINT.md, .claude/docs/modules/catalog.md), flint intake in [crates/catalog/src/probe/safetensors.rs:evaluate_config] and [crates/catalog/src/stream.rs] (`parse_flint_quantization` at line 221, `write_flint_install_streamed` at line 438), the shard rule in [crates/catalog/src/stream.rs:unindexed_shard_names], the `Status` values in [crates/catalog/src/entry.rs:Status], and "existing directory wins over alias" in [crates/catalog/src/store.rs:resolve_text_located].

## crates/catalog/README.md

Coverage: sampled (claims verified 17 of 19 checked, source not fully read)
- Override file location is wrong. README says `$TURBOSPARK_HOME/models.json` is merged. The code loads `<store root>/models.json` [crates/catalog/src/catalog.rs:Catalog::load], and both production callers pass the store root [crates/cli/src/bin/model.rs, crates/cli/src/front/models.rs]. The store root is `$TURBOSPARK_HOME`, else `$HOME/.turbospark` [crates/catalog/src/store.rs:default_root]. So with TURBOSPARK_HOME unset the override is read from `~/.turbospark/models.json`, and a `relocate_default_store` override also changes it.
- `discover.rs` is described as surveying "trending" repositories. The code surveys popular GGUF repositories via `client.popular_gguf_repos` [crates/catalog/src/recommend/discover.rs:discover]. "Trending" belongs to `HubClient::trending` [crates/catalog/src/hub.rs:HubClient::trending].
- Test list omits existing targets: tests/library_install.rs, tests/store_relocation_libraries.rs, tests/check_repo.rs, tests/hub_search_network.rs (the last is opt-in and network-gated).
- Omission: `recommend_catalog` drops vision-tower rows before ranking [crates/catalog/src/recommend/mod.rs:recommend_catalog]. Not in README, and the recommend EGM also omits it.
- Verified without drift: auth token order (explicit, HF_TOKEN / HUGGING_FACE_HUB_TOKEN, store `hf_token`, cache files) [crates/catalog/src/auth.rs:resolve_hf_token_with_source]; whoami-v2 validation [crates/catalog/src/auth.rs:validate_hf_token]; `probe` exits non-zero on refusal [crates/cli/src/bin/model_cmd/mod.rs:probe, Verdict::Refused]; range-chunk retention for immutable revisions only [crates/catalog/src/install.rs, crates/catalog/src/stream.rs:immutable_download_cache]; sidecar lists [crates/catalog/src/install.rs:OWNED_SIDECAR_NAMES, VISION_SIDECAR_FILES]; CLI subcommands list, list-speech, probe, recommend, pull [crates/cli/src/bin/model.rs]; `hub_validation.rs` owns no HTTP.
- Not verified: "tokenizer_config.json is optional" (the install requirement check was not read) and the "~26s" runtime comment.

## crates/catalog/examples/EGM.md

Coverage: sampled (claims verified 5 of 6 checked, examples/audio_pin.rs read only in part)
No drift detected.

## crates/catalog/src/EGM.md

Coverage: sampled (claims verified 12 of 13 checked, source not fully read; src/ holds about 40k lines)
No drift detected.

## crates/catalog/src/library/EGM.md

Coverage: sampled (claims verified 14 of 16 checked, module docs of registry, delete and transfer read; bodies of those files not fully read)
- `LibraryRegistry::delete_install` is wrong. The method is on `Store` [crates/catalog/src/library/delete.rs:Store::delete_install, impl Store at line 34]. `delete_image_install` is on `Store` too [crates/catalog/src/library/delete.rs:Store::delete_image_install]. Fix the Interface and Delete lines.
- Omission: the mount table is read by spawning `/sbin/mount` [crates/catalog/src/library/mount_table.rs:system mount read, Command::new("/sbin/mount")]. The Side Effects line says only "Reads the OS mount table", which hides the process spawn.
- Verified without drift: `INTERNAL_LIBRARY_ID`, `LIBRARIES_FILE`, `MARKER_FILE`, `MAX_MARKER_BYTES` (1 MiB, regular file check), `HOME_ONLY_FOLDERS` = ["embeddings", "ocr"], `replace_file` and `validate_alias` (crate-private), `STANDARD_LIBRARY_FOLDER`, `LibraryRegistry` methods, the internal library never written to the `libraries` array, per-write re-read in registry.rs module docs, tombstone-in-`.staging` delete, copy-verify-rename-repoint move order and refusal of links in move sources.

## crates/catalog/src/probe/EGM.md

Coverage: sampled (claims verified 10 of 11 checked, probe/mod.rs module docs, gguf.rs and types.rs read in part, safetensors.rs read in part)
- Sidecar and type names are correct. `KNOWN_SIDECARS` has 7 entries [crates/catalog/src/probe/types.rs:KNOWN_SIDECARS], `SLOT_COUNTS` is [8, 16, 32] [crates/catalog/src/probe/types.rs:SLOT_COUNTS]. `EXECUTABLE_GGUF_TYPES` lives in `model_io`, not this crate, so the doc should say `model_io::EXECUTABLE_GGUF_TYPES` [crates/catalog/src/probe/gguf.rs:evaluate_gguf].
- Verified without drift: `probe(client, repo, want_file, sidecar_repo)` signature [crates/catalog/src/probe/mod.rs:probe]; F32/F16/BF16 transcoded and exempt [crates/catalog/src/probe/gguf.rs:TRANSCODED_GGML_TYPES]; unsized types reported as unsized with a warning [crates/catalog/src/probe/gguf.rs:type_shares]; `Verdict::is_runnable`, `ProbeReport::refuse`, `ProbeReport::slot_cache_bytes` [crates/catalog/src/probe/types.rs]; sidecars taken from the file list.
- Not confirmed: "Expert granularity never refuses a repository". No expert-granularity refusal was found among the refuse call sites read in gguf.rs and safetensors.rs, but the full set was not checked.

## crates/catalog/src/recommend/EGM.md

Coverage: sampled (claims verified 10 of 11 checked, module docs of fit.rs, rank.rs and discover.rs read, bodies sampled)
- Omission: `recommend_catalog` ranks only `EntryKind::Model` rows and drops vision-tower rows, because they are not fit candidates [crates/catalog/src/recommend/mod.rs:recommend_catalog]. The Overview says "Ranks the curated table" with no such filter.
- Verified without drift: `recommend_catalog` takes no `Client` and does no network [crates/catalog/src/recommend/mod.rs:recommend_catalog]; `discover` and the `_probed` variants use `Client` [crates/catalog/src/recommend/discover.rs:discover]; `Evidence::of` [crates/catalog/src/recommend/rank.rs:Evidence::of]; `TIGHT_FRACTION` = 0.9 [crates/catalog/src/recommend/fit.rs:TIGHT_FRACTION]; `LADDER_WINDOWS` [crates/catalog/src/recommend/fit.rs:LADDER_WINDOWS]; `QUANT_LADDER` ordering [crates/catalog/src/recommend/discover.rs:QUANT_LADDER]; discovered rows pass the unchanged probe [crates/catalog/src/recommend/discover.rs module docs]; fit uses a working set, not phys_footprint or bits-per-weight [crates/catalog/src/recommend/fit.rs module docs].

## crates/cli/README.md

Coverage: sampled (claims verified: front-door routing, peer binary flags for image/music/clone/speak/ocr/model, pull and auth flags, steering flags in invocation, test file list, doc links; the expert-cache "auto" default and its 4 GiB reserve rule were not located in this crate or invocation, so they are unverified)
- `turbospark run MODEL "hello"` is documented as "Alias for turbospark-check --prompt". The code sends a prompt through the chat template as one user turn via `--messages-file /dev/stdin`; `--prompt` is used only with `--raw` [crates/cli/src/front/models.rs:check_args, run, plan_run]
- The front-door command list omits `humanize`, `steer`, `ps`, and `launch` (the agent launcher that `start <agent>` also reaches) [crates/cli/src/bin/turbospark.rs:Cmd::Humanize, Cmd::Steer, Cmd::Ps, Cmd::Launch]. `turbospark list` is a visible alias of `ls`, which calls turbospark-model `ls`, not `list` [crates/cli/src/bin/turbospark.rs:Cmd::Ls]
- The Tests section omits tests/daemon_root.rs, tests/library_lifecycle.rs and tests/model_library_cli.rs [crates/cli/tests]
- The Key Modules section omits the front/ and ui/ directories [crates/cli/src/front, crates/cli/src/ui]

## crates/cli/AGENTS.md

Coverage: sampled (verified: `--chat` refuses `--image` [crates/cli/src/chat.rs:run], exit-status mapping [crates/cli/src/main.rs:main], `run` flag arity from invocation::OPTIONS [crates/cli/src/front/models.rs:takes_value], the mference_check test covering the refusal, doc links; `--help`/`--version` without a model and the ASCII-only rule for the whole crate were not executed or checked)
No drift detected.

## crates/cli/EGM.md

Coverage: full read (23-line file); subdirectory links, the verification command, the install/delete delegation to turbospark-model, and the listed side effects were checked
No drift detected.

## crates/cli/src/EGM.md

Coverage: sampled (daemon.rs read in full except the get_running_pid to running_server span; main.rs read from line 30; agent.rs checked by symbol and a few lines only)
- Invariant "Pid, meta and log files are created private" is only partly true. The pid and meta files go through write_private_file (mode 0600) and the run dir through create_private_run_dir (0700). The log file is opened with plain `OpenOptions::new().create(true).append(true)` and gets no private mode [crates/cli/src/daemon.rs:start]
- Side effects say the pid, meta and log files are written "under the run dir". The log is `logs_dir()/server.log`, a sibling of the run dir, not inside it [crates/cli/src/daemon.rs:log_file, logs_dir]
- Omitted: `start` removes `--api-key VALUE` from the server argv and passes the key to the child as TURBOSPARK_API_KEY in its environment [crates/cli/src/daemon.rs:server_args_and_api_key, start]
- Omitted: `get_running_pid` deletes the pid and meta files when the recorded process is dead. `status` and `ps` reach it through `snapshot`, which is read-only from the user's view [crates/cli/src/daemon.rs:get_running_pid; crates/cli/src/front/server.rs:snapshot]. Only start and stop call `require_root`, so status and ps still resolve the state root from the CWD-relative fallback `.turbospark` when HOME and TURBOSPARK_HOME are unset [crates/cli/src/daemon.rs:root_dir, require_root]
- Omitted env read: TURBOSPARK_PREFIX_REUSE (value "quiet" silences the prefix-reuse note) [crates/cli/src/chat.rs:run]
- Omitted gate: main.rs declares `chat` and `generate` only on macOS (`#[cfg(target_os = "macos")]`) [crates/cli/src/main.rs:mod chat, mod generate]

## crates/cli/src/bin/EGM.md

Coverage: sampled (#[path] includes in turbospark.rs and bin/model.rs verified; steer.rs exit mapping verified; the other per-modality binaries checked by their flag lists only)
No drift detected.

## crates/cli/src/bin/model_cmd/EGM.md

Coverage: sampled (every listed symbol located; INTERRUPTED and the libc::signal SIGINT handler, check_alias call sites, image_source refusal to overwrite, KINDS and the /bin/df call verified; pull, move and library bodies not read)
No drift detected.

## crates/cli/src/bin/steer_cmd/EGM.md

Coverage: sampled (every listed symbol located; pack id check in TraitPack::parse, CAPTURE_ENV set and unset around each side, MAX_CAPTURE_CONTEXT cap, family refusal in resolve_model, .partial plus rename in write_atomically verified; walk, extract and check bodies only partly read)
No drift detected.

## crates/cli/src/front/EGM.md

Coverage: full read of non-test code in models.rs (532 lines, tests skipped) and server.rs (140 lines)
- Side effects omit the silent download path. `run` (through ensure_installed) pulls a missing curated model without asking when `--yes` is set or when stdin or stderr is not a terminal. Only an interactive terminal gets the confirm prompt. The EGM lists only "Prompts on stdin for confirmation" [crates/cli/src/front/models.rs:ensure_installed, interactive, confirm]

## crates/cli/src/generate/EGM.md

Coverage: sampled (dispatch and Outcome, image refusal for the verbatim prompt path, per-family vision attach functions, messages-file flow through prepare_images, env vars PHASES, DISPATCH_PROFILE, PREFILL_CHUNK and DEBUG_PROMPT_IDS verified; session.rs and format.rs bodies only partly read)
- Side effects say the module "Reads TURBOSPARK_EXPERT_DISK_IO (expert-bytes footer)". This crate never reads that variable. format.rs only names it in the hint printed when no expert-IO sample was taken ("n/a (set TURBOSPARK_EXPERT_DISK_IO=1)") [crates/cli/src/generate/format.rs:print_phases]
- Invariant names `Session::max_context` and `Session::rate`. Both are pub(crate) struct fields on Session, not methods [crates/cli/src/generate/session.rs:Session]. Wording only.

## crates/cli/src/ui/EGM.md

Coverage: sampled (module header and helper functions read, lines 1-135 of 354; the table and section code only checked by symbol)
- Invariant "Color is off under NO_COLOR and TERM=dumb, forced by CLICOLOR_FORCE" is overstated. CLICOLOR_FORCE (any value other than 0) is tested first and returns true before NO_COLOR or TERM=dumb are read, so it overrides both [crates/cli/src/ui/mod.rs:color_enabled]
- Side effects say the helpers write "to stdout/stderr". ok, note, warn and error all write to stderr (eprintln); stdout is used only by callers that print Paint-styled text [crates/cli/src/ui/mod.rs:ok, note, warn, error]

## crates/compute/EGM.md

Coverage: full read (doc is 7 lines; the link to ./src/EGM.md resolves).
No drift detected.

## crates/compute/AGENTS.md

Coverage: full read of the doc; all three linked files resolve; Cargo.toml and lib.rs read in full.
- "FP16 uses the `half` crate" is overstated. [crates/compute/Cargo.toml:dependencies] lists only turbospark-core (alias foundation), with no direct `half` dependency. The FP16 type is `foundation::LogitValue` [crates/compute/src/gdn.rs:F16], which turbospark-core defines as `Half` [crates/core/src/primitives.rs:LogitValue]. `half` is reached only indirectly through core. The comment at [crates/compute/src/gdn.rs:F16] (lines 26-27) also says "`half::f16` by its public path", which does not match the import.
- `#![forbid(unsafe_code)]`, the cargo check commands, and the BF16 bit-shift rule all match [crates/compute/src/lib.rs:forbid(unsafe_code)] and [crates/compute/src/quant.rs:bf16_to_f32].

## crates/compute/README.md

Coverage: sampled (claims verified 40 of 42 by symbol lookup, gotchas and test-file existence checked, source not fully read).
- Key Modules omits [crates/compute/src/short_conv.rs:short_conv_reference] (LFM3 gated short convolution, PORT-LOCAL, re-exported from lib.rs).
- Key Modules omits [crates/compute/src/steering_extract.rs:difference_of_means], [crates/compute/src/steering_extract.rs:CaptureSet], [crates/compute/src/steering_extract.rs:layer_stats], and [crates/compute/src/steering_extract.rs:derived_alpha_cap] (trait steering extraction and diagnostics).
- Tests list omits [crates/compute/tests/dash_oracle.rs] and [crates/compute/tests/gemma4_vision_reference.rs]. Both exist and are backed by generated oracles in tests/generated/.
- Verified with no drift: the downstream `compute` alias snippet (runtime, repack, cli, image, ffi use it; gpu uses the plain name, as the README says), the `ComputeStrategy` marker, the four FP16 rounding points [crates/compute/src/gdn.rs:h] (conv_out, conv tail rows, normed q/k, y), the NaN-sticky `worst()` fold [crates/compute/src/tolerance.rs:worst], and the BF16 NaN quieting [crates/compute/src/quant.rs:f32_to_bf16].

## crates/compute/src/EGM.md

Coverage: sampled (claims verified about 100 of 105 by name lookup against the pub item list, about 12 by reading code, source not fully read; the 4,367-line generated table file was not read).
- Tolerance interface omits [crates/compute/src/tolerance.rs:Tolerance::FP16_CHAINED_REDUCTION] (1e-2, chained FP16 reductions). The doc lists only IDENTITY, FP16_REDUCTION, QUANT_INT4, and QUANT_INT8.
- Interface omits several pub items that matter for GPU parity work, such as [crates/compute/src/steering_extract.rs:layer_stats] and [crates/compute/src/steering_extract.rs:derived_alpha_cap]. The steering_extract entry is otherwise present, so this is a partial omission.
- Verified with no drift: GROUP_SIZE = 64 [crates/compute/src/quant.rs:GROUP_SIZE], the int4 length assert [crates/compute/src/quant.rs:dequantize_int4_affine], the other tolerance constant values, `rms_norm_centered` computing x * (1 + w) [crates/compute/src/rms_norm.rs:rms_norm_centered], and the IQ table provenance header [crates/compute/src/quant_gguf_iq_tables.rs] and [crates/compute/src/quant_gguf_iq_lowbit_tables.rs] (generated from scripts/ggml_tables.c, which exists). The foundation alias and the quant_2bit shared f16 comment in lib.rs also match.
- Side Effects "None" is consistent: no `static`, `thread_local`, or OnceLock/LazyLock found in crates/compute/src.

## crates/compute/src/quant_gguf/EGM.md

Coverage: sampled (claims verified 30 of 30 by name and assert lookup, source not fully read).
- Constants omitted: the Q*_SUB_ELEMS constants (Q2_K_SUB_ELEMS, Q3_K_SUB_ELEMS, Q4_K_SUB_ELEMS, Q5_K_SUB_ELEMS, Q6_K_SUB_ELEMS) are exported from [crates/compute/src/quant_gguf/mod.rs] and re-exported at the crate root [crates/compute/src/lib.rs]. The doc mentions only *_BLOCK_ELEMS and *_BLOCK_BYTES.
- Verified with no drift: every listed function and constant exists in its file; the length and buffer asserts [crates/compute/src/quant_gguf/q4_0.rs:dequantize_q4_0] and [crates/compute/src/quant_gguf/q2_k.rs:dequantize_q2_k]; GEMV preconditions [crates/compute/src/quant_gguf/q5_k.rs:dequant_q5_k_gemv] and [crates/compute/src/quant_gguf/q6_k.rs:dequant_q6_k_gemv]; the pearson preconditions [crates/compute/src/quant_gguf/pearson.rs:pearson]; and the no-repack-dependency comments [crates/compute/src/quant_gguf/q8_0.rs], [crates/compute/src/quant_gguf/q4_k.rs], and [crates/compute/src/quant_gguf/q3_k.rs].

## crates/core/EGM.md

Coverage: full read (crates/core: Cargo.toml, all of src/ and tests/ read; about 1,200 source lines)
No drift detected.

## crates/core/README.md

Coverage: full read (crates/core: Cargo.toml, all of src/ and tests/ read; about 1,200 source lines)
- Omitted: the engagement module and its env var. [crates/core/src/engagement.rs:enabled] reads `TURBOSPARK_ENGAGEMENT` (cached in a OnceLock), and [crates/core/src/engagement.rs:engaged] (exported `#[macro_export]` macro, used as `foundation::engaged!` in crates/gpu/src/attention_decode.rs and crates/gpu/src/gdn.rs) prints `[engaged] NAME` to stderr. The README's Key Modules list has no engagement entry.
- Overstated: "Enforces ascending order and valid defaults at compile time via const assertions." [crates/core/src/runtime_config.rs:ALLOWED_CACHE_SLOTS] and [crates/core/src/runtime_config.rs:ALLOWED_CHUNK_SIZES] are the only sets with const assertions (lines 127-150). [crates/core/src/runtime_config.rs:ALLOWED_SPECULATION_BLOCKS] (1..=15), [crates/core/src/runtime_config.rs:DEFAULT_ADAPTIVE_CAP] and [crates/core/src/runtime_config.rs:MIN_ADAPTIVE_CAP] have no compile-time check. The src EGM.md states this correctly.
- Overstated: "Pure, deterministic logic with no filesystem or network I/O" under Safety. The crate reads an environment variable through [crates/core/src/engagement.rs:enabled]. This is not filesystem or network I/O, but the "pure" wording omits it.

## crates/core/AGENTS.md

Coverage: full read (crates/core: Cargo.toml, all of src/ and tests/ read; about 1,200 source lines)
No drift detected. (Links ../../.claude/docs/modules/core.md, ../../AGENTS.md and ../../docs/TESTING.md resolve. The `cargo check --target x86_64-unknown-linux-gnu` command was not run.)

## crates/core/src/EGM.md

Coverage: full read (crates/core/src: all eight modules and their unit tests read; about 1,200 source lines)
No drift detected. (Every [file:symbol] reference resolves. Checked against [crates/core/src/runtime_config.rs:max_of], [crates/core/src/prefill.rs:prefill_chunk_spans], [crates/core/src/prefill.rs:PrefillChunkCommitState::mark_dirty], [crates/core/src/steering.rs:SteeringMode], [crates/core/src/humanizer.rs:default_max_new], [crates/core/src/humanizer.rs:REUSE_FLAG_THRESHOLD] and [crates/core/src/engagement.rs:line]. The env var and OnceLock side effect match engagement.rs.)

## crates/coreml/AGENTS.md

Coverage: sampled (every rule checked against src/lib.rs, src/sys.rs, src/cache.rs, src/tensor.rs, src/stub.rs and tools/; sys.rs read in targeted ranges, not in full)
- Cargo.toml comment cites "AGENTS.md Gotcha 8" [crates/coreml/Cargo.toml:objc2]. This file has no numbered gotchas, so the reference does not resolve. (Cargo.toml is not a doc file; reported because it points here.)
- Cargo.toml comment says the macOS 15 gate is a respondsToSelector check in "sys::availability" [crates/coreml/Cargo.toml:objc2]. No such module exists. The gate is objc2 `available!` in [crates/coreml/src/sys.rs:supports]. The AGENTS rule "gated by supports" is correct; the Cargo.toml comment is stale.

## crates/coreml/EGM.md

Coverage: full read (10 lines, every claim checked)
- Tools bullet ("Python generators and bakers for ANE prefill, vision and sweep programs") omits the fixture generator [crates/coreml/tools/gen_coreml_fixtures.py:main], which the tools doc and AGENTS describe. Minor.

## crates/coreml/examples/EGM.md

Coverage: full read (both examples and the doc)
No drift detected.

## crates/coreml/src/EGM.md

Coverage: sampled (lib.rs, stub.rs and cache.rs read in full; error.rs read in part; sys.rs read in targeted ranges (CoreMlModel load/predict/state, compile_if_needed and pruning, compute_plan, SharedF16Plane entry points) plus grep; tensor.rs targeted reads. Source not fully read)
- (a) "same exported names" for sys and stub [crates/coreml/src/lib.rs:SharedF16Plane]: SharedF16Plane is re-exported only from sys, under `all(target_os = "macos", feature = "ane-prefill")`. stub.rs has no counterpart, so the exported names differ off macOS when the feature is on.
- (b) "Allocates page-aligned shared buffers" [crates/coreml/src/sys.rs:SharedF16Plane::alloc_size]: the code documents a page-rounded allocation size and an IOSurface bytes-per-row alignment. Nothing asserts page-aligned base addresses. Cite the source for that or reword.
- (c) Invariant "hashes package contents" omits that the cache key walk skips symlinks and hashes only regular files [crates/coreml/src/cache.rs:walk, cache_key]. The code comment says this is deliberate.
- (c) The capability-to-OS mapping is not stated: MultiFunction and State need macOS 15.0; AsyncCompile and ComputePlan need 14.4 [crates/coreml/src/lib.rs:Capability::required_macos]. The doc mentions only "newer than macOS 13".
- (c) Side Effects omit the cross-volume copy fallback when the compiler's temp dir cannot be renamed [crates/coreml/src/sys.rs:compile_if_needed], and the `.source` marker written beside each cache entry plus the one-hour orphan `.staging-*` pruning [crates/coreml/src/sys.rs:prune_stale].
- (c) Interface describes `CoreMlModel::load` as a plain loader. It refuses anything that is not an existing `.mlmodelc` directory [crates/coreml/src/sys.rs:CoreMlModel::load]; `open` is the entry point for packages.
- (c) Strided-read invariant omits that negative strides are refused [crates/coreml/src/tensor.rs:strided_extent]. Minor.

## crates/coreml/tools/EGM.md

Coverage: sampled (docstrings, CLI definitions and the key functions named in the doc were read for all five scripts; not every line. The uv verification command was not run)
- (c) Vision bake weights are not the install's weights. [crates/coreml/tools/bake_ane_vision_programs.py:const_f16_program] uses seeded random F16 constants and carries no biases (the module docstring says so). The Interface line "bakes a vision tower block's qkv/proj/fc1/fc2 projections" implies real weights. State that these are timing and placement probes.
- (c) Sweep `--const-int4` programs also use random weights [crates/coreml/tools/gen_ane_sweep_programs.py:const_int4_program]. The Interface omits the flags `--const-int4`, `--const-block`, `--expert-dims` and the packed-expert program kind [crates/coreml/tools/gen_ane_sweep_programs.py:main].
- (b) "rows are rounded down to 32" is wrong as written. [crates/coreml/tools/gen_ane_sweep_programs.py:main] refuses shape rows or K that are not multiples of 32. Only the ANE row count (`ma = max(32, int(rows * share) // 32 * 32)`) is rounded down.
- (c) Bake invariants omit two refusals: a group with a zero scale [crates/coreml/tools/bake_ane_prefill_programs.py:fp16_mapping], and `--lut` refusing a table more than 0.05 level off [crates/coreml/tools/bake_ane_prefill_programs.py:baked_program]. The `--tokens` default (2048) is also not stated. Minor.
- Verified with no drift: prefill K slices at 2048 with F16 accumulation [crates/coreml/tools/gen_ane_prefill_programs.py:matmul, build]; INT4 low nibble first, BF16 scales and biases per 64 columns [crates/coreml/tools/bake_ane_prefill_programs.py:Install.rows]; the 1e-5 or 0.05 acceptance rule [crates/coreml/tools/bake_ane_prefill_programs.py:fp16_mapping]; fixture closed forms and digest-only manifest [crates/coreml/tools/gen_coreml_fixtures.py:main, digest_tree].

## crates/embeddings/EGM.md

Coverage: sampled (claims verified 6 of 6 Cargo and crate-level claims; source not read in full)
No drift detected.

## crates/embeddings/AGENTS.md

Coverage: sampled (claims verified about 12 of 14; links, pins, env var names, feature list checked; numeric test claims not re-run)
- [src/weights.rs:quant_scheme] The rule "Refuse any quantization mode other than affine" holds only when a `mode` key is present. A `quantization` block with no `mode` key is accepted and treated as affine. Minor.

## crates/embeddings/MODELS.md

Coverage: sampled (pin table fully checked against every PINS entry; measured cosine and quantization-step figures not re-measured)
No drift detected. All 24 revisions, MLX-to-upstream pairings, gated flags and bit widths match [src/pins.rs:PINS]. The 5-bit and 6-bit EmbeddingGemma refusal matches [src/embeddinggemma/config.rs:VERIFIED_QUANT_BITS].

## crates/embeddings/src/EGM.md

Coverage: sampled (all listed re-exports, Weights, EmbeddingTable, pooling, normalize, similarity and error variants checked; lib.rs and weights.rs read in targeted ranges)
- [src/lib.rs:Result] The crate-level `pub type Result<T>` alias is not listed in Interface. Minor omission.
- [src/weights.rs:quant_scheme] Invariant "Only MLX affine quantization is accepted, other modes error" is true for an explicit non-affine `mode`, but a `quantization` block without `mode` is accepted as affine. Minor.

## crates/embeddings/src/bert/EGM.md

Coverage: sampled (all Interface symbols, the quantization gate and the XLM-R position function checked; mod.rs and encoder.rs not read in full)
No drift detected.

## crates/embeddings/src/bert/README.md

Coverage: sampled (pins, pooling policy, public API, refusal list, env var names and fixture names checked; measured tables and mutation results not re-run)
- [src/bert/config.rs:BertConfig::from_json] The refusal list says `hidden_act` other than `gelu` is refused. An absent `hidden_act` is also refused (the match falls through to `other`), while the HF BertConfig default is `gelu`. The README does not say absent values are refused. Minor.

## crates/embeddings/src/clap/EGM.md

Coverage: sampled (all Interface symbols, the 48 kHz rate, LongAudioCrop default Head, f64 and complex64 frontend and validate() checked)
No drift detected.

## crates/embeddings/src/clap/README.md

Coverage: sampled (API, refusals, all 11 named test functions, env var and default path, RoBERTa position and tanh pooler checked; numeric tables and timing not re-run; WAV resampler filter not checked)
No drift detected.

## crates/embeddings/src/embeddinggemma/EGM.md

Coverage: sampled (Interface symbols, verified quantization bits and group, GemmaTokenizer truncation and load path checked)
- [src/embeddinggemma/config.rs:GemmaConfig] The invariant cites a "262144-row vocabulary". That figure does not appear in embeddinggemma code or in the family README; it comes from the checkpoint. Not a code divergence, but the doc states a value the code does not contain.

## crates/embeddings/src/embeddinggemma/README.md

Coverage: sampled (pins, layout detection by tensor name, mean-pooling fallback, sliding band width 257, embed scale, refusal list, fixture file names checked; numeric tables and mutation results not re-run)
No drift detected.

## crates/embeddings/src/gte/EGM.md

Coverage: sampled (Interface symbols, rope_tables per-length recompute, BadConfig paths and embed_ids limit checked)
- [src/gte/config.rs:GteConfig::from_json] Invariant "Missing or malformed config keys fail with BadConfig" is overstated. Keys read with a default do not fail when absent: max_position_embeddings (2048), type_vocab_size (1), num_attention_heads (12), intermediate_size (3072), layer_norm_eps (1e-12), rope_theta (10000.0), hidden_act and position_embedding_type. Only keys read without a default, plus the model_type check, fail.

## crates/embeddings/src/gte/README.md

Coverage: sampled (pins, tokenizer optionality, rope_scaling and pack_qkv refusals, logn flags, max_seq_length resolution and fixture names checked; numeric tables and mutation results not re-run)
No drift detected.

## crates/embeddings/src/nomic_bert/EGM.md

Coverage: sampled (Interface symbols, ntk_base, padded_vocab_size and MLP_MULTIPLE_OF checked)
No drift detected.

## crates/embeddings/src/nomic_bert/README.md

Coverage: sampled (NTK formula and n > max_trained_positions boundary, rotary_scaling_factor default and null handling, refusal list, n_positions limit, pickle refusal in generator checked; numeric tables not re-run; heap-copy claim for the 94 MB word table not checked)
No drift detected.

## crates/embeddings/src/siglip2/EGM.md

Coverage: sampled (Interface symbols, TEXT_MAX_LENGTH tokenize guard, gelu_pytorch_tanh requirement, pixel_attention_mask checked)
No drift detected.

## crates/embeddings/src/siglip2/README.md

Coverage: sampled (processor refusals, vision_use_head, quantization refusal, square position grid, projection_size check, unpadded text tower, last-position pooling, test names checked; numeric tables not re-run)
No drift detected.

## crates/embeddings/tools/EGM.md

Coverage: sampled (argparse flags of all six generators, HF transformers and remote-code settings, torch import, fixture output paths checked; script bodies not read in full)
- [tools/gen_embeddinggemma_fixtures.py:main] The Interface line lists `--mlx, --qat, --model-dir` and omits `--tiny` and `--real`, which are the primary modes the family README regeneration command uses. Minor.

## crates/ffi/EGM.md

Coverage: full read (8 lines of prose, all claims checked: package name, subdirectories)
No drift detected.

## crates/ffi/AGENTS.md

Coverage: sampled (claims verified 10 of 12 checked; ABI guard rule, mirror rule, prefixReuse posture, make targets, doc links; source not fully read)
- Mirror rule incomplete: `OpenOptions.anePrefill` and `SessionInfo.anePrefill` are not documented in the header. `turbospark.h` mentions `expertSlotsWiring` (lines 272, 281, 562) but has no `anePrefill` option key; the only `anePrefill` text is `anePrefillSupported` in a capabilities example (line 1721). Rust side has them as [crates/ffi/src/wire.rs:OpenOptions] `ane_prefill` and [crates/ffi/src/wire.rs:SessionInfo] `ane_prefill`.
- Overstated rule: "every exported entry point must stay inside the ABI guard". [crates/ffi/src/api/core.rs:ts_last_error] and [crates/ffi/src/api/core.rs:ts_peak_footprint_bytes] are deliberately unguarded (their code comments say so).

## crates/ffi/README.md

Coverage: sampled (claims verified on Safety, Key Modules, Dev commands, Tests, Gotchas 1-3; source not fully read)
- Overstated: "Every entry point in `api/` wraps its implementation body in `abi::guard`..." Same two exceptions: [crates/ffi/src/api/core.rs:ts_last_error] and [crates/ffi/src/api/core.rs:ts_peak_footprint_bytes].

## crates/ffi/include/EGM.md

Coverage: sampled (all 9 `ts_` names and the 9 status codes checked against the header; header rules 1-4 compared line by line; header not fully read)
- Omits the platform contract from the header: on non-macOS, `ts_session_open` fails with `TS_ERR_UNSUPPORTED` while catalog, probe and install calls work. Code: [crates/ffi/src/api/session.rs:ts_session_open] (`#[cfg(not(target_os = "macos"))]`, line 45).
- Omits the rule 4 exception that `ts_install()`'s byte callback runs concurrently on worker threads ([crates/ffi/include/turbospark.h:ts_install]). The EGM rule 4 summary only says `ts_session_cancel()` is safe from any thread.

## crates/ffi/src/EGM.md

Coverage: sampled (named symbols checked: all resolve except the notes below; server idle, prefix reuse, STT and bridge constants verified against source; `src/` not fully read)
- Overstated invariant: "every entry point goes through `guard`". [crates/ffi/src/api/core.rs:ts_last_error] and [crates/ffi/src/api/core.rs:ts_peak_footprint_bytes] are unguarded by design.
- Omits the process-level scope of `ServerOptions.guardrails` and `prefixReuse` (one KV cache and posture shared by every client, not partitioned by API key), and that `prefixReuse` is applied before the startup session attaches ([crates/ffi/src/api/server.rs:ts_server_start], [crates/ffi/src/server.rs:Server]). Stated in crates/ffi/AGENTS.md and README but not here.
- Minor: [crates/ffi/src/heavy.rs:HeavyWorkGuard] is a re-export of `runtime::heavy::HeavyWorkGuard`, not a type defined in that file. The citation resolves, but a reader will look for a definition there.

## crates/ffi/src/api/EGM.md

Coverage: sampled (every `ts_` name listed in the Interface section resolves to an `extern "C"` fn; `ts_hf_token_set` storage path and `stt` cfg selection verified; `src/api/` not fully read)
- Overstated invariant: "Every entry point must run inside the ABI guard". Same exceptions: [crates/ffi/src/api/core.rs:ts_last_error], [crates/ffi/src/api/core.rs:ts_peak_footprint_bytes].
- Incomplete HF token surface in Interface: the exports also include [crates/ffi/src/api/models.rs:ts_hf_token_info_json], [crates/ffi/src/api/models.rs:ts_hf_endpoint_get] and [crates/ffi/src/api/models.rs:ts_hf_endpoint_set]. Validation is exported as `ts_hf_token_validate_json`, not `ts_hf_token_validate`.

## crates/ffi/src/generate/EGM.md

Coverage: sampled (event constants, humanize type names, `role_of`, `release_vision`, `SCRIPTED_HAS_NO_TOWER` and the humanizer raw-completion module doc verified; `generate/` not fully read)
No drift detected.

## crates/ffi/src/models/EGM.md

Coverage: sampled (all named install, probe, hub, fit, capability and control-vector symbols resolve; hub limits (20, 50, 2), install lock order (installs then move) and move/library refusal verified; `models/` not fully read)
No drift detected.

## crates/gpu/EGM.md

Coverage: full read (build.rs read in full; it is the only .rs file directly in the crate root).
- [build.rs:main] Doc says a missing Xcode Metal toolchain "fails the xcrun calls". Code only warns (cargo:warning) and falls back to runtime shader source, unless TURBOSPARK_METAL_PRECOMPILE_STRICT=1, which makes the build assert [build.rs:main]. The strict env var is not documented.
- [build.rs:main] Cross-target builds (CARGO_CFG_TARGET_OS != macos) write an empty precompiled_registry.rs and never invoke xcrun. Doc's Side Effects implies xcrun always runs.
- [build.rs:compile] Sources whose first line is "// turbospark: precise-math" are compiled without -ffast-math and with -fno-fast-math. Doc lists FLAGS only.
- [build.rs:main] Also reads DEVELOPER_DIR and SDKROOT (rerun-if-env-changed). Not documented (minor).

## crates/gpu/AGENTS.md

Coverage: sampled (about 20 of 22 claims checked; source not fully read). All 10 doc links resolve.
- [src/shaders/dequant_int4_fused.metal] Rule names the fused INT4 kernels `dequant_int4_gemv_fused_gate_up_act` and `dequant_int4_gemv_fused_down_epilogue`. Neither exists. The shader defines `dequant_int4_gate_up_act_gemv_simd` and `dequant_int4_down_epi_gemv_simd`. The Rust wrappers are encode_gate_up_act_gemv and encode_down_epi_gemv [src/dequant_int4_fused.rs]. The bit-exactness rule should name the real kernels.
- [src/ane_split.rs:GpuAneSplit] Doc names a type `GpuAneSplit`. No such type exists in crates/gpu or crates/runtime. The module exports encode_ane_dequant_rows, encode_ane_scale_rows, encode_ane_convert_out, encode_dequant_int4_gemm_steel_rows, HostEvent. The ANE split state is AneSplitState [crates/runtime/src/families/qwen/ane_split.rs:AneSplitState].
- [src/ane_split.rs] The rule "a non-finite value or NaN/Inf conversion flag signals a fallback to the Steel path" was not found in ane_split.rs. If the check exists, it is in the runtime consumer; the doc should cite where.

## crates/gpu/README.md

Coverage: sampled (about 25 of 28 claims checked; source not fully read). Numeric claims for the ~6 KiB autorelease cost were not checked.
- [src/shaders/] Doc says "50 `.metal` files". The directory has 51.
- [tests/] Doc says "97 integration test suites in tests/". The directory has 112 top-level .rs files (plus fixtures/, golden/, reference/ subdirectories). Some may be helpers; the count does not reconcile.
- [src/lib.rs:TokenId] Doc says the non-macOS build compiles to nothing but a TokenId re-export. This matches code (TokenId is ungated); it is listed here only because crates/gpu/src/EGM.md contradicts it.

## crates/gpu/examples/EGM.md

Coverage: sampled (both example files checked for the documented args, FNV hash, MLX version 0.32.3, and the case-generator path; full files not read).
- [examples/music3_mlx_linear_cmp.rs:main] Accepts an optional `--require-exact` flag (after the case directory). Not documented.

## crates/gpu/src/EGM.md

Coverage: sampled (about 25 claims checked: re-exported symbols, module gating, env vars, warmup gate, power_state unsafe placement; the ~22k lines of top-level source were not fully read).
- [src/lib.rs:TokenId] Doc says "on other platforms the crate exports nothing". lib.rs re-exports `foundation::TokenId` outside any cfg gate, so non-macOS builds export TokenId.
- Otherwise no drift detected in the checked claims: every listed type and function resolves through lib.rs, every "[cfg(target_os = macos)]" gate in lib.rs has its gate (checked by script), and the runtime warmup gate lives at [crates/runtime/src/real_forward_open.rs] as documented.

## crates/gpu/src/context/EGM.md

Coverage: sampled (all listed MetalContext, PassEncoder, CommittedPass, dispatch, buffer_io and GpuError symbols checked; pipeline_cache.rs read in part).
- [src/context/device.rs:MetalContext] Doc lists the env reads without their semantics. TURBOSPARK_METAL_PRECISE_MATH is presence-only (`std::env::var_os(...).is_some()`), so any value, including "0", enables precise math. TURBOSPARK_METAL_PRECOMPILED enables only on exactly "1". The doc should state both.
- [src/context/pass.rs:PassEncoder] Drop ends encoding when not already ended. Confirmed.
- [src/context/pipeline_cache.rs:supports_specialized_archives] macOS 15 gate and TURBOSPARK_METAL_PIPELINE_CACHE=1 gate confirmed.

## crates/gpu/src/moe_gguf/EGM.md

Coverage: sampled (listed symbols in mod.rs, kquants.rs, iq.rs, mxfp4.rs and q2_0.rs confirmed by name; kernels counted: moe_gguf.metal has 15 kernel entry points, matching the doc; bodies not read).
- [src/moe_gguf/mod.rs:PHASE2_FIXED_SLOTS] Invariant "top_k above that is rejected by assert" holds for encode_phase2 (mod.rs:158) and the mxfp4/iq/kquant entry points. The Q2_0 path is a separate 10-slot entry point [src/moe_gguf/q2_0.rs:encode_moe_phase2_q2_0_top10] backed by kernel `moe_phase2_down_reduce_k10_q2_0` [src/shaders/moe_gguf.metal]. The invariant as written is universal; it should say the 8-slot cap applies to the k8 path. Not checked: whether q2_0.rs asserts against 8.

## crates/gpu/src/music3/EGM.md

Coverage: sampled (all listed Music3Sequence operations, constructors and Music3Buffer methods confirmed by name in sequence.rs; bodies and the one-command-buffer claim not fully read).
No drift detected.

## crates/gpu/src/shaders/EGM.md

Coverage: sampled (file-to-kernel mapping, moe.metal (19 kernels) and moe_gguf.metal (15) counts, FC_ATTN_* constant list, and the kokoro.metal dispatch checked; kernel bodies not read).
- [src/shaders/attention.metal] Function-constant list omits FC_ATTN_ROWS_TILE (function_constant 75) and FC_ATTN_ROWS_HEADS (function_constant 76), used by attention_decode_gqa_rows_partial. AGENTS.md says these belong in the pipeline cache key, so the list should include them.

# Drift report: crates/image

## crates/image/EGM.md

Coverage: sampled (claims verified about 12 of 12 for this file; source of the crate is about 13.6k lines and was checked by targeted reads, not fully read)
No drift detected.

## crates/image/src/EGM.md

Coverage: sampled (claims verified about 45 of 50; source not fully read, targeted reads of install.rs, runtime.rs, builder.rs, packed.rs, metal_ops.rs and grep-confirmed definitions for every named symbol)
- Invariants omit the quantization-label check in the request path: `ImageRequest::validate` also calls `is_supported_quantization_label` and rejects unknown labels [crates/image/src/runtime.rs:ImageRequest::validate, crates/image/src/runtime.rs:is_supported_quantization_label]. The doc lists only model id, revision, prompt, noise provenance, batch, steps and guidance.
- Side effects say `pack_component` "creates output directories", but it first refuses an existing output directory [crates/image/src/packed.rs:pack_component]. The overwrite refusal is not documented.
- The PNG output carries an iTXt chunk keyed `turbospark` with the metadata JSON [crates/image/src/runtime.rs:encode_png_with_metadata]. Not mentioned anywhere in the doc, and downstream consumers may read it.
- Public API omitted from the interface list: `image_quantization_label` (re-exported from lib.rs) which decides the reported quantization label for MLX installs [crates/image/src/runtime.rs:image_quantization_label], and `ImageFamily::license` [crates/image/src/install.rs:ImageFamily::license].

## crates/image/src/shaders/EGM.md

Coverage: sampled (claims verified about 25 of 27; kernel list and thread caps checked against all `kernel` entries in image.metal, dispatch sizes checked in metal_ops.rs)
- Overstated: the doc says oversized VAE attention channel counts are "silently dropped, not rejected". That is true only at the kernel level. The host wrapper rejects channels above 512 and channels that are not a multiple of 8, and rejects areas that are not a multiple of 64, before dispatch [crates/image/src/metal_ops.rs:vae_attention]. The doc should say the host guard is the real gate and the kernel silently drops only if called directly.
- Incomplete: the doc says `image_attention` "supports head_dim <= 128". The host also requires `head_dim % 32 == 0` and `q_heads % kv_heads == 0` [crates/image/src/metal_ops.rs:attention].
- Verified without drift: 20 kernel names, `max_total_threads_per_threadgroup` values (512 for `image_attention`, 256 for the rest), the 256-thread dispatch for rms_norm, rope_adjacent, group_norm and vae_attention, the `4 * head_dim` threadgroup for attention, and the `include_str!` path to `metal_ops.rs:SOURCE`.

## crates/image/AGENTS.md

Coverage: sampled (links and commands checked; rules are not verifiable against code and were not judged)
- Stale-looking rule: "Keep preprocessing portable" refers to preprocessing that this crate does not hold for vision inputs. The only preprocessing here is `patchify.rs` (latent patch and unpatch), and portable vision preprocessing lives in `crates/vision-io` per the root CLAUDE.md. Low confidence: the rule may mean latent preprocessing only.
- Verified without drift: all four linked docs resolve (`.claude/docs/modules/image.md`, `docs/ZIMAGE_TURBO.md`, `docs/VISION.md`, `docs/TESTING.md`), and `cargo test -p turbospark-image` and the x86_64-unknown-linux-gnu check are the crate's commands.

## crates/image/README.md

Coverage: sampled (claims verified about 40 of 42; test list, env vars, ignore attributes, Cargo gating and linked docs checked)
- Overstated: the intro says the crate is a native pipeline for "Z-Image-Turbo, Qwen-Image-2.1, Qwen-Image-2.1-Turbo, and Krea 2 Turbo". The `generate` request path admits only the fixed Z-Image envelope (batch 1, 9 steps, guidance 0.0). Qwen-Image-2.1 (40 steps, guidance 1.0), Qwen-Image-2.1-Turbo and Krea 2 Turbo (8 steps, guidance 1.0) are rejected at `ImageRequest::validate` [crates/image/src/runtime.rs:ImageRequest::validate, crates/image/src/install.rs:ImageFamily::supported]. Install manifests for those families validate, but they cannot run through this crate.
- Omitted: the packed store writer refuses existing output directories [crates/image/src/packed.rs:pack_component]. Not required by the README, but the "Key modules" entry for `packed.rs` describes only the reader.
- Verified without drift: the `packed.rs` format claims (legacy affine INT4 group 64, MLX affine U32 planes at 2, 3, 4, 5, 6 and 8 bits [crates/image/src/packed.rs:MLX_AFFINE_BITS]), the `#[ignore]` status of `metal_parity` and `zimage_mlx_payload_network`, the `TURBOSPARK_IMAGE_INSTALL_DIR` variable, the eight test files, the `turbospark-image` CLI and FFI `ImageSession` consumers [crates/cli/src/bin/image.rs, crates/ffi/src/api/image.rs], and every linked doc path.

## crates/invocation/EGM.md

Coverage: full read of the doc; claims verified against src (package name, src/EGM.md link resolves).
No drift detected.

## crates/invocation/README.md

Coverage: sampled (claims verified 14 of 16 groups; src/options.rs and src/request.rs were grepped for items, not read in full; test files checked by test-function names and one file read, not every assertion).
- Overstated: "Confirms default values match documented runtime configuration defaults" [tests/request_defaults.rs:documented_defaults_are_applied]. The test hard-codes literal expected values; it does not read or compare against docs/CLI.md.
- Omission: the crate takes several accepted values and one default from the turbospark-core crate (`foundation`), not from this crate. Accepted sets for --expert-cache-slots and --prefill-chunk come from ALLOWED_CACHE_SLOTS and ALLOWED_CHUNK_SIZES [src/parser.rs:parse]; the prefill-chunk default comes from DEFAULT_CHUNK_SIZE [src/request.rs:PrefillChunk::default]; the cap and speculation constants re-export from foundation [src/request.rs:DEFAULT_ADAPTIVE_CAP, src/request.rs:ALLOWED_SPECULATION_BLOCKS]. The README lists only the Cargo dependency, not this coupling.
- Not verified: Gotcha 2 says filesystem existence is checked downstream by turbospark-cli or turbospark-runtime. turbospark-cli has some is_dir/exists checks (for example crates/cli/src/bin/speak.rs and crates/cli/src/bin/ocr.rs), but I did not confirm a model-path check on the main path or in turbospark-runtime. Treated as consistent, not confirmed.
- Verified without divergence: no unsafe code and no #![forbid(unsafe_code)] in src/lib.rs; no filesystem, env, or process I/O in src; OPTIONS, ParseFailure, InvocationRequest, module list; five-place rule matches src/parser.rs (no-value branch at parser.rs:parse line 123, value branch at line 315 with unreachable!); 5 integration test files; turbospark-check binary in crates/cli; downstream `invocation` alias used by crates/cli/Cargo.toml.

## crates/invocation/AGENTS.md

Coverage: full read of the doc; all "Read first" links checked for existence (.claude/docs/modules/invocation.md, docs/CLI.md, docs/TOOL_CALLING.md, docs/TESTING.md all resolve).
No drift detected.
- Note (not drift): the "server surface restates these rules rather than importing them" claim is consistent with the dependency graph; only crates/cli depends on turbospark-invocation.

## crates/invocation/src/EGM.md

Coverage: sampled (claims verified 25 of 28 groups). src/lib.rs, src/parser.rs, src/diagnostics.rs, src/usage.rs and src/failure.rs read in full (about 1,100 of 1,611 lines); src/options.rs and src/request.rs checked by targeted grep and reads of the named items, not in full.
- Omission: the mode and pairing rules are not in Invariants. Exactly one of --prompt, --messages-file, --humanize-file or --chat is required, else NoModeSelected; those four are pairwise mutually exclusive (except none with --system); --model is required (MissingRequired); --system is accepted only in chat mode (InvalidValue) [src/parser.rs:parse]. These are the refusals a caller most needs to know.
- Omission: the Failures bullet names ParseFailure without its six variants (UnknownOption, MissingValue, InvalidValue, MissingRequired, MutuallyExclusive, NoModeSelected) [src/failure.rs:ParseFailure].
- Omission: the Invariants say the other defaults are literals in this crate, but the accepted sets for --expert-cache-slots and --prefill-chunk (ALLOWED_CACHE_SLOTS, ALLOWED_CHUNK_SIZES) and the prefill-chunk default (DEFAULT_CHUNK_SIZE) come from foundation [src/parser.rs:parse, src/request.rs:PrefillChunk::default]. The doc lists only the cap and speculation constants as foundation-sourced.
- Omission (minor): steering-target and steering-gate refusal applies only when the flag was given explicitly, which the doc's "a knob without --steering" wording does not capture [src/parser.rs:parse, steering_target_explicit and steering_gate_explicit]. The "more knob values than paths" rule applies only to --steering-mode, --steering-scale and --steering-layers, since target and gate are single-valued.
- Verified without divergence: ParseOutcome variants (Success, Help, Version, Failure); help and version short-circuit at their scan position; Help and Version both route to stdout with exit 0 [src/diagnostics.rs:stream_routing]; ExitStatus codes 0 and 2 [src/diagnostics.rs:ExitStatus::code]; failure routed to diagnostic stream as Debug text plus usage [src/diagnostics.rs:stream_routing]; value-taking option consumes next token [src/parser.rs:parse]; inverted layer range refused [src/parser.rs:parse_layer_range]; the no-side-effect claim (no I/O in any src file read); every named public item in the Interface list exists and is exported from src/lib.rs.

## crates/lab/EGM.md

Coverage: sampled (cited commands, bins, env passthrough, integrity lists, lab dir, lock, atomic write and features checked by targeted reads; source not fully read)
No drift detected.

## crates/lab/AGENTS.md

Coverage: sampled (directory map and rules checked by targeted reads; source not fully read)
- Rule at lines 166-167 ("A prefill A/B runs both arms in one binary through the knob's kill switch") contradicts the rule at lines 122-124 ("Do not add an in-process arm switch") and the code: every arm is a fresh worker with a cleared env [src/orchestrate.rs:run_ab] [src/orchestrate.rs:PASSTHROUGH]. No in-binary kill-switch A/B path exists under crates/lab/src. Either it describes a harness outside this crate (name it) or it is stale.
- Lines 105-107 ("a sweep loads a tweaked copy with `--dir`") are false: [src/sweep.rs:sweep_cmd] never reads a pinned set, and its `--dir` selects a family profile directory (sweep.rs:162). `tslab pinned --dir` replaces the embedded set rather than overlaying it [src/cli.rs:pinned_cmd]. The same-id overlay [src/pinned.rs:with_overlay] has no caller outside tests.
- Lines 45-46 ("every response is `nosniff` plus a `default-src 'none'` CSP") are overstated: [src/dash/mod.rs:hardening] returns the 403 Host refusal before it sets either header.

## crates/lab/README.md

Coverage: sampled (command table, exit codes, output contract, verdict rule, dashboard, env and sandbox sections checked; source not fully read)
- Exit codes (line 99, "2 usage"): 2 is returned only for an empty argv [src/cli.rs:main]. An unknown command or bad argument returns Err from [src/cli.rs:dispatch], which [src/cli.rs:finish] turns into exit 1. `--help` exits 0. Usage errors do not exit 2.
- Commands table omits `hybrid-replay`, which [src/cli.rs:dispatch] routes and the "What it measures" section cites.
- Verdict rule (lines 89-92) omits the sign-agreement condition: [src/stats.rs:decide] requires every nonzero pair to agree below 6 clean pairs, and a sign test with p <= 0.05 at 6 or more, before WIN or LOSS.
- Dashboard (lines 133-135) says a loopback bind refuses foreign Host headers, which is correct, but omits that a non-loopback `--bind` turns that check off and serves with no authentication. [src/dash/mod.rs:serve_cmd] prints that warning.
- Verified without drift: envelope fields [src/schema.rs:Envelope], ledger KINDS [src/schema.rs:KINDS], TURBOSPARK_ provenance capture [src/provenance.rs], sandbox env list [src/orchestrate.rs:PASSTHROUGH], sibling worker path [src/orchestrate.rs:sibling_worker], metric names [src/workload.rs:Metric], the `DECISION:` line [src/experiment.rs:run_experiment_cmd], hash-diff exit 3 [src/cli.rs:hash_diff_cmd], and the GitBook and skill links.

## crates/lab/families/README.md

Coverage: sampled (embedding, schema string, knob value rule, provenance kinds, `--sweep`/`--dir` flags and the tests file checked; not every field rule traced)
No drift detected in the claims checked.

## crates/lab/pinned/README.md

Coverage: sampled (all 18 entries checked for repo, rev and license against the revisions table; validators checked against the field rules; `--dir` and overlay behavior traced)
- Lines 134-138 are false as written ("a sweep can layer tweaked copies over it through `--dir`"; "an id that collides with an embedded entry overrides it for that process"):
  - Sweep does not consume pinned sets at all [src/sweep.rs:sweep_cmd]. Its `--dir` is a family profile directory (sweep.rs:162).
  - `tslab pinned --dir DIR` loads only that directory through [src/pinned.rs:load_dir] (cli.rs, pinned_cmd), so embedded entries are absent rather than overridden.
  - The same-id replacement [src/pinned.rs:with_overlay] is called only from tests.
- Lines 1-9 (embedding and reader) are correct [src/pinned.rs:EMBEDDED].
- Field-rule sets match the code: [src/pinned.rs:REPOS], [src/pinned.rs:KERNEL_CLASSES], [src/pinned.rs:SETTING_KINDS], [src/pinned.rs:MEASURED_PROVENANCE].

## crates/lab/skill/EGM.md

Coverage: full read (the doc is short; SKILL.md, evals/ and references/ are not contract-bearing under this doc and were not read)
No drift detected.

## crates/lab/skill/scripts/EGM.md

Coverage: full read (both scripts read in full)
No drift in the stated interface or invariants. Checked: 10 GiB floor [src/host.rs:default_min_free_gb], MODEL_MARKERS fragments [src/host.rs:MODEL_MARKERS] against the preflight name list, lab dir precedence [src/paths.rs:lab_dir], and that no script clears quarantine.
- Not a divergence in the EGM, but preflight.sh line 29 comment cites `host.rs MODEL_PROCESS_NAMES`. That constant does not exist; the guard's list is `MODEL_MARKERS` (host.rs:68).
- Omission (minor): doctor.sh also reports the model store, xctrace, python3 and AC power as notes, which the EGM does not list.

## crates/lab/src/EGM.md

Coverage: sampled (command list, exit codes, machine lock, env passthrough, integrity lists, ledger and atomic write, and the cited symbols checked; source not fully read)
No drift detected.

## crates/lab/src/bin/EGM.md

Coverage: full read (both entry points read in full)
No drift detected.

## crates/lab/src/dash/EGM.md

Coverage: sampled (mod.rs read in full; index.rs refresh and api.rs routes checked by targeted reads)
- Invariant "Every response carries nosniff and a CSP" is overstated: [src/dash/mod.rs:hardening] returns the 403 Host refusal before inserting `nosniff` and the CSP, so that response carries neither header.
- Omission: [src/dash/mod.rs:serve_cmd] states that a non-loopback `--bind` turns off the Host check and that the dashboard has no authentication. Neither appears in Invariants or Side Effects.
- Verified: routes [src/dash/mod.rs:router], serve_cmd flags, `router` and `serve_router`, `App::new` and `App::with_index`, `LabIndex` build, refresh and record, and the JSON routes [src/dash/api.rs].

## crates/lab/src/kernel/EGM.md

Coverage: sampled (all 13 benches and their types located; ANE feature gating, music3 path include, `copies_for`, `uniform_union`, first-rep discard, machine lock and kernel_curve checked; vision_encode read)
- Invariant "Env knobs do not reach a bench" is overstated. The bench harness sets and clears no env, and benches run in the orchestrator process, so env reads inside the gpu crate still apply: device open reads `TURBOSPARK_METAL_PRECOMPILED` and `TURBOSPARK_METAL_PRECISE_MATH` [crates/gpu/src/context/device.rs], `TURBOSPARK_GDN_DK128` is read once in [crates/gpu/src/gdn.rs], `TURBOSPARK_ATTN_DECODE_KERNEL` in [crates/gpu/src/attention_decode.rs:736], and `TURBOSPARK_METAL_PIPELINE_CACHE` in [crates/gpu/src/context/pipeline_cache.rs]. The accurate statement is that the harness sets no knobs; the caller's environment still reaches the kernels. The same overstatement appears in the comment at [src/kernel/mod.rs:kernel_cmd].
- Side Effects lists only `kernel_curve` records. [src/kernel/mod.rs:kernel_cmd] also appends a `vision_encode` record for each vision bench result.

## crates/lab/tools/EGM.md

Coverage: full read (single script read in full)
- Invariant "Shader samples cover about a tenth of GPU time" is a figure from the module docstring. [tools/trace_tables.py:shaders] never computes coverage, so it is an observation about xctrace, not an invariant the code enforces.
- Omission (minor): [tools/trace_tables.py:encoders] counts only intervals on the "Compute" channel for the given pid. The EGM does not state this filter.

## crates/lab/workloads/audio/README.md

Coverage: full read (README and both job files read in full)
No drift detected. Checked: `AudioJob` location [src/workload.rs:AudioJob], task names (`music`, `speak`, `transcribe`), MegaTTS3 `reference_wav` and `consent` fields and `steps`, transcribe fields, and that the lab does not resample [src/worker_audio.rs:run].

## crates/model-io/EGM.md

Coverage: full read of Cargo.toml and lib.rs re-exports; all 4 claims (library name, dependency list, disk/mmap side effect, test command) checked.
No drift detected.

## crates/model-io/AGENTS.md

Coverage: sampled (claims verified 4 of 5 rule groups; manifest/mod.rs and resident_buffer.rs read, arch_validation.rs read only for its first 100 lines; source not fully read).
- [arch_validation.rs:validate_arch] Rule "reject missing or mismatched manifest fields" is not what the code does for `family`: an absent `arch.family` is compared as Gemma4 (`unwrap_or_else(ModelFamily::Gemma4)`), and every absent family-extension field takes the Gemma default. [manifest/mod.rs:peek_family] also returns `ModelFamily::Gemma4` when `arch.family` is missing. Either the rule or the code is wrong; the code is the existing behavior.
- Unsafe boundary omits the `sysconf(_SC_PAGESIZE)` call, which is an `unsafe` block in [resident_buffer.rs] (around line 109). The doc lists only mmap and the single statvfs call.

## crates/model-io/README.md

Coverage: sampled (flint.rs header and constructor body, gemma.rs, llama.rs, minimax.rs, gpt_oss.rs grep-checked; module list checked against src/ directory listing; test list checked against tests/ listing; source not fully read).
- [flint.rs:flint_40b_a2_5b] Doc says "top-6 INT8 sigmoid router". Code and doc comment say BF16 router ("BF16 router, no bias"); no INT8 router appears in flint.rs.
- [gpt_oss.rs:gpt_oss_20b] Doc says "(Harmony chat template)". No Harmony reference exists in gpt_oss.rs. Not verified elsewhere in this crate.
- [minimax.rs:minimax_m2] Doc says "split-GGUF and sigmoid routing". Sigmoid routing is present (`router_scoring_func: "sigmoid"`); no split-GGUF handling appears in minimax.rs.
- Arch baseline list omits `lfm3.rs` (`lfm3_1_7b`, `lfm3_layer_mask`), `unlimited_ocr2.rs` (`unlimited_ocr2_12l`), and the `deepseek_vl2_tiny` constructor in `deepseek2.rs`. All three are public re-exports in lib.rs.
- Key modules list omits `cgroup.rs` (`probe_cgroup_memory`, `parse_cgroup_limit`, both public re-exports used by sizing policy), `bounded_read.rs` (`read_capped`, size-capped metadata reads), and `error.rs` (`ModelError`).
- Safety section says unsafe is only mmap plus the single statvfs call. Omits the `sysconf(_SC_PAGESIZE)` call in [resident_buffer.rs] (around line 109).
- Tests list omits `tests/metadata_read_bounds.rs` and `tests/metadata_size_caps.rs`, both present in tests/.
- Gotcha 1 (newBufferWithBytesNoCopy pinning) is consistent with comments in [resident_buffer.rs] and [context_policy.rs] but the Metal wrapping itself is not in this crate; not verified.

## crates/model-io/src/EGM.md

Coverage: sampled (lib.rs, cgroup.rs, bounded_read.rs, error.rs, manifest/mod.rs, install_receipt.rs comment, resident_index constants, trunk_role.rs, dash_config.rs and speech_receipt.rs signatures read or grep-checked; the Interface list was checked against lib.rs re-exports; not every child file was read in full).
- Invariant "Unsafe is limited to ... plus the single statvfs call and the sysconf(_SC_PAGESIZE) call [resident_buffer.rs:unsafe_map] [safetensors.rs:SafetensorsFile] [disk_space.rs:available_bytes]": the sysconf call is in [resident_buffer.rs] (around line 109), not in [disk_space.rs:available_bytes]. Citation is wrong.
- Invariant "Expert strides must be page aligned [error.rs:ModelError::ExpertStrideNotPageAligned]" overstates the check. [manifest/mod.rs:page_size_bytes] hard-codes a 4096-byte floor, not the OS page size (16 KiB on this platform). The code comment says so explicitly.
- Side effects and interface entries otherwise match: `trunk_role::{has_marker, read, write_headless}`, `DashConfig::write_sidecar`, `SpeechInstallReceipt::write_to_dir`, `probe_self` reading `/sys/fs/cgroup/memory.max` and `memory.high` with no cfg gate.
- Source comment (not a doc file, noted for context) at [cgroup.rs:probe_self] says the probe resolves the process's cgroup through `/proc/self/cgroup`'s `0::` entry and returns None off Linux. The code hard-codes `/sys/fs/cgroup` and has no `/proc/self/cgroup` read and no cfg gate. The EGM side-effect bullet is accurate; the source comment is not.

## crates/model-io/src/arch_baselines/EGM.md

Coverage: sampled (mod.rs read fully, including the registry and all_known_architectures; constructor names checked against the `pub use` lines and lib.rs re-exports; flint.rs header read; other baseline bodies not read).
No drift detected.

## crates/model-io/src/arch_config/EGM.md

Coverage: sampled (mod.rs read fully; ModelFamily variant list and as_str/parse tables read; ArchConfig method names checked by grep; MAX_VISION_DEEPSTACK_MERGERS value confirmed as 3; config.rs predicate bodies not read).
No drift detected in the claims checked.

## crates/model-io/src/manifest/EGM.md

Coverage: sampled (manifest/mod.rs read fully; manifest/types.rs and manifest/quant.rs only grep-checked, validate_quant head read; source not fully read).
- Invariant "Flags outside `streamingPresent`, `turboQuantKV`, `aneSharedExpert` are an error" omits that `turboQuantKV: true` is itself refused with IndexCorrupt, with a message that it is not a real flag this port reads [manifest/mod.rs:validate]. The doc lists it as a recognized flag.
- Invariant "A manifest whose expert_stride is not page aligned is ExpertStrideNotPageAligned" overstates the check. The alignment is a hard-coded 4096-byte floor [manifest/mod.rs:page_size_bytes], not the real page size.
- Omitted behavior in [manifest/mod.rs:load]: `capability == "image-generation"` is refused with UnsupportedCapability.
- Omitted behavior in [manifest/mod.rs:validate]: `manifest.quant` is required when the expected arch matches a shipped baseline (`is_production_arch`).
- Omitted behavior in [manifest/mod.rs:validate]: per-layer packed expert files are required (`packed_experts/layer_NN.bin` or `layer_N.bin`, generated from `num_layers`).
- Omitted behavior in [manifest/mod.rs:validate]: when the arch has a vision tower, `packed_vision/layout.json` and `packed_vision/blobs.bin` are required (VISION_FILES).
- Omitted behavior in [manifest/mod.rs:validate_hadamard]: an optional `hadamard` section is validated (block power of two, sign widths, non-overlapping ranges, upload cap) and requires `hadamard.bin`.
- Omitted behavior in [manifest/mod.rs:peek_family]: a missing `arch.family` silently resolves to Gemma4 instead of erroring. This conflicts with the AGENTS.md rule to reject missing manifest fields.

## crates/repack/EGM.md

Coverage: full read (two bullets; both subdirectory links resolve and match src/ and examples/).
No drift detected.

## crates/repack/AGENTS.md

Coverage: sampled (claims verified 12 of about 20; Flint rebalance scope and example guards, canonicalize_flint_header U8 handling, mxfp4 0xFF refusal, test targets and linked docs all checked; Flint expert numbering and the streamed router path not checked).
- "the `softmax_offset` sinks are resident" [crates/repack/src/gemma4_checkpoint/classify.rs:classify_for_family]: no intake code in crates/repack names `softmax_offset`; the only match in the crate is the synthetic fixture [crates/repack/src/synthetic_flint.rs:build_synthetic_flint_install]. Could be a generic rule I did not find, so this is unconfirmed rather than proven false.
- "the raw BF16 router is force quantized to INT8" (unconfirmed for the streamed Flint path): the only caller of `quantize_gating_matrix_int8` is the non-streamed orchestrate path [crates/repack/src/gemma4_checkpoint/orchestrate.rs:orchestrate_gemma4_checkpoint_sharded]; the body of [crates/repack/src/gemma4_checkpoint/mod.rs:write_flint_install_streamed] shows no router-specific handling.

## crates/repack/README.md

Coverage: sampled (claims verified 10 of about 14; test file names, doc commands, the seven downstream `package = "turbospark-repack"` importers, the fish_drama3 verbatim copy and the build_manifest_json unconditional-field comment all checked; tokenizer sidecar list not traced to a copy site).
- "This crate contains 83 integration test files in `tests/`": crates/repack/tests holds 91 top-level .rs files (fixtures/ not counted). [crates/repack/tests:count]
- Key Modules omits two public parsers that src/EGM.md documents: [crates/repack/src/lfm3_config.rs:parse_lfm3_config] and [crates/repack/src/deepseek_vl2_config.rs:parse_deepseek_vl2_config].
- `gemma4_checkpoint/` is described as serving "Gemma 4, Qwen 3.6, and Flint" only; its own EGM also lists DeepSeek-VL2, LFM3, Qwen 3-VL and MuseGlimmer writers [crates/repack/src/gemma4_checkpoint/mod.rs:write_deepseek_vl2_install_streamed, write_lfm3_install_streamed, write_qwen3_vl_install_streamed, write_muse_glimmer_install_streamed].

## crates/repack/examples/EGM.md

Coverage: sampled (claims verified 8 of about 10; shard-prefix trailing slash check, existing-destination refusals, ShortRead on both FileRangeSource types, rebalance --undo exit 2 and expect-based usage panic, MiniCPM active-tower check and preprocessor_config copy all checked; flint trained-context write and rebalance second-application refusal not re-read).
No drift detected.

## crates/repack/src/EGM.md

Coverage: sampled (every cited symbol resolves in its file; claims verified about 20 of about 35, including quantizer preconditions, layer_base handling, 16 MiB control-vector cap, manifest cap, split.count bounds, trained-context u32 bound, LFM3 layer kinds and 4096 cap, F32/BF16-only dtypes, lm_head rule, UnlimitedOcr2 v1 refusal; flint and qwen36 refusal lists not fully re-read).
- Overview says the crate "quantizes matrices to affine INT1/2/4/8". The crate's own quantizers are INT4 and INT8 only [crates/repack/src/repack.rs:quantize_matrix_int4, quantize_matrix_int8] (plus gemma4 gating INT8 [crates/repack/src/gemma4_checkpoint/narrow.rs:quantize_gating_matrix_int8]). INT1 and INT2 appear only as resident dtype tags on the read side [crates/repack/src/resident_reader.rs:read_resident_entries] and as pre-quantized source tensors. Overstated.
- Interface lists fixture builders `build_synthetic_*` and `tiny_*_arch` for "llama" and "muse", which do not exist under those names. Actual names: [crates/repack/src/synthetic_llama.rs:build_synthetic_llama_real_install] (no `build_synthetic_llama_install`) and [crates/repack/src/synthetic_muse.rs:tiny_muse_glimmer_arch] with [crates/repack/src/synthetic_muse.rs:build_synthetic_muse_glimmer_install] (no `tiny_muse_arch` or `build_synthetic_muse_install`).

## crates/repack/src/gemma4_checkpoint/EGM.md

Coverage: sampled (every cited symbol resolves in its file; claims verified include the 27-block / 13-role vision tower, the 16 KiB page constant, affine group rules (4/8-bit at 64, 1/2-bit at 128), mxfp4 at 4-bit group 32, routed experts 3- or 4-bit only, mxfp4 0xFF refusal, hadamard `.signs` refusal, sibling staging name pattern, publish-by-rename, and the finish_streaming-before-manifest ordering; the cross-writer arm-parity invariant was not re-checked).
No drift detected.

## crates/repack/src/gguf_checkpoint/EGM.md

Coverage: sampled (claims verified about 14 of about 18, including DeepseekV4Flash refusal in both entry points, head-block exclusion, non-hex commit rejection, rope_freqs 1.0 / 1e29 thresholds and F32 requirement, MiniMax router F32, logical_shape panic, IQ4_NL PLE checks, index_qk_proj concatenation, header removal before n-gram rewrite, env use only under cfg(test); the resume size check and V-head de-tiling were not re-read).
No drift detected.

## crates/repack/src/gguf_config/EGM.md

Coverage: sampled (claims verified about 12 of about 15, including the 4096 layer caps, RMS epsilon pins for MiniMax-M2, Qwen2 and LFM3, the unported-family UnsupportedArchitecture arm, DeepSeek2 q_lora_rank refusal, the freq_base / freq_base_swa swap, Gemma, LFM3 and Qwen GDN mask rules, and the ssm inner_size divisibility rule; vocab row-width rule not re-read).
No drift detected.

## crates/repack/src/gguf_header/EGM.md

Coverage: sampled (claims verified about 10 of about 12, including the version 3 and GGUF magic gate, DEFAULT_MAX_HEADER_BYTES, DEFAULT_ALIGNMENT, MAX_ARRAY_DEPTH, the 32 B per tensor and 13 B per KV pre-allocation floor, the 1..=4 dimension rule, and the full error variant list; alignment and ggml block table contents not re-read).
No drift detected.

## crates/repack/src/gguf_names/EGM.md

Coverage: sampled (claims verified about 8 of about 11, including the GgufNameError variants, qwen35moe architecture string, A_log and dt_bias taking no `.weight` suffix, the canonical layer_prefix string, Gemma-only fused gate/up mapping, and the ZeroTailRopeFreqs variant; the rope_freqs refusal rule and the Mixtral pre-merge refusal were not re-read).
No drift detected.

## crates/repack/src/gturbo_writer/EGM.md

Coverage: sampled (full read of types.rs, streaming.rs and the install writers in layers.rs; manifest.rs build_manifest_json read, the family arch block only grepped; mod.rs not read). Claims verified about 14 of 16.
- Omission: the batch writers never set manifest `quant` or `hadamard`. write_gturbo_install_impl and write_gturbo_install_with_resident_index call build_manifest_json with hadamard_present=false and never insert a quant section [layers.rs:write_gturbo_install_impl, layers.rs:write_gturbo_install_with_resident_index]. The doc says set_quant is required for production-shape manifests but does not say the batch paths (write_gturbo_install, write_gturbo_install_with_resident_index_and_experts, write_gturbo_install_with_resident_index) cannot produce one.

## crates/repack/src/ranged_download/EGM.md

Coverage: sampled (mod.rs error, header and fetch code read; chunks.rs partition and fill_chunks read; http.rs client, cache, send_range, read_chunk, retry, download and token code read; http.rs test module and some builder bodies skipped). Claims verified about 30 of 34.
- Omission: with_pinned_identity changes the cache key. HttpRangeSource::cache_path hashes the pinned identity instead of the URL when one is set [http.rs:HttpRangeSource::cache_path]. The doc lists the builder but does not say it changes cache identity.
- Omission: pinned mode also sends If-Match when a validator is given and rejects a body with extra bytes past the expected length as SourceIdentity [http.rs:send_range, http.rs:read_chunk]. Only the Content-Range, length and ETag checks are in the doc invariant.
- Overstatement: "Retry-After wins" holds only for the delta-seconds form. The HTTP-date form is ignored by design, and the doc does not say so [http.rs:send_range].
- Omission: transport errors (Request, ShortRead and similar) retry on a 250 ms backoff doubling per attempt, capped at a 4-step shift [http.rs:with_retries]. The doc invariants cover only the status classes.

## crates/repack/src/synthetic_gguf/EGM.md

Coverage: sampled (gemma4_shape.rs and gemma4.rs claims read or grepped; gptoss.rs, minimax.rs and qwen2.rs claims grepped; mod.rs declarations grepped). Claims verified about 20 of 24.
No drift detected in the claims checked (QuantMix variants, SyntheticGgufShape presets and Default, Gemma fused gate/up and tied head, no attn_v on global layers, gpt-oss unfused biases and rank-2 expert biases, layer slide phase).

## crates/repack/src/synthetic_gguf/builder/EGM.md

Coverage: sampled (builder asserts, alignment handling, ggml type ids for IQ, MXFP4 and K-quant helpers, Q3_K type 11 grepped; write_value and build bodies not read in full). Claims verified about 12 of 12 checked.
No drift detected.

## crates/repack/src/synthetic_model/EGM.md

Coverage: sampled (arch.rs constants and name helpers, dense.rs swa mask, moe.rs seeds, expert sub-tensor count and 16 KiB stride, DEVIATIONS.md link). Claims verified about 15 of 15 checked.
No drift detected.

## crates/repack/src/synthetic_qwen/EGM.md

Coverage: sampled (2663 lines; public builder signatures and Result types, packed_triple bit guard, Drama 3 layer mask and bit overrides, vision sidecar preprocessor_config.json write, qwen4 n-gram shard layout checked by grep and targeted reads). Claims verified about 22 of 24.
- Unverified: the qwen4 n-gram invariant "heads * size <= total rows" was not confirmed against the arithmetic in qwen4.rs:ngram_tensors (3 shards of 2 rows each). Not reported as drift.

## crates/repack/src/synthetic_qwen/qwen4_decode/EGM.md

Coverage: sampled (arch.rs layer mask, budget assert and index_top_k derivation, attention_scale, ple_embed_dim, PLE layer id; mod.rs re-exports and streamed/raw-router builders; ngram.rs INT4 nibble order and multipliers; tensors.rs INT8 overrides, BF16 raw router, A_log and dt_bias naming). Claims verified about 20 of 22; GDN norm centering values not checked.
No drift detected in the claims checked.

## crates/runtime/src/families/EGM.md

Coverage: sampled (claims verified 14 of 14 checked: 12 `pub(crate) mod` declarations in mod.rs, 12 subdirectories present, all 12 child EGM.md links resolve, pub(crate) invariant; per-family one-line descriptions for museglimmer, qwen, qwen4, spark, synthetic, unlimited_ocr2 not verified against source)
No drift detected.

## crates/runtime/src/families/deepseek2/EGM.md

Coverage: sampled (claims verified about 25 of 35; attn.rs and moe.rs read by targeted search, state.rs build guards read in full; mha and steering internals not read)
- Omission: the YaRN magnitude is computed and deliberately NOT applied (`rope_mscale` is pinned to 1.0 and the value of `yarn_mscale()` is discarded), with a long comment explaining the cross-engine reason. The doc says only "yarn rope table" and omits this load-bearing decision [state.rs:RealDeepseek2State::build]
- Everything else checked matches: MLA shape guards (mask 5, q_lora_rank 0, kv_lora 512, 32/8/2 alignment), refused config flags, `top_k` vs `PHASE2_FIXED_SLOTS`, Q8_0 kv_b validation, `layer - lead` blob index, no-renorm softmax top-k, shared expert before phase 2, vocab and position errors as `Unsupported`, router histogram, mid-layer commit

## crates/runtime/src/families/flint/EGM.md

Coverage: sampled (claims verified about 25 of 30; state build guards, sigmoid top-k, sliding-window kv_start, env switches, debug constants checked; attention and prefill driver internals not fully read)
- Ambiguous: the doc says "Routed pipeline depth is 1", but the module comment in prefill.rs says the routed loop is "pipelined across two banks", and `encode_flint_layer_routed_moe_pipelined` uses `routed_pipeline_banks(...)` with a protect set when banks > 1 [prefill.rs:encode_flint_layer_routed_moe_pipelined, prefill.rs:chunked prefill module comment]. Reconcile the wording (depth 1 in flight vs two banks)
- Everything else checked matches: `ModelFamily::Flint` gate, sandwich/output-gate/K=V/softcap/tied/router-scoring refusals, RoPE skipped on mask 1 with `full_rope_theta == 0`, `softmax_offset` sizing, INT8 dtype 5 router size, `TURBOSPARK_BATCHED_GEMV` and `TURBOSPARK_ROUTED_BATCH` refusals, env vars and their defaults, `DEBUG_REPORT_ABOVE`

## crates/runtime/src/families/gemma4/EGM.md

Coverage: sampled (claims verified about 25 of 35; state build guards, router scale fold, vision refusal and span range, pending-routed depth, batched refusals and env seams checked; decode and prefill internals and the Interface function list beyond the cited ones not read)
- Overstated: the doc says "Only Gemma 4 is accepted", but the gate in `RealGemmaState::build` is flag-based (`!arch.ffn_sandwich_norms || !arch.router_scaled`) and never checks `ModelFamily`. Any install with both flags set passes, so the "only Gemma 4" claim and the family-keyed framing do not match the code [state.rs:RealGemmaState::build]
- Everything else checked matches: `num_experts == 0` dense selection, `MAX_STREAMED_EXPERTS` bound, 1/sqrt(hidden) fold into router scale, INT8 dtype 5 check, pilot probe, batched routed refusals (non-affine blobs, mapped residency, `MAX_PREFILL_EXPERT_BINDINGS`, pilot), quantized-KV refusal in batched attention, image-span refusal conditions including `window + len - 1` ring check, span attention range, routed pipeline depth 1 (`PendingRouted` debug_assert)

## crates/runtime/src/families/gptoss/EGM.md

Coverage: sampled (claims verified about 20 of 25; build guards, router bias add before top-k, unit per-expert ones, sink and bias tensor checks, env refusals, INT8 router size, decode cb1 commit checked; the body of `router_topk_gemma4` and the batched MXFP4 path not read)
No drift detected.

## crates/runtime/src/families/lfm3/EGM.md

Coverage: sampled (claims verified about 22 of 25; build guards, layer masks 0/1/6, taps range, BF16 conv weight, no lm_head, RMS_EPS, fused FFN INT4 gate, env defaults, chunked refusals, ring-capped width, reset; rope class selection read in full)
- Naming only, not a behavior change: the doc says sliding layers rotate all pairs at `rope_theta`, while `rope_params_for` takes the parameter `sliding_theta` [attn.rs:rope_params_for]
No other drift detected.

## crates/runtime/src/families/llama/EGM.md

Coverage: sampled (claims verified about 25 of 30; state build guards, MiniMax FP32 dtype 3 and finite bias checks, correction bias selection-only, batched-attention eligibility and capacity, dense `intermediate_size` rule, MoE and dense chunked refusals, phase 2 zero_hidden seed; all Interface symbols confirmed to exist; router and prefill bodies not fully read)
No drift detected.

## crates/runtime/src/families/museglimmer/EGM.md

Coverage: sampled (state.rs build read in full; mod.rs, attn.rs, mlp.rs, prefill.rs checked by targeted greps and partial reads; source not read in full)
No drift detected.

## crates/runtime/src/families/qwen/EGM.md

Coverage: sampled (claims checked with st and targeted reads; 14.7k-line source not read in full)
- Interface line 13 names `prefill_chunk_real_qwen_dense` with `_priming_mtp` / `_priming_dflash` suffixes. The code names are `prefill_chunk_real_qwen_dense_priming_mtp` and `prefill_chunk_real_qwen_dense_priming_dflash` [prefill.rs:prefill_chunk_real_qwen_dense_priming_mtp]. The shorthand does not resolve as written.
- Aside, code side and not a doc item: the plain-entry refusal text in [prefill.rs:prefill_chunk_real_qwen_dense] tells the user to use `prefill_chunk_priming_drafter`, which is not a runtime function (it exists only as a test helper in runtime/tests/speculative.rs).

## crates/runtime/src/families/qwen/dflash_draft/EGM.md

Coverage: sampled (claims checked by targeted reads; 1382-line directory not read in full)
- Side Effects cites [context_kv.rs:dflash_context_write] for "zeroes capture_rows after a write". context_kv.rs never touches `capture_rows`. The reset is `d.capture_rows = 0` in [mod.rs:dflash_run_block]. The citation points at the wrong function.

## crates/runtime/src/families/qwen4/EGM.md

Coverage: sampled (claims checked by targeted reads; 5.8k-line directory not read in full)
No drift detected.

## crates/runtime/src/families/spark/EGM.md

Coverage: sampled (build refusals, rope-by-class selection, GELU, buffer sizing and steering order checked; attn.rs, mlp.rs, prefill.rs not read in full)
No drift detected.

## crates/runtime/src/families/synthetic/EGM.md

Coverage: sampled (mod.rs and layer.rs checked by targeted reads; 620-line directory not read in full)
- Verification says `cargo test -p turbospark-runtime synthetic`. This directory has no #[test] functions. The only tests the filter selects are `synthetic_checkpoint_metal_matches_cpu_reference` [runtime/src/qwen3_asr_metal.rs] and `synthetic_metal_decoder_step_matches_cpu` [runtime/src/whisper/tests.rs], which belong to other families. The documented command runs no test for this flow.

## crates/runtime/src/families/unlimited_ocr2/EGM.md

Coverage: sampled (build refusals, prompt checks, MoE routing, commit count checked; source not read in full)
- Invariant says `top_k` above `MAX_STREAMED_EXPERTS` is refused. [state.rs:build] also refuses `top_k == 0`. The doc omits the zero case.
- Verification says `cargo test -p turbospark-runtime unlimited_ocr2`. This directory has no #[test] functions. The integration tests in runtime/tests/real_forward_unlimited_ocr2.rs that I checked (a_used_runner_replays_a_fresh_ones_generation, a_query_routes_to_image_blocks_past_its_local_window, candidates_restrict_a_query_to_its_page) do not contain the filter string in their names. The command may select zero tests. Not run; unverified.

## crates/runtime/AGENTS.md

Coverage: sampled (claims verified about 16 of about 35; source not fully read; all linked doc and code paths resolve)
- `TURBOSPARK_QWEN_FUSED_FFN=1` is described as the switch that activates fused INT4 FFN dispatch. Code is on by default and only `=0` turns it off (kill switch). [crates/runtime/src/families/qwen/mod.rs:fused_ffn]
- `TURBOSPARK_GDN_LAZY_TAPE=1` is described as the opt-in that records lazy tape updates. Code is on by default and only `=0` disables it. [crates/runtime/src/real_forward_rollback.rs:gdn_lazy_tape]
- `TURBOSPARK_INT4_MMA8` is said to gate only qwen35 dense ("Only qwen35 dense is gated"). The shared-tile gate `int4_mma8_takes` is also called from the qwen attention, qwen MoE, qwen4 attention and lfm3 MLP paths. [crates/runtime/src/real_forward_dispatch.rs:int4_mma8_takes] [crates/runtime/src/families/qwen/attn.rs] [crates/runtime/src/families/qwen/moe.rs] [crates/runtime/src/families/qwen4/attn.rs] [crates/runtime/src/families/lfm3/mlp.rs]

## crates/runtime/README.md

Coverage: sampled (claims verified about 20 of about 30; test-suite names not checked individually; source not fully read)
- Says the crate has "81 integration test suites in `tests/`". `crates/runtime/tests/` holds 105 `.rs` files (some may be shared helper modules, so the suite count was not confirmed). [crates/runtime/tests]
- Key Modules omit the native audio and Core ML stack and the encoder runner entirely: `native_audio.rs:AudioSession`, `native_audio/` (`AudioStreamSession`, `MultitalkerStreamSession`), `coreml/` (`ParakeetCoreMl`, `DiarizationCoreMl`, `UnlimitedOcrCoreMl`), `encoder/` (`EncoderRunner`), the TTS and music runners (`kokoro.rs`, `megatts3.rs`, `omnivoice.rs`, `music3.rs`, `fish_drama3*.rs`), the Qwen3-ASR runner (`qwen3_asr*.rs`). Also omitted: `heavy.rs:HeavyWorkGuard`, `hidden_trunk.rs:HiddenTrunk`, `startup.rs`. [crates/runtime/src/native_audio.rs:AudioSession] [crates/runtime/src/encoder/mod.rs:EncoderRunner] [crates/runtime/src/heavy.rs:HeavyWorkGuard]
- Gotcha 3 names `GdnStateManager` with no path. The type is referenced as `gpu::GdnStateManager` (crates/gpu), not defined in the runtime crate, so the reference should name its crate. [crates/runtime/src/families/qwen/attn.rs]

## crates/runtime/EGM.md

Coverage: sampled (all four subdirectory links resolve and the subdirectory list was checked against the tree; no source claims beyond that)
No drift detected.

## crates/runtime/src/EGM.md

Coverage: sampled (claims verified about 60 of about 100: every named Interface symbol confirmed to exist, env-var reads and several invariants checked; source not fully read)
- Admission row cites `raw_completion.rs:check_admission` for "prefix reuse is clamped to prompt_len - 1". `check_admission` (raw_completion.rs:126) holds the ZeroBudget and max_context checks; the clamp is at raw_completion.rs:245 (`.min(prompt_ids.len() - 1)`) and was not confirmed to sit inside `check_admission`. Fix the citation. [crates/runtime/src/raw_completion.rs:check_admission]

## crates/runtime/examples/EGM.md

Coverage: sampled (claims verified about 15 of about 30: all 19 example binaries present, macOS cfg gates, env reads; probe flags and warmup behavior not checked)
No drift detected.

## crates/runtime/scripts/EGM.md

Coverage: sampled (claims verified about 6 of about 12: flags `--arms` and `--runs`, `digests_agree`, `quiet_host` rule, `pmset` call; not every flag checked)
No drift detected.

## crates/runtime/tools/EGM.md

Coverage: sampled (claims verified about 6 of about 12: ffmpeg requirement, OUT_DIR default chain, SHA-256 abort, oversize abort; generator output files not checked)
No drift detected.

## crates/runtime/src/coreml/EGM.md

Coverage: sampled (claims verified about 14 of about 25: 1 MiB JSON cap, `*_UNITS` env reads, `TURBOSPARK_COREML_ALLOW_QUANTIZED_GPU` refusal, macOS 15 gates, 32000-sample append cap, `read_limited`, OCR `vocab_half` check; file-contract constants and OCR `PAGE_SIZE` not checked)
No drift detected.

## crates/runtime/src/encoder/EGM.md

Coverage: sampled (claims verified about 6 of about 10: supported model types, tokenizer load swallowed to None, 8-bit group size check, erf GELU flag, CLS-only pooling entry points; `check_encoder_dir` ordering and bias defaults not checked)
No drift detected.

## crates/runtime/src/native_audio/EGM.md

Coverage: sampled (claims verified about 6 of about 12: "stream is busy" string in both sessions, Unified admission 3x plus 256 MiB, Multitalker admission quarter plus 512 MiB, `multitalker-stream` thread name, named types; Unified finish reset and append minimum not checked)
No drift detected.

## crates/runtime/src/router_hist/EGM.md

Coverage: sampled (claims verified about 3 of about 8: env gates `TURBOSPARK_ROUTER_BATCHES` and `TURBOSPARK_ROUTER_BATCH_LIMIT`, default limit 1,000,000, `BatchCapture` and `BatchRow` exist; `batches.rs` internals not read)
No drift detected.

## crates/runtime/src/vision/EGM.md

Coverage: sampled (claims verified about 12 of about 30: all named Interface symbols exist, `VISION_SLOTS` = 2, `TURBOSPARK_VISION_RESIDENCY` and `TURBOSPARK_VISION_OVERFLOW` reads, qwen2.5 `block % 8 == 7` window rule, SigLIP final LayerNorm eps 1e-5, Uocr2 index 1 MiB and bin 2 GiB caps; the FP16 bound is checked through a `FP16_MAX` constant whose value was not confirmed as 65504; residency and sidecar claims not checked)
No drift detected.

## crates/runtime/src/whisper/EGM.md

Coverage: sampled (claims verified about 10 of about 22: config and tokenizer read caps, 24 GiB admission cap, decode cap of 440, `TURBOSPARK_WHISPER_DEVICE=cpu`, `TURBOSPARK_WHISPER_PROFILE=1`, named types and methods exist; 480000-sample padding, 8-repeat stop and stride value not checked)
No drift detected.

## crates/selection/EGM.md

Coverage: full read (overview file; its one link resolves to crates/selection/src/EGM.md).
No drift detected.

## crates/selection/src/EGM.md

Coverage: full read (all 7 src files, about 1477 lines, read in full; tests/ sampled by name and by the grep hits for symbols the doc names; claims verified N of M = all symbol names and all listed invariants checked).
- Omitted: penalty scope. [src/choose.rs:shape_candidates] applies repetition penalty over the full history (prompt included), while [src/penalty.rs:apply_presence_and_frequency_penalty] counts only the generated suffix `history[len - position..]`, with `position` clamped to `history.len()`. The doc's Invariants do not state this split, and callers in crates/runtime rely on `position` being the generated-token count.
- Omitted: pipeline and composition order in [src/choose.rs:shape_candidates]: penalties, softmax, rank (top-k cut), top-p against full-distribution mass, top-k cap on the survivors, min-p against `ranked[0]`, then reweight and draw. Pinned by tests/min_p.rs and tests/truncation_order.rs, but the doc only lists the helpers.
- Omitted: deterministic mode. [src/choose.rs:select] returns the argmax for `temperature == 0.0` (via [src/shaping.rs:ShapingConfig::is_deterministic]), ignoring penalties, truncation and seed; ties go to the lowest index ([src/choose.rs:argmax]). The doc never mentions this arm.
- Omitted: fallbacks that keep results defined. Empty survivor set returns argmax with probability 1 ([src/choose.rs:shape_candidates]). [src/choose.rs:reweight] divides by the top surviving probability before the power and returns uniform if the total underflows to zero. [src/choose.rs:draw] and [src/distribution.rs:ShapedDistribution::draw] return the last survivor on rounding at the tail.
- Omitted: panics versus errors. [src/distribution.rs:ShapedDistribution::new] panics (assert) on mismatched lengths or empty support rather than returning SelectionError. [src/truncation.rs:assert_u32_domain] panics when the candidate domain exceeds u32::MAX, reached from [src/truncation.rs:rank_top_k_u32_into] and fill_identity.
- Omitted: [src/penalty.rs:apply_repetition_penalty] clamps the penalty to the f32 range and saturates results to f32::MIN/MAX so non-finite scores cannot reach the softmax. Not in the doc.
- Omitted public API: [src/shaping.rs:ShapingConfig] getters (`temperature`, `top_k`, `top_p`, `repetition_penalty`, `seed`, `presence_penalty`, `frequency_penalty`, `min_p`, `is_deterministic`) and the `SelectionError` Display format ("invalid selection input: ..."). The doc names only the constructor and builders.

## crates/selection/README.md

Coverage: full read of the README; claims verified by targeted reads of choose.rs, derive.rs, penalty.rs, truncation.rs, distribution.rs and lib.rs, and by grep of Cargo.toml, downstream Cargo.toml files and the test file list. Test-file descriptions sampled by test function names (rejection_sampling, domain_guards, distribution, determinism, host_sampler_cost); 11 test files confirmed.
- [src/choose.rs:select] does not dispatch to rejection sampling. The Purpose and Key Modules text says `select` "Dispatches greedy argmax, categorical sampling, or rejection sampling". select only has the argmax arm and the categorical draw. Rejection sampling is caller-side: it uses [src/choose.rs:shaped_distribution] and [src/distribution.rs:residual] (see crates/runtime/src/speculative.rs).
- Gotcha 2 overstates reproducibility. "Deterministic Step Seeding ... guarantees bit-exact reproducible token sequences across identical runs" holds only when `seed` is `Some`. With `seed: None`, [src/derive.rs:derive_step_value] reads `SystemTime::now()` and draws differ between runs. The value is also a stateless hash of (seed, position) ([src/derive.rs:mix]), not a seeded generator carried across steps.
- Key Modules, truncation.rs: "Truncation algorithms including ... nucleus top-p filtering, and min-p thresholding" implies these are the live path. The live top-p and min-p logic is inlined in [src/choose.rs:shape_candidates]. [src/truncation.rs:truncate_by_probability_mass] and [src/truncation.rs:truncate_by_min_p] are reference implementations that only tests call (see src/EGM.md).
- Key Modules, penalty.rs: "Penalizes repeated tokens based on historical token frequency and occurrence" is overstated. [src/penalty.rs:apply_repetition_penalty] penalizes each distinct history id once, regardless of count, and it uses the full history including the prompt. Frequency counting exists only in [src/penalty.rs:apply_presence_and_frequency_penalty], and it uses the generated suffix only.
- Omitted: the prompt-versus-generated penalty split and the `position` contract described above ([src/choose.rs:shape_candidates]). This is the behavior a caller most needs to know.
- Omitted: the deterministic arm at `temperature == 0.0` ([src/shaping.rs:ShapingConfig::is_deterministic]) and its tie-break ([src/choose.rs:argmax]).

## crates/selection/AGENTS.md

Coverage: full read; both links resolve (.claude/docs/modules/selection.md and docs/TESTING.md exist); rule claims checked against [src/shaping.rs:ShapingConfig::is_deterministic] and [src/choose.rs:select].
No drift detected.

## crates/server/EGM.md

Coverage: sampled (claims verified 6 of 6 by source lookup; cargo test not run)
No drift detected.

## crates/server/AGENTS.md

Coverage: sampled (claims verified 9 of 11; FFI route surface and scripted vs real separation not checked)
- The rule that the in-process FFI host "reaches the same posture" (prefix reuse off by default, via `ServerOptions.prefixReuse`) does not hold. [crates/ffi/src/open.rs:442] sets `prefix_reuse = options.prefix_reuse.unwrap_or(true)`, and [crates/ffi/src/open_tests.rs:prefix_reuse_is_off_only_when_asked_and_defaults_to_on] asserts an absent option means on. [crates/ffi/src/server.rs:513] applies the server flag only when it is `Some`, so an absent `prefixReuse` keeps the session default (on). The server binary default is off ([src/args.rs:178]), which matches the doc. Checked by reading the default paths; not exercised at runtime.

## crates/server/README.md

Coverage: sampled (claims verified 14 of 17; main.rs and messages.rs read in part; the BERT/XLM-RoBERTa encoder claim is not checkable inside this crate because the server wraps a generic `EncoderRunner` in [src/encoder_model.rs:RealEncoderModel])
- Queue section says `queue.rs` has "no queue timeout or depth cap". False for depth: [src/queue.rs:DEFAULT_MAX_QUEUED] caps waiters at 64, `--max-queued` sets it (0 is unlimited, [src/args.rs:max_queued]), and a request past the cap gets 429 with Retry-After ([src/queue.rs:reserve], [src/handler/mod.rs:queue_full_response]). The timeout half was not checked.
- Binary Execution example labels `--pool-size 2` a "round-robin multi-model concurrency pool". `--pool-size` opens N runners of one install (`--model`'s ONE install, [src/args.rs:USAGE]; [AGENTS.md] says the same). Multiple models come from repeating `--model`. Pool dispatch is per install ([src/registry.rs:PoolRegistry]).
- "Uses zero-allocation translations between Anthropic and OpenAI protocols" is not supported. The translation code builds owned Strings, serde_json values and clones on every request ([src/messages.rs] has about two dozen such sites). Overstated.
- Omitted: on a server with no API key, every route except /health refuses a non-loopback Host ([src/host_check.rs:require_loopback_host]). The loopback allowance also includes `host.docker.internal` ([src/host_check.rs:is_loopback_host]). `--api-key` falls back to `$TURBOSPARK_API_KEY` ([src/args.rs:USAGE]). None of these three appears in the README.

## crates/server/src/EGM.md

Coverage: sampled (claims verified 30 of 30 symbol and invariant checks; lib.rs routing, queue, bind, auth and host_check read directly; registry.rs and observe.rs checked by symbol lookup only)
- Invariant "With no API key, every route except /health refuses a non-loopback Host" omits the `host.docker.internal` exception ([src/host_check.rs:is_loopback_host]). Low severity, but it is a deliberate allowance in the rebinding guard.

## crates/server/src/audio/EGM.md

Coverage: sampled (claims verified 19 of 20; audio/mod.rs, speech.rs consent block, jobs.rs, models.rs and ws.rs read in part; stt.rs decode path checked by lookup)
- Consent invariant says MegaTTS3, and OmniVoice when a reference is given, need `consent: true` ([src/audio/speech.rs:speech]). It omits Drama 3 (`fish_drama3`): with `reference_audio` present, the same function requires `consent: true` and `reference_text`. [src/audio/mod.rs:clones_voices] lists `fish_drama3` as a cloning family, so the doc's list is incomplete.
- Omitted realtime limits: [src/audio/ws.rs:MAX_BUFFER_SECONDS] (300 s of buffered audio), [src/audio/ws.rs:MAX_FRAME_BYTES] (1 MiB per frame) and [src/audio/ws.rs:IDLE_TIMEOUT] (60 s idle close). Low severity.

## crates/server/src/guardrails/EGM.md

Coverage: sampled (claims verified 8 of 9; extra_formats.rs header and `rescue()` dispatch read; `run_guarded` body in guardrails.rs not read)
- Dialect list is incomplete. The doc says the helper covers GLM, MiniMax invoke and Kimi K2, which are the dialects forge does not parse. [src/guardrails/extra_formats.rs:rescue] also tries `parse_qwen_xml` and `parse_gemma`. Gemma is not among forge's four strategies, and the Qwen XML parser overlaps forge's own. The module header of the same file ("Four families", GLM/MiniMax/Kimi/Longcat) does not mention either parser, so source and header disagree.

## crates/server/src/handler/EGM.md

Coverage: sampled (claims verified 14 of 14; handler/exec.rs read in full, handler/mod.rs and plan.rs read in part and checked by symbol lookup)
No drift detected.

## crates/server/src/responses/EGM.md

Coverage: sampled (claims verified 11 of 11; map.rs warnings, sse.rs build_response and mod.rs header read; event builders checked by name only)
No drift detected.

## crates/streaming/EGM.md

Coverage: full read (7-line file; package name and src link checked)
No drift detected.

## crates/streaming/README.md

Coverage: sampled (claims verified about 30 of about 32; src not fully read, targeted reads of every cfg gate, env var, and cited symbol)
- Crate Gotchas 2 says "even a single expert cache miss executes with wide I/O parallelism". Overstated: a miss is split into chunks only when the expert stride exceeds the 840 KiB miss chunk, so smaller blobs are one chunk and run inline on the caller. [pread_streamer.rs:execute_expert_cache_plan], [read_pool.rs:run_batch]
- Tests section omits `tests/offset_overflow.rs` (refusal of wrapping expert offsets on direct and cached loads). [tests/offset_overflow.rs]

## crates/streaming/AGENTS.md

Coverage: full read (25-line file; commands, script path, doc link, and the Linux cfg gate of the io_uring ring module checked)
- Not drift, a doc-to-doc inconsistency: the "Detailed module guide" link targets `.claude/docs/modules/streaming.md`, which describes itself as an archived copy whose active entry point is this AGENTS.md. The link resolves; the labels disagree.

## crates/streaming/src/EGM.md

Coverage: sampled (claims verified about 40 of about 45; src is 3073 lines across 11 files and was not fully read; targeted reads plus a pub-item skim)
- Side Effects says the streamer "sets `F_NOCACHE`" unconditionally. It is set only when `TURBOSPARK_EXPERT_NOCACHE=1` and the `fcntl` succeeds, and `nocache_active` reports the result. [pread_streamer.rs:PreadExpertStreamer], [disk_io.rs:nocache_requested]
- Interface omits public items that callers use: `PreadExpertStreamer::slot_data`, `slot_allocation` (raw pointer and length seam), `load_expert_into_slot`, `read_experts_into_bank`, `execute_speculative_reservation`, `advise_expert_misses`, `advise_expert_cache_plan_misses`, `non_resident_experts`, `layout`; `ExpertCache::plan` (panicking variant), `reserve_speculative_slots`, `publish_speculative_reservation`; `MappedExpertLayer::experts_per_layer`; `ExpertIoStats::accumulate`, `amplification`; `AlignedSlot::as_ptr`, `len`, `is_empty`. [pread_streamer.rs:PreadExpertStreamer], [expert_cache.rs:ExpertCache], [mapped_experts.rs:MappedExpertLayer], [disk_io.rs:ExpertIoStats], [aligned_slot.rs:AlignedSlot]

## crates/tokenizer/EGM.md

Coverage: sampled (claims verified 1 of 1, source not fully read)
No drift detected.

## crates/tokenizer/AGENTS.md

Coverage: sampled (claims verified 9 of 11, source not fully read)
No drift detected.
- Not verified: the added-token renumbering claim (external `tokenizers` behavior) and the Gotcha numbers (52, 49, 37), which point at a gotchas file outside this directory.

## crates/tokenizer/README.md

Coverage: sampled (claims verified 12 of 14, source not fully read)
- Test count drift: doc says "18 comprehensive test suites in `tests/`", but `crates/tokenizer/tests/` holds 25 .rs files [tests/:test suites]. The list also omits chatml_no_tool_call.rs, content_marker_spelling.rs, deepseek_vl2_image_render.rs, regex_backend_hostile_pattern.rs, sidecar_size_cap.rs, stop_matcher_long_stop.rs and vision_markers_muse.rs [tests/:test suites].
- `config.rs` is described as "Generation configuration dialect mapping". It actually holds `TokenizerConfig`, `GenerationConfig`, `TokenRef`, EOS parsing and chat-template sidecar extraction, not dialect mapping [crates/tokenizer/src/dialect/config.rs:TokenizerConfig].
- `reasoning.rs` is described as handling "reasoning token formatting". It defines `ReasoningEffort` (level, enable_thinking) and the `reasoning_support` / `accepted_reasoning_levels` probes, with no token formatting [crates/tokenizer/src/reasoning.rs:ReasoningEffort].

## crates/tokenizer/src/EGM.md

Coverage: sampled (claims verified 14 of 15, source not fully read)
- Omission: `MfDetokenizer::push` re-decodes the whole `stable_ids` history on every push, so generation is O(n^2) decode work by design [crates/tokenizer/src/detokenizer.rs:MfDetokenizer]. The doc's Interface entry for `push` / `flush` does not mention this cost.

## crates/tokenizer/src/chat_template/EGM.md

Coverage: sampled (claims verified 8 of 9, source not fully read)
- Omission: `verify_image_markers` also rejects `vision_start == image_pad` and any id that does not resolve in the vocabulary, before the render check [crates/tokenizer/src/chat_template/mod.rs:verify_image_markers]. The Invariants line describes only the exactly-one-start-and-one-pad render check.

## crates/tokenizer/src/dialect/EGM.md

Coverage: sampled (claims verified 9 of 11, source not fully read)
- Omission (side effect): all sidecar reads are size-capped. `tokenizer_config.json` and both chat-template sidecars are capped at 16 MiB (`MAX_SIDECAR_BYTES`), and `generation_config.json` at 1 MiB (`MAX_GENERATION_CONFIG_BYTES`). An over-cap file is treated as absent, so `chat_template.jinja` falls through to `chat_template.json` and then the config key, and a template-less dialect fallback can result [crates/tokenizer/src/dialect/mod.rs:read_capped]. The Side Effects section says only that files are read.

## crates/tokenizer/src/structured_decoder/EGM.md

Coverage: sampled (claims verified 8 of 10, source not fully read)
- Divergence: "`finish` returns `ToolCallParserError` when a tool call is left incomplete" is true only for the non-Harmony, non-Mistral dialects. `finish` also closes and emits the tool call for Harmony (`close_harmony_tool`) and Mistral (`close_mistral_tool`), which have no closing token in-stream [crates/tokenizer/src/structured_decoder/mod.rs:StructuredAssistantDecoder::finish]. The Invariants line omits this, and it is load-bearing: a consumer that never calls `finish` drops those calls.
- Omission: `StructuredAssistantDecoder::new` takes `prompt_ids` and starts in the Thought channel when the rendered prompt ends inside an open think frame (`prompt_opens_thought`) [crates/tokenizer/src/structured_decoder/mod.rs:StructuredAssistantDecoder::new]. The Interface entry lists `new` with no parameter description.
- Omission: after any parser error the decoder is permanently failed, and every later `consume` returns `ToolCallParserError::Malformed` [crates/tokenizer/src/structured_decoder/mod.rs:StructuredAssistantDecoder::consume]. The doc lists `take_failed_span_text` but not this state.

## crates/tokenizer/src/tool_call/EGM.md

Coverage: sampled (claims verified 9 of 10, source not fully read)
- Omission: function-name rules are stricter than "validated": non-empty, at most 64 bytes, ASCII alphanumeric plus `_` and `-` only [crates/tokenizer/src/tool_call/mod.rs:is_valid_function_name].

## crates/vision-io/EGM.md

Coverage: full read (doc and Cargo.toml read in full; lib.rs read in full; decode_image_file read)
No drift detected.

## crates/vision-io/src/EGM.md

Coverage: sampled (claims verified roughly 75 of 80; read in full: decode.rs, error.rs, preprocess.rs, normalize.rs, rounding.rs, smart_resize.rs, params.rs, patchify.rs, rope.rs; gemma4.rs and unlimited_ocr2.rs read in part; other family modules checked by symbol and signature only, source not fully read)
- Invariant "Patch sequences are in merge-window order and each patch row is (T, P_h, P_w, C)" is stated as a crate-wide invariant, but it holds only for the Qwen path. Muse emits raster order with rows (T, C, P_h, P_w) [muse.rs:muse_patch_rows] and Gemma 4 emits row-major [16][16][RGB] rows with no temporal axis [gemma4.rs:Gemma4Patches]. The crate-level module doc in lib.rs makes the same overstatement ("every per-patch sequence in this crate") [lib.rs:module docs].
- Errors: `ImageTooLarge` is listed as an error the crate reports, but nothing constructs it; it is only matched in Display [error.rs:VisionIoError::ImageTooLarge]. Oversized images fail inside the image crate's Limits and surface as `Decode` [decode.rs:reader]. The doc should say the variant is unused, or the cap line should say oversize returns `Decode`.
- Invariant "min_pixels above MAX_CONFIG_PIXELS is refused": accurate, but omits that max_pixels above MAX_CONFIG_PIXELS is silently clamped, not refused [params.rs:from_preprocessor_config_json].
- Interface: the re-export list for unlimited_ocr2 says "only these re-exports are reachable", but lib.rs also re-exports `uocr2_grid`, `UOCR2_BASE_SIZE`, `UOCR2_CROP_SIZE`, `UOCR2_MAX_CROPS` and `Uocr2Page` consumers' constants [lib.rs:unlimited_ocr2 re-export]. Those are omitted from the list.
- Interface omissions: several public items are not listed, for example `ov_select_best_resolution`, `ov_patch_output_size` [onevision.rs], `DeepseekOcrRow`, `DeepseekOcrView`, `DeepseekVl2Row` [deepseek_ocr.rs, deepseek_vl2.rs], and the public constants `GEMMA4_SOFT_TOKEN_BUDGETS`, `MINICPM_*_ID`, `OV_*`, `UOCR2_*`, `OCR_*`, `DSVL2_PATCH` [lib.rs re-exports]. The `resize`, `muse`, `rope` and `pos_embed` modules are public, so `resample_pil_with`, `PilFilter`, `muse_*` are reachable by module path only, which the doc does not mark (the rope entry does).

## crates/vision-io/README.md

Coverage: sampled (claims verified roughly 30 of 35; Cargo.toml, lib.rs, preprocess.rs, params.rs, patchify.rs, rope.rs, smart_resize.rs read; link targets and tests/ listing checked; test bodies not read)
- Test-oracle claim: "Every golden fixture under tests/generated/ is probed from the vendored mlx-vlm reference by scripts/qwen3vl_vision_oracle.py". `tests/generated/gemma4_preprocess_oracle.rs` header says it is generated by scripts/gemma4_vision_oracle.py [tests/generated/gemma4_preprocess_oracle.rs:header]. Only the qwen3 fixtures come from the named script.
- Scope: the intro says the crate serves the qwen3_5 tower plus Gemma 4 and Unlimited-OCR-2. Code also exposes CLIP, MiniCpm, DeepSeek-OCR, DeepSeek-VL2, Pixtral, Muse and OneVision pipelines [lib.rs:pub mod and pub use list].
- Key Modules: omits the modules listed above (clip.rs, minicpm.rs, deepseek_ocr.rs, deepseek_vl2.rs, pixtral.rs, muse.rs, onevision.rs). It also lists `unlimited_ocr2.rs` as a key module without saying it is private (`mod unlimited_ocr2;` in lib.rs), so its items are reachable only through re-exports [lib.rs:unlimited_ocr2].
- Key Modules, `preprocess.rs`: describes the chain as smart resize, bicubic, normalize, patchify. Omits that `ResizeRule::MuseTokens` bypasses this chain entirely [preprocess.rs:preprocess].
- Tests list is not exhaustive. Omitted: `tests/muse_params.rs`, `tests/muse_wrap.rs`, `tests/rope_theta.rs`, `tests/params_geometry.rs`, `tests/gemma4_preprocess_parity.rs`.

## crates/vision-io/AGENTS.md

Coverage: full read (doc is short; all four link targets checked for existence)
No drift detected.

## Notes for the caller (not doc drift)

- Source comments in rounding.rs, smart_resize.rs and patchify.rs say "see NOTICE". The NOTICE file exists only at the repo root (/Users/whit3rabbit/Documents/GitHub/turbospark/NOTICE), not in crates/vision-io.
- Index: `st` reported its index as stale and locked by another process; searches still ran against the stale index, so line numbers from `st` were confirmed by Read where cited.


# Crates drift check, third pass after the second doc-fix round (haiku-reported)

Third rerun of the same 34 jobs. Same caveats: mostly sampled, haiku-reported, not independently verified. README.md and AGENTS.md were not edited, so their findings repeat.

## crates/audio/AGENTS.md

Coverage: sampled (claims verified 14 of about 20 checkable; all relative links checked; quant, FFT, Slaney, Whisper and NeMo padding rules checked in source)
- Line 45 "Verify the local clone at ../mlx-audio": relative to crates/audio this resolves to crates/mlx-audio, which does not exist. The sibling checkout is at repo-root ../mlx-audio (GitHub/mlx-audio), and its HEAD is e1b19b9054bf163f5d812221a54fcc346f1890e9 as stated. The base directory of the path is ambiguous. (d)

## crates/audio/EGM.md

Coverage: full read (11 lines); claims checked against Cargo.toml [Cargo.toml:exclude], the directory listing and the tools/EGM.md, src/EGM.md, examples/EGM.md and scripts/EGM.md links.
No drift detected.

## crates/audio/README.md

Coverage: sampled (module map cross-checked against src/lib.rs; every role family listed was matched to a src directory; all src/ links resolve; test commands checked for the named test files)
No drift detected.

## crates/audio/MODELS.md

Coverage: sampled (claims verified about 35 of about 200 table and prose claims; all src/ and ../ links resolve; 101 source-directory count matches upstream after excluding __pycache__; table rows not individually read against source)
- Lines 255-256 say gpt2, cohere2, bailing_moe, switch_layers and falcon_h1/ssm are "still upstream-only", but the crate ships them: [src/lm/mod.rs], [src/lm/gpt2.rs], [src/lm/falcon_h1.rs], [src/lm/ssm.rs]. The arktts row (line 260) and indextts row (line 274) also say they are built on falcon_h1 and gpt2. The sentence is stale. (b)
- Line 157 says shared codecs belong under `models/codec/`. No such directory exists; codecs live at [crates/audio/src/codec/mod.rs] (src/codec). (d)
- Line 217 links `../../docs/SWIFT_BINDINGS.md#audio-binding-plan`. That anchor does not resolve: docs/SWIFT_BINDINGS.md has no "audio binding plan" heading, only "## Audio" at line 1784. (d)
- Layout tree (lines 145 and 148) lists `testdata/` twice. (d)
- CODEC table (lines 448-467) has no row for `spark_bicodec`, which [src/codec/mod.rs] declares and the tts/spark row (line 295) links as its codec port. (c)
- Aliases named in the table that do not occur in [crates/audio/src], [crates/model-io/src] or [crates/catalog/src]: `higgs_multimodal_qwen3` (line 273), `dramabox-tts` (line 269), `vibevoice_streaming` (line 298), `csm` and `marvis` (line 293), `samaudio` (line 415), `dialoguesidon` (line 408), `breeze-tts` (line 263). The alias mapping may live in runtime or elsewhere; it was not checked there. (a, scope limited)
- Line 188 says the speech catalog has nine pinned Whisper rows. The count of Whisper rows is correct, but [crates/catalog/src/speech_models.json] has ten rows in total; the tenth is `qwen3-asr-06b-8bit`, which the sentence does not mention. (c, minor)

## crates/audio/examples/EGM.md

Coverage: sampled (all 13 example files matched to the interface list; write and argument behavior checked by grep in each file; example bodies not read)
No drift detected.

## crates/audio/scripts/EGM.md

Coverage: sampled (generator, benchmark and NeMo tool file names checked; subprocess calls checked in benchmark_music3.py; one generator docstring and the benchmark docstring read; the "interleave variants" claim not checked)
No drift detected.

## crates/audio/scripts/nemo_218d0b4/README.md

Coverage: full read (27 lines); every listed script and subcommand checked, the NeMo commit checked in pyproject.toml and uv.lock, the referenced bench script and crate README checked to exist. The macOS arm64 and PyPI CPU wheel resolution claim was not checked.
No drift detected.

## crates/audio/src/EGM.md

Coverage: sampled (every named symbol checked to exist in its file or in the crate root re-exports; invariants and side effects checked in source; ops.rs, fft.rs, mel.rs and resample.rs bodies not read in full)
- Side Effects says the TableCache tables (`PLANS`, `BANKS`, `KERNELS`) "persist for the life of the process". The cache is bounded: it is cleared when it reaches `max_entries`, and tables marked not cacheable are never stored. [src/dsp.rs:TableCache::get_or_build], used by [src/fft.rs:RealFftPlan::cached], [src/mel.rs:mel_projector_cached] and [src/resample.rs] KERNELS. The claim is overstated. (b)
- Not this file, but in the same directory: the [src/lib.rs] module map says `quant` covers "2/3/4/6/8-bit", while [src/quant.rs:QuantScheme] validation accepts 2/3/4/5/6/8. The same map lists only Kokoro under `tts`, although about 40 families exist. Both are stale doc comments. (b, source comment)

## crates/audio/src/codec/EGM.md

Coverage: sampled (shared block signatures, the pub mod list in [src/codec/mod.rs], REFERENCE_COMMIT in all 16 family mod.rs files that declare it, and the side-effect claims checked; conv.rs, vq.rs and dac.rs bodies not read)
No drift detected.

## crates/audio/src/codec/README.md

Coverage: sampled (family list and every link checked; per-family architecture prose not checked against source)
No drift detected.

## crates/audio/src/codec/bigvgan/EGM.md

Coverage: sampled (interface names, SpeechError variants, reference pin, env and side-effect claims checked in mod.rs and tests.rs; forward pass not read)
No drift detected.

## crates/audio/src/codec/bigvgan/README.md

Coverage: sampled (test names, TURBOSPARK_BIGVGAN_MODEL, fixture and tool paths, snakebeta default and the resblocks.{i * num_kernels + j} indexing checked; sample-rate and upsample values are config-driven and have no code default to compare)
No drift detected.

## crates/audio/src/codec/dacvae/EGM.md

Coverage: sampled (interface names, validate checks, BadConfig/Input/Tensor variants, and the absence of env and fs writes in mod.rs checked; encode and decode bodies not read in full)
No drift detected.

## crates/audio/src/codec/dacvae/README.md

Coverage: sampled (default geometry, hop length, test count, env var, watermark and chunked-decode scope checked; static padding claim not checked)
- Verification says "cargo test -p turbospark-audio --lib codec::dacvae (3 tests)". tests.rs has five #[test] functions: config_defaults_match_reference, tiny_end_to_end_matches_reference, real_checkpoint_gate, config_rejects_degenerate_geometry, load_revalidates_hand_built_config [src/codec/dacvae/tests.rs]. (b)

## crates/audio/src/codec/descript/EGM.md

Coverage: sampled (interface names, decode_codes validation (count, length and index range), TURBOSPARK_DAC_MODEL gate and no-write claims checked; compress body not read)
No drift detected.

## crates/audio/src/codec/descript/README.md

Coverage: sampled (default config values, test names, fixture tool path and consumer claims checked; L2-normalized codebook claim not checked)
- Architecture table says strides are "[2, 4, 8, 8] (44.1 kHz) or [2, 4, 8, 8] / [8, 5, 4, 2]". The code default for encoder_rates is [2, 4, 5, 8] and for decoder_rates [8, 5, 4, 2] [src/codec/descript/mod.rs:DacConfig::from_json]. (b)
- The intro says DAC serves as the acoustic tokenizer for Chatterbox. Chatterbox uses the S3 codec, not DAC: [src/tts/chatterbox/mod.rs] imports `crate::codec::s3`, and MODELS.md lists chatterbox over s3. (b)

## crates/audio/src/codec/ecapa_tdnn/EGM.md

Coverage: sampled (interface names, Res2NetBlock divisibility check, BadConfig claim and no-env/no-write claim checked; forward pass not read)
No drift detected.

## crates/audio/src/codec/ecapa_tdnn/README.md

Coverage: sampled (test names, fixture and tool paths, block kernel and dilation values, scale and SE values checked; pooling and BatchNorm structure not read)
- Architecture table says input features are "80-channel log mel" and output embedding is "192 dimensions", and the pipeline ends in a "192-dimensional Speaker Embedding". The code defaults are input_size 60 and embed_dim 256 [src/codec/ecapa_tdnn/mod.rs:EcapaTdnnConfig::from_json]. The README says config_defaults_match_reference verifies defaults against mlx-audio, and that test asserts 60 and 256 [src/codec/ecapa_tdnn/tests.rs:config_defaults_match_reference]. The 80 and 192 values contradict the README's own test unless a checkpoint config overrides them, which the README does not say. (b)
- "Intermediate channels 512 or 1024": the default is 1024 only. (minor)

## crates/audio/src/codec/encodec/EGM.md

Coverage: sampled (claims verified 9 of 11, source not fully read)
- (c) `normalize` is an undocumented side-effect-free switch on the encode path. [crates/audio/src/codec/encodec/mod.rs:Encodec::encode] computes the RMS scale when `config.normalize` is set and then discards it (`let _ = scale;`), returning codes only. The EGM does not mention `normalize` or the scale.

## crates/audio/src/codec/encodec/README.md

Coverage: sampled (claims verified 8 of 10, source not fully read)
- (b) Architecture table row "Normalization | weight_norm on all convolutions" describes the reference. The Rust loader requires `norm_type == "weight_norm"` and expects weights already folded at conversion [crates/audio/src/codec/encodec/mod.rs:Encodec::load], [crates/audio/src/codec/encodec/mod.rs:plain_conv_dims]. Wording is misleading, not wrong.
- (b) Round-trip claim is incomplete: the doc on [crates/audio/src/codec/encodec/mod.rs:Encodec::decode_codes] says callers pass "the encode-time scale", but [crates/audio/src/codec/encodec/mod.rs:Encodec::encode] never returns it. See the EGM item above.
- Verified: the four test names in the Verification section exist in [crates/audio/src/codec/encodec/tests.rs] (config_defaults_match_reference, isolated_lstm_matches_reference, tiny_end_to_end_matches_reference, real_checkpoint_gate). Bandwidths [1.5, 3.0, 6.0, 12.0, 24.0], codebook 1024 x 128 and 24 kHz defaults match [crates/audio/src/codec/encodec/mod.rs:EncodecConfig::from_json].

## crates/audio/src/codec/fish_s1_dac/EGM.md

Coverage: sampled (claims verified 12 of 12 listed, source not fully read)
No drift detected.

## crates/audio/src/codec/fish_s1_dac/README.md

Coverage: sampled (claims verified 8 of 10, source not fully read)
- (a) "Scope and Divergences" says `encode_zq` / `decode_zq` (continuous-latent entry points) are not ported. The code exposes both as public API: [crates/audio/src/codec/fish_s1_dac/mod.rs:FishS1Dac::encode_zq] and [crates/audio/src/codec/fish_s1_dac/mod.rs:FishS1Dac::decode_zq], exercised by [crates/audio/src/codec/fish_s1_dac/tests.rs:encode_zq_decode_zq_roundtrip_matches_shapes]. The sibling EGM lists both as interface, so the two docs disagree. `from_latents` has no symbol in the crate, so that part of the claim holds.
- Verified: 44.1 kHz, encoder rates product 512 (hop 512), frame_length 2048, 4096 semantic and 9 x 1024 acoustic books, transformer on the last encoder block, match [crates/audio/src/codec/fish_s1_dac/mod.rs:FishS1DacConfig::build_ae] and [crates/audio/src/codec/fish_s1_dac/mod.rs:FishS1DacConfig::frame_length].
- Verified: `open` treats config.json as optional and reads `model.safetensors` ([crates/audio/src/codec/fish_s1_dac/mod.rs:FishS1Dac::open]). The env var `TURBOSPARK_FISH_S1_DAC_MODEL` appears only in [crates/audio/src/codec/fish_s1_dac/tests.rs:real_checkpoint_gate].
- Not checked: the 1e-3 relative gate value and the transformer layer counts inside the quantizer.

## crates/audio/src/codec/higgs_audio/EGM.md

Coverage: sampled (claims verified 8 of 8 listed, source not fully read)
No drift detected.

## crates/audio/src/codec/higgs_audio/README.md

Coverage: sampled (claims verified 6 of 8, source not fully read)
- (b) "Scope and Divergences" says the semantic encode path is not ported: "that tower is an STT model outside the codec crate and is not ported", "The sinc resampler ... is not ported", and "semantic fusion happens in a future STT integration". The code implements it. [crates/audio/src/codec/higgs_audio/mod.rs:HiggsAudioTokenizer::encode_tokens] runs [crates/audio/src/codec/higgs_audio/semantic.rs:sinc_resample] and [crates/audio/src/codec/higgs_audio/semantic.rs:SemanticEncoder], and the path loads when the config carries `semantic_model_config` ([crates/audio/src/codec/higgs_audio/mod.rs:HiggsAudioTokenizer::load]). Released-format configs still lack it, so decode-only is accurate for them, but the "not ported" wording is stale. The sibling EGM describes the gate correctly.
- Verified: defaults (24 kHz, 8 books x 1024, dim 64, latent 1024, fc2 width 256, acoustic hop 8x5x4x2x3 = 960) match [crates/audio/src/codec/higgs_audio/mod.rs:HiggsAudioConfig::defaults]. `quantizer_latent_dim` and `decoder_input_dim` exist as stated.

## crates/audio/src/codec/mimi/EGM.md

Coverage: sampled (claims verified 6 of 6 listed, source not fully read)
No drift detected.

## crates/audio/src/codec/mimi/README.md

Coverage: sampled (claims verified 9 of 11, source not fully read)
- (b) "Scope and Reference Deviations" item 2 "Name mapping" says the Rust loader "applies the reference's PyTorch rename rules for real checkpoints". No rename step exists in the crate. [crates/audio/src/codec/mimi/mod.rs:Mimi::load] looks up only the MLX-style names (for example `encoder.init_conv1d.conv.conv.weight` and `quantizer.rvq_first.input_proj.weight`). The loader doc comment at the same symbol repeats the false claim (lines 1024-1026). I did not verify the raw kyutai tensor names, so whether a real checkpoint loads is unknown; the mapping the doc describes is absent.
- (b) Verification section: "cargo test -p turbospark-audio --lib codec::mimi (4 tests)". [crates/audio/src/codec/mimi/tests.rs] defines 6 `#[test]` functions (mimi_202407_factory_defaults, tiny_end_to_end_matches_reference, isolated_transformer_matches_reference, real_checkpoint_gate, attention_matches_original_within_ulps_of_scale, decode_refuses_codes_outside_the_codebook).
- Verified: `encode_step` / `decode_step` are absent, matching the scope note. The rest RVQ quantizes its own projection of the input, not the first book's residual ([crates/audio/src/codec/mimi/mod.rs:SplitRvq::encode]), matching the README. Factory numbers (dim 512, ratios [8,6,5,4], 2048 bins x 256, context 250) match [crates/audio/src/codec/mimi/mod.rs:MimiConfig::mimi_202407].

## crates/audio/src/codec/mimo_audio_tokenizer/EGM.md

Coverage: sampled (claims verified 6 of 6 listed, source not fully read)
No drift detected.

## crates/audio/src/codec/mimo_audio_tokenizer/README.md

Coverage: sampled (claims verified 9 of 9 checked, source not fully read)
No drift detected. Checked: defaults (nfft 960, hop 240, 32 encoder layers with skip layer 3, avg pooler 2, 32 decoder layers, vocoder dim 256 with 16 layers) in [crates/audio/src/codec/mimo_audio_tokenizer/mod.rs:MimoConfig::defaults]; codebook sizes [1024, 1024] then 128 x 18 in the same function; vocoder band window (40, 10) read as [-40, +10] at the mask check in [crates/audio/src/codec/mimo_audio_tokenizer/mod.rs]; `istft.window` loaded when present in [crates/audio/src/codec/mimo_audio_tokenizer/mod.rs:MiMoAudioTokenizer::load]; LayerNorm vs RMSNorm via [crates/audio/src/codec/mimo_audio_tokenizer/mod.rs:LnormType]; the batch-only, no-segment_size scope.

## crates/audio/src/codec/moss_audio_tokenizer/EGM.md

Coverage: sampled (claims verified 7 of 7 listed, source not fully read)
No drift detected.

## crates/audio/src/codec/moss_audio_tokenizer/README.md

Coverage: sampled (claims verified 5 of 7, source not fully read)
- (b) "Parameters (default config)" table lists codebook dim 32 and "2 channels interleaved" at 48 kHz. These match the Nano checkpoint, not the code's fallbacks: [crates/audio/src/codec/moss_audio_tokenizer/mod.rs:MossConfig::from_json] defaults `codebook_dim` to 8 and `number_channels` to 1. Either label the table as checkpoint values or state the fallbacks.
- Verified: downsample rate 3840 default, 32 books x 1024 default `num_quantizers` / `codebook_size`, batch-only scope (no `AttentionStepCache` in the crate), `encode_frame` / `decode_frame` exist, and the weight-shard discovery in [crates/audio/src/codec/moss_audio_tokenizer/mod.rs:open_checkpoint] (`read_dir`).
- Not checked: the 48 kHz default sample rate fallback.

## crates/audio/src/codec/nemotron_voicechat/EGM.md

Coverage: sampled (claims verified 7 of 7 listed, source not fully read)
No drift detected.

## crates/audio/src/codec/nemotron_voicechat/README.md

Coverage: sampled (claims verified 8 of 10, source not fully read)
No drift detected in the claims checked: defaults (22 050 Hz, 384 base channels, multipliers [1, 2, 4], downsample [7, 7, 9], latent 512, n_fft 16, hop 4, 31 x 1024) in [crates/audio/src/codec/nemotron_voicechat/mod.rs:NemotronVoiceChatConfig::from_json]; the magnitude formula `100 * exp(-softplus(-logit + ln 100))` and DC/Nyquist imaginary zeroing in the decode head; strict `variance_list` loading in [crates/audio/src/codec/nemotron_voicechat/mod.rs:NemotronVoiceChatCodec::load]; the batch-only scope. Not checked: the "encode drops tail shorter than one token" and "tokens * 1764" length claims.

## crates/audio/src/codec/s3/EGM.md

Coverage: sampled (claims verified 8 of 8 listed, source not fully read)
No drift detected.

## crates/audio/src/codec/s3/README.md

Coverage: sampled (claims verified 8 of 10, source not fully read)
No drift detected in the claims checked: codebook sizes 4096 (v1, [crates/audio/src/codec/s3/mod.rs:S3Config::v1]) and 6561 (v2, [crates/audio/src/codec/s3/mod.rs:S3Config::v2]); 25 Hz profile selected by the `speech_tokenizer_v1_25hz` name ([crates/audio/src/codec/s3/mod.rs:V1_25HZ]); 2600-frame sliding stride ([crates/audio/src/codec/s3/mod.rs:FRAMES_PER_STRIDE]); codebook key discovery `quantizer.embed` and the two aliases ([crates/audio/src/codec/s3/mod.rs] codebook key list); the ignored `real_checkpoint_gate` reading `TURBOSPARK_S3_MODEL_DIR` ([crates/audio/src/codec/s3/tests.rs:real_checkpoint_gate]). Not checked: the ternary FSQ levels and the FSMN branch details.

## crates/audio/src/codec/snac/EGM.md

Coverage: sampled (claims verified 7 of 7 listed, source not fully read)
No drift detected. The `frames_per_level` consistency check exists ([crates/audio/src/codec/snac/mod.rs:SnacConfig::frames_per_level]).

## crates/audio/src/codec/snac/README.md

Coverage: sampled (claims verified 8 of 9, source not fully read)
- (b) "Config parsing follows the reference constructor defaults, so a checkpoint config.json that omits fields still loads." This is false for `attn_window_size`: when the key is absent, [crates/audio/src/codec/snac/mod.rs:SnacConfig::from_json] sets `Some(32)`, and [crates/audio/src/codec/snac/mod.rs:Snac::load] refuses any `Some` value with `SpeechError::Unsupported`. A config that omits the key is refused, not loaded.
- Verified: hop 512 (product of encoder rates at the checkpoint values), `decode` refuses noise-enabled configs ([crates/audio/src/codec/snac/mod.rs:Snac::decode]), the transpose-conv `output_padding: 1` quirk in the weight-norm path ([crates/audio/src/codec/wnconv.rs]), and 9 `#[test]` functions in [crates/audio/src/codec/snac/tests.rs]. The real-checkpoint gate falls back to the HF cache path and `TURBOSPARK_SNAC_MODEL` as described.

## crates/audio/src/codec/spark_bicodec/EGM.md

Coverage: sampled (claims verified 6 of 6 listed, source not fully read)
No drift detected.

## crates/audio/src/codec/spark_bicodec/README.md

Coverage: sampled (claims verified 8 of 9, source not fully read)
No drift detected in the claims checked: the identity `SamplingBlock` triples its input ([crates/audio/src/codec/spark_bicodec/mod.rs:SamplingBlock::forward]); the prenet/speaker width equality gate and the `sample_ratios` gate in [crates/audio/src/codec/spark_bicodec/mod.rs:BiCodecConfig::from_yaml] validation; empty semantic tokens rejected in [crates/audio/src/codec/spark_bicodec/mod.rs:BiCodec::detokenize]; `config.yaml` plus `model.safetensors` layout in [crates/audio/src/codec/spark_bicodec/mod.rs:BiCodec::open]; the 2e-4 detokenize tolerance (from the fixture trace, tests.rs). Not checked: the measured 8.9e-8 maximum diff.

## crates/audio/src/codec/stepaudio2/EGM.md

Coverage: sampled (claims verified most of the interface, invariants and side effects; mod.rs open/load/decode read in full, flow/hift/campplus not fully read)
- [mod.rs:StepAudio2Token2Wav::load] optional CAMPPlus encoder and `prepare_prompt` requirement (speaker_embedding required when no CAMPPlus is loaded) are omitted from the EGM.
- [mod.rs:StepAudio2Token2Wav::decode] takes `&mut self`; EGM does not say so (minor).

## crates/audio/src/codec/stepaudio2/README.md

Coverage: sampled (constants dit_depth 16, dit_hidden 512, inference_cfg_rate 0.7, upsample [8,5,3], AUDIO_LIMIT 0.99, rand_noise, test names and TURBOSPARK_STEPAUDIO2_MODEL, fixture and link paths verified; per-layer port notes not re-derived)
- No drift detected.

## crates/audio/src/codec/vocos/EGM.md

Coverage: sampled (interface symbols, open/load/decode/decode_from_codes read in full; invariants not all traced)
- [mod.rs:Vocos::decode] returns Unsupported for the Encodec front end; the EGM does not state this (omission, minor).
- [mod.rs:FeatureKind] Encodec variant and the caller-supplied codebooks of [mod.rs:Vocos::decode_from_codes] are not described in the EGM (omission, minor).

## crates/audio/src/codec/vocos/README.md

Coverage: sampled (exp/min(1e2) magnitude clamp, symmetric Hann iSTFT, head shape [n_fft+2, dim], intermediate default 1536, test names, fixture and link paths verified)
- Architecture table values (24 kHz, dim 512, 8 layers, n_fft 1024, hop 256) are checkpoint-specific; the code only holds defaults and config-driven values, and no real checkpoint config was checked. Not verified, not counted as drift.

## crates/audio/src/codec/wavvae/EGM.md

Coverage: sampled (interface symbols, open/encode/decode/Posterior::sample, megatts3 geometry and side effects verified in full read of mod.rs config, open, encode, decode)
- Invariant "Inference is CPU f32 over host Vec<f32> buffers; no GPU state is held here" is false. [mod.rs:WavVae::with_backend] stores `Option<Rc<dyn ComputeBackend>>` and a device weight cache `RefCell<HashMap<usize, Rc<dyn DeviceWeight>>>` ([mod.rs:ConvExec]), and the decoder convolutions run on that backend.
- Overview says the decoder "may run on an injected backend", which the invariant contradicts.

## crates/audio/src/codec/wavvae/README.md

Coverage: sampled (megatts3 geometry, reversed ratios [2,4,4,5,6], proj_to_z 512->64, clamp [-30,20], hop 960, Posterior fields, open path, has_encoder/Unsupported, real-checkpoint env vars, tools and testdata paths verified; tiny-fixture error numbers not re-run)
- [wavvae/tests.rs:real_checkpoint_matches_pytorch] reads `TURBOSPARK_WAVVAE_TOL` (default 5e-3); the README does not mention it (omission, minor).

## crates/audio/src/lid/EGM.md

Coverage: full read (mod.rs is 3 lines of doc comments; EGM claims checked against it)
- No drift detected.

## crates/audio/src/lid/README.md

Coverage: full read (README is short; link `../../MODELS.md#lid` resolves to `## LID` in crates/audio/MODELS.md)
- No drift detected.

## crates/audio/src/lm/EGM.md

Coverage: sampled (every Interface symbol located; Side Effects checked: no env, thread or static in src/lm; Execution fallback, device attention gate, token-id and sliding_window refusals, MoE slot order, cohere2 and falcon checks read in targeted reads; cache.rs internals not fully read)
- No drift detected.

## crates/audio/src/lm/README.md

Coverage: sampled (module table entries, backbone flavor claims for llama, qwen2, qwen3, ssm formula, 16 testdata/lm cases and 7 llama rope cases, regeneration script path verified; mutation-check numbers and cache-order tests not re-run)
- No drift detected.

## crates/audio/src/models/EGM.md

Coverage: full read (mod.rs is 28 lines; every re-export and alias checked)
- No drift detected.

## crates/audio/src/music/EGM.md

Coverage: full read (mod.rs is 6 lines; `pub mod minimax_music3` confirmed)
- No drift detected.

## crates/audio/src/music/README.md

Coverage: full read (link `../../MODELS.md#music` resolves to `## MUSIC` in crates/audio/MODELS.md; minimax_music3/README.md link resolves)
- No drift detected.

## crates/audio/src/music/minimax_music3/EGM.md

Coverage: sampled (all Interface symbols located; constants MAX_PROMPT_TOKENS, MAX_AUDIO_FRAMES, FRAME_RATE, AR_CFG_SCALE, AR_CFG_TOP_K, DIT_CFG_SCALE, SAMPLING_RATE, CHUNK_*, CROP_*, OVERLAP_LATENT_LENGTH, AR_SAMPLING_TOP_K and FlowProfile step counts verified; ~12.9k lines not fully read)
- Invariant "backend-specific state is not held in this crate" is overstated. [mod.rs:Model] holds `backend: Option<Rc<dyn ComputeBackend>>`; [depth.rs:DepthDecoder], [dit.rs:FlowMatchingTransformer], [fusion.rs:ConditionEncoder] hold `Option<Rc<dyn ComputeBackend>>`; [qwen3.rs:Qwen3] holds device KV buffers `Rc<dyn DeviceBuffer>` (qwen3.rs:154-155) that persist between decode steps.
- Side Effects omit interior mutability on `Model`: [mod.rs:Model] uses `RefCell<StageTimings>` and `Cell<FlowGuidance>`, which `set_flow_guidance`, `set_flow_solver`, `set_flow_profile` and the generate paths mutate through `&self`. The model is also single-threaded (Rc), which the EGM does not state (omission).
- `TURBOSPARK_MUSIC3_BENCH_TREES` ([timing.rs:bench_dir]) and `read_dir` in [mod.rs:load_official_tensors] match the EGM.

## crates/audio/src/music/minimax_music3/README.md

Coverage: sampled (request defaults 60 s / 30 steps / seed 0 and 360 s and 1..=30 limits, flow profile constants and defaults, test names, script, example, fixture, docs and MODELS.md links, music3 CLI flags in crates/cli/src/bin/music.rs verified; performance tables and numeric claims not re-measured; 12.9k lines not fully read)
- No drift detected.

## crates/audio/src/ops/EGM.md

Coverage: full read of EGM; `kernel_parity.rs` symbols and gating verified by grep (1187 lines not fully read)
- No drift detected.

## crates/audio/src/sts/EGM.md

Coverage: full read (mod.rs is 15 lines; re-exports verified; subdirectory EGM links deepfilternet, dialogue_sidon, mel_roformer, mossformer2_se, sam_audio all resolve)
- No drift detected.

## crates/audio/src/sts/README.md

Coverage: sampled (subdirectory README links resolve; family descriptions not verified against each family source, since those directories are outside this assignment)
- No drift detected.

## crates/audio/src/sts/deepfilternet/EGM.md

Coverage: sampled (claims verified 5 of 6; mod.rs is 46 KB, checked by targeted reads, not read in full)
- Invariant "[mod.rs:load]": there is no `DeepFilterNet::load`. Model loading happens in `DeepFilterNet::open` [crates/audio/src/sts/deepfilternet/mod.rs:open]. The only `fn load` items in the file are per-layer helpers [mod.rs:load at lines 64 and 186].
- Invariant on `sample_rate` says it feeds the normalization constant "[mod.rs:enhance]". The value is read in `open` and used to compute `a_raw` at mod.rs:926 inside `open`. `enhance` does not read it. The field is `#[allow(dead_code)]` [mod.rs:DeepFilterNet].

## crates/audio/src/sts/deepfilternet/README.md

Coverage: sampled (claims verified 4 of 6; source not fully read)
- "Supported Versions: DeepFilterNet v1, v2, and v3": [mod.rs:open] documents only the v3 layout. v2 loads through the `enc_concat` flag. No v1 handling was found in the file.
- Verification gate "v3 local witness: waveform correlation 0.964, max absolute error 0.021": no match under crates/audio/src/sts/deepfilternet (the only hit is the README). The number is not traceable to an in-tree test or fixture.
- "square-root Hann (Vorbis)" [mod.rs:vorbis_window]: the function computes a Vorbis power-complementary window, sin(pi/2 * sin^2(...)). It is not a square-root Hann window. The two names are conflated.

## crates/audio/src/sts/dialogue_sidon/EGM.md

Coverage: sampled (claims verified 9 of 9 checked; source not fully read)
No drift detected.

## crates/audio/src/sts/dialogue_sidon/README.md

Coverage: sampled (claims verified 8 of 9 checked; source not fully read)
- Chunked long-file inference is described as part of the reference `separate()`, and the Documented divergences section does not say the Rust port omits it. [crates/audio/src/sts/dialogue_sidon/mod.rs] has no chunk or overlap or crossfade logic (the only `chunk` hits are `chunks_exact_mut` and `normalize_chunk`). The "Not yet verified: chunked stitching" note covers testing only, not absence.
- Checked and matching: pre-emphasis 0.97 (mod.rs:1030), povey exponent 0.85 (mod.rs:47), peak normalization 0.9 with 160-sample pads (mod.rs:1097-1102), beta_end 0.02 and 1000 training steps (mod.rs:132-134), depthwise kernel 31 default (mod.rs:122), decoder via `codec::descript` (directory exists), all listed fixture files present under crates/audio/testdata/dialogue_sidon.

## crates/audio/src/sts/mel_roformer/EGM.md

Coverage: sampled (claims verified 6 of 6 checked; source not fully read)
No drift detected.

## crates/audio/src/sts/mel_roformer/README.md

Coverage: sampled (claims verified 6 of 8 checked; source not fully read)
- Window naming is wrong and contradicts itself. The Overview says "symmetric-Hann STFT" and the Architecture section says "periodic (symmetric) Hann". The code path uses `crate::dsp::hann_window` [crates/audio/src/dsp.rs:hann_window], which divides by `size` (periodic). The code comment at [crates/audio/src/sts/mel_roformer/mod.rs:forward-STFT, line 768] also says "Symmetric Hann window (np.hanning(n_fft + 1)[:-1])", which is the periodic form. The README and the comment should say periodic.
- Checked and matching: `open` requires config.json [mod.rs:open], packed `to_qkv` keys refused [mod.rs:load, lines 648-653], n_fft default 2048 [mod.rs:from_json], fixture files present under testdata/mel_roformer.

## crates/audio/src/sts/mossformer2_se/EGM.md

Coverage: sampled (claims verified 7 of 7 checked; source not fully read)
No drift detected.

## crates/audio/src/sts/mossformer2_se/README.md

Coverage: sampled (claims verified 8 of 9 checked; source not fully read)
- Enhance wrapper boundaries do not match the code. The README says the full path is used "below 20 s" and segmented "at 20 s and above", and chunked "above the 60 s threshold". The code uses the segmented path only when `len > sample_rate * one_time_decode_length` [mod.rs:decode_one_audio], which is strictly greater, so exactly 20 s takes the full path. The chunked path triggers when `duration >= auto_chunk_threshold` [mod.rs:enhance_with_mode], which is inclusive, so exactly 60 s is chunked. The code comment at [mod.rs:enhance_with_mode] ("(20 s) use the reference segmented overlap-discarding path") repeats the inclusive wording, so the code comment also disagrees with the code.
- Quantization claim "other bit widths are refused" is not traced. The loader hard-codes 8-bit, group 64 [mod.rs:load, the `linear` closure with `QuantScheme { bits: 8, group_size: 64 }`]. The refusal of other widths lives in the shared quant module and was not checked.
- Checked and matching: nonzero dither refused [mod.rs:kaldi_fbank], config defaults decode_window 4, chunk_overlap 0.25, auto_chunk_threshold 60.0 [mod.rs:Default for MossFormer2SeConfig], optional config.json in open [mod.rs:open], real-checkpoint gate `real_checkpoint_gate` [tests.rs:real_checkpoint_gate] reads TURBOSPARK_MOSSFORMER2_SE_MODEL.

## crates/audio/src/sts/sam_audio/EGM.md

Coverage: sampled (claims verified 8 of 9 checked; source not fully read)
- Not traced: the invariant that [shapes.rs] checks every config-derived stride at load. The module exists (mod.rs `mod shapes`), but its contents were not read.
- Checked and matching: all interface names exist (`SamAudioConfig::validate`, `OdeOptions`, `EmbedAnchors`-backed velocity path, `T5Encoder::forward`); `open` reads config.json, model.safetensors or tiny_weights.safetensors, and tiny_codec.safetensors [mod.rs:open].

## crates/audio/src/sts/sam_audio/README.md

Coverage: sampled (claims verified 5 of 7 checked; source not fully read)
No drift detected in the claims checked. Checked and matching: `OdeOptions` default is midpoint with step 2/32 [mod.rs:OdeOptions::default], temporal anchors (`EmbedAnchors`) exist, `codec::dacvae` directory exists, all listed fixture files present under testdata/sam_audio. Not traced: QK RMSNorm eps 1e-5, the RoPE theta formula, and the "no learned sigmoid gates" statement.

## crates/audio/src/vad/EGM.md

Coverage: full read of crates/audio/src/vad/mod.rs (409 bytes); sub-module claims not re-checked here
- Interface claims the re-exports include "the sortformer public API". [crates/audio/src/vad/mod.rs:pub use sortformer] re-exports only DiarizationOutput, DiarizationSegment, GenerateOptions, Sortformer, StreamingOptions, and StreamingProfile. Not re-exported: SortformerConfig, StreamingChunk, StreamingState, extract_mel_features, preds_to_segments, preds_to_segments_with_options, median_filter_probs, trim_silence. The Nemotron re-export is also partial: NemotronConfig, LatencyPreset, CoupledDiarizer, and EncoderExecutor are not re-exported.

## crates/audio/src/vad/README.md

Coverage: full read (short file); links checked by path only
- Links resolve: ../../MODELS.md is crates/audio/MODELS.md (exists); the three family READMEs exist. The `#vad` anchor was not checked.
- Checked and matching: Nemotron max 8 speakers [crates/audio/src/vad/nemotron_diarization/mod.rs:from_json, num_speakers default 8]; Sortformer 4 speakers and Silero dual 16 kHz/8 kHz branches [crates/audio/src/vad/silero_vad/mod.rs:load, vad_16k and vad_8k].

## crates/audio/src/vad/nemotron_diarization/EGM.md

Coverage: sampled (claims verified 9 of 9 checked; source not fully read)
No drift detected.

## crates/audio/src/vad/nemotron_diarization/README.md

Coverage: sampled (claims verified 8 of 10 checked; source not fully read)
- "80-bin Mel Spectrogram" in the diagram: the code default is `feature_size: 128` [crates/audio/src/vad/nemotron_diarization/mod.rs:MelConfig::default], and the coupled tests assert 128 mel bins [mod.rs, coupled test around line 2009]. config.json can override the default, so this is a mismatch only if the shipped checkpoint is 128. The README should name the checkpoint value or say it is config-driven.
- Example and tooling path does not resolve: `scripts/convert_multitalker_nemo.py convert-diar` is cited from the repo root. The script is at crates/audio/scripts/convert_multitalker_nemo.py. No scripts/ directory exists at the repo root.
- Checked and matching: coupled constants fifo_len 264, spkcache_update_period 222, 14-frame chunk [mod.rs:multitalker_coupled, lines 1308-1317]; test `coupled_chunk_preds_match_nemo` [mod.rs:coupled_chunk_preds_match_nemo]; both runtime files crates/runtime/src/nemotron_diar_metal.rs and crates/runtime/tests/nemotron_diar_metal.rs exist; 1e-4 tolerances in that test; 31 layers and subsampling 8 defaults [mod.rs:from_json]; `CoupledDiarizer::with_executor` [mod.rs:with_executor].

## crates/audio/src/vad/silero_vad/EGM.md

Coverage: sampled (claims verified 6 of 6 checked; source not fully read)
No drift detected.

## crates/audio/src/vad/silero_vad/README.md

Coverage: sampled (claims verified 4 of 5 checked; source not fully read)
- Verification claim "Maximum probability absolute error 4.8e-7" appears only in this README. No in-tree test or fixture under crates/audio/src/vad/silero_vad or testdata carries the figure, so it is not traceable to the repo.
- Checked and matching: LSTM hidden size 128 [mod.rs:load, lstm_dim] and state shape [mod.rs:VadStreamState]; 16 kHz and 8 kHz branches [mod.rs:initial_state, vad_16k and vad_8k]; streaming chunk sizes 512 and 256 appear in the per-rate config tuples [mod.rs:load].
- Not traced: the 188-chunk parity count, the CPU throughput and memory figures.

## crates/audio/src/vad/sortformer/EGM.md

Coverage: sampled (claims verified 9 of 9 checked; source not fully read)
No drift detected.

## crates/audio/src/vad/sortformer/README.md

Coverage: sampled (claims verified 2 of 4 checked; source not fully read)
- Checked and matching: hop 160 samples (10 ms at 16 kHz) [mod.rs:ProcessorConfig::default], Slaney mel scale [mod.rs:extract_mel_features, MelScale::Slaney].
- Not traced: "AOSC compression matches reference fixture byte-for-byte", the 1e-5 fixture tolerance, and the RTF and DER figures. These are test and benchmark claims, not code claims; no divergence asserted.

## crates/audio/tools/EGM.md

Coverage: sampled (file names and invariants checked; generator bodies not read)
- No divergence found in the names checked. Every listed generator, convert_megatts3.py, test_convert_megatts3.py, and the reference probes exist under crates/audio/tools. The test runs `import convert_megatts3` after inserting its own directory into sys.path [crates/audio/tools/test_convert_megatts3.py], so the documented `python3 -I` command works.
- `--real` uses `local_files_only` [crates/audio/tools/gen_snac_fixtures.py:main, the --real flag]. OUT is ../turbospark/crates/audio/testdata/snac [gen_snac_fixtures.py:OUT], consistent with the invariant.
- Not traced: "pinned commit" of mlx-audio. Most generators (69 of 71) reference the venv or mlx-audio, but the pinned commit was not compared.

## crates/audio/src/stt/EGM.md

Coverage: sampled (claims verified 14 of 16, source not fully read; mod.rs pub mod and re-export lists compared in full)
No drift detected.

## crates/audio/src/stt/README.md

Coverage: sampled (claims verified 8 of 8 integration and layout claims; family list not checked line by line against each family README)
No drift detected.

## crates/audio/src/stt/canary/EGM.md

Coverage: sampled (claims verified 14 of 16, source not fully read)
No drift detected.

## crates/audio/src/stt/canary/README.md

Coverage: sampled (claims verified 11 of 13, source not fully read)
- Test count drift: README says the port "adds 9 always-on tests and 1 ignored gated test" [mod.rs:tests] [tokenizer.rs:tests]. Current tree has 8 always-on plus 1 ignored in canary/mod.rs and 2 always-on in canary/tokenizer.rs, so 10 always-on plus 1 ignored under stt::canary. Low severity, count may be historical.

## crates/audio/src/stt/cohere_asr/EGM.md

Coverage: full read (cohere_asr/mod.rs read in full)
No drift detected.

## crates/audio/src/stt/cohere_asr/README.md

Coverage: full read (cohere_asr/mod.rs read in full; README pins and investigation claims are not verifiable from source and were checked only for consistency with the pinned constants)
No drift detected.

## crates/audio/src/stt/fireredasr2/EGM.md

Coverage: sampled (claims verified 18 of 20, source not fully read)
No drift detected.

## crates/audio/src/stt/fireredasr2/README.md

Coverage: sampled (claims verified 4 of 4 checkable claims; artifact digests not checked)
No drift detected.

## crates/audio/src/stt/fun_asr_nano/EGM.md

Coverage: sampled (claims verified 10 of 12, source not fully read)
No drift detected.

## crates/audio/src/stt/fun_asr_nano/README.md

Coverage: sampled (claims verified 5 of 5 checkable claims; artifact digests and fixture regeneration commands not executed)
No drift detected.

## crates/audio/src/stt/glmasr/EGM.md

Coverage: sampled (claims verified 12 of 13, source not fully read)
No drift detected.

## crates/audio/src/stt/glmasr/README.md

Coverage: sampled (claims verified 4 of 4 checkable claims: pinned revision, tolerances 1e-2 and 3e-2, test name, fixture generator; runtime and timing claims not checked)
No drift detected.

## crates/audio/src/stt/granite_speech/EGM.md

Coverage: sampled (claims verified 14 of 16, source not fully read)
No drift detected.

## crates/audio/src/stt/granite_speech/README.md

Coverage: sampled (claims verified 9 of 10 checkable claims: multipliers, pins, prompt, max_tokens, translation prompt; file digests and timing not checked)
No drift detected.

## crates/audio/src/stt/granite_speech5_ctc/EGM.md

Coverage: sampled (claims verified 12 of 13, source not fully read)
No drift detected.

## crates/audio/src/stt/granite_speech5_ctc/README.md

Coverage: sampled (claims verified 4 of 4 checkable claims: pin, frontend (HTK, 512 FFT, 400 Hann), CTC collapse, batching; fixture digest not checked)
No drift detected.

## crates/audio/src/stt/granite_speech_nar/EGM.md

Coverage: sampled (claims verified 13 of 14, source not fully read)
- Shape wording: Interface says `compute_features(samples)` returns `[T/2, 160]` features [mod.rs:compute_features]. The function returns a flat Vec with 2 * (n_samples / 320) rows of N_MELS = 80 values each, and no pairing or stacking step exists in compute_features; the flat buffer has the same element count and order as [T/2, 160]. The doc comment on compute_features also says "stacked paired" [mod.rs:compute_features]. Low severity: the wording describes a stacking that the code does not perform.

## crates/audio/src/stt/granite_speech_nar/README.md

Coverage: sampled (claims verified 9 of 10 checkable claims; artifact digests not checked)
- Slot placement: README says the hypothesis is "interleaved with blank insertion slots (odd positions, minimum length 8)". add_insertion_slots places blanks at even indices (0, 2, 4, ...) and CTC tokens at odd indices, matching the code comment and EGM.md. The README's "odd positions" contradicts the code [mod.rs:add_insertion_slots].

## crates/audio/src/stt/higgs_audio_3/EGM.md

Coverage: sampled (claims verified 14 of 16, source not fully read)
No drift detected.

## crates/audio/src/stt/higgs_audio_3/README.md

Coverage: sampled (claims verified 8 of 9 checkable claims: pin, prompts, token ids, VAD parameters and API, qwen3_asr shard extension; numeric stage witnesses and frame counts not re-run)
No drift detected.

## crates/audio/src/stt/lasr_ctc/EGM.md

Coverage: sampled (claims verified 13 of 14, source not fully read)
No drift detected.

## crates/audio/src/stt/lasr_ctc/README.md

Coverage: sampled (claims verified 9 of 10 checkable claims: pin, residual weights, eps from config, CTC decode order, frontend constants; fixture transcript not re-run)
No drift detected.

## crates/audio/src/stt/mega_asr/EGM.md

Coverage: sampled (claims verified approx. 12 of 20, source not fully read)
No drift detected.

## crates/audio/src/stt/mega_asr/README.md

Coverage: sampled (claims verified approx. 8 of 20, source not fully read; pinned sizes and SHA-256 table not checked against files)
No drift detected.

## crates/audio/src/stt/mms/EGM.md

Coverage: sampled (claims verified approx. 10 of 14, source not fully read)
No drift detected.

## crates/audio/src/stt/mms/README.md

Coverage: sampled (claims verified approx. 5 of 10, source not fully read; pinned sizes and SHA-256 table not checked against files)
No drift detected.

## crates/audio/src/stt/moonshine/EGM.md

Coverage: sampled (claims verified approx. 14 of 18, source not fully read)
No drift detected.

## crates/audio/src/stt/moonshine/README.md

Coverage: sampled (claims verified approx. 6 of 10, source not fully read; checkpoint revision and SHA-256 table have no code constant to compare against)
- [crates/audio/src/stt/moonshine/mod.rs:Moonshine::open] The pinned checkpoint revision `2ccd087d043f77a4a07bd412b6725266355b08f4` and file digests appear only in this README. Unlike the sibling families, moonshine has no profile constant in code, so the pin cannot be checked against code (informational, not a mismatch).

## crates/audio/src/stt/moss_music/EGM.md

Coverage: sampled (claims verified approx. 14 of 18, source not fully read)
No drift detected.

## crates/audio/src/stt/moss_music/README.md

Coverage: sampled (claims verified approx. 6 of 10, source not fully read; pinned sizes and SHA-256 table not checked against files)
No drift detected.

## crates/audio/src/stt/moss_transcribe_diarize/EGM.md

Coverage: sampled (claims verified approx. 10 of 16, source not fully read)
No drift detected.

## crates/audio/src/stt/moss_transcribe_diarize/README.md

Coverage: sampled (claims verified approx. 8 of 14, source not fully read; encoder and adaptor geometry checked only through config test values; pinned file table not checked against files)
No drift detected.

## crates/audio/src/stt/multitalker_asr/EGM.md

Coverage: sampled (claims verified approx. 14 of 16; Interface checked against public items in mod.rs, scheduler.rs, frontend.rs, seglst.rs; source not fully read)
- [scheduler.rs:LiveSession] Interface omits public methods `seglst`, `processed_frames`, `samples_received`, `diarization`.
- [scheduler.rs:CoupledSession] Interface omits public methods `new`, `run` (the batch entry point, named only in invariants), `seglst`, `diarization`.
- [mod.rs:MultitalkerAsr::drop_extra_pre_encoded] public method not listed in Interface.
- [mod.rs:Features] `frame_major` and `channel_major` not listed in Interface.
- [mod.rs:SpeakerStream] `tokens`, `decoded_frames`, `cache_len` not listed in Interface.
- [seglst.rs:SentenceBuilder] Interface names the type only; public `new`, `update`, `changes`, `entries` are not listed.

## crates/audio/src/stt/multitalker_asr/README.md

Coverage: sampled (claims verified approx. 12 of 18, source not fully read; benchmark and timing figures not checked, they are not reproducible from code)
- Reproduce/Tests section: `scripts/nemo_multitalker_trace.py` does not resolve from the repo root; the script is at `crates/audio/scripts/nemo_multitalker_trace.py` (the Reproduce block uses the full path, the Tests table does not).
- Confirmed as written: pinned revision `8749fc71fd6e2d88ef230159bbf2aea69b524ee1` [mod.rs:MULTITALKER_PARAKEET_STREAMING_V1]; frozen_e settings (4 speakers, threshold 0.5, cache gating over 2 chunks) [scheduler.rs:CoupledConfig::frozen_e]; `supported_punctuation` required [mod.rs:supported_punctuation]; catalog row `can_run: false` [crates/catalog/src/audio_families.json:stt/multitalker_asr]; no `ts_multitalker_stream_*` FFI in crates/ffi.

## crates/audio/src/stt/nemo/EGM.md

Coverage: sampled (claims verified approx. 9 of 10, alignment.rs read in targeted ranges only, not in full)
- [alignment.rs:merge_longest_contiguous] Invariant says the function returns `SpeechError::Input` when the best run is "shorter than half the overlap length". The code compares against `overlap_a.len() / 2`, a token count, not half the overlap duration. Wording mismatch only; the doc should say "half the overlap token count".

## crates/audio/src/stt/nemotron_asr/EGM.md

Coverage: sampled (claims verified approx. 10 of 14, source not fully read)
No drift detected.

## crates/audio/src/stt/nemotron_asr/README.md

Coverage: sampled (claims verified approx. 3 of 5, source not fully read)
No drift detected.

## crates/audio/src/stt/parakeet/EGM.md

Coverage: sampled (claims verified approx. 9 of 12, source not fully read)
No drift detected.

## crates/audio/src/stt/parakeet/README.md

Coverage: sampled (claims verified approx. 5 of 8, source not fully read; WER-free stage numbers in the README are not reproducible from code)
- Stage fixture command: `scripts/generate_parakeet_fixtures.py` does not resolve from the repo root; the script is at `crates/audio/scripts/generate_parakeet_fixtures.py`.

## crates/audio/src/stt/phonon/EGM.md

Coverage: sampled (claims verified approx. 12 of 16, source not fully read)
No drift detected.

## crates/audio/src/stt/phonon/README.md

Coverage: sampled (claims verified approx. 6 of 12, source not fully read; pinned file sizes and SHA-256 table not checked against files)
No drift detected.

## crates/audio/src/stt/qwen2_audio/EGM.md

Coverage: sampled (claims verified approx. 12 of 16, source not fully read)
No drift detected.

## crates/audio/src/stt/qwen2_audio/README.md

Coverage: sampled (claims verified approx. 5 of 10, source not fully read; pinned file table not checked against files)
- [crates/audio/testdata] Verification section names the shared smoke WAV as `crates/audio/testdata/qwen2_audio_reference.wav`. That file does not exist; the README's parenthetical notes the audio is actually `qwen3_forced_aligner_reference.wav`, so the primary path is wrong and should be corrected.
- Verification status table says `cargo check --target x86_64-unknown-linux-gnu -p turbospark-audio` passes. moss_transcribe_diarize/README.md says the same check could not run on the dev host (onig_sys needs a cross gcc). Both are dated claims and could both be true at different times, but the qwen2 claim is not reproduced here and should carry a date.

## crates/audio/src/stt/qwen3_asr/README.md

Coverage: sampled (claims verified 20 of 26, source not fully read: 4108 lines across 8 files)
No drift detected.
Not verifiable from source (pinned digests, sizes, and timing figures): SHA-256 table, phase timings, RSS figures, Hub revision hashes.

## crates/audio/src/stt/qwen3_asr/EGM.md

Coverage: sampled (claims verified 20 of 26, source not fully read: 4108 lines across 8 files)
No drift detected.

## crates/audio/src/stt/qwen3_forced_aligner/README.md

Coverage: full read (mod.rs, 584 lines; only source file in the directory)
- Local checks use the test filter `models::stt::qwen3_forced_aligner` (both the unit-test command and the ignored-fixture command). The test paths are `stt::qwen3_forced_aligner::tests::...` (the `models` module only re-exports `stt`), so the substring filter matches zero tests. Correct filter: `stt::qwen3_forced_aligner`. [mod.rs:tests]
- Not verifiable from source: artifact sizes and SHA-256 digests, fixture digests, the "within 1 ms" match claim (fixture has 9 items, consistent with README).

## crates/audio/src/stt/qwen3_forced_aligner/EGM.md

Coverage: full read (mod.rs, 584 lines; only source file in the directory)
- Invariant says config failures are `Unsupported` ("... else `Unsupported`"). Only the classify_num and timestamp_segment_time mismatch return `Unsupported`. model_type, timestamp_token_id, and timestamp_segment_time parse failures return `BadConfig`. [mod.rs:Qwen3ForcedAlignerConfig::from_json]

## crates/audio/src/stt/whisper/README.md

Coverage: sampled (claims verified 13 of 16, mod.rs 779 lines not fully read)
- Profile table lists `whisper-medium` (769M, d_model 1024, 24 enc / 24 dec layers, 16 heads). No medium profile or constructor exists; the code defines only tiny, base, small, large_v3, and distil_large_v3. [mod.rs:WhisperConfig]
- Decoding line says "Greedy or beam search". There is no beam search; `transcribe` is greedy argmax only. [mod.rs:WhisperModel::transcribe]
- Architecture line says the log-mel frontend is "Provided by [turbospark_audio::models::whisper]". `models::whisper` re-exports `stt::whisper`, which takes caller-supplied mel in `encode` and contains no Slaney filterbank. The frontend is `whisper_log_mel_window` in `crates/audio/src/whisper.rs` (also referenced by the module doc at stt/whisper/mod.rs:5). [whisper.rs:whisper_log_mel_window]

## crates/audio/src/stt/whisper/EGM.md

Coverage: sampled (claims verified 13 of 16, mod.rs 779 lines not fully read)
- Citation `[whisper.rs:WHISPER_WINDOW_SAMPLES]` does not point at a file in this directory. The constant is in the crate-root file `crates/audio/src/whisper.rs`. Low severity; the symbol exists. [whisper.rs:WHISPER_WINDOW_SAMPLES]

## crates/audio/src/stt/voxtral/README.md

Coverage: sampled (claims verified 14 of 17, mod.rs 1050 lines not fully read)
- Prompt bullet says the Tekken prompt uses "pinned English transcription prompt tokens". No pinned prompt constant exists. `Voxtral::transcribe` takes `prompt_ids` from the caller, and the EGM says the same. The module doc (mod.rs:6-7, "pinned here as token ids") has the same stale claim. [mod.rs:Voxtral::transcribe]
- Verification section states audio-tower features match "within 9e-3 relative tolerance" and transcripts are "byte-identical". No assertion with that tolerance exists in the module tests; the only checkpoint test is ignored and reads `TURBOSPARK_VOXTRAL_MODEL_DIR`. Result is not verifiable from source. [mod.rs:tests]

## crates/audio/src/stt/voxtral/EGM.md

Coverage: sampled (claims verified 14 of 17, mod.rs 1050 lines not fully read)
No drift detected.

## crates/audio/src/stt/sensevoice/README.md

Coverage: sampled (claims verified 18 of 19; frontend.rs and mod.rs checked by targeted reads, sanm.rs 745 lines not read)
No drift detected.
Not verifiable from source: model file size, SHA-256 of the WAV, and the Python reference transcript.

## crates/audio/src/stt/sensevoice/EGM.md

Coverage: sampled (claims verified 18 of 19; frontend.rs and mod.rs checked by targeted reads, sanm.rs 745 lines not read)
No drift detected.

## crates/audio/src/stt/vibevoice_asr/README.md

Coverage: sampled (claims verified 12 of 14, source 2738 lines not fully read)
No drift detected.
Not verifiable from source: upstream load-path failure narrative, reference timings, prompt-id count (83), and the artifact table.

## crates/audio/src/stt/vibevoice_asr/EGM.md

Coverage: sampled (claims verified 12 of 14, source 2738 lines not fully read)
- Overview says "Streaming, diarization controls, and hotwords are refused." No refusal check exists for diarization or hotwords; they are absent from the API. The README itself says "refused by omission" for streaming, so the EGM wording overstates the code. [mod.rs:VibeVoiceAsr]

## crates/audio/src/stt/voxtral_realtime/README.md

Coverage: sampled (claims verified 17 of 19, source 3803 lines not fully read)
- Overview says the real-time `StreamingSession` "is out of scope and refused with a reason". No `StreamingSession` type or refusal path exists in code; the name appears only in the module doc (mod.rs:16). Streaming is absent, not refused. [mod.rs:VoxtralRealtime]
- Not verifiable from source: parity figures (1.3e-6, 6.1e-5, 1.1e-5), the two port bugs described, and artifact digests.

## crates/audio/src/stt/voxtral_realtime/EGM.md

Coverage: sampled (claims verified 17 of 19, source 3803 lines not fully read)
No drift detected.

## crates/audio/src/stt/wav2vec/README.md

Coverage: sampled (claims verified 9 of 10, source 2422 lines not fully read)
No drift detected.
Not verifiable from source: artifact digests, the 2.0e-4 and 5.0e-3 parity tolerances, and the reference transcript.

## crates/audio/src/stt/wav2vec/EGM.md

Coverage: sampled (claims verified 9 of 10, source 2422 lines not fully read)
No drift detected.

## crates/audio/src/stt/whistle/README.md

Coverage: sampled (claims verified 14 of 15, source 3511 lines including fixtures not fully read)
No drift detected.
Not verifiable from source: checkpoint SHA-256 values, Hugging Face revisions, and the oracle agreement claims.

## crates/audio/src/stt/whistle/EGM.md

Coverage: sampled (claims verified 14 of 15, source 3511 lines including fixtures not fully read)
No drift detected.

## crates/audio/src/tts/EGM.md

Coverage: sampled (module wiring, all child EGM links, alias list and no-alias list checked against mod.rs; all 11 directories read at the EGM level)
- Interface lists "Families exported as modules only (no alias)" without `pocket_tts`, but mod.rs declares `pub mod pocket_tts` with no `pub use`, and this file's own Invariants say pocket_tts has no alias. [mod.rs:pub mod pocket_tts]

## crates/audio/src/tts/README.md

Coverage: sampled (links resolve, `../../MODELS.md#tts` anchor exists at crates/audio/MODELS.md line 244, family index compared against the pub mod list)
- Family index lists 19 families and omits 20 declared families that each have their own README: arktts, bark, chatterbox, chatterbox_turbo, confucius4, dia, dramabox, echo_tts, fish_drama3, fish_qwen3_omni, irodori_tts, longcat_audiodit, melotts, omnivoice, pocket_tts, tada, vibevoice, voxcpm, voxcpm2, voxtral_tts. [mod.rs:pub mod]

## crates/audio/src/tts/arktts/EGM.md

Coverage: sampled (SamplingParams defaults, codec upsample strides, empty-generation error, decode-only codec and symbols checked)
No drift detected.

## crates/audio/src/tts/arktts/README.md

Coverage: full read of README; sampled source (SamplingParams, codec strides [8,8,4,2], falcon_h1 seam, fixture paths). Test counts ("15 arktts tests", "1296-test crate suite") were not run.
- Broken link: `../../../../tools/gen_arktts_fixtures.py` resolves to crates/tools/gen_arktts_fixtures.py, which does not exist. The file is at crates/audio/tools/gen_arktts_fixtures.py, so the link should be `../../../tools/gen_arktts_fixtures.py`. [README.md link]

## crates/audio/src/tts/bailingmm/EGM.md

Coverage: sampled (all named symbols located; Model::open prefix handling, max_decode_steps rejection, and the step-3 stop threshold checked)
No drift detected.

## crates/audio/src/tts/bailingmm/README.md

Coverage: full read of README; sampled source (crate::lm and ming_qwen2/dit/audio_vae layout, fixture tool path, stop rule)
No drift detected.

## crates/audio/src/tts/bark/EGM.md

Coverage: sampled (named consts and types located; Model::generate forcing of max_coarse_history 1, sliding_window_len 60, and no coarse KV cache verified at mod.rs:Model::generate)
No drift detected.

## crates/audio/src/tts/bark/README.md

Coverage: full read of README; sampled source (tests.rs, fixture dir testdata/tts_bark and tools/gen_bark_fixtures.py exist, NonCausalSelfAttention has no mask, max_coarse_history pin, 60-token default window)
No drift detected. (ALLOWED_PROMPTS is an upstream Python name named only as an open item; not a Rust symbol and not a claim about this crate.)

## crates/audio/src/tts/breeze_tts/EGM.md

Coverage: sampled (named symbols located; GenerateParams::validate checks, reserved-codec masking, EOS class at vocab_size checked)
No drift detected.

## crates/audio/src/tts/breeze_tts/README.md

Coverage: full read of README; sampled source (GenerateParams defaults 750 / 0.9 / 1.0 / 50 / 1.0 match Default impl at mod.rs:514; fixture dirs and tools/gen_breeze_tts_fixtures.py exist)
No drift detected.

## crates/audio/src/tts/chatterbox/EGM.md

Coverage: sampled (named symbols located; env var TURBOSPARK_DEBUG_CONV, 6561/6562 markers, 480/480 trim fade, s3gen/hift constants checked)
- Invariant "ids at or above 6561 are dropped" cites [mod.rs:drop_invalid_tokens], but drop_invalid_tokens only slices between the first start marker and the first stop marker. The `< SPEECH_VOCAB_SIZE` filter is applied in the caller in mod.rs (about lines 544-553). Behavior matches; the citation points at the wrong symbol.

## crates/audio/src/tts/chatterbox/README.md

Coverage: full read of README; sampled source (env vars TURBOSPARK_CHATTERBOX_MODEL_DIR and TURBOSPARK_S3_MODEL_DIR, #[ignore] real_checkpoint_gate, debug_flow_stages and debug_hift_stages, tools/gen_chatterbox_fixtures.py and tools/dbg_chatterbox_traces.py, src/codec/s3 link, trim_top_db 20 and rate 1.3)
No drift detected.

## crates/audio/src/tts/chatterbox_turbo/EGM.md

Coverage: sampled (S3GEN_SIL = 4299, token filter below 6561, meanflow default of 2 steps, empty-text rejection in inference_turbo checked; env var claim checked by grep, none found in non-test code)
No drift detected.

## crates/audio/src/tts/chatterbox_turbo/README.md

Coverage: full read of README; sampled source (four post-upsample conformer layers at conformer.rs, S3GEN_SIL, with_position_table_at_root in lm/gpt2.rs, noised_mels splice, stft_window recomputed in hift.rs, empty-text rejection). StatsPool epsilon and the other gotcha claims were not checked.
No drift detected.

## crates/audio/src/tts/confucius4/EGM.md

Coverage: sampled (22050 output rate, 16 kHz intake, language_token, 26-point linspace with cfg 0.7, BOS/EOS 8192/8193, 400-sample refusal in dsp.rs checked)
No drift detected.

## crates/audio/src/tts/confucius4/README.md

Coverage: full read of README; sampled source (sample rates, 1.72 length regulator, tests.rs, tools/gen_confucius4_fixtures.py, codec/bigvgan path)
- Names a `LANGUAGE_TOKEN` subset as the API. No such item exists; the public API is `language_token(lang)`. The name appears only in a doc comment at [mod.rs:language_token].
- "Rust gates ... (6 tests)": tests.rs has 8 `#[test]` attributes and none are `#[ignore]`d. [tests.rs]

## crates/audio/src/tts/dia/EGM.md

Coverage: sampled (KEEP_VOCAB 1025, EXTRA_STEPS_AFTER_EOS 30, MASK_DENIED -1e9, float32-only dtype check, delay_pattern check)
No drift detected.

## crates/audio/src/tts/dia/README.md

Coverage: full read of README; sampled source (GenerateParams defaults temperature 1.3, top_p 0.95, cfg_filter_top_k 35; delay pattern [0,8..15]; 30-frame tail drop and [0,1023] clamp in audio.rs; codec/descript link; tools/gen_dia_fixtures.py; testdata/tts_dia)
No drift detected.

## crates/audio/src/tts/dramabox/EGM.md

Coverage: sampled (GenerateParams defaults cfg 2.5, stg 1.5, stg_block 29, steps 30; decode_waveform_refusal; SeededNoise; all_perturbed handling)
No drift detected.

## crates/audio/src/tts/dramabox/README.md

Coverage: full read of README; sampled source (all_perturbed branch at dit.rs ~300, causal time bound at diffusion.rs:174-176, flip_sin_to_cos at dit.rs ~420, waveform refusal at mod.rs ~1109, DRAMABOX_MEASURE in tests.rs:53, fixture dir and tools/gen_dramabox_fixtures.py)
No drift detected.

## crates/audio/src/tts/echo_tts/EGM.md

Coverage: sampled (DEFAULT_TRUNCATION_FACTOR 0.96, t schedule from 0.999, model. weight prefix, delete_blockwise_modules refusal, zero one-patch speaker default, codec fish_s1_dac link)
No drift detected.

## crates/audio/src/tts/echo_tts/README.md

Coverage: full read of README; sampled source (config defaults latent 80, model 2048, 40 steps, sequence 640, heads 16, text/speaker 1280; 2048-sample crop; codec encode_zq/decode_zq; fixture and tool paths). Blocks=24 and the speaker chunk default were not checked.
- "Rust gates: cargo test ... echo_tts (9 tests)": echo_tts/tests.rs has 10 `#[test]` attributes and none are `#[ignore]`d. Test run not executed to confirm the count.

## crates/audio/src/tts/fish_drama3/EGM.md

Coverage: sampled (most interface and invariant claims grep-checked and read at the cited lines; generate.rs and tower.rs not read in full; exact N of M not tallied)
No drift detected.

## crates/audio/src/tts/fish_drama3/README.md

Coverage: sampled (sampling presets, RAS values, PauseGuard constants, rollback limit, file paths and fixture paths checked; the real-checkpoint and parity numbers are not re-run)
No drift detected.

## crates/audio/src/tts/fish_qwen3_omni/EGM.md

Coverage: sampled (symbols, RAS constants, speed and budget formulas, semantic bias guard, and no-env/no-thread claim checked; mod.rs (48 KB) not read in full)
No drift detected.

## crates/audio/src/tts/fish_qwen3_omni/README.md

Coverage: sampled (RAS window and temperature values, REFERENCE_COMMIT, and the budget formula checked; fixture generator and parity figures not re-run)
No drift detected.

## crates/audio/src/tts/higgs_audio/EGM.md

Coverage: sampled (delay-pattern, sampler, fade, prompt-string, and stop-reason symbols, codebook stride, and attention-head checks confirmed; mod.rs (53 KB) not read in full)
No drift detected.

## crates/audio/src/tts/higgs_audio/README.md

Coverage: sampled (refused config flags, DEFAULT_CODEC_REPO, sampler and fade symbols checked; the "no real-checkpoint gate" statement is a claim about unrun work and was not verified either way)
No drift detected.

## crates/audio/src/tts/higgs_audio_v3/EGM.md

Coverage: sampled (AUDIO_PLACEHOLDER_ID, DEFAULT_MAX_NEW_FRAMES, DEFAULT_SAMPLE_RATE, SamplerState, sample_independent, greedy_sample_audio, PromptBuilder/PromptParts/ReferenceCodes, and the four tensor-prefix layouts checked; mod.rs (37 KB) not read in full)
No drift detected.

## crates/audio/src/tts/higgs_audio_v3/README.md

Coverage: sampled (default max frames, sample-rate default, load_with_embed_table, layout probes, and the "no clipping" claim checked; the claim that use_delay_pattern and audio_token_id are never read was not confirmed, since the grep output was truncated)
No drift detected.

## crates/audio/src/tts/indextts/EGM.md

Coverage: sampled (GenerateParams defaults, REF_MEL_BINS, DEFAULT_SAMPLE_RATE, position_row bounds, top_k vocab refusal, max-mel-token bound, and rel_pos and kernel-parity checks confirmed; mod.rs (25 KB) and the sub-modules not read in full)
No drift detected.

## crates/audio/src/tts/indextts/README.md

Coverage: sampled (rel_pos refusal, macaron refusal, log_mel_spectrogram, sampler defaults, and file paths checked)
- Name typo: "ECPA-TDNN" (Architecture, the BigVGAN bullet) should read "ECAPA-TDNN", the name used in EGM.md and the speaker encoder (`crates/audio/src/tts/indextts/speaker.rs:SpeakerEncoder`).

## crates/audio/src/tts/irodori_tts/EGM.md

Coverage: sampled (alternating-CFG refusal, ref_max_seconds, SeededFlowNoise, build_duration_features, sample_rate, decode_latents, encode_conditions_full, build_kv_cache, forward_with_conditions, and duration clamp checked; mod.rs (136 KB) not read in full)
No drift detected.

## crates/audio/src/tts/irodori_tts/README.md

Coverage: sampled (codec path link, alternating-CFG refusal, dacvae folder loading, and FlowNoise and ModernBERT types checked; the parity figures and the AdaLN and RoPE bug notes were not re-derived)
No drift detected.

## crates/audio/src/tts/kitten_tts/EGM.md

Coverage: sampled (install_voice, generate_segment_with_durations, synthesize_with, fake-quant helpers, SAMPLE_RATE, REFERENCE_COMMIT, frontend re-exports, npz-to-safetensors mapping, per-request Rng, and finite-speed checks confirmed; mod.rs (105 KB) and frontend.rs (50 KB) not read in full)
No drift detected.

## crates/audio/src/tts/kitten_tts/README.md

Coverage: sampled (STYLE_SPLIT of 128, ActivationQuant and fake_quant_dynamic_u8, the voices.safetensors name probe, TextPreprocessor::process, and ModelConfig fields checked; the phonemizer boundary and pipeline passes were not traced end to end)
No drift detected.

## crates/audio/src/tts/kokoro/EGM.md

Coverage: sampled (open_with_backend, diagnostic_vocoder (doc hidden), VOICE_FILES, MAX_FRAMES_PER_PHONEME, round_duration (ties to even), pub seed, execution weight cache (RefCell), trace, checkpoint callback, and the no-env/no-thread claim checked; mod.rs (82 KB) not read in full)
No drift detected.

## crates/audio/src/tts/kokoro/README.md

Coverage: sampled (every tests/reference and fixture path in the reproduce block exists, provenance.json link exists, and KokoroRunner is defined at crates/runtime/src/kokoro.rs; the GPU, precision, and Metal parity claims are in other crates and were not checked)
No drift detected.

## crates/audio/src/tts/kokoro/frontend/EGM.md

Coverage: sampled (MAX_PHONEME_CHARS, usable length formula, 65536-byte input limit, [510, 1, 256] voice shape, ENGINE OnceLock, bundled module, style_row, and no-env/no-thread claim checked; mod.rs (22 KB), normalize.rs, and tagger.rs not read in full)
No drift detected.

## crates/audio/src/tts/kugelaudio/EGM.md

Coverage: sampled (speech token ids, VALID_SPEECH_TOKENS, FINAL_LATENT_LOGIT_MARGIN of 5.0, ddpm step bounds (1..=100000), scheduler beta-schedule acceptance and cosine-only betas, prediction types, backbone prefix probe, NaN scaling-factor rule, and step restore in generate checked; scheduler.rs, diffusion_head.rs and acoustic_decoder.rs not read in full)
No drift detected.

## crates/audio/src/tts/kugelaudio/README.md

Coverage: sampled (link targets ../../lm and ../../ops exist, speech token ids, the decoder-only loader, and the [out, kernel, in] conv layout note checked; decode geometry and checkpoint claims not re-run)
No drift detected.

## crates/audio/src/tts/llama/EGM.md

Coverage: sampled (all special-token constants, CODES_PER_GROUP, DEFAULT_TOKENIZER_NAME, GenerateParams defaults, i32 token-id bound, encode_audio_to_codes 1:2:4 check, and the rng import path checked; mod.rs (17 KB) and tests not read in full)
No drift detected.

## crates/audio/src/tts/llama/README.md

Coverage: sampled (special-token table ids, sampler.rs link, music::minimax_music3::rng path, and defaults checked; the zero-noise decode and the sampled-run divergence claims were not re-run)
No drift detected.

# Drift report: audio TTS families (job audio_tts_3)

Method: every doc read in full. Every Interface identifier in each EGM.md
resolves to a definition in its directory, and every cited [file:symbol]
resolves. Invariants, env vars, side effects and numeric constants were
checked by targeted reads and st/grep, not by reading all source. Test
commands and tool/fixture paths were checked for existence.

## crates/audio/src/tts/longcat_audiodit/EGM.md

Coverage: sampled (all Interface names and cited symbols resolve; invariants, constants, LC_DEBUG side effect and APG constants checked by targeted reads; source not fully read)
No drift detected.

## crates/audio/src/tts/longcat_audiodit/README.md

Coverage: sampled (EGM-level claims plus CJK/alphabetic duration constants, APG eta and momentum, RoPE base and the fixture/tool paths checked; source not fully read)
No drift detected.

## crates/audio/src/tts/megatts3/EGM.md

Coverage: sampled (all Interface names and cited symbols resolve; env/thread/logging scan clean; DiT tone rule, MAX_LATENT_FRAMES, sampler 1..=1000, draw order, layer bounds and open paths checked; source not fully read)
No drift detected.

## crates/audio/src/tts/megatts3/README.md

Coverage: sampled (quality presets, DiT/duration/aligner/chunker constants, API signatures, crossfade, reserve constant and fixture/tool paths checked; Metal and runtime/FFI/server claims are outside this directory and were not verified; source not fully read)
- [sampler.rs:sample_controlled] README "Steps and quality" says `steps` takes any count from 1 to 100. The code enforces `1..=MAX_STEPS` with `MAX_STEPS = 1000` [sampler.rs:MAX_STEPS]. No 100 cap exists in [facade.rs:synthesize_controlled]. If the 100 bound is intended, it must be enforced in a caller layer (HTTP/FFI), which this directory does not read. Otherwise the README is wrong. EGM.md states 1..=1000, so the two docs disagree.
- README names `benchmark_megatts3.py` with no path. The file is at `crates/audio/scripts/benchmark_megatts3.py`, not under `crates/audio/tools/` (the README's tools directory). Low severity; a reader looking in tools/ will not find it.

## crates/audio/src/tts/melotts/EGM.md

Coverage: sampled (all Interface names and cited symbols resolve; flow count of four coupling+flip pairs at even indices, speaker resolution order, sdp_ratio guard and layer_norm_eps default checked; source not fully read)
No drift detected.

## crates/audio/src/tts/melotts/README.md

Coverage: sampled (flow and speaker wiring, BERT layer-norm epsilons, Rng::normal, sdp_ratio guard, fixture generator and testdata paths checked; source not fully read)
No drift detected.

## crates/audio/src/tts/moss_tts/EGM.md

Coverage: sampled (all Interface names and cited symbols resolve; Model::load refusal of v1.5, sampling order (penalty, top-k, top-p, categorical, greedy no draw), INT64_MAX sentinel and n_embd % n_head guard checked; source not fully read)
No drift detected.

## crates/audio/src/tts/moss_tts/README.md

Coverage: sampled (variant dispatch, default tokenizer repos (v1 and v2), use_static_local_kv_cache unused, legacy refusal, fixture generator and testdata paths checked; source not fully read)
No drift detected.

## crates/audio/src/tts/moss_tts_nano/EGM.md

Coverage: sampled (all Interface names and cited symbols resolve; sampler and loader invariants spot-checked; source not fully read)
No drift detected.

## crates/audio/src/tts/moss_tts_nano/README.md

Coverage: sampled (GenerateParams defaults (375 frames, text and audio sampler values, penalty 1.2), prompt constants, sampler candidate handling and fixture paths checked; source not fully read)
No drift detected.

## crates/audio/src/tts/omnivoice/EGM.md

Coverage: sampled (all Interface names and cited symbols resolve; config from_json, Rc backend, seed+step RNG, num_steps guard and pipeline module checked; source not fully read)
No drift detected.

## crates/audio/src/tts/omnivoice/README.md

Coverage: sampled (tokens-per-second formula, quantization format gate (omnivoice-rowwise only), layout names, fixture and tool paths checked; source not fully read)
No drift detected.

## crates/audio/src/tts/omnivoice/duration/EGM.md

Coverage: full read (directory holds only tests.rs; `mod tests;` at duration.rs:1098 and `mod duration;` privately declared in omnivoice/mod.rs:39 confirmed)
No drift detected.

## crates/audio/src/tts/outetts/EGM.md

Coverage: sampled (all Interface names and cited symbols resolve; decode chunk 4096 and -18 LUFS / -1 dB limits, completion-prompt separator mutation, from_ids/from_lookup checked; source not fully read)
No drift detected.

## crates/audio/src/tts/outetts/README.md

Coverage: sampled (GenerateParams defaults (temperature 0.4, top-p 0.9, min-p 0.05, top-k 40, penalty 1.1 over 64, max 1200), model_type dispatch, DEFAULT_SAMPLE_RATE checked; source not fully read)
No drift detected.

## crates/audio/src/tts/pocket_tts/EGM.md

Coverage: sampled (all Interface names and cited symbols resolve; zero-head and divisibility checks, conditioning multiple, bos seed, noise width and StopReason checked; source not fully read)
No drift detected.

## crates/audio/src/tts/pocket_tts/README.md

Coverage: sampled (eos threshold default -4.0, frames_after_eos default (3 + 2 short-text branch), lsd_decode, prepare_text_prompt checked; source not fully read)
No drift detected.

## crates/audio/src/tts/qwen3/EGM.md

Coverage: sampled (all Interface names and cited symbols resolve; special token ids and offsets from TOKENIZER_LENGTH=151669, 7-code grouping and prefill rule checked; source not fully read)
No drift detected.

## crates/audio/src/tts/qwen3/README.md

Coverage: sampled (special-token table (all ids match), GenerateParams defaults (0.6, 0.8, top-k off, penalty 1.3 over 20, 1200), fixture paths checked; source not fully read)
No drift detected.

## crates/audio/src/tts/rumik_oss/EGM.md

Coverage: sampled (all Interface names and cited symbols resolve; quantizer-index resync rule, non-unit clear, min_tokens gate, two stop-predictor tensor layouts, DEFAULT_SPEAKERS and MIMI_REPO checked; source not fully read)
No drift detected.

## crates/audio/src/tts/rumik_oss/README.md

Coverage: sampled (stop-predictor layout, frame-rate and quantizer defaults, prompt helpers, fixture and tool paths checked; source not fully read)
No drift detected.

## crates/audio/src/tts/sesame/EGM.md

Coverage: sampled (claims verified ~25 of ~30; mod.rs read in targeted ranges, tests.rs not read)
No drift detected.

## crates/audio/src/tts/sesame/README.md

Coverage: sampled (claims verified ~10 of ~15; mod.rs read in targeted ranges, tests.rs not read)
No drift detected.

## crates/audio/src/tts/soprano/EGM.md

Coverage: sampled (claims verified ~20 of ~25; mod.rs, decoder.rs and text.rs read in targeted ranges, tests.rs not read)
No drift detected.

## crates/audio/src/tts/soprano/README.md

Coverage: sampled (claims verified ~12 of ~15; mod.rs, decoder.rs and text.rs read in targeted ranges, tests.rs not read)
No drift detected.

## crates/audio/src/tts/spark/EGM.md

Coverage: sampled (claims verified ~15 of ~18; mod.rs read in targeted ranges, tests.rs not read)
No drift detected.

## crates/audio/src/tts/spark/README.md

Coverage: sampled (claims verified ~6 of ~8; mod.rs read in targeted ranges, tests.rs not read)
No drift detected.

## crates/audio/src/tts/tada/EGM.md

Coverage: sampled (claims verified ~18 of ~22; mod.rs, gray_code.rs read in targeted ranges, diffusion_head.rs and tests.rs not read)
No drift detected.

## crates/audio/src/tts/tada/README.md

Coverage: sampled (claims verified ~8 of ~12; mod.rs and gray_code.rs read in targeted ranges)
No drift detected.

## crates/audio/src/tts/vibevoice/EGM.md

Coverage: sampled (claims verified ~18 of ~22; mod.rs and scheduler.rs read in targeted ranges, acoustic_decoder.rs and diffusion_head.rs not read)
No drift detected.

## crates/audio/src/tts/vibevoice/README.md

Coverage: sampled (claims verified ~8 of ~12; mod.rs and scheduler.rs read in targeted ranges)
No drift detected.

## crates/audio/src/tts/voxcpm/EGM.md

Coverage: sampled (claims verified ~18 of ~22; mod.rs, audio_vae.rs, dit.rs, encoder.rs, minicpm.rs read in targeted ranges)
No drift detected.

## crates/audio/src/tts/voxcpm/README.md

Coverage: sampled (claims verified ~8 of ~12; mod.rs, dit.rs, audio_vae.rs read in targeted ranges)
No drift detected.

## crates/audio/src/tts/voxcpm2/EGM.md

Coverage: sampled (claims verified ~20 of ~24; mod.rs, audio_vae.rs, dit.rs read in targeted ranges, tests.rs not read)
No drift detected.

## crates/audio/src/tts/voxcpm2/README.md

Coverage: sampled (claims verified ~15 of ~20; mod.rs, audio_vae.rs, dit.rs read in targeted ranges, tests.rs not read)
- Open items say the `mean_mode` checkpoint variant is refused, but the code accepts it and maps it to `dit_mean_mode` [mod.rs:SubConfig]. The README's own "V2 differences" section describes it as live, and tests.rs pins the normalization. The refusal claim is false.
- Names the DiT type `VoxCPMLocDiTV2`; the Rust type is `VoxCpmLocDiTv2` [dit.rs:VoxCpmLocDiTv2].

## crates/audio/src/tts/voxtral_tts/EGM.md

Coverage: sampled (claims verified ~20 of ~24; mod.rs read in targeted ranges, tests.rs not read)
No drift detected.

## crates/audio/src/tts/voxtral_tts/README.md

Coverage: sampled (claims verified ~12 of ~15; mod.rs read in targeted ranges)
No drift detected.

## crates/audio/src/tts/zonos2/EGM.md

Coverage: sampled (claims verified ~18 of ~22; mod.rs read in targeted ranges, tests.rs not read)
- Omits `TtsSamplingParams` fields `top_p` (default 0.0), `repetition_window` (50) and `repetition_codebooks` (8) from its defaults list [mod.rs:TtsSamplingParams]. Minor.

## crates/audio/src/tts/zonos2/README.md

Coverage: sampled (claims verified ~12 of ~16; mod.rs read in targeted ranges, tests.rs not read)
- Layout section names `Zonos2Config.from_dict`; the Rust entry point is `ModelConfig::from_json` [mod.rs:ModelConfig]. `from_dict` appears only in a doc comment at mod.rs:ModelConfig.

## crates/bench/EGM.md

Coverage: sampled (claims verified about 35 of 38; crate is about 22k lines including tests, source not fully read)
- [src/model_mode.rs:run_model_mode, src/real_model.rs:run_protocol_case_speculating] Doc says "a drafter forces greedy shaping". False. `--speculative` does not change shaping. `run_model_mode` refuses `--speculative` unless `--shaping greedy` is given (exit 2), and `run_protocol_case_speculating` refuses `Sampled` shaping instead of promoting it. `--speculative-drafter` only selects the drafter [src/main.rs:main].
- [src/model_mode.rs:run_model_mode] Omits that `--model` mode reads `TURBOSPARK_ROUTED_BATCH` and `TURBOSPARK_BATCHED_GEMV` and echoes them in its banner, marked INERT when prefill is sequential. Relevant to the Invariants and Side Effects sections.

## crates/bench/AGENTS.md

Coverage: full read (68 lines; every Read-first link and directory-map path resolves)
No drift detected.

## crates/bench/README.md

Coverage: sampled (claims verified about 30 of 33; test files checked by listing, source not fully read)
- [tests/:directory] "This crate contains 81 test files in `tests/`" is wrong. There are 91 top-level `.rs` files in `tests/`, plus 8 helper module directories (`humanizer_common`, `lfm3_common`, `oracle_common`, `quality_common`, `qwen2_dense_common`, `qwen3_dense_common`, `qwen3vl_common`, `vision_oracle_rounds`).
- [tests/:directory] The categorized lists omit 10 files: `dflash_generation_gate.rs`, `dflash_tree_pin.rs`, `lfm3_memory_oracle.rs`, `lfm3_prefill_arm_probe.rs`, `lfm3_q8_memory_oracle.rs`, `lfm3_q8_quality_gate.rs`, `lfm3_quality_gate.rs`, `moe_route_capture.rs`, `turn_snapshot_gate.rs`, `turn_snapshot_ttft_probe.rs`.
- [src/main.rs:main, src/scripted.rs:run_real_mode] "Binary Execution Modes" omits the `--real` synthetic-model mode (macOS, real GPU forward pass on a generated install). Only the default scripted mode is described, and its "no GPU forward passes" text is correct only for that default.
- [src/meeting_eval_main.rs:score, src/meeting/mod.rs:score_panel] Names, defaults and `--clips` values match the code. No drift.

## crates/bench/scripts/EGM.md

Coverage: sampled (import script body and `frozen_e_score.py` read; `prepare_frozen_e_clips.py` checked by targeted grep only)
- [scripts/import_frozen_e_evidence.py:main, scripts/import_frozen_e_evidence.py:SOURCES] Doc says "Only references.json and E_seglst.json are read from the evidence dir". False. `main` also reads `evaluation_manifest.json` for the clip list, and `manifest.json` records SHA-256 of all three `SOURCES` (`evaluation_manifest.json`, `E_seglst.json`, `TurboSpark_Nemotron_E_Frozen_Qualification_v3.ipynb`).
- Otherwise the named symbols (`FROZEN_E`, `MATCHED_SIX`, `sha256`, `MEETING_SHA256`, `tokens`, `edit`, `cpwer`), the stop-on-source-mismatch and exit-at-end behavior, the 120 s cut, PCM_16 output, and the stdlib-only import dependency match the code. No other drift.

## crates/bench/src/EGM.md

Coverage: sampled (symbols and gating claims checked by targeted reads; source about 3.8k lines, not fully read)
No drift detected.

## crates/bench/src/meeting/EGM.md

Coverage: sampled (`mod.rs` and `cpwer.rs` key sections and `seglst.rs` `group_by_session` read; `seglst.rs` otherwise not fully read)
No drift detected.

## crates/catalog/EGM.md

Coverage: sampled (claims verified 14 of 15, source not fully read)
No drift detected.

## crates/catalog/README.md

Coverage: sampled (claims verified 40 of 50, source not fully read)
- Omits the library subsystem entirely: no mention of external-drive libraries, move, delete, import, `libraries.json`, or the `library.json` marker [src/library/mod.rs:INTERNAL_LIBRARY_ID, src/library/registry.rs:LibraryRegistry].
- Omits `check.rs`, which provides the compatibility check API (`check_repo`, `arch_hint`) re-exported from the crate root [src/check.rs:check_repo, src/lib.rs:check_repo].
- Overstated: Key Modules says `recommend/rank.rs` tiers "based on system memory and hardware configuration". rank.rs orders by `Key` (runs, evidence, tok_s, install_bytes, suspicious) [src/recommend/rank.rs:Key]. Memory enters only through the fit verdict [src/recommend/fit.rs:fit].
- Wrong location: `stream.rs` is described as containing the flint intake `write_flint_install_streamed`. That function is defined in repack [crates/repack/src/gemma4_checkpoint/mod.rs:write_flint_install_streamed]; stream.rs only calls it [src/stream.rs:write_flint_install_streamed].
- Tests list omits library_install.rs, store_relocation_libraries.rs, check_repo.rs, hub_cache.rs and hub_search_network.rs [tests/library_install.rs, tests/check_repo.rs].

## crates/catalog/AGENTS.md

Coverage: sampled (claims verified 12 of 13, source not fully read)
- Wrong name: the flint rule says `repack::parse_flint_quantization` "lives in `mlx_export`, not `quantization`". No module named `mlx_export` exists in crates/repack/src. The function is defined in crates/repack/src/flint_config.rs [crates/repack/src/flint_config.rs:parse_flint_quantization]. `mlx_export` is the JSON config block key it reads, not a module.

## crates/catalog/examples/EGM.md

Coverage: sampled (claims verified 6 of 6, source not fully read)
No drift detected.

## crates/catalog/src/EGM.md

Coverage: sampled (claims verified 20 of 21, source not fully read)
No drift detected.

## crates/catalog/src/library/EGM.md

Coverage: sampled (claims verified 16 of 18, source not fully read)
- Imprecise: the Import line lists `import_moved_root` as a free function. It is a method in the `impl LibraryRegistry` block [src/library/import.rs:import_moved_root].

## crates/catalog/src/probe/EGM.md

Coverage: sampled (claims verified 8 of 8, probe/mod.rs gates read, gguf.rs and safetensors.rs partly read)
No drift detected.

## crates/catalog/src/recommend/EGM.md

Coverage: sampled (claims verified 11 of 13, source not fully read)
- Overstated: Invariant "Quantization choice walks QUANT_LADDER, picking the largest published file that plausibly fits" does not match [src/recommend/discover.rs:choose_quantization]. QUANT_LADDER is only a filter (a file must have a ladder rank) and a size tie-breaker. The pick is the largest file at or under the physical budget. When none fits, the function falls back to the smallest runnable file (`runnable.first()`), which the doc omits.
- Omitted: sharded GGUF files (names containing `-of-0`) are excluded from quantization choice [src/recommend/discover.rs:choose_quantization].

## crates/cli/EGM.md

Coverage: full read of the doc; claims checked against the [[bin]] table in crates/cli/Cargo.toml and the child EGM.md files.
No drift detected.

## crates/cli/AGENTS.md

Coverage: sampled (checked --chat/--image refusal [chat.rs:run], prompt template path [front/models.rs:check_args], flag arity [front/models.rs:takes_value via invocation::OPTIONS], view verbs and --json [model_cmd/views.rs], doc links; main.rs exit-status mapping and the no-unsafe claim in src/ui not read).
No drift detected.

## crates/cli/README.md

Coverage: sampled (flags and subcommands checked against invocation OPTIONS, bin/turbospark.rs Cmd enum, bin/model.rs, bin/steer.rs, crates/server args, agent list [agent.rs:is_agent], expert cache auto sizing [model-io/src/expert_cache_policy.rs:auto_slots_at] and its constants, and the tests directory; measured tok/s and RAM figures are not verifiable from code and were not checked).
- [src/front/models.rs:plan_run] and [src/front/models.rs:check_args]: line 141 says `turbospark run <model> "hello"` is an "Alias for turbospark-check --prompt". Code sends a positional prompt through the chat template (`--messages-file /dev/stdin`, one user turn) unless `--raw` is given; only `--raw` produces `--prompt`. The README is wrong for the default path. Note that the root CLAUDE.md states the same shorthand "encodes the text verbatim", which matches the README and not the code. The crate AGENTS.md matches the code.
- Omission: the Tests section (lines 198-206) does not list tests/daemon_root.rs, tests/library_lifecycle.rs, or tests/model_library_cli.rs, which exist in crates/cli/tests/.

## crates/cli/src/EGM.md

Coverage: sampled (all Interface symbols located in daemon.rs and agent.rs; pid/meta 0600 and run dir 0700 [daemon.rs:write_private_file, daemon.rs:create_private_run_dir]; --api-key stripping [daemon.rs:server_args_and_api_key]; log path sibling of run dir [daemon.rs:logs_dir, daemon.rs:run_dir]; require_root; get_running_pid cleanup; TURBOSPARK_PREFIX_REUSE [chat.rs:run]; main.rs cfg gates; start/stop bodies read in part).
No drift detected.

## crates/cli/src/bin/EGM.md

Coverage: sampled (#[path] includes [bin/turbospark.rs], model delegation [bin/model.rs], steer exit codes [bin/steer.rs:main], steer_data contents, child EGM links; the "thin shell" characterization was not measured, and bin/music.rs and bin/speak.rs run 700+ lines).
No drift detected.

## crates/cli/src/bin/model_cmd/EGM.md

Coverage: sampled (every Interface symbol located; alias ASCII check [mod.rs:check_alias]; /bin/df use and KINDS [transfer.rs]; image staging refusal [image_source.rs:materialize]; shared SIGINT flag [mod.rs:INTERRUPTED] used by audio.rs, drama3.rs, transfer.rs; auth stdin read [auth.rs:read_token]; bodies of mod.rs pull and pull_image not read).
- Omission: [mod.rs:recommend] (mod.rs:951, dispatched from bin/model.rs:323) is a public catalog entry point that the Interface list does not name, though the Overview says "catalog listing, probe, and install entry points".
- Minor: [render.rs:human_bytes] resolves only as a `pub use catalog::human_bytes` re-export (render.rs:14), not a function render owns.

## crates/cli/src/bin/steer_cmd/EGM.md

Coverage: sampled (every Interface symbol located; family refusal [walk.rs:resolve_model]; pack id check [packs.rs:parse]; CAPTURE_ENV set and removed [walk.rs:capture_side]; MAX_CAPTURE_CONTEXT cap; .partial plus rename [extract.rs:write_atomically]; extract and check run bodies not fully read).
No drift detected.

## crates/cli/src/front/EGM.md

Coverage: sampled (every Interface symbol located; HIDE_RESOLVED_REQUEST env set [models.rs:check_command]; interactive() gate [models.rs:interactive]; loopback probe [agent.rs:http_health_ok] used by [server.rs:snapshot]; the rest of ensure_installed after its Verbatim and Catalog arms and the run() body were not read, so the "--yes or non-terminal downloads without asking" claim is unverified).
No drift detected.

## crates/cli/src/generate/EGM.md

Coverage: sampled (every Interface symbol and attach_* family located; env var reads [format.rs:print_phases], [mod.rs:resolve_chunk_tokens], [mod.rs:run_one_turn]; TURBOSPARK_EXPERT_DISK_IO only in a footer hint; Outcome semantics [mod.rs:Outcome] and chat --image refusal [chat.rs:run]; image-install refusal [session.rs:reject_image_install]; the per-turn prepare_images call site not read).
- Omission: the humanize mode is not documented. [mod.rs:try_generate] dispatches `Mode::HumanizeFile` to [mod.rs:run_humanize] (mod.rs:55, :65), selected by `--humanize-file` [crates/invocation/src/options.rs] and the `turbospark humanize` subcommand [bin/turbospark.rs]. [mod.rs:run_prompt] is also not named. The Overview describes only chat-template streaming.

## crates/cli/src/ui/EGM.md

Coverage: sampled (color decision order and env vars [mod.rs:color_enabled], Paint methods, ok/note/warn/error stream and marks, columns/clip/section, Table/Cell/CellStyle/Align located; --json bypass checked against views.rs print_json).
No drift detected. Aside, not drift in this doc: the ui/mod.rs module doc (line 9) lists `*` among the marks, and no helper emits it. The EGM already states this correctly.

## crates/compute/EGM.md

Coverage: full read
No drift detected.

## crates/compute/README.md

Coverage: sampled (about 40 of about 45 extracted claims checked; source not fully read)
- Key Modules omits two public modules: [crates/compute/src/short_conv.rs:short_conv_reference] (LFM3 gated short convolution, PORT-LOCAL) and [crates/compute/src/steering_extract.rs:difference_of_means] with [steering_extract.rs:layer_stats] and [steering_extract.rs:derived_alpha_cap] (difference-of-means direction extraction for turbospark-steer).
- Tests section omits [crates/compute/tests/dash_oracle.rs] (pins dash.rs to mlx-vlm) and [crates/compute/tests/gemma4_vision_reference.rs] (pins vision_gemma4.rs).
- Tests entry for tests/kv_quant.rs says it "verifies TurboQuant compression error bounds", but the fixture it includes, [crates/compute/tests/generated/kv_quant_oracle.rs], is headed as generated with the numpy fallback and says to re-run without it before trusting the fixture beyond structural review. The caveat is omitted.

## crates/compute/AGENTS.md

Coverage: full read
- Rule "FP16 uses the `half` crate" is false. The crate depends only on turbospark-core (aliased `foundation`) [crates/compute/Cargo.toml:foundation], and has no `half` import anywhere under crates/compute. FP16 goes through `foundation::LogitValue` [crates/compute/src/quant_1bit.rs:f16_to_f32], [crates/compute/src/quant_1bit.rs:f32_to_f16], and the alias `F16` in [crates/compute/src/gdn.rs] and the quant_gguf modules. The rule as written would push a contributor to add a second FP16 representation, which the same file forbids.
- Verified: BF16 bit-shift rule, the read-first links (.claude/docs/modules/compute.md, docs/BENCHMARKS.md, docs/TESTING.md all resolve), `#![forbid(unsafe_code)]` [crates/compute/src/lib.rs], and the test and cargo check commands. The cargo check command was not run.

## crates/compute/src/EGM.md

Coverage: sampled (existence of every named symbol and all five Tolerance constants checked; invariants and side effects spot-checked; source not fully read)
- Invariant "Affine quantization uses 64-element groups" is overstated. It holds for [crates/compute/src/quant.rs:GROUP_SIZE] (INT4/INT8). The 1-bit and 2-bit affine formats use 128-element groups, [crates/compute/src/quant_1bit.rs:BONSAI_GROUP_SIZE] and [crates/compute/src/quant_2bit.rs:TERNARY_GROUP_SIZE], and the 1-bit doc says `group_size` is a per-checkpoint property carried on the row [quant_1bit.rs:Int1AffineRow].

## crates/compute/src/quant_gguf/EGM.md

Coverage: sampled (all named symbols, asserts, and the duplicated-layout doc claims checked; source files read in part, not in full)
- Omits the behavior of [crates/compute/src/quant_gguf/pearson.rs:pearson], which returns 0.0 when either input is constant. Callers' thresholds depend on this (the doc comment says an undefined correlation must not read as a match), so the EGM should state it next to the "non-empty, equal length" invariant.

## crates/core/EGM.md

Coverage: full read (doc, Cargo.toml, all 9 src files, 2 test files; crate is 1313 lines).
- Crate summary omits the chunk_sizing and engagement modules (the doc lists only token types, runtime knob sets, prefill planning, steering modes and humanizer). [crates/core/src/lib.rs:chunk_sizing] [crates/core/src/lib.rs:engagement]

## crates/core/README.md

Coverage: full read (doc, Cargo.toml, all src and test files, plus targeted greps of downstream crates).
- Key Modules omits the engagement module, the exported `engaged!` macro (`#[macro_export]`, used as `foundation::engaged!` in crates/gpu) and the `TURBOSPARK_ENGAGEMENT` env var. [crates/core/src/engagement.rs:engaged] [crates/core/src/engagement.rs:enabled]
- Safety says "Pure, deterministic logic with no filesystem or network I/O." Filesystem and network claim holds, but `enabled()` reads `TURBOSPARK_ENGAGEMENT` and `engaged!` writes `[engaged] NAME` to stderr when it is set. The Safety section should say so. [crates/core/src/engagement.rs:enabled] [crates/core/src/engagement.rs:engaged]
- Crate Gotcha 1 says runtime parameters are checked against sorted allowed sets "rather than open integer ranges". This contradicts the doc's own bullets: `ALLOWED_SPECULATION_BLOCKS` is a `RangeInclusive<u32>` (1..=15) and `DEFAULT_MAX_CONTEXT` has no allowed set. [crates/core/src/runtime_config.rs:ALLOWED_SPECULATION_BLOCKS] [crates/core/src/runtime_config.rs:DEFAULT_MAX_CONTEXT]
- runtime_config bullet says the module "Enforces ascending order and valid defaults at compile time via const assertions". The assertions cover only `ALLOWED_CACHE_SLOTS` and `ALLOWED_CHUNK_SIZES` and their defaults. `ALLOWED_SPECULATION_BLOCKS`, `DEFAULT_ADAPTIVE_CAP` and `MIN_ADAPTIVE_CAP` have no compile-time check. (src/EGM.md states this correctly.) [crates/core/src/runtime_config.rs:ALLOWED_CACHE_SLOTS]
- Crate Gotcha 2 says negative token IDs are "reserved for sentinel error codes or unassigned positions". Nothing in crates/core defines or checks that convention, so the claim is not established by this crate. Not checked across the workspace. [crates/core/src/primitives.rs:TokenId]
- Verified without drift: `TokenId = i32`, `LogitValue` = `half::f16`, cache slot and chunk set literals and defaults (16, 128), `ALLOWED_SPECULATION_BLOCKS = 1..=15`, `DEFAULT_ADAPTIVE_CAP = 4`, `MIN_ADAPTIVE_CAP = 3`, `DEFAULT_MAX_CONTEXT = 4096`, `MAX_CHUNK_TOKENS = 4096`, the steering enum, `STEERING_MODE_NAMES`, humanizer items, the `foundation` alias, zero internal workspace dependencies (Cargo.toml has only `half`, plus dev-dep `sha2`), and the `turbospark_compute::steering` and `turbospark_gpu::encode_steer_direction` references. The Renorm "analytic norm repair" claim matches crates/compute/src/steering.rs.

## crates/core/AGENTS.md

Coverage: full read (doc is 25 lines; its three linked docs and the `cargo test -p turbospark-core` package name were checked).
No drift detected.

## crates/core/src/EGM.md

Coverage: full read (all 7 source files in src plus both test files; every interface symbol was located).
- Side Effects says "Otherwise pure; no disk or network access. None beyond the above." This omits the stderr write from the `engaged!` macro, which runs whenever `TURBOSPARK_ENGAGEMENT=1`. [crates/core/src/engagement.rs:engaged]
- Invariant for `PrefillChunkCommitState::mark_dirty` lists only the positive-token-count requirement. The code also asserts that `start_position + token_count` does not overflow `usize`, which the doc omits. [crates/core/src/prefill.rs:PrefillChunkCommitState]
- Interface says `RewriteAudit::warnings` produces "stderr report lines". The function returns `Vec<String>` and prints nothing; the caller writes to stderr. Wording only. [crates/core/src/humanizer.rs:RewriteAudit]
- Outside the doc, but in the module docs this EGM indexes: the `prefill.rs` header (line 3) says the module was "Ported from `Runtime/Prefill/PrefillRuntimeConfig.swift`". No file of that name exists in the repository, so the reference does not resolve. [crates/core/src/prefill.rs:prefill_chunk_spans]
- Verified without drift: the Overview's consumers (runtime, invocation, cli and server all depend on `turbospark-core`), `runtime_config` symbols and const assertions, `prefill_chunk_spans` clamping, empty-plan and overflow-panic behavior, `max_of` deriving `MAX_CHUNK_TOKENS`, steering encodings (Ablate 0, Add 1, Clamp 2, Renorm 3) and `can_grow`, `default_max_new` clamp and saturation, `REUSE_FLAG_THRESHOLD = 0.22`, `PROMPT_FINGERPRINT`, the `TURBOSPARK_ENGAGEMENT` OnceLock cache, and the `lib.rs` module wiring.

## crates/coreml/EGM.md

Coverage: full read of doc; 6 of 6 structural claims verified (sub-doc links, tests/testdata have no EGM.md, tools contents, tests/fixtures.rs is cfg(target_os = "macos")); source read in part.
No drift detected.

## crates/coreml/AGENTS.md

Coverage: sampled (claims verified 24 of 25; lib.rs, cache.rs, stub.rs, sys.rs load/predict/state/compile/prune/compute_plan read; sys.rs marshalling helpers and tests not read).
- "Programs are generated via tools/gen_ane_prefill_programs.py and validated against manifest.json": this crate does not validate against manifest.json. The check runs in the runtime [crates/runtime/src/families/qwen/ane_split.rs:manifest program validation, line 333]. Here, sys.rs:check_plane validates against the model's feature specs only. Doc is true only for the runtime.

## crates/coreml/examples/EGM.md

Coverage: full read (both examples and the doc read in full).
No drift detected.

## crates/coreml/src/EGM.md

Coverage: sampled (claims verified 27 of 33; lib.rs, stub.rs, cache.rs, error.rs variants, tensor.rs strided_extent and f16 conversion read; sys.rs load, predict, state, compile, prune, compute_plan, SharedF16Plane read; sys.rs lines 130-405 (marshalling helpers), 1340-1400 and the tests not read).
- Cache key overstated: doc says the key "hashes package contents (regular files only, symlinks skipped on purpose) so a changed package recompiles". Code hashes Manifest.json bytes, then each file's relative path, size and mtime, then the OS build string. Weight bytes are not hashed. A single-file .mlmodel is hashed in full [cache.rs:cache_key]. The OS build string in the key is omitted [cache.rs:cache_key], [sys.rs:os_tag].
- Omitted invariant: predict_with_state refuses a CoreMlState created by a different CoreMlModel instance (process-unique owner id) [sys.rs:CoreMlModel::predict_with_state]. The doc covers only copy_from.
- Omitted behavior: compute_plan refuses a path that is not a compiled .mlmodelc directory, and refuses networks that are not ML programs or pipelines [sys.rs:compute_plan], [sys.rs:walk_structure].
- Omitted behavior: check_inputs refuses duplicate input names [lib.rs:check_inputs]. The "every predict input must be declared" line is incomplete without it.
- Omitted behavior: predict_shared requires every input and output to be bound, and refuses a result whose output backing Core ML did not write into [sys.rs:predict_shared].
- Wording: "Allocates page-rounded shared buffers". The code aligns bytes-per-row with IOSurfaceAlignProperty and does not round to pages itself; the page-aligned base is an IOSurface property noted in a comment [sys.rs:SharedF16Plane::new]. Low severity.

## crates/coreml/tools/EGM.md

Coverage: sampled (claims verified 17 of 19; read in full: gen_coreml_fixtures.py, gen_ane_sweep_programs.py main and const_int4_program, bake_ane_vision_programs.py const_f16_program; read in part: bake_ane_prefill_programs.py (Install.rows, fp16_mapping, lut_table, baked_program, constants), gen_ane_prefill_programs.py (matmul, build, main head). Not read: bake_ane_prefill_programs.py bake_classes/main bodies, bake_ane_vision_programs.py main.)
- Side effect misdescribed: doc says "Each script writes programs or packages into an output directory argument [gen_coreml_fixtures.py:main]". gen_coreml_fixtures.py takes no output argument. It writes to a fixed OUT (crates/coreml/testdata), removes any existing package of the same name first (shutil.rmtree), and rewrites testdata/manifest.json. The destructive overwrite is omitted.
- Citation mismatch: the Baked INT4 layout claim (row-major codes low nibble first, BF16 scales and biases per 64 columns) cites [bake_ane_prefill_programs.py:baked_program]. The nibble unpacking and scale/bias reads are in Install.rows; baked_program only consumes them [bake_ane_prefill_programs.py:Install.rows].

## crates/embeddings/AGENTS.md

Coverage: sampled (checked test gating, excludes, env var names, family doc links, and the audio and docs link targets; cargo commands were not run)
- Overstated: "Default suite: ... frontend (tokenizer, image, mel) parity. No install needed." Real tokenizer-vs-reference tests are #[ignore]d for gte [crates/embeddings/src/gte/tests.rs:real_tokenizer_matches_sentence_transformers_ids], embeddinggemma [crates/embeddings/src/embeddinggemma/tests.rs:real_tokenizer_matches_sentence_transformers_ids], siglip2 [crates/embeddings/src/siglip2/tests.rs:real_tokenizer_ids_match_processor], and clap [crates/embeddings/src/clap/tests.rs:real_tokenizer_ids]. Only nomic_bert runs a tokenizer parity test without an install [crates/embeddings/src/nomic_bert/tests.rs:pinned_tokenizer_matches_sentence_transformers_ids].

## crates/embeddings/EGM.md

Coverage: sampled (checked Cargo.toml features, excludes, lib name, dependency list, and the subdirectory list against src/ and tools/)
No drift detected.

## crates/embeddings/MODELS.md

Coverage: sampled (compared every revision and converted_from entry against src/pins.rs; Hub facts such as blob ids and search results cannot be checked offline)
No drift detected. All revisions and MLX-to-upstream pairings match [crates/embeddings/src/pins.rs:PINS], and the 5- and 6-bit EmbeddingGemma pins match the refusal in [crates/embeddings/src/embeddinggemma/config.rs:VERIFIED_QUANT_BITS].

## crates/embeddings/src/EGM.md

Coverage: sampled (checked re-exports in lib.rs, the pub items listed for pooling, similarity, normalize, weights, st_config, tokenize, error, and the invariants on constants and quantization mode)
No drift detected.

## crates/embeddings/src/bert/EGM.md

Coverage: sampled (checked every listed type and method exists, the quantization constants, the XLM-R position rule, and the refusal of unimplemented config values)
No drift detected.

## crates/embeddings/src/bert/README.md

Coverage: sampled (checked the public API list, modules.json requirement, pooling-source policy, refused config keys including do_lower_case and include_prompt, and the test tolerances; the numeric tables, timings and mutation results were not re-run)
No drift detected.

## crates/embeddings/src/clap/EGM.md

Coverage: sampled (checked every listed type and method, LongAudioCrop default, validate() methods, and the f64 and complex64 spectrum rounding)
- Citation mismatch: the invariant "shorter ones tiled then zero-padded" cites [crates/embeddings/src/clap/features.rs:LongAudioCrop], but that enum only defines Head, Center and Offset(n) and holds no tiling or padding logic. Cite the extraction routine instead [crates/embeddings/src/clap/features.rs:ClapFeatureExtractor].

## crates/embeddings/src/clap/README.md

Coverage: sampled (checked the API, LongAudioCrop default and embed_audio using Head, classify softmax, 512-token text limit [crates/embeddings/src/clap/config.rs:max_tokens], WAV path, refusals, and the test names listed under Verified stages; numeric tables and timings were not re-run)
- Nonexistent test named in the mutation table: `positions_start_after_padding_idx` does not exist anywhere under crates/embeddings/src/clap/ [crates/embeddings/src/clap/tests.rs]. The other names in that table (tiny_text_tower_matches_reference, and the other Verified stages names) do exist.

## crates/embeddings/src/embeddinggemma/EGM.md

Coverage: sampled (checked every listed type and method, Layout variants, the quantization constants, and the sliding band rule)
- Omitted invariant: the code refuses any config where use_bidirectional_attention is not true, because a causal config would embed with the wrong mask [crates/embeddings/src/embeddinggemma/config.rs:GemmaConfig::from_json]. The doc does not state it.

## crates/embeddings/src/embeddinggemma/README.md

Coverage: sampled (checked sliding band width 512/2+1 = 257 [config.rs:band_width], local RoPE base for sliding layers [backbone.rs], layout detection by tensor name and mean-pooling fallback [mod.rs:EmbeddingGemma::load], truncate_dim range, the activation and quantization refusals; numeric tables and timings were not re-run)
- Minor omission: the refusal list does not mention that use_bidirectional_attention must be true (code refuses false, [crates/embeddings/src/embeddinggemma/config.rs:GemmaConfig::from_json]). "causal attention" in the refusal list covers it only by implication.

## crates/embeddings/src/gte/EGM.md

Coverage: sampled (checked every listed type and method, rope_tables recomputed per forward [gte/encoder.rs:Encoder::forward_traced], BadConfig for required keys and the model_type check, and defaults for max_position_embeddings, type_vocab_size and rope_theta [gte/config.rs:GteConfig::from_json])
No drift detected.

## crates/embeddings/src/gte/README.md

Coverage: sampled (checked optional tokenizer.json, embed_ids limit against max_position_embeddings [gte/mod.rs:embed_ids], rope_scaling/layer_norm_type/pack_qkv/logn/type_vocab_size refusals, word table mapped when F32 [gte/encoder.rs:WORD_TABLE], and the test tolerances; numeric tables and timings were not re-run)
No drift detected.

## crates/embeddings/src/nomic_bert/EGM.md

Coverage: sampled (checked every listed type and method, MLP_MULTIPLE_OF, padded_vocab_size rounding with default 1 [nomic_bert/config.rs:padded_vocab_size], and ntk_base [nomic_bert/encoder.rs:ntk_base])
No drift detected.

## crates/embeddings/src/nomic_bert/README.md

Coverage: sampled (checked the NTK formula and base-1000 boundary [encoder.rs:ntk_base], rotary_scaling_factor absent/null/0 handling [config.rs:NomicBertConfig::from_json], word table copied to heap [encoder.rs:Encoder::load], the refused flags, and the test tolerances; numeric tables and timings were not re-run)
- Incomplete refusal list: the code also refuses `parallel_block_tied_norm = true` [crates/embeddings/src/nomic_bert/config.rs:NomicBertConfig::from_json], which the README's list of refused flags omits.

## crates/embeddings/src/siglip2/EGM.md

Coverage: sampled (checked every listed type, function and constant, the tokenize position-count guard [siglip2/mod.rs:Siglip2::tokenize], the gelu_pytorch_tanh-only rule [siglip2/config.rs:HIDDEN_ACT], and the pixel_attention_mask helper)
No drift detected.

## crates/embeddings/src/siglip2/README.md

Coverage: sampled (checked the quantized-checkpoint, non-Fast processor, resample, vision_use_head and projection_size refusals, the Mapped token table, the logit/probability API, and the test tolerances; numeric tables and timings were not re-run)
No drift detected.

## crates/embeddings/tools/EGM.md

Coverage: sampled (checked argparse flags of all six generators, trust_remote_code only in gte and nomic_bert, HF_HUB_OFFLINE forcing, torchvision import in the siglip2 generator, and module-level torch import in every generator; generators were not run, no venv here)
No drift detected.

## crates/ffi/AGENTS.md

Coverage: sampled (claims verified 9 of 9 checked; Swift side, the verification reference and .claude/docs/modules/ffi.md body not read)
No drift detected.

## crates/ffi/EGM.md

Coverage: sampled (claims verified: 157 exported ts_ functions, guard coverage, unguarded pair, staticlib+rlib crate-type, no workspace dependents, lib.rs macOS cfgs)
No drift detected.

## crates/ffi/README.md

Coverage: sampled (claims verified: ABI_VERSION and TS_ERR_* codes, prefixReuse/guardrails process-level wiring, server thread and tokio runtime, c_surface.rs and listed domain test files exist, Makefile swift-test-real MODEL var, ts_session_cancel atomic flag; Whisper/Qwen3 ASR/Parakeet and Engine::Real not checked)
No drift detected. Daemon environment forwarding is omitted here but is reported under crates/ffi/src/api/EGM.md.

## crates/ffi/include/EGM.md

Coverage: sampled (turbospark.h is about 121 KB and was not read in full; verified TS_ABI_VERSION 2 and TS_ERR_* 0-8 against abi.rs, header rules 3 and 4 against the header text, ts_session_open non-macOS UNSUPPORTED in api/session.rs:open_platform, ts_chip_supports_ane_prefill)
No drift detected.

## crates/ffi/src/EGM.md

Coverage: sampled (claims verified: ABI guard exceptions, server prefixReuse applied in api/server.rs:ts_server_start and server.rs:Server::attach, STT MAX_APPEND_BYTES = 2 s of 16 kHz f32, 1 MiB queue caps, idle SWEEP_INTERVAL and DEFAULT_IDLE_TTL with the disabled-policy on-demand-only branch, default 127.0.0.1 bind, DEFAULT_MAX_ATTACHED_MODELS = 8, lib.rs macOS cfgs; not read: session.rs Engine::Real arm, telemetry bodies, vision.rs body)
- [heavy.rs:HeavyWorkGuard] is described only as a re-export. The heavyweight-work permit shared by audio jobs and chat generation (src/api/audio_stream.rs module doc, src/api/audio.rs:ts_heavy_work_try_acquire and ts_heavy_work_release) is an invariant the doc does not state.

## crates/ffi/src/api/EGM.md

Coverage: sampled (all 157 exported names compared to turbospark.h, no mismatch in either direction; every name the doc lists exists; guard coverage checked on all exports; daemon, hf token and store code paths read; bodies of most listed functions not read)
- [api/daemon.rs:ts_daemon_start] forwards TURBOSPARK_API_KEY and TURBOSPARK_HOME into the spawned daemon's environment (daemon.rs lines ~335-338 and ~600-614), and reads PATH and HOME when resolving it (daemon.rs ~251, ~259). The doc says only "starts or stops the daemon" and lists no env handoff in Side Effects.
- Exports not named in the doc and not covered by its "Other public surface" line: [api/audio.rs:ts_heavy_work_try_acquire] and [api/audio.rs:ts_heavy_work_release] (shared heavy-work permit); [api/models.rs:ts_store_relocate] (store relocation); [api/libraries.rs:ts_store_import_moved_root], [api/libraries.rs:ts_model_forget], [api/libraries.rs:ts_library_set_default], [api/models.rs:ts_model_delete], [api/models.rs:ts_model_resolve_path], [api/models.rs:ts_hub_check_json], [api/models.rs:ts_expert_residency_for_install_json]; [api/server.rs:ts_server_attach_image_model], [api/server.rs:ts_server_attach_embedding_model], [api/server.rs:ts_server_attach_audio_model], [api/server.rs:ts_server_attach_audio_speech_model] and the matching detach, complete and poll functions (the doc names only "image/embedding attach"); [api/session.rs:ts_session_release_vision], [api/session.rs:ts_session_phases_json]; [api/generate.rs:ts_session_fit_window_with_tools_json]; [api/image.rs:ts_image_buffer_free]; [api/embedding.rs:ts_embedding_install].
- Side Effects gap: [api/models.rs:ts_store_relocate] and [api/libraries.rs:ts_store_import_moved_root] change where the managed store lives. The doc lists move and delete side effects for [libraries.rs:ts_model_move] and [libraries.rs:ts_model_delete_install] but not for store relocation. Behavior not read.
- Not drift: [api/models.rs:ts_hf_token_set] stores the token at `<store root>/hf_token` with mode 0600 set at creation (catalog/src/store.rs:Store::set_hf_token), and it honors the process-local root set by ts_store_root_set (catalog/src/store.rs:default_root). "Model store root" is accurate, but it is the store root, not the `models/` subdirectory.

## crates/ffi/src/generate/EGM.md

Coverage: sampled (TS_EVENT_* values 0-4 match generate/mod.rs; role_of rejects unknown roles as the doc says; turn_block holds the session engine lock during streaming; humanize type names exist; humanizer raw-completion path and output_budget body not read)
No drift detected.

## crates/ffi/src/models/EGM.md

Coverage: sampled (every listed function exists; hub limits 20/50/2 match hub.rs constants; install lock order installs-then-move matches install.rs; refuse_while_installing_for covers model moves as the doc says; TURBOSPARK_RESIDENCY_INSTALL and TURBOSPARK_STEERING_VECTOR appear only in #[ignore] or skip-if-unset tests, so not production env vars; catalog and fit bodies not read)
No drift detected.

## crates/gpu/EGM.md

Coverage: sampled (build.rs main, collect, tool, compile, digest, and strict/non-macOS paths read; symbol and flag claims verified; syn expand body not read)
No drift detected.

## crates/gpu/AGENTS.md

Coverage: sampled (claims verified N of M: about 25 of 30 checked; env vars, MAX_GQA_GROUP, MAX_CHUNKS, FC_GDN_DK, 128-thread norm, BatchedScratch fields, delta_pipeline, script and test paths, runtime gates; DecodeScratch match and the RoPE/MLX details not read)
- Fused INT4 kernel names do not exist: doc says `dequant_int4_gemv_fused_gate_up_act` and `dequant_int4_gemv_fused_down_epilogue`; the Metal kernels are `dequant_int4_gate_up_act_gemv_simd` and `dequant_int4_down_epi_gemv_simd` [src/shaders/dequant_int4_fused.metal:dequant_int4_gate_up_act_gemv_simd], [src/dequant_int4_fused.rs:encode_gate_up_act_gemv], [src/dequant_int4_fused.rs:encode_down_epi_gemv]
- Omitted mode: `TURBOSPARK_INT4_MMA8=verify` (drift-bound arm, only the verify path) is read by the runtime alongside `=1` [crates/runtime/src/real_forward_dispatch.rs:int4_mma8_mode]

## crates/gpu/README.md

Coverage: sampled (test-file list checked file by file for existence; shader and test counts counted; vision/ANE/INT4 flag claims, rope/rms/chunks_for names, and gotchas 1 and 3 verified; "~6 KiB per buffer" and the `gpu` alias use in downstream crates not verified against code, though the alias itself is used by crates/runtime/Cargo.toml)
- Test suite count is stale: doc says 97 integration test suites in `tests/`; `crates/gpu/tests/` has 112 top-level `.rs` files (some may be shared helpers, so the true suite count needs a precise definition) [crates/gpu/tests]
- Shader count is stale: doc says 50 `.metal` files under `shaders/`; there are 51 [src/shaders]

## crates/gpu/examples/EGM.md

Coverage: sampled (both example CLIs checked for their subcommands and flags, the MLX version pin 0.32.3, and the generator script path; the FNV/timing bodies not read)
No drift detected.

## crates/gpu/src/EGM.md

Coverage: sampled (about 80 of 90 named symbols confirmed to exist with the stated names; env var reads, cfg gating, lib.rs re-exports, NSProcessInfo, runtime forbid(unsafe_code), warmup gate location, KV max_context asserts, FP16 batch check, and ane-prefill feature gating verified; encode bodies not read)
No drift detected.

## crates/gpu/src/context/EGM.md

Coverage: sampled (all MetalContext, PassEncoder, and CommittedPass methods listed were confirmed; Drop impls, env var reads, and the macOS 15 archive gate verified; cache path resolution read)
- Omitted default: when `TURBOSPARK_METAL_CACHE_DIR` is unset, the disk cache falls back to `$HOME/Library/Caches/TurboSpark/metal` (the doc names the env var but not the default) [src/context/pipeline_cache.rs:PipelineDiskCache]

## crates/gpu/src/moe_gguf/EGM.md

Coverage: sampled (all listed encode functions and constants confirmed to exist; PHASE2_FIXED_SLOTS, the top_k assert, the Q2_0 top_k == 10 requirement, and MXFP4 constants verified; kquants, iq and mxfp4 kernel bodies not read)
No drift detected.

## crates/gpu/src/music3/EGM.md

Coverage: sampled (all listed Music3Sequence operations and Music3Device constructors confirmed; Music3Buffer write/read error returns read; the one-command-buffer-per-sequence claim not checked)
No drift detected.

## crates/gpu/src/shaders/EGM.md

Coverage: sampled (all 15 FC_ATTN_* function constants in attention.metal match the doc; kernel counts for moe.metal (19) and moe_gguf.metal (15) match; named rope, moe, gdn and music3 kernels exist; chunk rounding and M_q = 1 header comments verified; every .metal file is covered by the doc)
No drift detected.

## crates/image/AGENTS.md

Coverage: sampled (claims verified 9 of 9 checked: 4 read-first links resolve, rule anchors checked against lib.rs cfg gates; cargo commands not executed; crate source not fully read)
No drift detected.

## crates/image/EGM.md

Coverage: sampled (claims verified 8 of 9 checked; src is about 9.5k non-test lines, targeted reads only)
- [runtime.rs:ImageRequest::validate] and [install.rs:ImageFamily::supported]: the doc says the request path "only admits the Z-Image one". ImageRequest::validate (runtime.rs:385-411) has no family check. It compares batch, steps, and guidance against the hard-coded Z-Image constants IMAGE_BATCH, IMAGE_STEPS, IMAGE_GUIDANCE (install.rs:23-26). The observable result matches (Qwen-Image-2.1 at 40 steps and Turbo or Krea at 8 steps are rejected), but the gate is a constant comparison, not a family gate. A change to those constants would silently change family behavior. Wording should say so. (b, minor)

## crates/image/README.md

Coverage: sampled (claims verified 20 of 22 checked; test names, ignore attributes, fixture paths, and env vars checked by search; crate source not fully read; cross-target cargo check not executed)
- [tests/transformer_math_parity.rs:test_rope_three_axis_synthetic_parity, test_transformer_synthetic_block_parity, test_transformer_checkpoint_block_parity] Line 49 says `cargo test -p turbospark-image` runs "math parity against checked-in fixtures", and line 70 says the file "Validates diffusion transformer blocks and attention." All three math parity tests in this file are `#[ignore]` (transformer_math_parity.rs:36, 146, 261) and read from `target/ig0/runs/...` or an external shard, which is not in the repo and is absent in this checkout. Only test_patchify_unpatchify_contract runs by default. (b, overstated)
- [tests/conditioning_parity.rs, tests/scheduler_parity.rs] Line 49 "checked-in fixtures" is false. `git ls-files crates/image` lists no fixture data. These tests are not ignored; they print NOTE and return early when external files are missing (conditioning_parity.rs:37, 79, 106; scheduler_parity.rs:66, 110), so the default run passes without checking them. The README should say the default run is a smoke pass and parity runs need the external fixtures. (b, overstated)
- [install.rs / shaders] Line 20 says the macOS app runs these families "through the app's Swift MLX sessions (the vendored ZImage and QwenImage packages)". swift/TurboSparkApp/Package.swift vendors only Vendor/ZImage and Vendor/QwenImage (lines 23-24, 70-71). No Krea 2 Turbo or Qwen-Image-2.1-Turbo Swift path was found in this check, so the Krea part is unverified. (could not confirm)
- Line 46-54 Platform claim that the portable core compiles on Linux x86_64 was not checked by running `cargo check --target x86_64-unknown-linux-gnu`. (could not confirm)

## crates/image/src/EGM.md

Coverage: sampled (claims verified about 30 of 33 checked; src is about 9.5k non-test lines, targeted reads and pub-item skim only)
- [metal_ops.rs:TURBOSPARK_IMAGE_RESIDENCY] The Environment bullet lists only TURBOSPARK_IMAGE_STREAM_PROGRESS and TURBOSPARK_IMAGE_FORCE_SYNC_DISPATCH. TURBOSPARK_IMAGE_RESIDENCY is read at metal_ops.rs:94-99, but only inside `#[cfg(test)]`, so it is test-only. The doc should either list it as test-only or leave it out with a note. (c, minor)
- [lib.rs:pub use] Interface omits public items the crate exports: ImageBudgetError (lib.rs:58-61, the error returned by admit_image_budget that the doc lists only indirectly), the macOS diagnostic trace types ImageFirstStepBoundary, ImageFirstStepTrace, ImageIntraBlockTrace (lib.rs:43-46), ImageStage, ImageMetadata, SchedulerMetadata (lib.rs:58-66), PackedQuantization, PackedTensor, PackedTensorReport, PACKED_INDEX_NAME, PACKED_DATA_NAME (lib.rs:47-50), ImageInstallSpec, ImageInstallReport (lib.rs:29-31). (c, minor; the doc does not claim the list is exhaustive, but ImageBudgetError and the trace types are the ones a caller reaches for)
- [runtime.rs:ImageRequest::validate] Same wording issue as crates/image/EGM.md: the doc at line 25 reads as a family gate. (b, minor)

## crates/image/src/shaders/EGM.md

Coverage: full read of image.metal (842 lines, under the 1500-line threshold); host-side claims sampled (metal_ops.rs:21, 773, 815, 1198-1207, 1234), claims verified 17 of 18 checked
- [image.metal:image_scale_shift, image.metal:image_gate_add, image.metal:image_scheduler_step] Invariant line 22 says the elementwise kernels "take a constant uint & count", and the rest "take a params struct". Only image_add, image_silu_mul, image_silu, image_round_bf16, and image_scheduler_step take the `constant uint &count` form. image_scale_shift and image_gate_add take `constant uint2 &shape` (image.metal:525-533, 547-556), and image_scheduler_step also takes a scalar `constant float &delta` (image.metal:536-544). Neither image_scale_shift nor image_gate_add takes a params struct. (b, minor)

Everything else in this file matches the source: all 20 kernel names exist, the 512 cap applies only to image_attention (image.metal:416), the 256-thread kernels are the ones listed, image_vae_attention holds numerator[16] over 32 lanes (image.metal:655), the host bounds 1..=512, multiple of 8, and area multiple of 64 (metal_ops.rs:1198-1207), and image_attention needs 4 * head_dim threads with head_dim <= 128 (metal_ops.rs:773, 815; image.metal:426).

## crates/invocation/EGM.md

Coverage: sampled (claims verified 25 of 27; parser.rs, lib.rs, diagnostics.rs, failure.rs, usage.rs read in full; options.rs and request.rs checked by targeted grep and reads; tests not read)
No drift detected.

## crates/invocation/README.md

Coverage: sampled (claims verified 25 of 27; parser.rs, lib.rs, diagnostics.rs, failure.rs, usage.rs read in full; options.rs and request.rs checked by targeted grep and reads; tests not read)
No drift detected.
Not confirmed: the claim that turbospark-cli or turbospark-runtime checks model and message-file existence downstream was not traced end to end (cli/src/agent.rs shows is_file and is_dir checks, which is consistent but not proof).

## crates/invocation/AGENTS.md

Coverage: sampled (claims verified 25 of 27; parser.rs, lib.rs, diagnostics.rs, failure.rs, usage.rs read in full; options.rs and request.rs checked by targeted grep and reads; tests not read; cargo commands not executed)
No drift detected.

## crates/invocation/src/EGM.md

Coverage: sampled (claims verified 25 of 27; parser.rs, lib.rs, diagnostics.rs, failure.rs, usage.rs read in full; options.rs and request.rs checked by targeted grep and reads; tests not read)
- Omitted behavior: repeated `--system` values are joined with a newline in supplied order [parser.rs:parse].
- Omitted behavior: `--stop`, `--image`, `--steering` and the steering knob lists are repeatable and order-preserving, and the i-th knob value pairs with the i-th `--steering` path [parser.rs:parse, request.rs:steering_knob].
- Omitted behavior: `--vision-sidecar` takes an opaque path, and the literal `auto` is read as a path, not a keyword [parser.rs:parse].
- Omitted behavior: `--help` and `--version` short-circuit only when reached, so an earlier per-token failure (unknown option, missing value, invalid value) still wins over a later help or version token [parser.rs:parse].
- Omitted invariant: a sub-1.0 `--top-p` with `--top-k 0` is accepted; the earlier refusal was removed [parser.rs:parse]. docs/CLI.md line 316 still says the threshold requires `--top-k > 0`, so that doc diverges from the code (outside this doc file).
- Stale code comment, not a doc file: the InvalidValue doc names a "sampling cross-check violation" that the parser no longer emits [failure.rs:ParseFailure].

## crates/lab/README.md

Coverage: sampled (claims verified about 45 of 52, source not fully read: crates/lab/src is about 30k lines)
- Exit codes section says "1 error, 2 usage". Code does not do that: any returned error, including an unknown command or a bad flag, goes through finish() and exits 1; exit 2 is reached only for an empty argv [src/cli.rs:finish] [src/cli.rs:main] [src/cli.rs:dispatch]. The same wrong line is in the USAGE string [src/cli.rs:USAGE]. src/EGM.md states the code behavior correctly.
- Command table omits `hybrid-replay`, which dispatch accepts [src/cli.rs:dispatch]. The "Offline pricing and replays" bullet does name it, so the table is incomplete rather than wrong.
- Verdict rule omits the sign-test requirement. Below 6 clean pairs every pair must agree in sign; from 6 pairs the one-sided exact sign test must reach 0.05 [src/stats.rs:decide] [src/stats.rs:UNANIMITY_BELOW] [src/stats.rs:SIGN_TEST_ALPHA]. The doc lists only the pair count, median, and bootstrap interval.
- "INCONCLUSIVE means the host was busy" is imprecise. In decide(), INCONCLUSIVE is returned only when clean pairs are fewer than MIN_PAIRS; a sign disagreement or an interval spanning zero yields NOISE [src/stats.rs:decide]. I did not trace orchestrate.rs to confirm how host-busy rounds are counted before they reach decide().

## crates/lab/EGM.md

Coverage: sampled (claims verified about 30 of 33; symbol existence checked by grep across crates/lab/src; invariants checked against integrity.rs, paths.rs, atomic.rs, orchestrate.rs, cli.rs)
- No drift detected.

## crates/lab/AGENTS.md

Coverage: sampled (claims verified about 25 of 28; test file names and bench registry checked against tree and src/kernel/mod.rs)
- Rule "Commands print through say!, never println!" has one exception the doc does not mention: finish() prints the single --json envelope with println! [src/cli.rs:finish]. This is the intended path, so the rule should say so.
- Otherwise no drift detected.

## crates/lab/src/EGM.md

Coverage: sampled (claims verified about 35 of 40; every cited symbol name grepped for existence, invariants and exit codes read in code)
- No drift detected. Exit-code, PASSTHROUGH, ORCHESTRATOR_ENV, FORBIDDEN_PREFIXES, and atomic hard-link claims all match code.

## crates/lab/src/bin/EGM.md

Coverage: full read (both entry points are short and read in full)
- No drift detected. Both binaries exit 2 with a message on non-macOS [src/bin/tslab.rs:main] [src/bin/tslab_worker.rs:main].

## crates/lab/src/dash/EGM.md

Coverage: sampled (claims verified about 12 of 14; route table and serve defaults read in dash/mod.rs; CSP and nosniff header values not read)
- No drift detected in the checked claims. Routes, default port 7878, default bind 127.0.0.1, GET-only registration, no MachineLock use under src/dash, and the non-loopback warning path match.

## crates/lab/src/kernel/EGM.md

Coverage: sampled (claims verified about 14 of 17; bench names, registry, helper functions, arm labels, feature gates, and the music3 golden-file include checked; "runs in the orchestrator process, no worker" not read end to end)
- No drift detected in the checked claims. Registry order and membership, MachineLock in kernel_cmd, and the ane-prefill gating match the doc.

## crates/lab/skill/EGM.md

Coverage: full read (folder holds no Rust source; the only contract claim is the layout, and it matches the tree)
- No drift detected.

## crates/lab/skill/scripts/EGM.md

Coverage: full read (both scripts read in full)
- No drift detected. doctor.sh checks, exit status, the lab dir resolution order, and the preflight "no CPU-load check" statement match the scripts. The 10 GiB threshold matches [src/host.rs:default_min_free_gb], and the name-fragment MODEL_MARKERS list matches [src/host.rs:MODEL_MARKERS].

## crates/lab/tools/EGM.md

Coverage: sampled (claims verified about 8 of 9; function names and the main() usage contract checked in trace_tables.py; the one-tenth GPU sample share is a docstring claim and was not recomputed)
- No drift detected in the checked claims.

## crates/lab/families/README.md

Coverage: sampled (all 26 family JSON files in crates/lab/families checked by script against the documented field rules: schema string, family equal to file stem, kind, doc and upstream findings with verdict "reading", finding id prefix; reader validators read in src/family_profile.rs; tests not run)
- No drift detected. Prefix tags, "!" negation and "|" alternation, id slug characters, date format, ASCII rule, and the ledger/doc/upstream provenance rules all match src/family_profile.rs. Not checked: the "findings named in the same file" rule for recommended[].

## crates/lab/pinned/README.md

Coverage: sampled (all 18 pinned JSON files checked by script: schema, id equal to repo and stem, kernel_class, settings kind, role; revision table compared to source.rev and rev_date; reader validators read in src/pinned.rs; tests not run)
- Field rule says `role` "starts with primary or secondary", and the doc says these rules are "enforced by the reader and its tests". The reader only rejects an empty role [src/pinned.rs:validate]. The data does not follow the rule: 24 model rows across 8 files in pinned/mlx-serve/ (attn_kv.json, gdn_decode.json, keyed_sample.json, prefill_fusions.json, row_attn.json, row_exact.json, spec_round.json, verify_lanes.json) have roles that begin with free text. The primary_model() lookup depends on starts_with("primary") [src/pinned.rs:primary_model].
- Format section says `source.rev` is "the 40-hex commit". The reader accepts 7 to 40 hex characters [src/pinned.rs:validate, rev_ok]. Minor.
- Field rule says `files` "exist at that rev". The reader checks only that each path is repo-relative [src/pinned.rs:validate]. The doc heading claims enforcement that does not exist. Not verifiable without the upstream clones.
- Revision table matches the source.rev and rev_date values in all 18 files.

## crates/lab/workloads/audio/README.md

Coverage: full read (doc is short; AudioJob fields, transcription input check, and speak/consent behavior checked in src/workload.rs and src/worker_audio.rs)
- No drift detected. The transcription 16 kHz mono requirement is enforced [src/worker_audio.rs:pcm_16k_mono]. The "AR stage dominates" timing remark was not measured and is not verified.

## crates/model-io/EGM.md

Coverage: sampled (claims verified 9 of 9 at this level: crate name, Cargo deps and `foundation` alias, doc links, test command; source not read beyond the crate root)
No drift detected.

## crates/model-io/AGENTS.md

Coverage: sampled (claims verified about 14 of 15: read-first links resolve, index and manifest constants exist and are imported by repack [crates/repack/src/resident_writer.rs, crates/repack/src/gturbo_writer/manifest.rs], unsafe sites located; `cargo check --target x86_64-unknown-linux-gnu` not run)
- Unsafe rule omits the `libc::sysconf(_SC_PAGESIZE)` call in `resident_buffer.rs` [src/resident_buffer.rs:page_size_bytes]. The source-level EGM (src/EGM.md) records it, so the two docs disagree.

## crates/model-io/README.md

Coverage: sampled (claims verified about 45 of 55: module file list, public type names, baseline constructors, error variants, constants, test file list, unsafe sites; source of most modules not fully read)
- Flint router described as "INT8 sigmoid router". The Flint baseline header says the router is BF16 ("p = sigmoid(h @ router^T) # BF16 router, no bias") and the checkpoint is MLX affine INT4 g64 [src/arch_baselines/flint.rs:flint_40b_a2_5b] (INT8 is not mentioned in that file). Verify which tensor the INT8 claim refers to before keeping it.
- `arch_baselines/llama.rs` described as covering "Llama 3, Mixtral, and related Dense/MoE variants". The file exports only `mixtral_8x7b`; the header says a dense Llama 3.1 differs only in shape fields read from GGUF by `arch_from_gguf` (`gguf_config.rs`, outside this crate) [src/arch_baselines/llama.rs:mixtral_8x7b]. The README overstates what this file provides.
- `arch_baselines/gpt_oss.rs` described as the GPT-OSS-20B "(Harmony chat template) architecture". No Harmony reference exists in the crate [src/arch_baselines/gpt_oss.rs:gpt_oss_20b]. The template belongs to another crate, or the claim is stale.
- `arch_baselines/minimax.rs` described as "split-GGUF and sigmoid routing". The file sets `router_scoring_func: "sigmoid"`; no split-GGUF handling appears in the file or the crate [src/arch_baselines/minimax.rs:minimax_m2].
- Tests list omits `tests/metadata_read_bounds.rs` and `tests/metadata_size_caps.rs` [tests/metadata_size_caps.rs].
- Key Modules list omits `cgroup.rs` (`probe_cgroup_memory`, `parse_cgroup_limit`, a public re-export) and `bounded_read.rs` (size-capped metadata reads used by manifest and install receipt loading) [src/lib.rs:pub use].
- Safety section omits the `sysconf(_SC_PAGESIZE)` call in `resident_buffer.rs` [src/resident_buffer.rs:page_size_bytes].

## crates/model-io/src/EGM.md

Coverage: sampled (claims verified about 40 of 45: re-export list against lib.rs, `ModelError` variants, cgroup paths, receipt tamper note, sysconf and statvfs sites, index constants, trunk marker and speech receipt write sites; not every listed function body read)
- Interface omits several public items re-exported from the crate root: `build_layer_multipliers`, `derive_head_vocab_and_offsets`, `find_nth_prime_after` [src/ngram_hash.rs], `ExpertSlotWiring`, `ResolvedExpertResidency`, `largest_context_within*`, `committed_breakdown*`, `CommittedBytes` [src/expert_cache_policy.rs, src/context_policy.rs], `EncoderQuantization` [src/encoder_config.rs], and the `TRUNK_ROLE_FILE`, `SPEECH_RECEIPT_FILENAME`, `SIDECAR_RECORD_FILE`, `DASH_SIDECAR` file-name constants [src/lib.rs:pub use]. Lower priority than the omissions above; the Interface block is a summary, so this is a completeness note.
- Unsafe invariant lists `mmap`, `statvfs` and `sysconf`; the README and AGENTS.md do not list `sysconf`. See the two items above.

## crates/model-io/src/arch_baselines/EGM.md

Coverage: sampled (claims verified about 25 of 25 at the constructor and registry level; `arch_validation` field-by-field check not traced)
No drift detected.

## crates/model-io/src/arch_config/EGM.md

Coverage: sampled (claims verified about 15 of 16: `ModelFamily` persisted strings, `ArchConfig` predicates and `has_*_layers` methods, `MAX_VISION_DEEPSTACK_MERGERS` = 3, re-export list; round-trip of every variant not exhaustively checked)
No drift detected.

## crates/model-io/src/manifest/EGM.md

Coverage: sampled (claims verified about 16 of 17: `load`/`validate`/`peek_family` signatures, constants, `known_flags`, `turboQuantKV` refusal, `image-generation` refusal, stride check, vision and Hadamard file requirements, `Gemma4` fallback, 4 MiB default cap, `validate_quant` visibility; `validate_quant` slot rules not read)
- "quant is required when the arch matches a shipped baseline" overstates the check. `is_production_arch` matches only `num_layers` and `hidden_size` against any baseline in `all_known_architectures()`, not the full arch [src/manifest/mod.rs:is_production_arch]. A toy manifest with the same two values would be treated as production.

## crates/repack/README.md

Coverage: sampled (claims verified 17 of 21, source not fully read)
- [crates/repack/tests:count] Says "This crate contains 83 integration test files in `tests/`". There are 91 top-level .rs files in crates/repack/tests/ (the count is stale).
- [crates/repack/src/lfm3_config.rs:parse_lfm3_config] and [crates/repack/src/deepseek_vl2_config.rs:parse_deepseek_vl2_config] are public config parsers that the Key Modules list omits. synthetic_lfm3.rs (LFM3 tiny-install fixture) is also not listed under the synthetic generators.

## crates/repack/AGENTS.md

Coverage: sampled (claims verified 9 of 11, source not fully read)
- Not verified in production code: "the `softmax_offset` sinks are resident" (Flint). The only `softmax_offset` reference in crates/repack/src is the test fixture [crates/repack/src/synthetic_flint.rs:tiny_flint_arch]. Residency is implied by the default resident bucket in classify, but no production symbol names it.
- All five doc links resolve (.claude/docs/modules/repack.md, docs/GTURBO.md, docs/MODELS.md, docs/FLINT.md, .claude/docs/diagnostics.md). The mxfp4 nibble-order, 0xFF refusal, flint 2^6 rebalance, marker file, and position-numbered expert blobs claims match code.

## crates/repack/EGM.md

Coverage: full read
No drift detected. Both subdirectory links (./src/EGM.md, ./examples/EGM.md) resolve.

## crates/repack/examples/EGM.md

Coverage: sampled (claims verified 15 of 17, source not fully read; examples total about 390 lines)
No drift detected.
(Checked: four example .rs files; `--undo` and exit code 2 in [crates/repack/examples/flint_rebalance.rs:main]; shard prefix trailing-slash check in [crates/repack/examples/fish_drama3_intake.rs:main]; four sidecars in [crates/repack/examples/flint_local_intake.rs:SIDECARS]; existing-destination refusal in both intake examples; preprocessor_config.json copy and the no-vision-tower error in [crates/repack/examples/minicpm_tower_local.rs:run].)

## crates/repack/src/EGM.md

Coverage: sampled (claims verified about 25 of 40, source not fully read; src/ top level is about 10k lines)
- [crates/repack/src/hf_checkpoint.rs:decode_tensor_f32] The Invariant "lm_head only when embeddings are untied" is cited to decode_tensor_f32, but that function only decodes F32/BF16 and never touches lm_head. The untied condition (`!dims.tie_word_embeddings`) is in the orchestration code around hf_checkpoint.rs lines 135 and 249. Fix the citation.
- Not checked, so not reported as drift: the `describe_headless_tts_trunk` routing, and the statement that the fixture group size is 64 via [crates/repack/src/synthetic_tensors.rs:int4_triple] (the file does not name GROUP_SIZE directly; the quantizer's GROUP_SIZE is 64 in crates/compute/src/quant.rs).

## crates/repack/src/gemma4_checkpoint/EGM.md

Coverage: sampled (claims verified about 18 of 30, source about 12k lines not fully read)
No drift detected.
(Checked: all interface symbols exist in the crate; FP16 overflow refused, not clamped, in [crates/repack/src/gemma4_checkpoint/narrow.rs:convert_raw_to_fp16]; GTURBO_PAGE_BYTES = 16384 in [crates/repack/src/gemma4_checkpoint/shards.rs:GTURBO_PAGE_BYTES]; the expert stride rounded to the page size in [crates/repack/src/gemma4_checkpoint/expert_blobs.rs:expert_stride_from_headers]; affine/mxfp4 group rules in [crates/repack/src/gemma4_checkpoint/config.rs:is_supported_affine_shape] and [config.rs:MXFP4_GROUP_SIZE]; routed expert 3/4-bit rule in [expert_blobs.rs:routed_expert_bits]; 0xFF mxfp4 refusal in [mlx_mxfp4.rs]; .signs refusal and hadamard.bin written before finish_streaming in [mod.rs:write_gemma4_install_streamed_in_place]; vision_gemma4 27 depths and 13 roles in [vision_gemma4.rs:SUPPORTED_DEPTHS, GEMMA4_BLOCK_ROLES]; the 13-variant bucket enum in [classify.rs:Gemma4Bucket]; the staging-dir and rename/rollback publish in [mod.rs:sibling_work_dir, publish_staged_install].)

## crates/repack/src/gguf_checkpoint/EGM.md

Coverage: sampled (claims verified about 14 of 16, source about 3.4k lines not fully read)
- [crates/repack/src/gguf_checkpoint/mod.rs:write_gguf_install_streamed_resumable] Invariant (line 24) says a leftover `layer_NN.bin` is adopted only when provenance matches AND its size equals the header-derived size, "otherwise leftovers are deleted", citing [resume.rs:prepare] and [plan.rs:layer_file_bytes]. Actual behavior: resume::prepare deletes leftovers only when provenance does not match. The size check is in write_gguf_install_streamed_resumable (on_disk == plan::layer_file_bytes). When provenance matches but the size differs, the layer is rewritten by writer.write_layer, not deleted. Fix the citation and the "deleted" wording for that path.
- Checked and matching: no env reads in production code (env::temp_dir appears only in #[cfg(test)] helpers in resume.rs and ngram.rs); the DeepseekV4Flash refusal; ResumeProvenance::new 40-hex rule; the ngram header remove, temp, and sync_all sequence; FUSED_GATE_FIRST; the minimax_gguf_sizing, verify_plan, zero_tail_pattern, qwen4exp_tensor_is_transcoded, and gguf_manifest_quant symbols; the MiniMax F32 router rule in [plan.rs:classify]; the IQ4_NL constants in [ngram.rs].

## crates/repack/src/gguf_config/EGM.md

Coverage: sampled (claims verified about 14 of 20, source about 1.2k lines not fully read)
No drift detected.
(Checked: 4096 layer cap via MAX_MODEL_LAYERS in [gguf_config/mod.rs]; the Unported-family list returning UnsupportedArchitecture in [mod.rs:arch_from_gguf]; tie_word_embeddings only when output.weight is absent; rope.freq_base to full_rope_theta and freq_base_swa to rope_theta; MiniMax-M2, Qwen2 (1e-6) and LFM3 (1e-5) epsilon pins; DeepSeek2 q_lora_rank > 0 refusal; the Spark25 and LFM3 dimension consistency checks, and the Qwen4Exp partial_rotary derivation; the lfm3 KV-head rule (0 is kind 6 only on a full layer, otherwise BadValue) in [masks.rs:lfm3_layer_mask]; Meta accessor methods; qwen_gdn_moe, gemma4, gpt_oss masks and kv_heads / linear_attention in attention.rs and masks.rs.)

## crates/repack/src/gguf_header/EGM.md

Coverage: sampled (claims verified 12 of 13, source about 740 lines not fully read)
No drift detected.
(Checked: DEFAULT_ALIGNMENT=32, DEFAULT_MAX_HEADER_BYTES=64 MiB, MAX_ARRAY_DEPTH=64, SUPPORTED_VERSION=3 in [gguf_header/types.rs]; all GgufHeaderError variants named in the doc exist in [types.rs:GgufHeaderError]; the 32 B and 13 B minimum sizes in [parser.rs:parse_header]; Cursor::take and Cursor::value; helper methods on GgufHeader and GgufTensorInfo; ggml_type_name and ggml_type_block in [ggml.rs].)

## crates/repack/src/gguf_names/EGM.md

Coverage: sampled (claims verified 12 of 14, source about 930 lines not fully read)
No drift detected.
(Checked: GgufMapping variants (Resident, Routed, RoutedFusedGateUp, Ignored, ZeroTailRopeFreqs) and GgufNameError variants in [gguf_names/mod.rs]; rope_freqs.weight refused for all families except Lfm3 (lfm3 top-level checked first) in [mod.rs:map_gguf_name]; PreMergeExperts refusal for Llama in [llama.rs:is_pre_merge_expert]; the five unported families returning None in [mod.rs]; qwen35moe and qwen35 strings in [mod.rs:gguf_architecture]; the layer_prefix string; A_log and dt_bias with no .weight suffix in [qwen.rs]; all per-family map_* functions named in the doc exist in their files.)

## crates/repack/src/gturbo_writer/EGM.md

Coverage: sampled (claims verified about 22 of 24; layers.rs and streaming.rs read in full, manifest.rs header and hash path read, its JSON body checked by grep only, types.rs checked by grep)
- No drift detected.

## crates/repack/src/ranged_download/EGM.md

Coverage: sampled (claims verified about 30 of 33; key regions of http.rs, chunks.rs, and mod.rs read, not every line)
- Omission: `DownloadError::Request` is also returned for local filesystem failures and for the "refusing to overwrite" check, not only for transport errors [http.rs:PendingDownload::new, http.rs:PendingDownload::publish]. Callers that match `Request` as a network error will see local I/O errors too. The doc's variant list does not say so.

## crates/repack/src/synthetic_gguf/EGM.md

Coverage: sampled (claims verified about 18 of 20; gemma4.rs, gptoss.rs, minimax.rs, qwen2.rs and gemma4_shape.rs checked by targeted grep and reads, not in full)
- Invariant cites `[gemma4_shape.rs:SyntheticGgufShape]` for the rule that dimensions tile each quant type's block size. SyntheticGgufShape has no such check. Only the doc comments mention tiling there. The enforcement is the panicking asserts in `[builder/tensors.rs:q8_0_tensor, q4_k_tensor, q6_k_tensor, q3_k_tensor, iq_tensor, mxfp4_tensor]`.

## crates/repack/src/synthetic_gguf/builder/EGM.md

Coverage: sampled (claims verified about 14 of 15; mod.rs build path and tensors.rs assertions read directly)
- Invariant says "Row length `dims[0]` must tile the block size ... [tensors.rs:q8_0_tensor, ...]". `q8_0_tensor` does not check `dims[0]`. It asserts that the product of all dims is a multiple of 32 [tensors.rs:q8_0_tensor]. The other helpers (q4_k, q6_k, q3_k, iq, mxfp4) do check `dims[0]`.

## crates/repack/src/synthetic_model/EGM.md

Coverage: sampled (claims verified about 14 of 16; arch.rs grep, moe.rs sub-tensor and stride code, dense.rs SWA mask read directly)
- Interface says `build_synthetic_gemma4_install`, `build_synthetic_gemma4_swa_install`, `build_synthetic_gemma4_moe_install` and `build_synthetic_gemma4_moe_streamed_install` "return the ArchConfig". Each actually returns `Result<ArchConfig, WriterError>` [synthetic_model/dense.rs:build_synthetic_gemma4_install, synthetic_model/moe.rs:build_synthetic_gemma4_moe_install]. The invariant "the returned ArchConfig" is right only inside the Ok arm.

## crates/repack/src/synthetic_qwen/EGM.md

Coverage: sampled (claims verified about 20 of 24; public items and builder signatures checked by grep, dense_tensors.rs, drama3.rs, vision.rs and qwen4.rs read in targeted ranges)
- Interface says the dense GDN builders (and `build_synthetic_qwen_gdn_dense_install` in particular) "return ArchConfig". The signature is `Result<ArchConfig, Box<dyn std::error::Error>>` [synthetic_qwen/dense.rs:build_synthetic_qwen_gdn_dense_install]. The invariant section states the Result return correctly, so only the interface line is off.

## crates/repack/src/synthetic_qwen/qwen4_decode/EGM.md

Coverage: sampled (claims verified about 18 of 20; arch.rs, tensors.rs, ngram.rs and mod.rs checked by targeted grep and reads, the indexer-budget validation path not read)
- No drift detected.

## crates/runtime/src/families/EGM.md

Coverage: full read of EGM.md and families/mod.rs (14 lines); all 12 child module links resolve; the 12 declared modules match the doc's list. Claims about the child flows were not re-checked here (see the child sections).
No drift detected.

## crates/runtime/src/families/deepseek2/EGM.md

Coverage: sampled (claims verified about 25 of about 30, source not fully read; directory is about 1550 lines across 5 files, state.rs and mod.rs read by targeted search and reads).
No drift detected. Checked: produce_real_deepseek2 / _inner, encode_steering use, layer_tensor, RealDeepseek2State and build_mha, encode_mla_attention_block, encode_mha_attention_block, Q_PROJ and the other suffix consts, softmax_topk_no_renorm, encode_shared_expert, encode_deepseek2_layer_moe, validate_q8_0_matrix, yarn_mscale not applied, rope_mscale, PHASE2_FIXED_SLOTS, PreadExpertStreamer, the ffnSandwichNorms/attentionKEqV refusal list, the kv_lora == 512 and mask 5 checks, the dense-lead and build_mha refusals, the MissingTensor failures, vocab/logits-length checks, and the flat vision injection write.

## crates/runtime/src/families/flint/EGM.md

Coverage: sampled (claims verified about 20 of about 25, source not fully read; directory is about 1870 lines across 6 files).
- Overview calls the chunked prefill path "layer-batched" and says it shares per-token encoders. prefill.rs refuses TURBOSPARK_BATCHED_GEMV and TURBOSPARK_ROUTED_BATCH for flint because "its resident GEMVs stay per token" and "its routed half stays per token" [prefill.rs:prefill_chunk_real_flint], and runs attention through [prefill.rs:encode_flint_chunk_token]. The wording overstates the batching; it reads as a per-token path driven layer by layer per micro-batch.
- Not checked: the 49-layer count and layers 0 and 1 for the published checkpoint (manifest data, not source), and routed_pipeline_banks sizing (2 * top_k threshold) [moe_prefill_pipeline.rs:routed_pipeline_banks].
- Verified with no drift: RealFlintState::build refusals (ModelFamily::Flint, sandwich, untied lm_head, non-sigmoid router, mask > 1, sliding_window > 0, full_rope_theta == 0, lead range), router dtype 5 and size check, softmax_offset per head, sigmoid_topk_renorm (stable rank, sigmoid over sum times scale), encode_swiglu widths 6144 and 1024, DEBUG_REPORT_ABOVE = 2000, env vars TURBOSPARK_FLINT_DEBUG, TURBOSPARK_FLINT_ROUTER_DUMP, TURBOSPARK_FLINT_CHUNKED_PREFILL (OnceLock, default on), kv_start for sliding window.

## crates/runtime/src/families/gemma4/EGM.md

Coverage: sampled (claims verified about 20 of about 25, source not fully read; directory is about 3950 lines across 10 files).
No drift detected. Checked: RealGemmaState and build, ensure_batched, produce_real_gemma4 / _inner, prefill_chunk_real_gemma4, prefill_micro_batch_gemma4 (prefill.rs:300), prefill_micro_batch_gemma4_dense, encode_gemma4_routed_half_per_token, the attn/attn_batch/moe/moe_batch/shared_expert entry points, router_offsets_gemma4, encode_gemma4_layer_tail(_dense), RoutedSlot re-export, gemma4_span_refusal (BATCHED_GEMV, ROUTED_BATCH, quantized KV, ring < window + len - 1), span_attention_range, encode_gemma4_pilot_probe (TURBOSPARK_PILOT_PROBE), the attn_batch quantized-KV refusal [attn_batch.rs:encode_gemma4_layer_attn_and_router_batched], the batched routed refusals for non-affine INT4 blobs, MAX_PREFILL_EXPERT_BINDINGS and the pilot probe [moe_batch.rs:encode_gemma4_layer_routed_moe_batched], dense = num_experts == 0, and the flag-based acceptance (ffn_sandwich_norms and router_scaled).
Not checked: the sqrt(hidden) omission for image rows in vision_prefill.rs.

## crates/runtime/src/families/gptoss/EGM.md

Coverage: sampled (claims verified about 12 of about 20, source not fully read; directory is about 2080 lines across 7 files).
- Env seam list in Side Effects omits TURBOSPARK_PILOT_PROBE. The batched routed path also refuses the router pilot probe by name [moe_batch.rs:encode_gpt_oss_layer_routed_moe_batched], and the doc's Invariants mention the pilot probe but the Side Effects env-seam bullet does not. Minor omission.
- Verified with no drift: layer_tensor, RealGptOssState::build and its refusals (family, Gemma flags, softcap, untied head, mask > 1, window <= 0, yarn factor <= 0), ensure_batched, BatchedRoutedScratch, produce_real_gpt_oss, encode_attention_block, encode_gpt_oss_layer_attn_and_router, encode_gpt_oss_layer_moe (router bias added to logits before top-k, router_topk_gemma4 with per-expert ones, INT8 dtype 5 router size check), encode_gpt_oss_layer_routed_moe_pipelined and _batched, prefill_chunk_real_gpt_oss with the TURBOSPARK_BATCHED_GEMV refusal, kv.advance, and the TURBOSPARK_ROUTED_BATCH / TURBOSPARK_EXPERT_RESIDENCY refusal text.
Not checked: the two command buffers per decode layer, and the BF16 sink count check.

## crates/runtime/src/families/lfm3/EGM.md

Coverage: sampled (claims verified about 15 of about 22, source not fully read; directory is about 1730 lines across 6 files).
No drift detected. Checked: RealLfm3State build refusals (ModelFamily::Lfm3, rope_neox_subdim, lm_head.weight refusal, conv.conv.weight BF16 [hidden, taps] check), reset, Lfm3Batched, RMS_EPS, layer_tensor, produce_real_lfm3, set_lfm3_fused_ffn, short_conv_tail_abs_max, encode_conv_block, rope_params_for with full_rotated_pairs, encode_mlp_block and the int4_mma8_takes gate, lfm3_fused_ffn (TURBOSPARK_LFM3_FUSED_FFN, OnceLock), prefill_chunk_real_lfm3 and the TURBOSPARK_FFN_HIST refusal, lfm3_chunked_prefill (TURBOSPARK_LFM3_CHUNKED_PREFILL, OnceLock), LFM3_PREFILL_ROWS, lfm3_prefill_width, encode_attention_rows, commit_kv_rows, and the TURBOSPARK_PREFILL_STEEL / TURBOSPARK_PREFILL_ROWS_ATTN references.
Not checked: the layer mask 0/1/6 refusal text and the sandwich-norm flags refusal list in full.

## crates/runtime/src/families/llama/EGM.md

Coverage: sampled (claims verified about 14 of about 22, source not fully read; directory is about 3380 lines across 10 files).
No drift detected. Checked: RealLlamaState and build (Gemma-extension refusals, softcap, non-full-attention refusal, expert count agreement, MAX_STREAMED_EXPERTS, dense intermediate_size, MiniMax FP32 dtype 3 bias read, untied head for MiniMax), batch_attention_eligible and batch_attention_route_eligible (family, num_experts == 0, all-full mask, unquantized KV, live_rows > 1), capacity == MAX_PREFILL_BATCH check [state.rs:build], encode_llama_layer_dense, encode_llama_layer_moe (sigmoid_topk with bias selection), encode_llama_layer_routed_moe_pipelined, router::encode / sigmoid_topk / fp32_tensor, the mRoPE arms, prefill_chunk_real_llama_dense and _moe with the TURBOSPARK_ROUTED_BATCH and TURBOSPARK_BATCHED_GEMV refusals, the TURBOSPARK_STANDIN lab_hooks use, and the routed_batch_prefill / batched_gemv_prefill / routed_pipeline runner flags.
Not checked: the Qwen2 / Qwen3-VL / MiniMax-M2 family routing list in full, and the image-pad host write detail beyond the scratch.x write.

## crates/runtime/src/families/museglimmer/EGM.md

Coverage: sampled (about 22 of 26 claims verified across 6 files of about 1,260 lines; source not fully read)
- The doc calls the four scalars "published", but the constants are documented in code as RECALLED values where config.json is silent, guarded only by crates/repack/tests/museglimmer_config.rs, and the code says to change them there in lockstep. The doc omits this guard and the caveat [state.rs:RealMuseState, state.rs:RMS_EPS]

## crates/runtime/src/families/qwen/EGM.md

Coverage: sampled (about 45 of 60 claims verified across 25 files of about 13,350 lines; source not fully read; routed-expert slot ordering in the mapped arm and the DFlash tie rules were not traced)
- Invariant "Drafters allocate nothing unless asked at open" is false for the default environment. Unset TURBOSPARK_MTP_DRAFT maps to Auto, which builds the MTP head at AUTO_DEPTH 2 whenever the install carries one, and unset TURBOSPARK_DFLASH_DRAFT maps to Auto, which builds the DFlash drafter on dense installs carrying dflash.fc.weight [mtp_state.rs:MtpDraftPolicy::from_env, mtp_state.rs:MtpState::build (Auto arm), dflash.rs:DflashState::build (Auto arm)]
- Env reads omitted from the Side Effects list: TURBOSPARK_PREFILL_STEEL, TURBOSPARK_MTP_BATCHED_PRIME and TURBOSPARK_PREFILL_ROWS_ATTN are read on the shared open path that builds this runner [real_forward_open.rs:369-371]; TURBOSPARK_STANDIN skips FFN or attention blocks in the decode loop [lab_hooks.rs:68, produce.rs:produce_real_qwen_inner]; TURBOSPARK_ROUTER_LOGITS_LAYER is read for router logit capture [router_hist.rs:96, moe.rs:encode_qwen_layer_moe]; TURBOSPARK_ATTN_DECODE_KERNEL selects the decode attention kernel [gpu/src/attention_decode.rs:736, batched_layers.rs:encode_full_attention_block_batched]

## crates/runtime/src/families/qwen/dflash_draft/EGM.md

Coverage: sampled (about 14 of 18 claims verified across 6 files of about 1,380 lines; source not fully read; tie-break and tree-chain equivalence not traced)
- "Requires TURBOSPARK_DFLASH_DRAFT set before open for real_dflash to exist" is wrong. Unset means Auto, which builds real_dflash on any dense install that carries dflash.fc.weight. The variable is needed only to force a block size or to turn the drafter off with 0 [dflash_state.rs:DflashDraftPolicy::from_env, dflash.rs:DflashState::build (Auto arm)]. The error text in dflash_prime_from_capture repeats the same instruction [mod.rs:dflash_prime_from_capture]
- The env read the doc cites at mod.rs:dflash_prime_from_capture is actually in dflash_state.rs:DflashDraftPolicy::from_env (citation does not resolve to the read)
- Omitted: the Off/Auto/Fixed policy and the serving block size chosen by TURBOSPARK_INT4_MMA8 (DFLASH_SERVING_BLOCK vs DFLASH_SERVING_BLOCK_TILE) [dflash_state.rs:dflash_serving_block]

## crates/runtime/src/families/qwen4/EGM.md

Coverage: sampled (about 30 of 36 claims verified across 13 files of about 5,830 lines; source not fully read)
- Invariant "build refuses ... PLE on a QSA layer" is not enforced. The only PLE layer-kind check is `if arch.layer_is_linear(ple_layer) { ... }` at state.rs:RealQwen4State::build (around lines 1145-1156), and its body is comments only, so a PLE layer on a QSA layer passes build. The comment says the check exists to make that case a loud refusal, so either the refusal is missing or the doc overstates it [state.rs:RealQwen4State::build]
- TURBOSPARK_QWEN4_TAP_LAYERS is cited to GdnNormCapture, but the read is in TapCapture::from_env [state.rs:TapCapture::from_env]. The doc also omits the default layer list "5,15,25,35,45" and the QTAP file format described in that code
- Omitted: the runtime setter for TURBOSPARK_QSA_FORCE_DENSE (the doc comment at state.rs:785-789 names RealForwardRunner::set_qsa_force_dense) and its diagnostic purpose (KL against the dense path)

## crates/runtime/src/families/spark/EGM.md

Coverage: sampled (about 18 of 22 claims verified across 5 files of about 1,100 lines; source not fully read)
No drift detected.

## crates/runtime/src/families/synthetic/EGM.md

Coverage: sampled (about 12 of 13 claims verified; produce_inner read in full, layer.rs and other claims spot-checked by grep; source not fully read)
- Verification list is non-exhaustive: crates/runtime/tests/real_forward_qwen35.rs also contains the_synthetic_flows_arithmetic_is_frozen, which the doc omits (minor)

## crates/runtime/src/families/unlimited_ocr2/EGM.md

Coverage: sampled (about 14 of 15 claims verified across 3 files of about 1,100 lines; source not fully read; router tensor size checks and pread/mapped offsets not traced)
No drift detected.

## crates/runtime/EGM.md

Coverage: sampled (subdirectory links and their targets verified; top-level structure only, no code claims)
- No drift detected.

## crates/runtime/AGENTS.md

Coverage: sampled (claims verified 14 of about 40; module and doc links all resolve; env defaults and numeric thresholds checked by grep and Read; Qwen MoE, MiniCPM, lazy tape, Flint rebalance factor and expert count not verified)
- Says `TURBOSPARK_QWEN_FUSED_FFN=1` activates the fused INT4 FFN dispatches (opt-in). The code default is ON: the flag is read as `!= "0"`, so only `0` disables it, and the doc comment there calls `=0` the kill switch [families/qwen/mod.rs:fused_ffn].
- Says MegaTTS3 sessions open on Metal by default and that the CPU reference path runs only as an opt-in fallback when Metal cannot open. `MegaTts3Runner::open` takes an explicit `Device` and propagates Metal init errors (`map_err(metal_error)?`) with no automatic CPU fallback [src/megatts3.rs:MegaTts3Runner::open]. Metal is the default only in `examples/megatts3_bench.rs` (`--device` default). A fallback may exist in a caller (ffi/Swift); not confirmed here.

## crates/runtime/README.md

Coverage: sampled (all module, test-suite and family paths resolve; `ScriptedLogitProducer`, `forbid(unsafe_code)`, package/alias name checked; `PhaseCounters`, `DecodeScratch`, `GdnStateManager` reset behavior and the llama family member list not verified)
- Says the crate has "81 integration test suites in `tests/`". There are 105 top-level `.rs` files in `tests/` and all 105 contain `#[test]` or `fn main` [tests/]. The count is stale.
- The families list omits `lfm3/`, which exists under `src/families/` and is not mentioned anywhere in the README [src/families/mod.rs].
- Says `whisper/` provides the "audio mel frontend windowing". The mel computation comes from the audio crate (`audio::whisper::whisper_log_mel_window`, imported in [src/whisper/mod.rs]); `whisper/` only windows and calls it. Minor wording overstatement.

## crates/runtime/examples/EGM.md

Coverage: sampled (claims verified 19 of about 30: binary count, `#![cfg(target_os = "macos")]` gates on the four fish_drama3 probes and whisper/parakeet, the `_say`/`_runner`/`_step_cost` ungated state, CLI flags for frozen_e_metal, music3_bench, megatts3_bench, moonshine_backend_probe, OCR single/multi prompt, `TURBOSPARK_DRAMA3_SERIAL`, `TURBOSPARK_MUSIC3_AR_ONLY`, `OCR_MAX_NEW_TOKENS`, `!Send` note on the Music 3 runner; per-binary behavior not read; 4.4k lines)
- No drift detected.

## crates/runtime/scripts/EGM.md

Coverage: sampled (quiet_host rule, `digests_agree` reported not enforced, arm rotation, binary path and `pmset` use checked against the two drivers; 365 lines, not fully read)
- No drift detected.

## crates/runtime/tools/EGM.md

Coverage: sampled (OUT_DIR fallback chain and `--help` absence checked in `fetch_sound_recall_clips.py`; SHA-256 pin behavior, ffmpeg spawn and `gen_ocr_pages` macOS requirement not verified; 406 lines)
- No drift detected.

## crates/runtime/src/EGM.md

Coverage: sampled (about 30 of about 70 cited symbols resolved to their file; `#![forbid(unsafe_code)]` and macOS gating of the listed modules checked against `lib.rs`; `RuntimeError` and `RealForwardError` variants; `open_with_options` pins for speculation, steering and session slots; `open_with_kv_quant` defaulting residency to Auto; `TURBOSPARK_METAL_KERNEL_WARMUP` only `"1"`; `TURBOSPARK_ROUTER_HIST` writes on Drop; env reads; 30k lines, not fully read)
- No drift detected.

## crates/runtime/src/coreml/EGM.md

Coverage: sampled (all five `*_UNITS`/quantized-GPU env names, `<install>/.compiled` cache root, 1 MiB JSON cap, macOS 15 and MultiFunction gates, Unlimited OCR GPU pin and Neural Engine rejection, OCR config checks and Core ML file contract verified; 5.9k lines)
- No drift detected.

## crates/runtime/src/encoder/EGM.md

Coverage: sampled (`SUPPORTED_MODEL_TYPES`, missing `model_type` refusal, `1_Pooling` pass-through, tokenizer load swallowed to `None` via `.ok()`, 8-bit group size check against `GROUP_SIZE`, `use_tanh_gelu: false`; per-layer bias defaults and encode-time errors not verified; 510 lines, not fully read)
- No drift detected.

## crates/runtime/src/native_audio/EGM.md

Coverage: sampled (`busy` and `cancelled` handling, terminal cancel, `"cancelled"` ABI marker, Unified admission `3x + 256 MiB`, Multitalker admission `weights + weights/4 + 512 MiB`, named threads `native-audio` and `multitalker-stream`, Unified finish reset to a new stream, Multitalker Metal-free header; 540 lines)
- Says Multitalker `finish` "closes the session". The worker loop in `multitalker_stream.rs` has no closed flag and does not exit after `Op::Finish`; the only "closed" statement is a doc comment on `finish`. Whether `LiveSession::finish` refuses later appends was not checked (lives outside this directory). Unconfirmed divergence [src/native_audio/multitalker_stream.rs:finish].

## crates/runtime/src/router_hist/EGM.md

Coverage: full read (`batches.rs`, 178 lines, including its two unit tests); the env gating in parent `router_hist.rs` checked by grep only
- Says "Empty route slices and records outside a begin/end scope are dropped". In `record`, an empty route slice also sets `truncated = true` before returning, so an empty record marks the whole capture truncated. Out-of-scope records return without touching `truncated`. The doc omits the empty-slice `truncated` side effect [src/router_hist/batches.rs:BatchCapture::record].

## crates/runtime/src/vision/EGM.md

Coverage: sampled (`VISION_SLOTS = 2`, FP16 narrowing bound `65504` in `gemma4.rs:to_f16` and `lfm3_vl.rs:to_f16`, Qwen2.5 full-attention every 8th block, LFM3-VL `t != 1` refusal and 65,536-patch cap, Uocr2 2 GiB bin cap, SigLIP final LayerNorm eps 1e-5, `TURBOSPARK_VISION_RESIDENCY` read, all sibling tower files present; 14.7k lines)
- No drift detected.

## crates/runtime/src/whisper/EGM.md

Coverage: sampled (`MAX_OPEN_BYTES` 24 GiB hard cap, config 1 MiB and tokenizer 16 MiB limits, safetensors header 1 MiB, 480000-sample windows and 29 s stride, 440-token cap, 8-repeat stop, `TURBOSPARK_WHISPER_DEVICE=cpu` and `TURBOSPARK_WHISPER_PROFILE=1`, static open lock, English-only language acceptance, silent Metal-to-CPU fallback verified; 4.3k lines)
- No drift detected.

## crates/selection/AGENTS.md

Coverage: full read of src/ and docs; tests/ checked by file list and grep only.
No drift detected.

## crates/selection/EGM.md

Coverage: full read of src/ and docs; tests/ checked by file list and grep only.
No drift detected.

## crates/selection/README.md

Coverage: full read of src/ and docs; tests/ checked by file list and grep only.
- Key Modules, choose.rs bullet says `select` "Dispatches greedy argmax, categorical sampling, or rejection sampling." False: [choose.rs:select] only chooses between the deterministic argmax arm and a reweighted draw. Rejection sampling is not dispatched in choose.rs; the caller builds the p/q distributions with [choose.rs:shaped_distribution] and corrects with [distribution.rs:residual].
- Public entry point [choose.rs:shaped_distribution] is omitted from the README entirely, though it is re-exported in [lib.rs:shaped_distribution] and is the path rejection sampling depends on.
- Gotcha 2 "Deterministic Step Seeding" overstates: "guarantees bit-exact reproducible token sequences across identical runs" holds only when a seed is supplied. With `seed: None`, [derive.rs:derive_step_value] reads `SystemTime::now()` and the draw varies between runs. The README also says the generator is "seeded per generation step"; the code keys the draw on (seed, position) in [choose.rs:draw] and [derive.rs:derive_step_value], and the clock fallback is not mentioned.
- Omitted invariants: [shaping.rs:ShapingConfig::new] rejects top_k above [shaping.rs:MAX_TOP_K] (256), non-finite or non-positive repetition penalties, and top_p outside (0, 1]. The builders [shaping.rs:ShapingConfig::with_presence_penalty], [shaping.rs:ShapingConfig::with_frequency_penalty] (range [-2, 2]) and [shaping.rs:ShapingConfig::with_min_p] (range [0, 1), 0 disables) also validate. None of these bounds appear in the README.
- Gotcha 3 "Hot-Path Partial Ranking" describes a "sequential kth-largest cut followed by sorting only the admitted set". That is only [truncation.rs:rank_top_k_u32_into], the path [choose.rs:shape_candidates] uses. The public [truncation.rs:rank_top_k] uses `select_nth_unstable_by` and has no sequential cut. The README does not say which function it means.

## crates/selection/src/EGM.md

Coverage: full read of all seven src/*.rs files and this doc; tests/ checked by grep only.
- Minor omission: the sentence "`softmax` and the `truncate_by_*` helpers are not called by `select`" also holds for [truncation.rs:rank_indices] and [truncation.rs:rank_top_k], which are listed under Truncation and ranking but are not named as non-live. A reader can assume they are on the live path.
- Every other named symbol, invariant, fallback, panic site and side effect checked against source ([choose.rs:select], [choose.rs:shape_candidates], [choose.rs:argmax], [choose.rs:reweight], [choose.rs:SCRATCH], [derive.rs:derive_step_value], [penalty.rs:apply_repetition_penalty], [distribution.rs:residual], [truncation.rs:assert_u32_domain]) matches.

## crates/server/EGM.md

Coverage: sampled (claims verified 4 of 5, source not fully read)
No drift detected.

## crates/server/AGENTS.md

Coverage: sampled (claims verified about 9 of 11, source not fully read; catalog alias rule and tool-guardrail buffering checked only against the --help text in crates/server/src/args.rs)
No drift detected.

## crates/server/README.md

Coverage: sampled (claims verified about 40 of 44, source not fully read)
- [crates/server/src/queue.rs:DEFAULT_MAX_QUEUED] Line 52 says the admission gate has "no queue timeout or depth cap". The code caps waiters at DEFAULT_MAX_QUEUED (64), settable with --max-queued ([crates/server/src/args.rs:parsed.max_queued], [crates/server/src/main.rs] set_max_queued), and returns 429 with Retry-After past the cap ([crates/server/src/handler/mod.rs:queue_full_response]). The "depth cap" part is false. No queue timeout was found in queue.rs (only test timeouts), so that part may hold.
- [crates/server/src/messages.rs:messages] Intro says "Uses zero-allocation translations between Anthropic and OpenAI protocols." Nothing in messages.rs or responses/map.rs or any comment supports this claim. Remove or qualify it.
- [crates/server/src/guardrails.rs:inspect] Key Modules says guardrails provide "schema repair". The code validates parsed arguments against the request schema (validate_tool_arguments in inspect) and re-asks once (GuardrailConfig::default retries = 1). No repair step was found. coerce_value in [crates/server/src/guardrails/extra_formats.rs:coerce_value] converts scalar text during rescue, which is parsing, not schema repair. Replace "schema repair" with "schema validation".
- [crates/server/src/args.rs:USAGE] (c) Key Modules lists the flag parser by example only and omits behavior-changing flags: --max-queued, --prefix-reuse (default off, with a shared-cache privacy caveat that appears only in crates/server/AGENTS.md), --session-slots (requires --prefix-reuse on), --guardrails, --max-concurrent-requests (alias for --session-slots), and the --steering and --kv-bits families. Low priority if the list is meant to be partial.

## crates/server/src/EGM.md

Coverage: sampled (claims verified about 45 of 47, source not fully read)
No drift detected.

## crates/server/src/audio/EGM.md

Coverage: sampled (claims verified about 25 of 27, source not fully read)
- [crates/server/src/audio/mod.rs:clones_voices] Invariants give the consent families as MegaTTS3, OmniVoice, and Drama 3 (fish_drama3) only. clones_voices also accepts "qwen3_tts" ([crates/server/src/audio/mod.rs:clones_voices]), and no consent or reference_text branch for qwen3_tts appears in [crates/server/src/audio/speech.rs:speech]. The doc does not list qwen3_tts as a cloning family and does not state its consent rule. Confirm the intended behavior before documenting it.

## crates/server/src/guardrails/EGM.md

Coverage: sampled (claims verified 7 of 7, source not fully read; extra_formats.rs was grepped and read in part, not in full)
No drift detected.

## crates/server/src/handler/EGM.md

Coverage: sampled (all named symbols confirmed to exist; SSE keep-alive, RunControl, and cancelled_before_start behavior verified; source not fully read)
No drift detected.

## crates/server/src/responses/EGM.md

Coverage: sampled (claims verified about 10 of 10, source not fully read)
No drift detected.

## crates/streaming/EGM.md

Coverage: full read (doc is 6 lines; its one link resolves; `src` symbols it names all exist). Directory `crates/streaming` as a whole: sampled, see the `src/EGM.md` section.
No drift detected.

## crates/streaming/README.md

Coverage: sampled (claims verified about 50 of 55, source not fully read). Checked: env vars, `POOL_THREADS` = 8, `F_RDADVISE`/`posix_fadvise(WILLNEED)`, `MADV_WILLNEED`, `read_volatile` touches, `proc_pid_rusage`/`F_NOCACHE`, `posix_memalign`, `libc::syscall` io_uring, cfg gating in Gotcha 4, `layer_NN.bin` naming (repack writes two-digit names, model-io accepts both), `packed_vision/blobs.bin`, `streaming` alias in `crates/runtime/Cargo.toml`.
- `aligned_slot.rs` is described under Key Modules as "Page-aligned buffer allocations required for unbuffered direct disk I/O". Overstated: the shipping `pread` path opens no `O_DIRECT` descriptor (only the unwired `linux_uring.rs` path does, see `linux_uring.rs:dio_aligned`), and `aligned_slot.rs:AlignedSlot` documents its purpose as a no-copy base pointer for a GPU backend to wrap.
- Tests section omits `tests/offset_overflow.rs` (asserts offset arithmetic refuses to wrap in the streamer chain). Omission.

## crates/streaming/AGENTS.md

Coverage: sampled (claims verified about 50 of 55, source not fully read). Checked: `../../.claude/docs/modules/streaming.md` exists, `scripts/expert_policy_replay.py` exists, both `cargo` check and test commands name this package, Gotcha on `linux_uring.rs` Linux-only gating matches the `#[cfg(target_os = "linux")]` on `sys`.
No drift detected.

## crates/streaming/src/EGM.md

Coverage: sampled (claims verified about 50 of 55; source is about 3,050 lines so `expert_cache.rs` lines 30-110 and 270-389, most of `pread_streamer.rs`, and `mapped_experts.rs` were checked by targeted reads and grep, not read in full).
- Overview says the crate provides "io_uring on Linux" as a read source. Overstated: no streamer reads through the ring, `Mode::parse("auto")` resolves to `Pread`, and nothing selects `Mode::Uring` outside tests [linux_uring.rs:Mode] [linux_uring.rs:Mode::parse]. The README already says "nothing selects it yet".
- Cache entry calls `ExpertCache::plan_if_possible` the "non-panicking plan entry". It still panics via `assert!(experts.len() <= self.slot_count)` at the top of the function; only the "not enough evictable slots" case returns `None` [expert_cache.rs:ExpertCache::plan_if_possible].
- Side Effects list `mlock`ing slot pages alongside opening and pread. `PreadExpertStreamer::open` does not wire anything; wiring is the separate explicit call `wire_slots` [pread_streamer.rs:PreadExpertStreamer::open] [pread_streamer.rs:PreadExpertStreamer::wire_slots].
- Interface omits public items: `ExpertCache::slot_count` [expert_cache.rs:ExpertCache::slot_count], `ExpertCache::non_resident_experts` [expert_cache.rs:ExpertCache::non_resident_experts], `ExpertCachePolicy::describe` [expert_cache.rs:ExpertCachePolicy::describe], `Mode::from_env_default` and `Mode::parse` [linux_uring.rs:Mode::from_env_default]. Minor.
- Code comment, not this doc: the `linux_uring.rs` module header says "`Mode` is selected once per process from `TURBOSPARK_LINUX_IO`", but `Mode::from_env` has no production caller. The EGM's "parse-only seam" wording is the accurate one [linux_uring.rs:Mode].
- Verified without drift: `ExpertCachePolicy::DEFAULT` is `Lfu`; `TURBOSPARK_EXPERT_CACHE_DECAY` parsing and halving after increments; `slot_count > 0` asserts in `ExpertCache::new` and `PreadExpertStreamer::open`; speculative in-flight slots excluded from plans; `prefetch_expert` refused under `F_NOCACHE`; `POOL_THREADS` = 8 and lazy pool spawn after the single-chunk inline path in `run_batch`; `Sqe` 64-byte const assert; `TURBOSPARK_READ_QOS`, `TURBOSPARK_EXPERT_DISK_IO`, `TURBOSPARK_EXPERT_NOCACHE` env reads; all `PreadExpertStreamer` and `MappedExpertLayer` method names listed exist; `MappedPagePreparationMode` and `MappedPagePreparationStats` exist; re-exports in `lib.rs` match the list.

## crates/tokenizer/AGENTS.md

Coverage: sampled (claims verified 12 of 14; source not fully read). Not verified: "FFI guard and server poison recovery contain the panic", "only the decoder arm is compiler-enforced".
No drift detected.

## crates/tokenizer/EGM.md

Coverage: full read (index doc; linked child src/EGM.md also read).
No drift detected.

## crates/tokenizer/README.md

Coverage: sampled (claims verified 16 of 19; source not fully read). Not verified: downstream `tokenizer` alias usage in other crates' Cargo.toml.
- Doc says "This crate includes 18 comprehensive test suites in `tests/`", but tests/ holds 25 .rs files. The list omits 7: chatml_no_tool_call.rs, content_marker_spelling.rs, deepseek_vl2_image_render.rs, regex_backend_hostile_pattern.rs, sidecar_size_cap.rs, stop_matcher_long_stop.rs, vision_markers_muse.rs [crates/tokenizer/tests]
- Gotcha 3 overstates UTF-8 handling: "buffers incomplete UTF-8 sequences and emits text only when valid codepoints are formed". [detokenizer.rs:MfDetokenizer::flush] emits `String::from_utf8_lossy` output, so an unfinished trailing sequence comes out as U+FFFD at end of stream. The hold-back in [detokenizer.rs:MfDetokenizer::push] covers byte-fallback ids and trailing U+FFFD runs, not general incomplete sequences.
- Gotcha 1 says to resolve ids "using `token_to_id`, `end_of_turn_id`". `end_of_turn_id` is a public field of [dialect/mod.rs:MfTokenizer], not a lookup method. Wording only.
- Key Modules omit the template sidecar precedence (chat_template.jinja, then chat_template.json, then the tokenizer_config key) [dialect/mod.rs:MfTokenizer::load_from_dir]. Omission only.
- Key Modules describe `json_value.rs` as used inside Jinja template contexts only. Tool-call parsers also build values with [json_value.rs:JsonValue] (`parse`) and share [json_value.rs:MAXIMUM_DEPTH] with [tool_call/gemma.rs:GemmaToolCallParser::parse]. Omission only.

## crates/tokenizer/src/EGM.md

Coverage: sampled (claims verified 12 of 13; source not fully read).
- Side effects list `TURBOSPARK_CHAT_DATE` but omit the fallback: when it is unset, [jinja_date.rs:strftime_now] reads the system clock in UTC, so a template calling strftime_now is non-deterministic by default. Omission only.
- Verified without drift: public re-export list in [lib.rs], `MAXIMUM_DEPTH` = 64 [json_value.rs:MAXIMUM_DEPTH], `#![forbid(unsafe_code)]`, `parse_pinned` day range 1..=31 (2026-02-31 accepted), seconds defaulting to 0, `Tokenizer` file loading.

## crates/tokenizer/src/chat_template/EGM.md

Coverage: sampled (claims verified 9 of 10; source not fully read). Not verified: "fallback renderers are pure".
No drift detected.

## crates/tokenizer/src/dialect/EGM.md

Coverage: sampled (claims verified 10 of 12; source not fully read). Not verified: `NO_SUCH_TOKEN_ID` behavior for every dialect.
- Invariant "generation_config EOS lists are capped at 256 ids" describes truncation, but the code rejects instead. [dialect/config.rs:EosTokenIds] `visit_seq` returns "too many EOS token ids" at the 257th element, and [dialect/mod.rs:read_generation_config] maps any parse error to `None`. The whole generation_config.json is then ignored, so no extra EOS ids are added, including the valid ones. A checkpoint with more than 256 EOS ids silently loses its full stop set.
- Consumer field list omits public fields `end_of_turn_id`, `bos_prefix_id` and `think_start_id` [dialect/mod.rs:MfTokenizer]. `encode(add_bos = true)` uses `bos_prefix_id` [dialect/mod.rs:MfTokenizer::encode], which the doc does not mention. Omission only.
- Verified without drift: sidecar size caps (16 MiB and 1 MiB) and fall-through on over-cap files [dialect/mod.rs:read_capped], load order of template sidecars, `MissingSpecialToken` from [dialect/resolve.rs:required_id], filtering of extra EOS ids to the vocabulary range [dialect/mod.rs:MfTokenizer::new], MiniMax EOS doubling as pad [dialect/minimax.rs].

## crates/tokenizer/src/structured_decoder/EGM.md

Coverage: sampled (claims verified 8 of 8 checked; source not fully read).
No drift detected.

## crates/tokenizer/src/tool_call/EGM.md

Coverage: sampled (claims verified 8 of 9; source not fully read). Not verified: Kimi-K2.5 grammar provenance beyond the module doc.
- Invariant "Function names must be non-empty, at most 64 bytes, ASCII alphanumeric plus `_` and `-`" is stated as universal. [tool_call/gemma.rs:GemmaToolCallParser::parse] does not call [tool_call/mod.rs:is_valid_function_name]. It checks the name against `allowed_tools` and returns `UnknownTool`. The doc comment on `is_valid_function_name` ("the shared name check every dialect applies") is also wrong for Gemma. The Harmony decoder arm does call it [structured_decoder/harmony.rs:is_valid_function_name use].
- Verified without drift: all six parsers reference `MAXIMUM_BYTES` (256 KiB), `ParsedToolCall`, GLM and Kimi marker consts, `DSML_MARK`, `deepseek_dsml_mark`, and each parser has `new` and `parse`.

## crates/vision-io/EGM.md

Coverage: full read (doc); all 4 claims verified against Cargo.toml and src/decode.rs.
No drift detected.

## crates/vision-io/src/EGM.md

Coverage: sampled (claims verified about 60 of about 75; src is about 6,000 lines, so symbols were checked by grep and the cited functions read, not every file read in full).
- [src/lib.rs:crate docs] Not in this EGM, but the crate-level doc says "Every per-patch sequence in this crate" is merge-window order. False for Muse (raster order, muse.rs:muse_patch_rows) and Gemma 4 (row-major, gemma4.rs:Gemma4Patches). The EGM's own invariant ("Qwen path only") is correct.
- [src/lib.rs:pub use] Public constants and types reachable from the crate root are not listed in the EGM Interface: `GEMMA4_PATCH`, `GEMMA4_PATCH_DIM`, `GEMMA4_POOL`, `GEMMA4_DEFAULT_SOFT_TOKENS`, `GEMMA4_SOFT_TOKEN_BUDGETS` [gemma4.rs]; `CLIP_PATCH`, `CLIP_PATCH_DIM` [clip.rs]; `OCR_BASE_SIZE`, `OCR_DOWNSAMPLE`, `OCR_GRID`, `OCR_IMAGE_TOKENS`, `OCR_PATCH`, `DeepseekOcrRow`, `DeepseekOcrView` [deepseek_ocr.rs]; `DSVL2_PATCH`, `DSVL2_PATCH_DIM`, `DeepseekVl2Row` [deepseek_vl2.rs]; `PIXTRAL_PATCH`, `PIXTRAL_PATCH_DIM` [pixtral.rs]; `OV_PATCH`, `OV_PATCH_DIM`, `OV_SIDE`, `OV_TILE`, `OV_TILE_PATCHES`, `OV_MAX_AREA_TILES` [onevision.rs]; `MINICPM_PATCH`, `MINICPM_SCALE_RESOLUTION`, `MINICPM_MERGE`, `MINICPM_SLICE_END_ID`, `MINICPM_NEWLINE_ID`, `MINICPM_IMAGE_ID_START`, `MINICPM_IMAGE_ID_END`, `MINICPM_MARKER_TOKENS` [minicpm.rs]. The Interface lists names, so this is an omission (c), not a false claim.

## crates/vision-io/README.md

Coverage: full read of the doc; the pipeline, the four gotchas, the test list, the Key Modules list, and the dependent crates were checked against source. Claims about the pixel budget, the resize order, and the rope theta matched code.
- [tests/generated/gemma4_preprocess_oracle.rs] README "Tests" says every golden fixture under tests/generated/ is probed by scripts/qwen3vl_vision_oracle.py. The Gemma 4 fixture header says `@generated by scripts/gemma4_vision_oracle.py` (that script exists at repo root).
- [tests/generated/preprocess_oracle.rs:header] README says the mlx-vlm reference is "vendored". No copy exists in the repo. The fixture header points to an absolute path outside the repo (mlx-v/mlx-vlm).
- [tests/generated/preprocess_oracle.rs:synth_image, scripts/qwen3vl_vision_oracle.py:synth_image, src/decode.rs:module doc] Gotcha 3 says fixtures are "generated from PNG rather than JPEG sources". Fixtures start from synthetic pixels generated identically on both sides (decode.rs module doc, synth_image in the oracle script). The no-JPEG rule holds; the PNG wording does not describe the source.
- [tests/ listing] Test list is incomplete. Files not mentioned: gemma4_preprocess_parity.rs, muse_params.rs, muse_wrap.rs, params_geometry.rs, rope_theta.rs.
- [src/lib.rs:pub mod, src/*.rs] "Key Modules" omits every family pipeline module except gemma4.rs and unlimited_ocr2.rs: clip.rs, minicpm.rs, deepseek_ocr.rs, deepseek_vl2.rs, pixtral.rs, muse.rs, onevision.rs. Muse in particular uses a different resize filter (PIL Lanczos, resize_lanczos_pil) and raster patch order, and neither appears in the README.
- Verified with no drift: the `patch_size * merge_size` rounding factor (params.rs:spatial_factor), merge-window and (T, P_h, P_w, C) row order (patchify.rs:patch_rows), 48x48 pos-embed grid, rope base 1e4 vs trunk 1e7 (rope.rs:VISION_ROPE_THETA), `image_dimensions`/`decode_image_bytes` limits (decode.rs), version 0.8.0 and no dependent crate aliasing the package (workspace Cargo.toml and dependents).

## crates/vision-io/AGENTS.md

Coverage: full read (27 lines). The four `Read first` links resolve (.claude/docs/modules/vision-io.md, docs/VISION.md, docs/VISION_PHASE0.md, docs/TESTING.md). The two `Checks` commands were not executed, so they are unverified, not confirmed.
No drift detected.

