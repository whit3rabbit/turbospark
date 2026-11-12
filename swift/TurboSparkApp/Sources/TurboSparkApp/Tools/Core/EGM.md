# Tools/Core

## Overview
The gate between a model's proposed tool call and anything that runs. It parses calls out of reply text or engine-native calls, validates them against the schemas this turn actually offered, rates their risk, decides allow, ask, or deny under the project's permission mode and the app-wide network policy, and supplies the shared subprocess runner. It also holds the sandbox and host-scope rules, the MCP permission-rule grammar, the argument-spelling resolver that keeps the approval card, risk classifier and executor reading the same value, and the bundled linear "permission gate" model (advisory only). Agent mode, where a local model answers soft asks, lives in the child directory.

## Interface
- Parsing and dispatch gate
  - [ToolCallParser.swift:ToolCallParser] is the single reader of calls: `parse(from:projectURL:)`, internal `parseCandidates`, `nativeCandidates` for engine-parsed `GenerationToolCall`, `parseJSONArguments`. Calls start as `.pendingApproval`.
  - [ToolCallDispatchGate.swift:ToolCallDispatchGate] `evaluate(...)` returns `ToolCallDispatchGateResult`. `ToolCallDispatchResolution` (`resolve`, `handle`) picks retry, finish-as-prose, or dispatch. `ToolCallValidationRefusal.userFacingSummary` localizes refusals.
  - [SwiftToolCallValidation.swift:TurnAvailableTools] holds the per-turn snapshot (`definitions`, `promptDefinitions`, `deferredMcpTools`) and `validate(toolName:argumentsJSON:)` returning `ToolCallValidationResult`. Also `ToolCallJSONValue` and `ToolCallValidationRefusal`.
  - [ToolCallNormalization.swift:ToolCallNormalization] and the `TurnAvailableTools` extension (`resolvedDefinition(for:)`, `canonicalizedArguments`) map Claude Code, opencode and Pi spellings onto offered definitions.
- Schema types [OpenAIToolSchema.swift:OpenAITool, OpenAIFunction, JSONSchema, JSONSchemaProperty, IndirectWrapper, OpenAIToolSerializer].
- Argument resolution [ToolArgumentResolver.swift:ToolArgumentResolver, ToolArgumentConflict, McpWireArguments].
- Built-in tool definitions [ToolSearchTools.swift:ToolSearchToolDefinitions, DeferredToolDescriptor, ToolSearchCatalog, ToolSearchExecutor], [ToolObservationTools.swift:ToolObservationToolDefinitions] (`recall_tool_output`).
- Permission decision
  - [AppToolPermissionEngine.swift:AppToolPermissionEngine] `evaluate(...)` returns `ToolPermissionDecision` (allow, ask with assessment, deny).
  - [AppToolPermissionEngine.swift:SessionApprovalStore] per-chat "always allow" grants for tools, command prefixes, and exact hosts.
  - [AppToolPermissionEngine.swift:ToolPermissionContext] task-local `fallbackMode` for projectless chats.
  - [McpPermissionRules.swift:McpPermissionRule] and `AppProjectPermissions` extension (`mcpDenyMatches`, `mcpAllowMatches`, `addingMcpRule`).
- Risk [ToolRiskClassifier.swift:ToolRiskClassifier, ToolRiskAssessment, ToolRiskLevel] with `assessRisk`, `assessTerminalCommand`, `assessMcpTool`, `adjusting`, `isSensitivePath`.
- Sandbox [AppToolSandbox.swift:AppToolSandbox, SandboxConfig, AppHostScope]: `validateWritePath`, `validateDomain`, `hostScope`, `isPrivateOrMetadataHost`, `resolvedForContainment`.
- Subprocess [ProcessExecutor.swift:ProcessExecutor] (`run`, `terminateAndReap`, `killTreeNow`, `killAllLiveChildrenNow`, `descendantPIDs`) and `CappedOutputBuffer`.
- Advisory ML gate [CommandGate.swift:CommandGate] (`score`, `veto`, `advisoryReason`, `vetoEnabled`, `vetoThreshold`), with [HashedFeatureVectorizer.swift:HashedFeatureVectorizer] and [CommandLexicalFeatures.swift:CommandLexicalFeatures].
- Subdirectories: [AgentMode](./AgentMode/EGM.md) (local-model permission classifier and its fallback counters).

## Invariants
- Order in `evaluateProjectPolicy` is load-bearing. Browser gate first, then read-only (absolute), category deny, the computer-use access request (always hard-gated ask), MCP deny rule, full access, the high-risk gate, permissive, MCP allow rule, session approval, MCP server auto-approve and repo-import ask, ask mode, auto, and a trailing allow [AppToolPermissionEngine.swift:evaluateProjectPolicy]. Deny and high-risk are checked before any session grant, so an approval for one call never covers a later high-risk call under the same tool name. A persisted allow rule buys back the ask, not the risk ceiling.
- The network policy wraps everything. Its deny returns before any project mode (full access included). Its ask hard-gates an otherwise allowed call unless that exact host was approved for the chat [AppToolPermissionEngine.swift:evaluate]. Approved hosts are exact, never suffixes [SessionApprovalStore.allowHost].
- An MCP server imported from a repository config asks in `auto` and `agentAuto` unless explicitly `autoApprove`, and that ask is `hardGated` so Agent mode cannot answer it. Server annotations can only raise risk. If a server has not finished discovery, permissive, auto and the trailing allow ask instead of allowing.
- `hardGated` marks a deterministic guard (denylist, structural shell check, sensitive path, sandbox rule). Nil reads as soft. An unknown decoded risk level reads as `.high`, and the decoder is tolerant so one bad field cannot quarantine a chat archive [ToolRiskClassifier.swift:ToolRiskAssessment.init(from:)].
- Terminal risk: denylist regexes only supply reasons. The allowlist decides. Anything that is not a single simple invocation of a recognized command is `.high` and hard-gated [ToolRiskClassifier.swift:assessTerminalCommand]. A `batch` is rated as its strongest child. The MCP bridge is rated on the tool it will dial. User-scope `propose_skills` is always high.
- The ML veto may only add friction. `vetoEnabled` defaults false, and `advisoryReason` only decorates an already-`.ask` verdict. `veto` must be called only after the allowlist said yes [CommandGate.swift].
- Feature parity with the trainer is a compatibility constraint: the `\b` token anchors, murmur3 x86_32 at seed 0 with signed result, `abs` computed in 64 bits, char_wb short-word rule, the 20 lexical features and their keyword lists, and `lexicalCount == CommandLexicalFeatures.count`. The `permission-gate.bin` layout is `PGT1` magic, little-endian, validated on load, and a failed load yields nil scores [CommandGate.swift:loadModel, HashedFeatureVectorizer.swift, CommandLexicalFeatures.swift].
- One argument, one winner. Two spellings with different values throw `ToolArgumentConflict`, and risk, card, and executor all go through `ToolArgumentResolver`, so what is shown is what runs. `name` names an MCP target only when no explicit tool key exists. A typed value reaches an MCP server only if its string projection still equals the string the gate saw [ToolArgumentResolver.swift:McpWireArguments].
- Normalization resolves only to a definition this turn offered, never overwrites a value sent under the canonical key, and the result still passes schema validation [ToolCallNormalization.swift]. Validation refuses unsupported schema keywords instead of ignoring them [SwiftToolCallValidation.swift:unsupportedSchemaIssue].
- Native engine calls replace text parsing for that turn. The XML arm wins and the markdown arm is only a fallback. A call prefixed by the localized "Incomplete tool call" label is inert text. Only a completed stream dispatches. Prose resolutions have no path to the executor [ToolCallDispatchGate.swift, ToolCallParser.swift].
- Sandbox containment resolves the deepest existing ancestor with `lstat` and follows dangling symlinks (depth 16), so a write under a symlinked directory cannot escape. Writes need the workspace root or an explicit allow and must not hit `deniedWritePaths`. `hostScope` treats all of 169.254/16 and `.internal` as metadata, parses IPv6 to bytes (mapped, NAT64, 6to4), accepts decimal, octal and hex IPv4, and fails closed to private for an unparseable colon literal [AppToolSandbox.swift].
- Subprocess: one polled reader thread per pipe so a full pipe cannot deadlock and a backgrounded child cannot pin the thread. The deadline loop checks `Task.isCancelled` explicitly and uses a non-throwing sleep. Cancellation throws `CancellationError` after the kill. Termination snapshots descendants before the signal, then SIGTERM, SIGKILL, and a kill of every snapshot member. Stdin writes set `F_SETNOSIGPIPE` on the descriptor. Output is capped at 1,000,000 bytes by default and truncation is reported [ProcessExecutor.swift].
- Deferred MCP tools are rebuilt from live servers and cache on every request, filtered by deny rules and by the calling agent's `callerToolFilter`, and `tool_call` refuses names outside that set [ToolSearchTools.swift].

## Side Effects
- Spawns child processes and signals them: `Process`, `kill`, and a `sysctl kern.proc.all` walk [ProcessExecutor.swift]. Keeps a static table of live children guarded by an `NSLock`.
- Disk reads: `permission-gate.bin` from the app bundle [CommandGate.swift], `lstat` and symlink resolution [AppToolSandbox.swift], `GlobalMcpFileStore.load()` [ToolSearchTools.swift:ToolSearchExecutor, AppToolPermissionEngine.swift default argument], localization bundles [ToolCallDispatchGate.swift, ToolCallParser.swift].
- UserDefaults read of `AppLanguage.storageKey` [ToolCallDispatchGate.swift:localizedUserFacingSummary].
- Reads shared singletons: `AppNetworkPolicyStore.current`, `McpToolCatalogCache.shared`, `CustomToolManager.shared`, `BrowserPermissionRuleStore`.
- Mutable global state: `SessionApprovalStore.shared` (actor), `CommandGate.vetoEnabled` and `vetoThreshold` (`nonisolated(unsafe)`, set at launch), the lazy `CommandGate.model`.
- Calls `AppToolRegistry.executeDeferredMcpCall` for deferred MCP calls [ToolSearchTools.swift].
- No network calls, Keychain access, or NotificationCenter use here.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
