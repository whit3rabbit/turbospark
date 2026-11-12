# Workflows

## Overview
This directory implements the workflow scripting engine of the TurboSpark app. A restricted, non-evaluating subset of JavaScript (facade v1) is lexed, parsed, and statically checked into an AST plus an approval manifest. The interpreter then runs the checked AST against a capability sink (model actors, pinned commands, read-only world reads). Every capability call is journaled so a run can be resumed or replayed deterministically. Scripts never execute as JavaScript and never choose executables or paths themselves.

## Interface
No child directories have an EGM.md.

- Authoring surface: `WorkflowFacade.version`, `Member`, `entryForm`, `validateRunDispatch(commandKeyForm:argumentForm:)`, `loadAPIReference()` [WorkflowFacade.swift:WorkflowFacade]
- Lex and parse: `WorkflowLexer(bytes:maximumTokens:).lex()`, `WorkflowToken`, `WorkflowSymbol`, `WorkflowSourceMap`, `WorkflowParseFailure` [WorkflowLexer.swift:WorkflowLexer]
- AST and outcomes: `WorkflowScriptAST` (currentVersion 1), `WorkflowScriptParseResult`, `WorkflowCheckOutcome`, `WorkflowDefinitionCheckOutcome`, `WorkflowCheckedProgram`, `WorkflowScriptDiagnostic`, `WorkflowScriptDiagnosticRule` [WorkflowAST.swift:WorkflowScriptAST]
- Checking: `WorkflowScriptChecker.parse`, `checkDefinition`, `check(source:name:args:)`, `Limits.production` [WorkflowScriptChecker.swift:WorkflowScriptChecker]
- Static validators: `WorkflowScriptLiteralValidation.validate` [WorkflowScriptLiteralValidation.swift:WorkflowScriptLiteralValidation], `WorkflowScriptSemanticValidation.validate` and `validateDefinitionArguments` [WorkflowScriptSemanticValidation.swift:WorkflowScriptSemanticValidation], `WorkflowScriptManifestCompiler.build` [WorkflowScriptManifestCompiler.swift:WorkflowScriptManifestCompiler]
- Core types: `WorkflowRunState`, `WorkflowErrorKind`, `WorkflowError`, `WorkflowSiteKey`, `WorkflowRequestIdentity.make`, `WorkflowCanonicalSerialization`, `WorkflowResultShape`, `WorkflowLaunchManifest`, `WorkflowRunDescriptor`, `WorkflowSavedDefinition`, `WorkflowSavedDefinitionStore` [WorkflowTypes.swift:WorkflowCanonicalSerialization]
- Execution: `WorkflowInterpreter.execute`, `requestCancel`, `clearPendingCancel`, `WorkflowInterpreterLimits`, `WorkflowCommandSink`, `WorkflowCancellationToken` [WorkflowInterpreter.swift:WorkflowInterpreter]
- Run engine: `WorkflowRunEngine.start`, `cancel`, `progress`, `perform`, `WorkflowCapabilities`, `WorkflowEngineLimits`, `WorkflowRunProgress` [WorkflowRunEngine.swift:WorkflowRunEngine]
- Graph scheduling: `WorkflowGraph.make`, `WorkflowScheduler.runGraph`, `validateDAG`, `WorkflowNode`, `WorkflowGraphOutcome` [WorkflowScheduler.swift:WorkflowScheduler]
- Actors: `WorkflowActorBridge.ask`, `SubagentDispatcherPort`, `WorkflowAskLimits`, `WorkflowActorCapabilityAdapter`, `WorkflowAppSubagentDispatcher` [WorkflowActorBridge.swift:WorkflowActorBridge]
- Answer shapes: `WorkflowResultShapeCodec` (decode, validate, parseAnswer, canonical/JSON conversion), `WorkflowActorTranscriptCodec` [WorkflowResultShapeCodec.swift:WorkflowResultShapeCodec]
- Journal: `WorkflowJournal`, `WorkflowJournalStore`, `WorkflowJournalEvent`, `WorkflowActorTranscriptSnapshot`, `WorkflowJournalResumeSnapshot` [WorkflowJournal.swift:WorkflowJournal]
- World reads: `WorkflowWorld`, `WorkflowWorldRead`, `WorkflowGitReadOp`, `WorkflowWorldLimits`, `WorkflowWorldExecutorPort`, `WorkflowWorldSystemExecutor` [WorkflowWorld.swift:WorkflowWorld]

## Invariants
- The checker never evaluates source. Only `async function workflow() { ... }` is accepted, and a checked program exists only when parse, literal, semantic, and manifest build all report no diagnostics [WorkflowScriptChecker.swift:WorkflowScriptChecker.check]
- A partial AST with diagnostics is never valid [WorkflowAST.swift:WorkflowScriptParseResult]. The AST version is pinned and the interpreter refuses other versions [WorkflowAST.swift:WorkflowScriptAST]
- Scripts cannot pick executables. `run` needs a literal command key declared earlier, values matching declared slots, never executable, workingDirectory, or argv [WorkflowFacade.swift:WorkflowFacade.validateRunDispatch]
- Command executables are absolute and pinned by canonical path plus file SHA-256. The manifest is nil if any diagnostic exists [WorkflowScriptManifestCompiler.swift:WorkflowScriptManifestBuilder.pinExecutable]
- The manifest site table order (source offset, then insertion order) must stay in step with the interpreter site map [WorkflowScriptManifestCompiler.swift:WorkflowScriptManifestBuilder.makeSiteTable]
- Dynamic command values are rejected over the byte limit or when they contain NUL. Workspace input paths must be relative with no '.' or '..' [WorkflowInterpreter.swift:WorkflowInterpreterAttempt.executeAwaited]
- One interpreter attempt runs at a time (serial executor). A fresh start must call `clearPendingCancel` [WorkflowInterpreter.swift:WorkflowInterpreter.requestCancel]
- Static analysis is work-bounded (65536 items for literals) and reports `staticAnalysisBudgetExceeded` [WorkflowScriptLiteralValidation.swift:WorkflowScriptLiteralValidation.maximumStaticAnalysisWorkItems]
- Every graph is joined exactly once on every path and never inside a loop. Graphs have at most 100 nodes, unique ids, and no cycles [WorkflowScriptSemanticValidation.swift:WorkflowScriptSemanticValidator.analyzeJoinFlow]
- Scheduler retries only `modelFailure`, bounded by node.maxRetries (0...3). A batch holds at most one node per actor. Dependents of failed nodes are recorded as blocked [WorkflowScheduler.swift:WorkflowScheduler.validateDAG]
- Replay never repeats uncertain work: a `requestStarted` with no outcome (and no later cancellation record) refuses to re-execute [WorkflowRunEngine.swift:WorkflowRunEngine.executeJournaled]
- Ask requests use their actor lane, all others the main lane. Each lane runs one request at a time, and journal appends are serialized [WorkflowRunEngine.swift:WorkflowRunRequestLane]
- Resume fails closed on canonicalization version mismatch, missing or duplicate actor transcripts, or hash mismatch [WorkflowJournal.swift:WorkflowJournal.resumeSnapshot]
- Canonical serialization (versioned prefix, sorted keys, NFC strings) feeds journal hashes. Changing it invalidates stored runs [WorkflowTypes.swift:WorkflowCanonicalSerialization]
- Reserved system sites use negative Int.min offsets and are valid only on the main lane [WorkflowTypes.swift:WorkflowJournalSystemSites]
- Repair asks are clamped to 0...2, and transcript snapshots above 256 KiB fail with `resourceLimit` [WorkflowActorBridge.swift:WorkflowAskLimits]
- Actor answers must be a JSON object with no undeclared fields, reporting at most 16 issues [WorkflowResultShapeCodec.swift:WorkflowResultShapeCodec.validateFields]
- World reads are read-only and workspace-contained: paths are canonicalized and symlink escapes refused, and git runs `/usr/bin/git` with a scrubbed environment [WorkflowWorld.swift:WorkflowWorld.resolve]
- Repo-defined git filter drivers are neutralized before status or diff [WorkflowWorld.swift:WorkflowWorld.filterNeutralizingOverrides]
- Lexer refuses template strings, optional chaining, arrows, bitwise and loose-equality operators, and caps tokens [WorkflowLexer.swift:WorkflowLexer.nextToken]

## Side Effects
- Reads `workflow-facade-v1.md` from the module bundle [WorkflowFacade.swift:WorkflowFacade.loadAPIReference]
- Hashes executable files (64 KiB chunks) and stats the filesystem when pinning commands [WorkflowScriptManifestCompiler.swift:WorkflowScriptManifestBuilder.pinExecutable]
- Appends run state, phase, request, and failure events to the journal store [WorkflowRunEngine.swift:WorkflowRunEngine.appendJournal]. The scheduler appends begin and resolve events per attempt [WorkflowScheduler.swift:WorkflowScheduler.executeBatch]
- Runs live capabilities (models, commands, reads) via `capabilities.perform` [WorkflowRunEngine.swift:WorkflowRunEngine.executeJournaled]
- Runs model subagent turns through the injected dispatcher, one child Task per dispatch [WorkflowActorBridge.swift:WorkflowActorBridge.dispatch]
- Starts a deadline monitor Task and uses NSLock-guarded cancellation state [WorkflowInterpreter.swift:WorkflowCancellationToken]
- Node dispatches run in a `withTaskGroup` [WorkflowScheduler.swift:WorkflowScheduler.dispatchAttempts]
- Spawns `/usr/bin/git`, reads and enumerates workspace files, and calls `realpath` [WorkflowWorld.swift:WorkflowWorldSystemExecutor.execute]

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
