# Codemode

## Overview
Implements the opt-in `codemode` tool: the model writes one JavaScript script that batches many deferred MCP tool calls (`await tools.<name>(args)`), and only the script's printed output and return value come back. Each script runs in a fresh worker process (a copy of the app binary in serve mode) that hosts a JavaScriptCore context confined by a Seatbelt profile. The host side supervises the process, enforces limits and the deadline, and routes every nested call back through the normal tool path so hooks and permissions still apply. The directory also builds the prompt listing of typed `tools.<name>(args)` signatures.

Data flow: `CodemodeSandbox.run` parses the source and store snapshot, `CodemodeWorkerSupervisor` spawns the worker and speaks `CodemodeWire` lines, the worker's `CodemodeWorkerContext` evaluates `CodemodePrelude` plus the wrapped script, and nested calls return to `CodemodeNestedRunner`.

## Interface
- Feature flag and tool definition [CodemodeToolDefinitions.swift:CodemodeSettings] (`codemode.enabled`, default off), [CodemodeToolDefinitions.swift:CodemodeToolDefinitions] (`all` is empty while the flag is off; `codemode` takes one `code` string).
- Prompt listing [CodemodeToolDefinitions.swift:CodemodeCatalog.promptListing] (falls back to `ToolSearchCatalog.promptListing` when disabled, not offered, or no entries), typed signatures via [CodemodeDeclarations.swift:CodemodeSchemaRenderer.argumentsDeclaration].
- Run entry and result formatting [CodemodeSandbox.swift:CodemodeSandbox.run], [CodemodeSandbox.swift:CodemodeSandbox.promptOutput], [CodemodeSandbox.swift:CodemodeSandbox.archivalCalls], [CodemodeSandbox.swift:CodemodeSandbox.ParsedSource] (the optional `// @options:` line), [CodemodeSandbox.swift:CodemodeSandbox.StoreBox] (per-chat store), [CodemodeSandbox.swift:CodemodeResultStop].
- Nested call routing [CodemodeNestedRunner.swift:CodemodeNestedRunner.callHandler].
- Value types and limits [CodemodeTypes.swift:CodemodeToolEntry], [CodemodeTypes.swift:CodemodeResult], [CodemodeTypes.swift:CodemodeLimits], [CodemodeTypes.swift:CodemodeError], [CodemodeTypes.swift:CodemodeCallRecord], [CodemodeTypes.swift:CodemodeIdentifier] (JS name sanitizing and collision handling).
- Host supervisor [CodemodeWorkerSupervisor.swift:CodemodeWorkerSupervisor] (`run`, `terminateAndWait`, `CallHandler`, `physicalFootprint`).
- Wire protocol [CodemodeWire.swift:CodemodeWire] (`RunRequest`, `CallResult`, `Call`, `Output`, `Done`, `Crash`, `encodeLine`, `decodeWorkerMessage`).
- Worker side [CodemodeWorkerContext.swift:CodemodeWorkerContext], [CodemodeWorkerSandbox.swift:CodemodeWorkerSandbox] (`profile`, `apply`, `runProbe`, `probeArgument`), [CodemodePrelude.swift:CodemodePrelude] (`source`, `wrapper`, `scriptURL`).
- Subdirectories: none. A non-EGM `README.md` also sits in this directory.

## Invariants
- A nested call must take the same path as a model-issued call: [CodemodeNestedRunner.swift:callHandler] only accepts names in the granted descriptor set, refuses `codemode` itself (no recursion), and goes through `AppToolRegistry.executeDeferredMcpCall` so hooks, approvals, and the permission engine run. `ToolPermissionContext.fallbackMode` is captured outside the closure because detached tasks read it as nil.
- The sandbox profile is a deny-list over `(allow default)` (deny network, exec, fork, file-write, reads under `/Users` and `/Volumes`, signals, mach-lookup, iokit-open, preference reads), with the app's own resolved bundle and executable roots allowed back after the deny. Later rules win, so order matters. [CodemodeWorkerSandbox.swift:profile]
- Applying the profile is irreversible and fails closed: if it cannot be applied the worker must not run the script. Paths must be symlink-resolved before confinement. [CodemodeWorkerSandbox.swift:apply], [CodemodeWorkerSandbox.swift:ownReadRoots]
- The worker environment is the MCP child allowlist with no passthrough, so launching-shell secrets never reach the process running model-written code. [CodemodeWorkerSupervisor.swift:init]
- Wire is newline-delimited JSON; tool arguments and results cross as JSON strings, and unknown message types decode to nil. A line over `maximumLineBytes` fails the run. One process serves exactly one execution. [CodemodeWire.swift], [CodemodeWorkerSupervisor.swift:receive]
- Worker stdout is read by one blocking thread to EOF, and completion is only declared after both stdout and stderr closed (or a 2 second grace after exit); a `readabilityHandler` was dropped because it lost the terminal `done` line. [CodemodeWorkerSupervisor.swift:startReader], [CodemodeWorkerSupervisor.swift:completedResult]
- An oversized nested result is rejected, never truncated. Calls over `maximumCalls` get a rejection, calls past `maximumConcurrentCalls` queue, and calls arriving after the result settled are ignored. In-flight calls are cancelled and recorded `cancelled` when the script ends. [CodemodeWorkerSupervisor.swift:handleCall], [CodemodeWorkerSupervisor.swift:runAdmittedCall], [CodemodeWorkerSupervisor.swift:assemble]
- Store writes apply only when the run succeeded, and only with a non-nil `storeKey`. `StoreBox` keeps at most 256 keys per chat and 64 chats (least recently used evicted), preferring to keep keys the script just wrote. [CodemodeSandbox.swift:run], [CodemodeSandbox.swift:StoreBox.apply]
- The prelude closes over the bridges and deletes their globals; `tools`, `console`, `text`, `exit`, `store`, `load` are non-writable, non-configurable. Settlement is first-write-wins (`finished` flag and `CodemodeOutcomeBox`); an output-cap breach reports failure before throwing, and no call may cross the bridge after settlement. The return value shares the output budget. [CodemodePrelude.swift:source], [CodemodeWorkerContext.swift:CodemodeOutcomeBox]
- The `// @options:` line is the first non-blank line and is blanked in place so reported line numbers still match the source; the wrapper opener shares line 1 with the script for the same reason. Unknown option fields or a malformed line are source errors. [CodemodeSandbox.swift:ParsedSource.parse], [CodemodePrelude.swift:wrapper]
- Call telemetry (`calls`) is archival only and must never enter the prompt projection. [CodemodeSandbox.swift:archivalCalls]
- The listing keeps `jsName` unique (first wins; collisions get a numeric suffix) and dedupes advertised names case-insensitively. [CodemodeTypes.swift:CodemodeIdentifier.entries]
- Schema rendering is a bounded, lossy summary (depth, union, enum, `$ref` expansion limits; falls back to `Record<string, unknown>`), not a validator. [CodemodeDeclarations.swift]

## Side Effects
- Process spawn: one worker per run via `Process`, with stdin/stdout/stderr pipes; termination is stdin close, then `ProcessExecutor.terminateAndReap`. [CodemodeWorkerSupervisor.swift:init], [CodemodeWorkerSupervisor.swift:terminateAndWait]
- Process-global `signal(SIGPIPE, SIG_IGN)`. [CodemodeWorkerSupervisor.swift:ignoreBrokenPipes]
- Reader threads for stdout and stderr, a 20 ms polling loop, and detached tasks for nested calls. [CodemodeWorkerSupervisor.swift:run], [CodemodeWorkerSupervisor.swift:handleCall]
- Kernel query `proc_pid_rusage` roughly every 100 ms to kill a worker over the memory cap. [CodemodeWorkerSupervisor.swift:physicalFootprint]
- UserDefaults read and write of `codemode.enabled`. [CodemodeToolDefinitions.swift:CodemodeSettings]
- In worker: `dlsym` of `sandbox_init` and `sandbox_free_error`, JavaScriptCore context creation, stdout wire writes; the probe also writes a temp file, spawns `/usr/bin/true`, opens a loopback socket, and does a Mach lookup. [CodemodeWorkerSandbox.swift:apply], [CodemodeWorkerSandbox.swift:runProbe]
- Nested calls execute real MCP tools (which spawn MCP server processes) through `AppToolRegistry`. [CodemodeNestedRunner.swift]
- In-memory per-chat store in the `StoreBox` actor; nothing is persisted to disk. [CodemodeSandbox.swift:StoreBox]

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
