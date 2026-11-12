# Terminal

## Overview
The model-facing shell: the Bash, BashOutput, KillShell, and REPL tool schemas, the foreground and background execution path under zsh, and the policy code around it (network confinement, working-directory persistence, output shaping, and command classification for the `.auto` permission mode). Foreground commands run through `ProcessExecutor` (defined elsewhere). Background commands run through `BackgroundShellManager`, which owns its own pipes and reader threads.

## Interface
- Tool schemas: `TerminalToolDefinitions.bash`, `.bashOutput`, `.killShell`, `.repl`, `.all`; `REPLInput`, `REPLOutput` payload structs (no REPL executor in this directory) [TerminalTools.swift:TerminalToolDefinitions].
- Execution entry (internal): `ShellCommandRunner.run`, `.runForTool` (returns `ToolOutput` with `presentation` and un-compacted `archivalOutput`), `.backgroundOutput`, `.killBackground`, `.clampedTimeoutSeconds`, `.wrappedCommand` [ShellCommandRunner.swift:ShellCommandRunner].
- Background registry: `BackgroundShellManager.shared` (`launch`, `record(id:chatID:)`, `knownShellIDs`, `outputSnapshot`, `waitAndOutput`, `kill`, `killAll`, `killAllForShutdown`, `runningRecords`, `forgetFinishedRecords(chatID:)`, `setChangeObserver`, `resetForTests`), `BackgroundShellRecord`, and the public value snapshot `BackgroundShellSummary` [BackgroundShellManager.swift].
- Cwd persistence: `ShellCwdTracker.shared` (`startDirectory(for:chatID:)`, `record(finalDirectory:root:chatID:)`, `isInside`) [ShellCwdTracker.swift:ShellCwdTracker].
- Network confinement: `ShellNetworkSandbox.profile(allowLocalhost:)`, `.launch(executable:arguments:policy:)` [ShellNetworkSandbox.swift:ShellNetworkSandbox].
- Output shaping and environment: `ShellOutputFormatting` (`stripANSI`, `compact`, `compactWithSpill`, `spillRootURL`, `isUnderSpillRoot`, `pruneSpillFiles`, `hasUnquotedCompoundOperator`, `benignExitNote`, `shellEnvironment`) [ShellOutputFormatting.swift:ShellOutputFormatting].
- Classification (public): `TerminalCommandClassifier` (`isSingleSimpleInvocation`, `headCommand`, `isAutoApprovable`, `isCollapsible`, `interpretExitCode`, and the command sets) [TerminalCommandClassifier.swift:TerminalCommandClassifier].
- Child directories: none.

## Invariants
- `isAutoApprovable` is the only gate for silent execution under `.auto`; `isCollapsible` is presentation only and must never feed a security decision. Keep them separate [TerminalCommandClassifier.swift:isCollapsible].
- `isAutoApprovable` is a positive allowlist: the command must be a single simple invocation (no `| ; & $ \` > < ( ) { }`, newline, or CR), with no quote, backslash, or `=` in the head word. An interpreter (sh, bash, python, node, env, xargs, ...) with `-c`, `-e`, `--eval`, or `--command` is refused. `git` is approvable only for the listed subcommands [TerminalCommandClassifier.swift:isSingleSimpleInvocation, isAutoApprovable].
- Network Block all fails closed: if `/usr/bin/sandbox-exec` is missing the command is refused, never run unconfined. The profile is `allow default` plus `deny network*`; never add a wildcard `local ip "localhost:*"` outbound allow (it is an outbound hole). Applied to foreground and background launches alike [ShellNetworkSandbox.swift:launch].
- Foreground commands are wrapped so the exit status is captured first and re-raised last, with the `pwd -P` capture written to a temp file. Newlines, not `;`, separate the suffix so trailing comments or heredocs cannot swallow it. Background commands run raw and never move the cwd anchor [ShellCommandRunner.swift:wrappedCommand, launchBackground].
- Timeout is clamped to 1...600 s (default from `ProcessExecutor`). A timeout throws (code 42) and carries partial output; a nonzero exit throws (code 40) unless `benignExitNote` applies (exit 1 from a lone grep/rg/diff/test, no unquoted compound operator) [ShellCommandRunner.swift, ShellOutputFormatting.swift:benignExitNote].
- `ShellCwdTracker` keys by (chat, project root), is process-lifetime only (never persisted), re-stats the remembered directory, and resets to the root when the final directory is outside it (symlink-resolved compare, so /tmp equals /private/tmp) [ShellCwdTracker.swift].
- Background shells are chat-scoped: `record(id:chatID:)` returns nil for another chat's id, and the error text only lists the caller's ids. Limits: 20 running, 50 finished per chat, 200 finished total; running records are never pruned [BackgroundShellManager.swift].
- `markKilled` and `finalize` only act while the state is `.running`, so a killed shell is never relabeled `failed` by its termination handler. `kill` sets state before terminating and runs the SIGTERM/SIGKILL ladder off the caller's thread [BackgroundShellManager.swift:kill].
- Reader threads must start only after `process.run()` succeeds, and the parent's write ends and stdin must be closed; on launch failure the record is removed and all handles closed. Each reader closes its handle at EOF to avoid fd exhaustion. Do not switch to `readabilityHandler` [BackgroundShellManager.swift:launch].
- The termination handler waits up to 2 s for readers before finalizing, so `completed` does not coexist with unread output [BackgroundShellManager.swift:launch].
- The spill directory lives under the current profile's `AppStorageRoot.directory/spill`, never a machine-wide folder. `isUnderSpillRoot` backs the `read_file` allowlist and must keep resolving symlinks. Only 20 spill files are kept. Model-visible output is capped at 30,000 chars (20,000 head, 8,000 tail) [ShellOutputFormatting.swift].
- `shellEnvironment` sets `GIT_EDITOR=true`, `GIT_PAGER=cat`, `PAGER=cat`, `TERM=dumb`, `NO_COLOR=1`, `TURBOSPARK_APP=1` to prevent hangs [ShellOutputFormatting.swift:shellEnvironment].
- `BashOutput` waits 0...120 s (default 30), and cancellation returns the snapshot rather than throwing [ShellCommandRunner.swift:backgroundOutput, BackgroundShellManager.swift:waitAndOutput].
- Error codes under domain `TurboSparkTool`: 4 missing id, 9 unknown id, 40 failed, 41 too many shells, 42 timeout, 43 sandbox unavailable.

## Side Effects
- Process spawn: `/bin/zsh -c` (optionally via `/usr/bin/sandbox-exec`) for foreground and background commands; SIGTERM/SIGKILL via `ProcessExecutor.terminateAndReap` and `killTreeNow` [ShellCommandRunner.swift, BackgroundShellManager.swift].
- Disk: temp cwd-capture file in `NSTemporaryDirectory()` (removed afterward); spill files under the profile storage root [ShellCommandRunner.swift:runForeground, ShellOutputFormatting.swift:writeSpillFile].
- Threads and queues: one `DispatchQueue.global` reader per pipe, global queue for kill ladders, and the change observer is dispatched on the main queue [BackgroundShellManager.swift].
- Reads `AppNetworkPolicyStore.current` and the parent process environment.
- None for Keychain, UserDefaults, or NotificationCenter.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
