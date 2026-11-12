# Tools/Hooks

## Overview
The lifecycle hook subsystem, modeled on Claude Code's hooks contract. It discovers hooks (app-created custom hooks, user and project config files, plugin manifests), gates them behind a content-hash trust list, dispatches lifecycle events to the matching hooks (shell command or HTTP webhook), parses each hook's output into a structured outcome, and folds all outcomes for one event into a single verdict that callers act on (allow, ask, deny, block, prevent continuation, updated input, extra context).

Flow: `AppHookStore` holds the hooks and trust state. `AppHookExecutionEngine.dispatch` filters candidates (enabled and trusted), checks matchers (`AppHookMatcher.swift`), runs each (`AppHookCommandRunner.swift`), `AppHookResponseParser` interprets output, and `AppHookDecisionAggregator` produces the verdict.

## Interface
- Models [AppHookModels.swift:AppHookEvent] (13 lifecycle events), [AppHookModels.swift:AppHookType] (`command`, `http`, `prompt`), [AppHookModels.swift:AppHookShell], [AppHookModels.swift:AppHookSourceType], [AppHookModels.swift:AppHookCommand] (including `contentHash`), [AppHookModels.swift:AppHookOptionSpec], [AppHookModels.swift:AppHookSourceGroup], [AppHookModels.swift:AppHookPermissionBehavior], [AppHookModels.swift:AppHookPreToolUseDecision], [AppHookModels.swift:AppHookExecutionResult].
- Store (main actor, `ObservableObject`) [AppHookStore.swift:AppHookStore] with `shared`, `refresh(projectDirectory:)` [AppHookStore+Discovery.swift:refresh], trust API (`isHookTrusted`, `trustHook`, `untrustHook`, `trustAllInGroup`), `toggleHookEnabled`, `updateOptionValue`/`getOptionValue`, custom hook CRUD (`addCustomHook`, `updateCustomHook`, `deleteCustomHook`), `discoveryDiagnostics`, `includeClaudeGlobalConfig`, and the internal [AppHookStore.swift:AppHookDispatchSnapshot].
- Engine [AppHookExecutionEngine.swift:AppHookExecutionEngine] with `dispatch(event:...)`, `evaluatePreToolUse`, `hasMatchingHook`, `markGhostSession`, `blockingEvents`.
- Command and HTTP runners [AppHookCommandRunner.swift:executeCommandHook], [AppHookCommandRunner.swift:executeHttpHook], placeholder helpers `expandEnvPlaceholder`, `shellQuoted`, `optionEnvSuffix` (the last is shared with the options sheet).
- Output interpretation [AppHookResponse.swift:AppHookResponseParser.parseHookOutput], [AppHookResponse.swift:AppHookOutcome] (`structured`, `plainText`, `blocked`, `nonBlockingError`, `unavailable`), [AppHookResponse.swift:AppHookResponse].
- Verdict [AppHookDecisionAggregator.swift:AppHookDecisionAggregator.aggregate], [AppHookDecisionAggregator.swift:AppHookVerdict].
- Matching [AppHookMatcher.swift:AppHookMatcherEvaluator.matches], [AppHookMatcher.swift:matchesCondition], [AppHookToolNameAliases.swift:AppHookToolNameAliases.equivalentNames].
- Hook stdin and webhook body builder [AppHookStdinPayload.swift:AppHookStdinPayload.build].
- Secret storage [HookOptionKeychain.swift:HookOptionSecretStoring], [HookOptionKeychain.swift:HookOptionKeychain].
- Subdirectories: none.

## Invariants
- Trust is by content hash. `contentHash` covers source type, source path, plugin name, event, type, command, `if`, matcher, shell, timeout, and async flag, so a byte-identical hook from a different source (for example a cloned repo) is not trusted by an earlier approval. Only `.custom` hooks are trusted by default. Dispatch runs only enabled hooks whose hash is trusted. [AppHookModels.swift:contentHash], [AppHookStore.swift:isHookTrusted], [AppHookExecutionEngine.swift:candidates]
- A hook that cannot answer is never consent. Timeout, spawn failure, a missing project directory, an invalid URL, or an HTTP transport failure yield `.unavailable`, which the aggregator turns into `.ask` on permission events (deny still outranks it, and it outranks allow) and into feedback elsewhere. The prompt hook type yields a non-blocking outcome and is never evaluated. [AppHookCommandRunner.swift], [AppHookDecisionAggregator.swift:aggregate], [AppHookExecutionEngine.swift:executeSingleHook]
- Every matching hook runs; there is no short-circuit. Permission events rank deny over ask over allow. For other events the first block wins, while later hooks' context and messages still aggregate. `continue: false` sets `preventContinuation` on any event and is distinct from `isBlocked`. [AppHookDecisionAggregator.swift]
- Exit code 2 blocks only for `PreToolUse`, `UserPromptSubmit`, and `Stop`; elsewhere it is non-blocking feedback. Any other nonzero exit is non-blocking. [AppHookResponse.swift:blockingExitTwoEvents]
- `async: true` is ignored for `blockingEvents` (`PreToolUse`, `UserPromptSubmit`, `Stop`, `PermissionRequest`) because the caller acts on the decision; other events run detached. [AppHookExecutionEngine.swift:blockingEvents]
- Ghost sessions never expose prompt or tool content to hooks: content-bearing events return no results, and `evaluatePreToolUse` denies when a matching PreToolUse hook exists rather than disclose the payload. [AppHookExecutionEngine.swift:dispatch], [AppHookExecutionEngine.swift:evaluatePreToolUse]
- Command hooks get a minimal environment (`PATH`, `HOME`, `LANG`, `TMPDIR` plus hook variables), never the parent environment. Sensitive option values reach the hook by environment only, never substituted into the `-c` argv (placeholders are rewritten to an env reference). `${CLAUDE_PROJECT_DIR}`, `${CLAUDE_PLUGIN_ROOT}`, `${CLAUDE_PLUGIN_DATA}` are expanded to quoted env references according to the surrounding quote context. A hook naming `${CLAUDE_PROJECT_DIR}` with no project directory is refused. [AppHookCommandRunner.swift:executeCommandHook], [AppHookCommandRunner.swift:expandEnvPlaceholder]
- `SessionEnd` timeout is capped at 1.5 seconds for command and HTTP hooks. A non-positive timeout falls back to 600 seconds. Hand-authored hooks default to 30 seconds, discovered hooks to 600 on purpose (Claude Code parity); do not unify. [AppHookCommandRunner.swift], [AppHookModels.swift:AppHookCommand.init], [AppHookStore+Discovery.swift:parseHookObject]
- Persistence must not truncate custom hooks: `saveCustomHooks` is skipped until the first `refresh`, `addCustomHook` refreshes first, and corrupt custom-hook files are quarantined by `AppJSONStore` instead of overwritten. [AppHookStore+Persistence.swift:saveCustomHooks], [AppHookStore.swift:addCustomHook]
- Sensitive option values are written to the Keychain and stripped from the plaintext options file (mode 0600); legacy plaintext values are migrated on `synchronizeSensitiveOptionValues`, and keys once Keychain-backed stay out of the file even if discovery stops reporting them sensitive. [AppHookStore+Persistence.swift]
- Discovered hooks the user disabled are tracked by content hash (`disabledHashes`) and re-applied after every refresh. `lastProjectDirectory` must be reused by every `recomputeSourceGroups` call. [AppHookStore+Discovery.swift:refresh], [AppHookStore.swift:toggleHookEnabled]
- Discovery rules: the Claude global config (`~/.claude/settings.json` and `.local`) is read only under `includeClaudeGlobalConfig`; a non-default profile reads only its own `hooks.json`; plugin hook paths must stay inside the plugin directory (`PathContainment`); an `agent` hook type maps to the unevaluated `.prompt` type rather than a shell command; unparseable entries are dropped with a diagnostic. [AppHookStore+Discovery.swift]
- Matcher order: empty or `*` matches all; a pattern of letters, digits, `_`, `-`, space, `,`, `|` is an exact or list match; anything else is an unanchored regex; all comparisons go through the tool name alias table so Claude Code and app tool names are interchangeable. `if` conditions glob only a trailing `*` after whitespace normalization. [AppHookMatcher.swift]
- A dispatch bound to a project directory uses one atomic `AppHookDispatchSnapshot` (hooks, trust, options, sensitive keys). [AppHookStore.swift:dispatchSnapshot]
- The Keychain account is a SHA-256 of storage path, source ID, and key, so profiles do not share secrets. [HookOptionKeychain.swift:baseQuery]

## Side Effects
- Process spawn of `/bin/bash`, `/bin/zsh`, `/bin/sh`, or `/usr/local/bin/pwsh` through `ProcessExecutor.run`, with the event payload as stdin. [AppHookCommandRunner.swift:executeCommandHook]
- Network: HTTP POST of the event payload to the hook URL via `URLSession.shared`. [AppHookCommandRunner.swift:executeHttpHook]
- Disk under `AppStorageRoot.subdirectory("Hooks")`: `trusted_hook_hashes.json`, `disabled_hook_hashes.json`, `custom_hooks.json`, `hook_options_values.json` (atomic writes, 0600 on the options file). [AppHookStore+Persistence.swift]
- Disk reads of `~/.turbospark`, optionally `~/.claude`, project `.turbospark`/`.claude` config files, `hooks/hooks.json`, and enabled plugin hook files; reads `PluginManager.shared`. [AppHookStore+Discovery.swift]
- Keychain read, write, update, and delete (service `TurboSpark.hook-options`, accessible after first unlock). [HookOptionKeychain.swift]
- `Task.detached` for async non-blocking hooks; process-wide ghost session set guarded by a lock. [AppHookExecutionEngine.swift]
- `@Published` state changes on the main actor drive the Hooks settings UI. [AppHookStore.swift]

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
