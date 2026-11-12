# Custom

## Overview
User-defined custom tools: definition model, discovery from disk (global and project folders), parsing of JSON and YAML files, execution (command, script, or HTTP), and the trust gate that keeps repository-authored tools from running until the user approves that exact definition. Global tools are authored by the user and always trusted. Project tools are authored by whoever wrote the repo and are treated like repository hooks.

## Interface
- Model: `CustomToolExecutionType` (command, script, http), `CustomToolExecution`, `CustomToolDefinition` (with `effectiveCategory` and `openAITool`) [CustomToolDefinition.swift].
- Discovery and persistence: `CustomToolManager.shared` (`globalTools`, `reloadGlobalTools`, `resolveEffectiveTools(for:)`, `untrustedProjectTools(for:)`, `trustProjectTool(named:projectURL:)`, `saveTool`, `deleteTool`, `globalToolsDirectory`, `userHomeToolsDirectory`) [CustomToolManager.swift:CustomToolManager].
- Parsing: `CustomToolParser.parse(fileURL:scope:)`, `.parseDictionary`, `CustomToolParseError` [CustomToolParser.swift].
- Execution: `CustomToolExecutor.execute(tool:arguments:projectRootURL:)`; internal `baseEnvironment`, `substituteArguments`, `environmentKey` [CustomToolExecutor.swift:CustomToolExecutor].
- Trust: `CustomToolTrustStore` (`shared` is a replaceable static var, `fingerprint(of:)`, `isTrusted`, `trust`, `revoke`) [CustomToolTrustStore.swift].
- Approval sheet logic: `ProjectToolReview`, `ProjectToolApprovalOutcome`, `ProjectToolApproval.review(of:)`, `.pending(for:rejected:manager:)`, `.approve(_:projectURL:manager:store:)` [ProjectToolApproval.swift].
- Child directories: none.

## Invariants
- The declared category is a request, not a grant. `effectiveCategory` is `.terminal` for command and script tools and at least `.web` for http tools. Permissions and risk must use `effectiveCategory` [CustomToolDefinition.swift:effectiveCategory].
- Project-scoped tools (`<project>/.turbospark/tools`, `<project>/.agents/tools`) are never offered or executed until trusted. An untrusted definition, even one marked disabled, is skipped entirely so it cannot suppress a global tool of the same name. Project tools override global ones of the same name; a trusted disabled project tool removes the global one [CustomToolManager.swift:resolveEffectiveTools].
- A custom tool may never use a built-in tool name (`AppToolCatalog.isBuiltInToolName`); such files are skipped for both global and project scans [CustomToolManager.swift].
- The trust key is a SHA-256 over the standardized project path, source path, lowercased name, effective category, the sorted-key JSON of the execution block and of `parameters`, the description, and the enabled flag. Adding `parameters`, description and enabled invalidated pre-existing trust records once. Any edit to what the tool runs (for example via `git pull`) invalidates trust. Do not drop fields from the fingerprint [CustomToolTrustStore.swift:fingerprint].
- `ProjectToolApproval.approve` re-checks the fingerprint against the current on-disk definition, so a file rewritten between review and click returns `.changedSinceReview` and trusts nothing. Reject is not stored; an untrusted tool just stays off [ProjectToolApproval.swift].
- `saveTool` auto-trusts a project-scoped tool the user authored in the app (re-parsing the saved file so the source path matches the key). Plugin scope cannot be saved [CustomToolManager.swift:saveTool].
- `CustomToolExecutor.execute` filters model arguments to those declared in `parameters.properties`; an undeclared key cannot influence substitution or the environment [CustomToolExecutor.swift:execute].
- Command templates never receive argument bytes in shell source. `{{key}}`, `${key}`, and `$KEY` markers are rewritten in one pass to `${TOOL_ARG_<KEY>}` references (quote-context aware), and values travel via environment variables. Unknown markers are left untouched so real env vars still work. Keep it single-pass, or a value containing another marker would get substituted into [CustomToolExecutor.swift:substituteArguments].
- Script tool arguments are substituted into argv elements (no shell involved) in ONE pass by `substituteLiteral`, so a value containing another argument's marker is never re-scanned. The temp script is deleted after the run [CustomToolExecutor.swift:execute].
- Tool names must match `[A-Za-z0-9_-]{1,64}` (`CustomToolDefinition.isSafeName`) because they become `<name>.json` and `custom_<name>_<id>.sh`. `CustomToolParser.parse` rejects other names (the tool is skipped), `saveTool` throws code 4 in `TurboSparkCustomTool`, and script execution throws code 90 [CustomToolDefinition.swift:isSafeName].
- The only production caller of `CustomToolExecutor.execute` is `AppToolRegistry`, which resolves the tool through `resolveEffectiveTools` (trust-filtered, re-read from disk per call) immediately before executing. The executor itself does not check trust, so any new caller must resolve through the manager [AppToolRegistry.swift].
- The child environment is an allowlist: HOME, LANG, TMPDIR, a PATH extended with Homebrew locations, plus the tool's own `environment`. No parent secrets are inherited [CustomToolExecutor.swift:baseEnvironment].
- Command and script tools launch through `ShellNetworkSandbox.launch` (fails closed under Block all). Http tools honor only the Block all mode (with the localhost exception) because the endpoint is the user's own configuration [CustomToolExecutor.swift].
- A nonzero exit or a timeout throws (code 33), so partial output is not reported as success. Output is compacted via `AppToolRegistry.compactOutput` [CustomToolExecutor.swift:failIfUnsuccessful].
- Parsing: JSON uses a strict decode with a tolerant dictionary fallback. Strict decode failure of `parameters` falls back to `AppToolCatalogMcp.parameterSchema(from:)` so the validation gate refuses unrepresentable shapes instead of validating nothing. The YAML path is a line scanner that reads only `name:`, `description:`, `command:` and makes a `.terminal` command tool [CustomToolParser.swift].
- Error codes under domain `TurboSparkTool`: 30 missing command, 31 missing script, 32 bad HTTP URL, 33 non-zero exit or timeout, 34 network blocked, 90 unsafe tool name. Manager errors use domain `TurboSparkCustomTool`.

## Side Effects
- Disk reads: scans `AppStorageRoot.subdirectory("tools")`, `UserProfileStore.userScopeSubdirectory("tools")`, and project `.turbospark/tools` and `.agents/tools` for `.json`, `.yaml`, `.yml` files [CustomToolManager.swift].
- Disk writes: `saveTool` writes `<name>.json` atomically (pretty, sorted keys); `deleteTool` removes the file; `CustomToolTrustStore` persists `trusted_project_tools.json` under `AppStorageRoot` (atomic) [CustomToolManager.swift, CustomToolTrustStore.swift].
- Process spawn: `/bin/zsh -c` or the configured interpreter via `ProcessExecutor.run`, in the project root; temp script file in the system temp directory with mode 0755 [CustomToolExecutor.swift].
- Network: http tools send the filtered arguments as a JSON body through `URLSession.shared` with the tool's timeout [CustomToolExecutor.swift].
- Reads `AppNetworkPolicyStore.current`. `CustomToolManager.shared` loads global tools at initialization.
- None for Keychain, UserDefaults, or NotificationCenter.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
