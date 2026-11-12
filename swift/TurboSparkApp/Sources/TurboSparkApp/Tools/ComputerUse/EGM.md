# ComputerUse

## Overview
The model-facing macOS computer-use tool family (`computer_list_apps`, `computer_request_access`, `computer_snapshot`, `computer_screenshot`, `computer_click`, `computer_type`, `computer_key`, `computer_scroll`, `computer_open_app`). It decides whether the tools are offered this turn, projects strict model arguments onto typed `ComputerUseCommand` values, enforces per-app grants and tiers, and calls an injected `ComputerUseControlPort` (the helper process boundary, defined in `ComputerUseKit` and elsewhere). The generic category ask/deny gate runs earlier in the agent loop via `AppToolPermissionEngine`.

## Interface
- Availability: `ComputerUseToolAvailability` (`isEnabled`, `manifest`, `mediaCapability`, `isOffered`, `supports(toolName:)`, `disabled`) [ComputerUseToolDefinitions.swift:ComputerUseToolAvailability].
- Schemas: `ComputerUseToolDefinitions` (`requestAccessToolName`, `commandKindsByToolName`, `toolNames`, `all`, `tools(availableFor:)`) [ComputerUseToolDefinitions.swift:ComputerUseToolDefinitions].
- Execution: `ComputerUseToolRuntime` (availability, `ComputerUseSession`, `ComputerUseControlPort`, `ManagedAssetStore`), `ComputerUseToolExecutionOutcome` (completed with optional media, denied, invalidInput, unavailable, failed, cancelled), `ComputerUseToolExecutor.execute(call:runtime:)`, `readOnlyToolNames` [ComputerUseToolExecutor.swift].
- Child directories: none.

## Invariants
- The family is offered only when all three hold: the user enabled it, the helper manifest is current (`manifest.isCurrent`), and the loaded model accepts image-bearing tool results. Otherwise the model sees none of the tools (unlike the browser, which hides only its screenshot tool) [ComputerUseToolDefinitions.swift:isOffered].
- Argument projection is strict: unknown keys are rejected per tool, integers must round-trip as strings, `computer_type` text is 1 to 4,000 characters, `computer_key` is at most 40 characters and lowercased, scroll amount is 1 to 50, click count is 1 or 2 and the button left or right [ComputerUseToolExecutor.swift:parseCommand, allowedKeys].
- Every command that names an app requires an existing grant (`session.tier(for:)`), else `denied` ("call computer_request_access first"), and `ComputerUseAppPolicy.allows(command, tier:)` must permit the action for that tier. Bundle ids must pass `isPlausibleBundleID` [ComputerUseToolExecutor.swift:execute].
- `computer_request_access` accepts 1 to 8 comma-separated bundle ids plus a reason. Reaching execution means the user approved the call's card showing the apps and reason verbatim. Apps the policy refuses are reported as refused. A successful grant calls `port.resumeAfterStop()`, the only thing that re-arms input after the kill switch [ComputerUseToolExecutor.swift:requestAccess, execute].
- A `stoppedByUser` port error revokes every grant (`session.revokeAll()`), so continuing needs a fresh approval. Preserve this fail-closed path [ComputerUseToolExecutor.swift:execute].
- `computer_click` takes either a snapshot `reference` or `x`, `y`, and `screenshot_id`, never both. Pixel clicks map through the latest recorded screenshot frame for that app; a stale id or out-of-bounds point is refused instead of guessed [ComputerUseToolExecutor.swift:clickTarget].
- Accessibility snapshots are returned wrapped in `<untrusted_app_snapshot>` with `&`, `<`, `>` escaped and a notice not to treat contents as instructions [ComputerUseToolExecutor.swift:output].
- Screenshots go through `AppToolMediaPolicy.storeScreenshot` and are recorded in the session with the stored pixel size, so later pixel clicks scale correctly. A command and result that do not match produce `failed` [ComputerUseToolExecutor.swift:output].
- `readOnlyToolNames` (list_apps, snapshot, screenshot) are the tools the permission engine lets through on `.ask`, since the app grant was the approval; deny still refuses them [ComputerUseToolExecutor.swift].

## Side Effects
- Drives other macOS apps through the helper (`port.perform`): accessibility reads, window screenshots, clicks, typing, key presses, scrolling, and launching or fronting apps. The helper implementation is outside this directory.
- Disk: screenshots are written to `ManagedAssetStore` (default name `computer-screenshot.png`) via `AppToolMediaPolicy.storeScreenshot` [ComputerUseToolExecutor.swift:output].
- State: mutates the `ComputerUseSession` (grants, revocation, screenshot frames) [ComputerUseToolExecutor.swift].
- None for network, Keychain, UserDefaults, NotificationCenter, or direct process spawn within this directory.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
