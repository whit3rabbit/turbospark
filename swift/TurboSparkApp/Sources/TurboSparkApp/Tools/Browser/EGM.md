# Browser

## Overview
The model-facing browser tool surface and the media policy for image-bearing tool results. It defines the eight `browser_*` tool schemas, strictly projects a model call into a typed `BrowserControlCommand`, runs it through the permission engine, and hands it to an injected backend (`BrowserToolRuntime.perform`). It also validates, downsizes, and stores screenshots as managed assets, and re-resolves stored screenshot references for providers that accept images in tool results. The browser backend itself, `BrowserControlCommand`, `BrowserPermissionContext`, and `BrowserOrigin` live elsewhere.

## Interface
- Availability and schemas: `BrowserToolAvailability` (`supports(_:)`, `disabled`), `BrowserToolDefinitions` (`all`, `commandKindsByToolName`, `tools(availableFor:)`, `commandKind(for:)`) [BrowserToolDefinitions.swift].
- Execution: `BrowserToolRuntime` (availability, permission context, `perform` closure), `BrowserToolExecutionOutcome` (completed, pendingApproval, denied, unsupported, invalidInput, unavailable, failed, cancelled), `BrowserToolExecutor.execute(call:in:runtime:currentActionApproved:globalServers:)`, `.command(for:arguments:)`, `.modelOutput(for:)` [BrowserToolExecutor.swift].
- Media policy: `AppToolMediaCapability` (`textOnly`, `imageBearingToolResults(...)`), `AppToolMediaPolicy` (limits constants, `storeScreenshot`), `AppToolMediaProcessingOutcome`; internal `PreparedAppToolImage`, `AppToolMediaPreparation`, `prepareScreenshotPNG` [AppToolMediaPolicy.swift].
- History projection: `AppToolMediaHistoryAdapter.project(_:capability:materialize:)` and `AppToolMediaHistoryProjection` [AppToolMediaHistoryAdapter.swift].
- Child directories: none.

## Invariants
- Fail closed on availability: a tool is offered or executed only when the backend is enabled, the manifest exists, its `commandSurfaceVersion` equals `BrowserControlProtocol.currentVersion`, and the command is `.supported`. Screenshots additionally need image-bearing tool result support. `supports` is checked again right before `perform` [BrowserToolDefinitions.swift:supports, BrowserToolExecutor.swift:execute].
- Argument projection is strict: unknown keys are rejected, timeouts must be finite, positive, and at most `BrowserAutomationSession.maximumTimeoutSeconds`, integers must round-trip as strings, and booleans accept only true/false/1/0 (the dispatch layer serializes JSON booleans as 1/0). `browser_wait` requires either `load_state` or `reference` plus `condition`, never both [BrowserToolExecutor.swift:parseCommand].
- Navigation must be an HTTP or HTTPS URL that parses into a `BrowserOrigin`; userinfo is refused. The permission context is built for the parsed destination origin, and a caller's `currentActionApproved` transfers only if the raw `url` argument's origin equals that destination, so a padded URL cannot ride on another origin's approval [BrowserToolExecutor.swift:execute].
- Every browser call is forced to category `.browser`, risk-assessed, and evaluated by `AppToolPermissionEngine` with `owner: .agent`. A backend result whose `commandKind` differs from the request is a failure. Cancellation maps to `.cancelled` [BrowserToolExecutor.swift].
- Model output must never echo typed text or full URLs: navigation reports only the canonical origin, and `browser_type` reports only the reference. Snapshots are wrapped in `<untrusted_browser_snapshot>` with `&`, `<`, `>` escaped and prefixed with a notice not to treat contents as instructions [BrowserToolExecutor.swift:modelOutput].
- The parser accepts an optional `full_page` for `browser_screenshot`, but the published schema declares no properties (viewport only, since the WebKit backend rejects full-page). Keep the schema and parser in step if full-page support changes [BrowserToolDefinitions.swift, BrowserToolExecutor.swift].
- Media limits: provider limits are clamped down to the application ceilings (24,000,000 pixels, 8 MiB encoded, 4,096 px per side), never raised. Only PNG is accepted. An oversized image is re-encoded with up to 32 iterations, shrinking the max side by 20 percent each time, and is refused if it still does not fit. Never persist an image that exceeds the provider's capability [AppToolMediaPolicy.swift:prepareScreenshotPNG].
- History projection returns no images (only an omitted count) for providers without image-bearing tool result support. It re-checks MIME (`image/png`), byte count, pixel limits, a valid managed-asset id, and that the asset materializes, skipping anything else; `omittedReferenceCount` reports how many were dropped [AppToolMediaHistoryAdapter.swift:project].

## Side Effects
- Disk: `AppToolMediaPolicy.storeScreenshot` writes PNG data into `ManagedAssetStore.shared` (named `browser-screenshot.png` by default); the history adapter materializes assets via the caller-supplied closure [AppToolMediaPolicy.swift:storeScreenshot, AppToolMediaHistoryAdapter.swift].
- Browser actions: `runtime.perform` drives the controlled browser tab (navigation, clicks, typing, key presses, scrolling, screenshots, snapshots, waits). The implementation is outside this directory.
- Image processing: ImageIO thumbnail creation and PNG encoding (CPU work).
- Reads `ToolRiskClassifier` and `AppToolPermissionEngine` for decisions.
- None for network, Keychain, UserDefaults, NotificationCenter, or process spawn within this directory.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
