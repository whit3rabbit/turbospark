# ComputerUse (app side)

## Overview
The app half of the computer-use feature, which lets a vision-capable local model see and drive Mac apps the user allows. This directory holds the pure policy and per-chat state (which apps are granted at which tier, which screenshot the model last saw), the Settings pane, and the client that launches and talks to the separate helper app `TurboSpark Computer Use.app`. The helper does the Accessibility, capture and input work (see [TurboSparkComputerUse](../../TurboSparkComputerUse/EGM.md)). The wire types and framing are shared through [ComputerUseKit](../../ComputerUseKit/EGM.md). Nothing here synthesizes input itself.

## Interface
- Policy: `ComputerUseAppTier` (refused < read < click < full, `Comparable`) and `ComputerUseAppPolicy` with `tier(forBundleID:)` and `allows(_:tier:)` [ComputerUseAppPolicy.swift:ComputerUseAppTier] [ComputerUseAppPolicy.swift:ComputerUseAppPolicy].
- Per-chat state: `ComputerUseSession` actor (`grant`, `tier(for:)`, `grantedApps`, `revokeAll`, `recordScreenshot`, `frame(for:screenshotID:)`) and `ComputerUseScreenshotFrame` with `screenPoint(x:y:)` mapping image pixels to screen points [ComputerUseSession.swift:ComputerUseSession] [ComputerUseSession.swift:ComputerUseScreenshotFrame].
- Settings model: `ComputerUseSettings` (`enabled`, default false, lenient decode) stored under `computerUse` in settings.json [ComputerUseSettings.swift:ComputerUseSettings].
- Helper client: `HelperComputerUsePort` actor conforming to `ComputerUseControlPort`: `installedHelperURL(in:)`, `launch(helperURL:)` returning the port and a `Welcome` (manifest plus status), `perform(_:)`, `status()`, `requestPermission(_:)`, `resumeAfterStop()`, `shutdown()` [HelperComputerUsePort.swift:HelperComputerUsePort].
- UI: `ComputerUseSettingsPaneView` (feature toggle, permissions section, Refresh) and `ComputerUsePermissionRow` [ComputerUseSettingsPaneView.swift:ComputerUseSettingsPaneView] [ComputerUseSettingsPaneView.swift:ComputerUsePermissionRow].
- Child directories: none.

## Invariants
- A `.refused` app is never stored by `grant`, whatever the caller approved. The tier is always recomputed from the bundle id, never taken from the caller [ComputerUseSession.swift:grant].
- Refused set: any id with the prefix `com.whit3rabbit.turbospark`, plus System Settings, Keychain Access, Passwords, SecurityAgent, Screen Sharing and common password managers. The reason is that approval cards live in TurboSpark, so an agent able to click its own app could approve itself [ComputerUseAppPolicy.swift:refusedBundleIDs] [ComputerUseAppPolicy.swift:tier(forBundleID:)].
- Browsers are read tier (the in-app browser tools enforce origin approvals this surface cannot). Terminals, editors and `com.jetbrains.*` are click tier. Everything else is full [ComputerUseAppPolicy.swift:readOnlyBundleIDs] [ComputerUseAppPolicy.swift:clickOnlyBundleIDs].
- Click tier allows only left clicks with count 1 or 2, scrolling and opening apps. Typing and keys need full tier. Right click is excluded because its context menu has Paste. `listApps` needs no grant [ComputerUseAppPolicy.swift:allows(_:tier:)].
- Grants and frames never outlive the session: `revokeAll` clears both [ComputerUseSession.swift:revokeAll]. The file comment says the app drops the session on chat switch, model switch or kill switch. That wiring lives outside this directory.
- `frame(for:screenshotID:)` returns a frame only when the id equals the latest screenshot for that bundle, so a stale id cannot aim clicks. `screenPoint` returns nil for non-finite, negative or out-of-image coordinates, and scales by `windowFrame / image pixel size` (the image the model received, not the Retina capture) [ComputerUseSession.swift:frame(for:screenshotID:)] [ComputerUseSession.swift:screenPoint(x:y:)].
- Computer use is off by default, and `enabled` decodes leniently to false [ComputerUseSettings.swift:ComputerUseSettings].
- The helper is launched through `NSWorkspace.openApplication` (not `Process`) so macOS attributes grants to the helper, with `createsNewApplicationInstance`. One helper per connection [HelperComputerUsePort.swift:launch(helperURL:)].
- Handshake: the first request must be `.hello(token:protocolVersion:)` with the session token, and any non-`.welcome` reply closes the port and throws [HelperComputerUsePort.swift:launch(helperURL:)].
- The session directory is mode 0700 with a short `tscu-` name (a `sockaddr_un` path holds under 104 bytes) and the token file is mode 0600. `tryConnect` rejects overlong paths [HelperComputerUsePort.swift:makeSessionDirectory] [HelperComputerUsePort.swift:tryConnect].
- The socket is strictly one request then one response. The actor serializes callers and the blocking IO runs on a private dispatch queue, off the cooperative pool. Any IO failure maps to a generic "helper is not responding" error [HelperComputerUsePort.swift:send].
- macOS permission prompts come only from `requestPermission`, which is wired to the user's Allow button. Enabling the feature only reads status [ComputerUseSettingsPaneView.swift:request] [HelperComputerUsePort.swift:requestPermission].
- Settings UI strings use `Text(..., bundle: .module)` for localization parity. One `Section` per computed property (a macOS 14 type-checker workaround noted in the file).

## Side Effects
- Process launch: starts the helper app via `NSWorkspace.shared.openApplication` with `--session-dir` [HelperComputerUsePort.swift:launch(helperURL:)].
- Disk: creates and later removes a session directory under the temporary directory, writes the token file [HelperComputerUsePort.swift:makeSessionDirectory] [HelperComputerUsePort.swift:close].
- Sockets: connects an AF_UNIX stream socket to the helper, retrying every 100 ms for up to 10 seconds, and closes it on shutdown [HelperComputerUsePort.swift:connect].
- Security framework: `SecRandomCopyBytes` for the 32-byte session token [HelperComputerUsePort.swift:randomToken].
- NotificationCenter: the pane observes `NSApplication.didBecomeActiveNotification` to refresh helper status after the user returns from System Settings [ComputerUseSettingsPaneView.swift:body].
- Calls into `AppModel` (`setComputerUseEnabled`, `refreshComputerUseHelperStatus`, `requestComputerUsePermission`) from the pane.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
