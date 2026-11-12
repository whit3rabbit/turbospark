# TurboSparkComputerUse (helper executable)

## Overview
The source of `TurboSpark Computer Use.app`, a menu-bar accessory app bundled inside TurboSpark. It is a separate process so macOS attributes the Accessibility and Screen Recording grants to it rather than to TurboSpark. It serves exactly one authenticated connection from the app over a Unix socket, executes computer-use commands (app listing, accessibility snapshots, window screenshots, clicks, typing, key presses, scrolling, opening apps), and exits when that connection ends. Policy (which apps and tiers are allowed) lives in the app, see [ComputerUse](../TurboSparkApp/ComputerUse/EGM.md). Wire types, framing and peer checks come from [ComputerUseKit](../ComputerUseKit/EGM.md). The helper enforces only per-action safety (frontmost app, window under the point, kill switch), not grants.

## Interface
- Entry point: `main.swift` parses `--session-dir <dir>` (exit 64 on usage error) or `--self-test` (prints the manifest JSON for `scripts/verify-app-bundle.sh` and exits 0), sets accessory activation policy, and runs `HelperController` [main.swift].
- `HelperController` (`@MainActor`): menu bar status item with "Stop Agent Control" and "Quit Computer Use Helper" items, and wiring of the service, kill switch and server [HelperController.swift:HelperController].
- `HelperService` (`@MainActor`): `manifest`, `welcome()`, `status()`, `handle(_:)`, `stop()`, `installKillSwitch()`, `onStoppedChange`, plus `HelperError` [HelperService.swift:HelperService] [HelperService.swift:HelperError].
- `HelperServer`: socket listener and authenticated request loop (`start()`, `acceptTimeoutSeconds`) [HelperServer.swift:HelperServer].
- `AXSnapshotter`: bounded accessibility tree text with element refs (`snapshot(app:)`, `element(bundleID:reference:)`, `frame(of:)`, limits `maximumDepth` 30, `maximumNodes` 500, `maximumCharacters` 30,000) [AXSnapshotter.swift:AXSnapshotter].
- `InputSynthesizer`: `click`, `type`, `press`, `scroll`, `bringToFront`, `frontWindowCenter`, and static `windows()` and `frontWindow(pid:)` [InputSynthesizer.swift:InputSynthesizer].
- `WindowCapturer`: `capture(pid:)` returning PNG data and the window frame [WindowCapturer.swift:WindowCapturer].
- Child directories: none.

## Invariants
- One connection, one process lifetime: the server accepts once (30 s accept timeout, exit 0), serves until the socket closes or `.shutdown`, then the process exits [HelperServer.swift:run] [HelperServer.swift:start].
- Authentication happens before any command. The session directory must be a real directory owned by this uid with no group or other permission bits (`lstat`, mode `& 0o077 == 0`), and the token file must hold at least 32 characters. The token is unlinked when read and never travels in argv. The first request must be `.hello` with the current protocol version, and `ComputerUsePeerPolicy.accepts` must pass (uid, signed bundle path of the containing app, token). A refusal sends `.unauthorized` and exits 77. A bad session exits 77 and a failed bind exits 71 [HelperServer.swift:takeToken] [HelperServer.swift:run].
- The peer's path and signature validity come from the socket's audit token (`LOCAL_PEERTOKEN`) through `SecCodeCopyGuestWithAttributes`, so a recycled pid cannot stand in. Failure yields an invalid peer [HelperServer.swift:peerInfo].
- The socket file is chmod 0600, the backlog is 1, and an overlong `sockaddr_un` path is refused [HelperServer.swift:listen].
- Requests run one at a time on the main actor. The blocking server thread hops to it with a semaphore [HelperServer.swift:onMain] [HelperService.swift:HelperService].
- Input commands (click, type, key, scroll) require: not stopped (`stoppedByUser` otherwise), Accessibility trusted, and the target app brought to front (up to 20 x 50 ms polls, else `notFrontmost`). `lastInputAt` is then stamped [HelperService.swift:perform].
- Pointer actions require the topmost window under the point (ignoring the helper's own windows) to belong to the target pid, else `outsideWindow`. Typing and keys require the app to still be frontmost before every chunk. Text is posted in chunks of at most 20 UTF-16 units [InputSynthesizer.swift:requireTopWindow] [InputSynthesizer.swift:requireFrontmost] [InputSynthesizer.swift:type].
- A left single click on an element ref tries `AXPress` first, then falls back to the element's frame midpoint. A missing ref gives `staleReference`. Refs live only until the next snapshot of the same app (stored per bundle id) [HelperService.swift:perform] [AXSnapshotter.swift:snapshot(app:)].
- Secure text field values are never read (rendered `<masked>`). Labels clip at 120 characters, values at 200, indentation caps at 12 levels, and the tree is cut with a truncation line at the node or character limits [AXSnapshotter.swift:snapshot(app:)].
- Screenshots capture one app's frontmost normal window (layer 0, over 40 pt each way), never the display, and are preceded by `CGPreflightScreenCaptureAccess` because a `SCShareableContent` query would prompt. Capture size uses the filter's point-pixel scale, with no cursor [HelperService.swift:perform] [WindowCapturer.swift:capture(pid:)] [InputSynthesizer.swift:frontWindow(pid:)].
- Status reads never prompt. macOS permission prompts occur only in `requestPermission` (Accessibility prompt, else open the Privacy pane, and likewise for Screen Recording). The app sends it only from the user's Allow click [HelperService.swift:status] [HelperService.swift:requestPermission].
- Kill switch: a global Escape key monitor (keyCode 0x35) stops agent control only if input was posted within the last 10 seconds and the event does not carry `ComputerUseHelperLayout.syntheticEventMarker`. Every posted event carries that marker (set in `eventSourceUserData`). The Stop menu item also stops. `.resume` clears it. The status icon and title reflect the state [HelperService.swift:installKillSwitch] [InputSynthesizer.swift:post] [HelperController.swift:update].
- Events use a private `CGEventSource` state and post to `.cghidEventTap`.
- The helper does not enforce app grants or tiers. It trusts the app, which is why peer authentication matters.

## Side Effects
- Sockets and disk: creates, binds and unlinks `<session-dir>/<socket>`, deletes the token file, and writes diagnostics to stderr [HelperServer.swift:run] [HelperServer.swift:log].
- Posts synthetic mouse, keyboard and scroll events through CoreGraphics, moves the cursor, and activates apps [InputSynthesizer.swift].
- Reads other apps' accessibility trees and on-screen window lists, and captures windows with ScreenCaptureKit [AXSnapshotter.swift] [WindowCapturer.swift].
- NSWorkspace: lists running apps, opens apps (`activates = true`), and opens `x-apple.systempreferences:` privacy URLs [HelperService.swift:openApp] [HelperService.swift:openPrivacyPane].
- Global NSEvent monitor, an NSStatusItem, a dedicated server thread, and `exit()` on quit or connection end [HelperService.swift:installKillSwitch] [HelperController.swift] [HelperServer.swift:start].
- None for network, Keychain, UserDefaults or NotificationCenter.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
- `scripts/verify-app-bundle.sh` runs the helper's `--self-test` (per the comment in `main.swift`).
