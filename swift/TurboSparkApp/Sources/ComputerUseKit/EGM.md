# ComputerUseKit

## Overview
The wire contract between the TurboSpark app and the computer-use helper app. Both link this module, so they cannot disagree about a message's shape. It is Foundation-only on purpose: the helper links nothing else of TurboSpark, so its binary (and with it its macOS Accessibility and Screen Recording grants) changes only when this module or the helper's own source changes (per the comment in `Package.swift`). It holds the message enums, length-prefixed JSON framing over a blocking socket, a key-combo parser, and the peer authentication policy the helper applies. Consumers: [ComputerUse](../TurboSparkApp/ComputerUse/EGM.md) (app side) and [TurboSparkComputerUse](../TurboSparkComputerUse/EGM.md) (helper).

## Interface
- Messages: `ComputerUseRequest` (`hello`, `status`, `requestPermission`, `command`, `resume`, `shutdown`), `ComputerUseResponse` (`welcome`, `status`, `result`, `error`, `ok`), `ComputerUseErrorKind`, `ComputerUsePermissionKind`, `ComputerUseHelperStatus`, `ComputerUsePortError` [ComputerUseWire.swift:ComputerUseRequest] [ComputerUseWire.swift:ComputerUseResponse].
- Layout constants: `ComputerUseHelperLayout` (bundle id `com.whit3rabbit.turbospark.computeruse`, bundle and executable names, `relativeBundlePath`, socket and token file names, `syntheticEventMarker`) [ComputerUseWire.swift:ComputerUseHelperLayout].
- Commands: `ComputerUseCommand`, `ComputerUseCommandKind`, `ComputerUseTarget` (`reference` or `point`), `ComputerUseMouseButton`, `ComputerUseScrollDirection`, `ComputerUseRect`, `ComputerUseRunningApp`, `ComputerUseCapture`, `ComputerUseResultValue`, `ComputerUseBackendManifest`, `ComputerUseProtocol.currentVersion` [ComputerUseProtocol.swift:ComputerUseCommand] [ComputerUseProtocol.swift:ComputerUseBackendManifest] [ComputerUseProtocol.swift:ComputerUseProtocol].
- Port abstraction: `ComputerUseControlPort` protocol (`perform`, default no-op `resumeAfterStop`) [ComputerUseProtocol.swift:ComputerUseControlPort].
- Framing: `ComputerUseFraming` (`encode`, `decodeFrames`, `write`, `read`, `disableSigpipe`, `maximumFrameBytes`, `FramingError`) [ComputerUseFraming.swift:ComputerUseFraming].
- Keys: `ComputerUseKeyCombo` with `Modifiers` and `parse(_:)` [ComputerUseKeyCombo.swift:ComputerUseKeyCombo].
- Peer auth: `ComputerUsePeer`, `ComputerUsePeerPolicy` (`accepts`, `containingAppBundle(helperBundleURL:)`) [ComputerUsePeerPolicy.swift:ComputerUsePeerPolicy].
- Child directories: none.

## Invariants
- `ComputerUseProtocol.currentVersion` (1) must be bumped whenever a command or result changes shape. A manifest with another version is not current, so `supports(_:)` is false for every command and the helper gets no tools (fail closed) [ComputerUseProtocol.swift:currentVersion] [ComputerUseProtocol.swift:supports(_:)].
- `hello` must be the first message on a connection and the helper drops a connection that starts otherwise. One response per request, in order [ComputerUseWire.swift:ComputerUseRequest] [ComputerUseWire.swift:ComputerUseResponse].
- Frames are a 4-byte big-endian length then JSON, capped at 48 MiB in `encode`, `decodeFrames` and `read`, which throw `tooLarge` rather than allocate. A zero-byte read gives `closed`, and `EINTR` is retried. `disableSigpipe` must be applied by both ends to every socket they own, otherwise a dead peer raises SIGPIPE and kills the process [ComputerUseFraming.swift].
- Coordinates are global screen points with a top-left origin (the CGEvent and AX space). A `point` target is already mapped from screenshot pixels. A `reference` comes from the latest snapshot [ComputerUseProtocol.swift:ComputerUseTarget].
- `ComputerUseCommand.bundleID` is nil only for `listApps`. Adding a command requires adding it to `ComputerUseCommandKind`, `kind`, `bundleID`, and the policy and helper switch statements (the helper's manifest advertises `ComputerUseCommandKind.allCases`) [ComputerUseProtocol.swift:ComputerUseCommand].
- Peer policy accepts only when the peer uid equals the helper's, the peer's code signature is valid, the peer's canonicalized (symlink-resolved) bundle path equals the canonicalized expected containing app bundle, the expected token is non-empty, and the token matches in constant time. A helper not installed at `<App>.app/Contents/Helpers/<Helper>.app` has no containing bundle and accepts nobody. Known limit stated in the file: the app is ad-hoc signed, so "validly signed" cannot pin a developer identity [ComputerUsePeerPolicy.swift:accepts] [ComputerUsePeerPolicy.swift:containingAppBundle(helperBundleURL:)].
- `syntheticEventMarker` (0x54534355) tags every event the helper posts so its kill switch ignores them. Both sides must keep the value in step [ComputerUseWire.swift:syntheticEventMarker].
- Key combos use the ANSI layout's virtual key codes (letters by position). Modifier names outside cmd/command/meta/super, shift, alt/option/opt and ctrl/control make `parse` return nil, as does an unknown key name. Text input goes through `type`, not this table [ComputerUseKeyCombo.swift:parse(_:)].
- `ComputerUseControlPort.resumeAfterStop` is called only when the user approves a new app grant, never by the model [ComputerUseProtocol.swift:ComputerUseControlPort].

## Side Effects
- Framing performs blocking `read`, `write` and `setsockopt` calls on caller-supplied file descriptors [ComputerUseFraming.swift:write] [ComputerUseFraming.swift:read] [ComputerUseFraming.swift:disableSigpipe].
- Peer policy resolves symlinks on paths (filesystem lookups) [ComputerUsePeerPolicy.swift:canonical].
- None for network, Keychain, UserDefaults, process spawn, notifications or timers. Everything else is pure value types.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
