# swift/

This is the macOS product surface. It includes the Swift package, bindings,
TurboSparkApp, and release scripts.

Keep the seams explicit. Rust ABI changes, staged headers, Swift wrappers, and
app packaging can fail independently even when one layer builds cleanly.

Short seams. Verify each one before release.

## Read first

- [Design rules](../DESIGN.md): ALWAYS read before any UI or design change
  (views, controls, copy, color, type, layout, menus, shortcuts, motion).
  It covers accessibility, theming, localization, and layout.
- [Detailed archived Swift guide](../.claude/docs/modules/swift.md)
- [Swift tool implementation](docs/SWIFT_TOOLS.md)
- [Swift skills](docs/SWIFT_SKILLS.md)
- [Swift storage](docs/storage.md)
- [Swift memory architecture](docs/SWIFT_MEMORY.md)
- [Swift localization](docs/SWIFT_LOCALIZATION.md)
- [Swift keyboard and accessibility](docs/KEYBOARD_SHORTCUTS.md)
- [Swift API workspace](docs/SWIFT_API_WORKSPACE.md)
- [Automatic app updates](docs/AUTO_UPDATE.md)
- [Download manager](docs/DOWNLOAD_MANAGER.md)
- [Rust binding contract](../docs/SWIFT_BINDINGS.md)
- [Verification reference](../.claude/docs/verification.md)

## Build order

```sh
make swift-lib
make swift-test
make swift-test-app
make swift-test-qwenimage
make swift-test-real MODEL=/path/to/model
make swift-app
make app-bundle
make dmg
```

`make swift-lib` stages the static library and handwritten header into the
Swift package. Run it before SwiftPM tests when the Rust ABI or generated
staging files changed.

## Rules

- Keep the Rust ABI, staged header, Swift wrapper, and capability predicates
  synchronized. A Rust-only test does not validate the handwritten header.
- The app is a full multi-chat product with persistence, projects, tools,
  model discovery, attachments, localization, and phase inspection. Do not
  describe it as a minimal demo.
- Keep app bundle and DMG creation in the release scripts. Verify the mounted
  DMG contents, not only the packager exit code.
- Localization changes go through the string catalog and ten parity gates
  (`swift/docs/SWIFT_LOCALIZATION.md`). Key-only initializers outside `Text`
  must never take bare string literals. Use `AppStrings.localized("Key")` for
  UI strings without `Text`. `Localization/Info.plist` declares
  `CFBundleLocalizations` for all 21 languages so menu bar and system dialogs
  follow the in-app picker after relaunch.
- Store durability: `AppJSONStore` quarantines unreadable or corrupt JSON
  files (`corrupt-...`) instead of overwriting them on save, preventing silent
  data loss.
- Generation menu shortcuts: `Cmd+Return` and `Cmd+.` are scoped by workspace.
  `runsChatGenerationCommand` enables chat generation only in Chat. Full-bleed
  workspaces (`.images`, `.audio`, `.humanize`) declare `ownsGenerationShortcuts`
  to bind those chords to their own run/stop controls.
- The API destination has three modes. Text and Image share the TurboSpark
  Rust listener, address, and key. TypeSafe manages a separate loopback
  `openkindd` process, URL, and Keychain key through the OpenKind Swift
  package. Keep its native playground on the typed `SystemRequest` client;
  form and raw JSON must produce the same request contract.
- The Image API and Images screen share one resident MLX image session and
  serialize generation through `ImageJobCoordinator`. Stop and detach must
  cancel queued and active served image jobs. Keep base64 PNGs out of traffic
  text previews.
- The app bundle and mounted DMG must contain an executable `openkindd`.
  Pin the OpenKind Swift package and daemon to the same published revision
  for release; a sibling package path is for development only.
- Keep user data and model storage scoped through the documented profile and
  storage abstractions. Do not invent another `~/.turbospark` path.
- Accessibility and keyboard changes must follow the existing shortcuts and
  VoiceOver contracts.
- Every UI or design change follows [DESIGN.md](../DESIGN.md): build from
  its shared control vocabulary (`Theme/AppControls.swift`,
  `Theme/AppPreferenceOptions.swift`, `Theme/AppStrings.swift`,
  `Theme/TurboSparkMotion.swift`), not local variants, and run its
  verification list.
- The Sparkle updater starts only inside a real .app bundle (see
  `SparkleUpdateController`); never bypass that guard for `swift run` builds.
  Update trust is the EdDSA signature, not Apple code signing -- the bundle
  is ad-hoc signed. Bundle-script signing order ends innermost-first inside
  Sparkle.framework (XPCServices, Autoupdate, Updater.app, then the
  framework) before the app itself. The brew cask's `auto_updates true` and
  the in-app updater are one contract: either may change, not just one.

## Checks

```sh
make swift-lib
make swift-test
```

`make swift-test` covers only `swift/TurboSpark`. Run `make swift-test-app`
(TurboSparkApp; relinks the staged archive with localized symbols, so run it
after, not interleaved with, `make swift-test`) and
`make swift-test-qwenimage` (Vendor/QwenImage) for app or vendor changes.

Use the real-install and app-bundle gates from the linked verification pages
for runtime or release changes.
