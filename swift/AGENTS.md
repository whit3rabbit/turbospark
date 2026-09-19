# swift/

This is the macOS product surface. It includes the Swift package, bindings,
TurboSparkApp, and release scripts.

Keep the seams explicit. Rust ABI changes, staged headers, Swift wrappers, and
app packaging can fail independently even when one layer builds cleanly.

Short seams. Verify each one before release.

## Read first

- [Detailed archived Swift guide](../.claude/docs/modules/swift.md)
- [Swift tool implementation](docs/SWIFT_TOOLS.md)
- [Swift skills](docs/SWIFT_SKILLS.md)
- [Swift storage](docs/storage.md)
- [Swift localization](docs/SWIFT_LOCALIZATION.md)
- [Swift keyboard and accessibility](docs/KEYBOARD_SHORTCUTS.md)
- [Rust binding contract](../docs/SWIFT_BINDINGS.md)
- [Verification reference](../.claude/docs/verification.md)

## Build order

```sh
make swift-lib
make swift-test
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
- Localization changes go through the string catalog and parity workflow.
  Do not add ad hoc strings to a view.
- Keep user data and model storage scoped through the documented profile and
  storage abstractions. Do not invent another `~/.turbospark` path.
- Accessibility and keyboard changes must follow the existing shortcuts and
  VoiceOver contracts.

## Checks

```sh
make swift-lib
make swift-test
```

Use the real-install and app-bundle gates from the linked verification pages
for runtime or release changes.
