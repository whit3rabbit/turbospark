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
- [Design system](../docs/design-system/README.md): tokens, component
  guides, and contrast figures for every theme. Use it with DESIGN.md for
  exact values. Never paste its hex values into Swift: use the theme
  accessors it maps them to.
- [Detailed archived Swift guide](../.claude/docs/modules/swift.md)
- [Swift tool implementation](docs/SWIFT_TOOLS.md)
- [Swift skills](docs/SWIFT_SKILLS.md)
- [Swift storage](docs/storage.md)
- [Swift memory architecture](docs/SWIFT_MEMORY.md)
- [Syntext code search](docs/SYNTEXT.md) and
  [conversation recall (Syntext context engine)](docs/SYNTEXT_CONTEXT.md)
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
- Resources live in the `TurboSparkAppResources` target
  (`Sources/TurboSparkAppResources/Resources`), and the `TurboSparkApp` target
  declares none. `Bundle.module` in the app is a hand-written forwarder
  (`Theme/ResourceBundle.swift`) to `AppResources.bundle`, which checks
  `Bundle.main.resourceURL` first. Do not add `resources:` to `TurboSparkApp`:
  SwiftPM then generates its own `Bundle.module`, which collides, and on Xcode
  16.4's native build system it looks only at the `.app` root and a
  build-machine path, which crashed v0.2.0 and v0.3.0 on launch. The bundle
  cannot move to the root instead: `codesign` rejects anything there but
  `Contents/`, symlinks included.
- A bundle that packages is not a bundle that starts. Run
  `scripts/verify-app-bundle.sh` (it launches the built app and checks every
  Mach-O's deployment target) after any change to bundle layout, resources,
  `Package.swift`, or the Xcode pin. CI runs it before the DMG is built.
- syntext is pinned `exact:` from 2.7.1 on. Never pin an exact tag up to
  2.7.0: those tags pin the previous release's xcframework, and
  `exact: "2.5.0"` linked 2.4.0 with no error. For one of those, pin its
  `swift-vX.Y.Z` commit by `revision:`. After any bump,
  `.build/workspace-state.json` must name the matching zip
  (`docs/SYNTEXT.md` section 6). An archive member built above the 14.0 floor
  is only an ld warning, which `verify-app-bundle.sh` cannot see.
- Never write or read the Keychain at launch for a feature the user has not
  turned on. The app is ad-hoc signed, so every new build is a different
  identity from the one that created an item, and macOS asks again on each
  update. Storage is opt-in per credential (the TypeSafe key:
  `TypeSafeKeychain.isKeychainEnabled`, the server key:
  `ServerKeychain.resolveOptIn`, which keeps an item an earlier version
  stored rather than dropping the key). Probe for an item with an
  attributes-only query, never by reading the secret. Show the system's reason
  on failure (`TypeSafeKeychain.describe`), not a bare "could not save".
  Do not launch the installed or an older build to compare behavior while
  testing credential storage: builds before v0.3.2 create the TypeSafe Keychain
  item at start when it is missing and read it on every start, which recreates
  the state under test.
- macOS privacy prompts (microphone, screen and system audio recording) are
  opt-in the same way. Only `RecordingPermissions` calls
  `AVCaptureDevice.requestAccess` or `CGRequestScreenCaptureAccess`, from the
  access sheet's Allow button. `AudioCapture` and the controller only check
  status (`microphoneAccessAllowed`, `systemAudioAccessAllowed`) and throw,
  and every `SCShareableContent` query sits behind that check, because a
  query prompts when access is missing. A new capture entry point uses
  `.recordingAccessGate` (DESIGN.md, Asking for macOS access). A declined
  permission must leave import, transcription, and the other source usable.
  `RecordingPermissionsTests` scans `Sources/` for the prompt APIs.
- Test permission flows in the bundled app (`make app-bundle`), not
  `swift run`. An unbundled build is attributed to the app that launched it
  (Terminal, Xcode), so its prompts and System Settings rows name that app.
  `tccutil reset Microphone com.whit3rabbit.turbospark` (and `ScreenCapture`)
  returns the bundle to never asked. It resets the installed app too.
- The API destination has three modes. Text and Image share the TurboSpark
  Rust listener, address, and key. TypeSafe manages a separate loopback
  `openkindd` process, URL, and API key through the OpenKind Swift package.
  The key is in memory per launch unless the user opts in to Keychain storage. Keep its native playground on the typed `SystemRequest` client;
  form and raw JSON must produce the same request contract.
- The Image API and Images screen share one resident MLX image session and
  serialize generation through `ImageJobCoordinator`. Stop and detach must
  cancel queued and active served image jobs. Keep base64 PNGs out of traffic
  text previews.
- The app bundle and mounted DMG must contain an executable `openkindd`.
  Pin the OpenKind Swift package and daemon to the same published revision
  for release; a sibling package path is for development only.
  The daemon build (`openkind-server --features mlx`) needs `SDKROOT` and
  `BINDGEN_EXTRA_CLANG_ARGS="-isysroot $SDKROOT"` under Xcode 27, or
  mlx-sys's bindgen fails on `stdlib.h`. `make-app-bundle.sh` sets both.
- Keep user data and model storage scoped through the documented profile and
  storage abstractions. Do not invent another `~/.turbospark` path.
- Accessibility and keyboard changes must follow the existing shortcuts and
  VoiceOver contracts.
- Every UI or design change follows [DESIGN.md](../DESIGN.md): build from
  its shared control vocabulary (`Theme/AppControls.swift`,
  `Theme/AppSurfaces.swift`, `Theme/AppProgressBar.swift`,
  `Theme/AppPreferenceOptions.swift`, `Theme/AppStrings.swift`,
  `Theme/TurboSparkMotion.swift`), not local variants, and run its
  verification list.
- Voice cloning consent covers one exact reference region
  (`AudioCloneOptions.consentedReference`) and lapses on any retrim. A UI
  seam that sets the reference (drop, import, inline recording, reuse from a
  take) never sets or copies consent. A voice reference recording captures
  the microphone through local flags and never writes the Record page's
  capture settings. `AudioVoiceCloneWorkspaceTests` pins all three.
- The Voice Cloning page only presents models with native Metal or MLX
  acceleration (`clonesNatively`). Portable CPU-only models remain gated to
  the Advanced portable experiment and must not be presented as voice clone
  options until an accelerated backend is implemented. OmniVoice and MegaTTS3
  are Metal-accelerated and present on the Voice Cloning page.
- Audio model residency lifecycle is managed by `AudioWorkspaceController`
  (`isModelLoaded`, `canLoadModel`, `canUnloadModel`). Unloading is wired into
  `AppModel.canUnloadAnyModel` and the top bar eject button, ensuring audio
  models can be loaded and unloaded consistently with text and image sessions.
- The Sparkle updater starts only inside a real .app bundle (see
  `SparkleUpdateController`); never bypass that guard for `swift run` builds.
  Update trust is the EdDSA signature, not Apple code signing -- the bundle
  is ad-hoc signed. Bundle-script signing order ends innermost-first inside
  Sparkle.framework (XPCServices, Autoupdate, Updater.app, then the
  framework) before the app itself. The brew cask's `auto_updates true` and
  the in-app updater are one contract: either may change, not just one.
- FluidAudio is pinned `exact: "0.15.5"` ([FLUIDAUDIO.md](../docs/FLUIDAUDIO.md)).
  From v0.15.6 it links a Rust staticlib whose unwind personality the app
  link cannot take. Only `Audio/FluidAudio/` imports it, and
  `FluidAudioEngine` sets `ModelHub.offlineMode` and a blocked
  `ModelRegistry.baseURL` before any use. Left alone, FluidAudio downloads
  Hugging Face `main` and deletes a model folder after a failed load.
- Never call a FluidAudio download, cache or purge API
  (`FluidAudioIsolationTests` scans for them). Installs come from the pinned
  Rust audio catalog. A family runs only once `FluidAudioFamily.hasAdapter`
  is true for it. Check a model's inputs before handing it to FluidAudio:
  a mismatch is a `preconditionFailure` that ends the app, not an error.

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

## Open issues

- **Auto-started server has no API key on a new install**
  ([#150](https://github.com/whit3rabbit/turbospark/issues/150)). The server key is
  session-only unless the user opts in to Keychain storage
  (`ServerKeychain.resolveOptIn`), so `serverAPIKeyInput` is empty at launch.
  With "start server on launch" on (`serverAutoStartOnLaunch`,
  `AppModel+Persistence.swift`, the `server == nil` auto-start branch), the
  server then starts with `apiKey: nil` and any local process can drive the
  loaded models over loopback. The Server pane warns about it, but an
  auto-start never shows that pane. Someone who already had a stored key is
  unaffected, because the first launch adopts the existing item. Options:
  generate a per-session key at auto-start (as TypeSafe does) and put it in
  the field so Copy works, or refuse to auto-start without a key. This needs a
  decision before either is built.
- **A fresh build directory fails to link on Xcode 27.0 (beta, build 27A266a).**
  `swift build` into a new `--scratch-path` stops with "Build input file cannot
  be found: .../CSyntext.o". `CSyntext` is a header-only C target in the
  external `syntext` package and the new build system expects an object for it.
  A 336-byte placeholder `CSyntext.o` left in the shared `.build/out/Products/*`
  hides it, which is why it only shows up on a clean build. CI uses released
  Xcodes and is not affected. Workaround for a scratch build: copy that file
  into `<scratch>/out/Products/Debug/`. Revisit when Xcode 27 ships or syntext
  adds a source file to the target.
