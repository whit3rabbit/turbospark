# TurboSparkAppResources

## Overview
A resource-only SwiftPM target that holds the macOS app's bundled assets and the hand-written lookup for finding them. It exists as its own target for two reasons recorded in the code and `Package.swift`: a target with resources always gets a generated `Bundle.module` that cannot be overridden in place, and the `TurboSparkApp` executable must declare no resources itself or SwiftPM would generate a `Bundle.module` that collides with the forwarding one in `Theme/ResourceBundle.swift`. The single Swift file resolves the resource bundle without trusting the toolchain's generated accessor, which on Xcode 16's native build system only looks at the `.app` root and at a build-machine path, and crashed releases on other Macs. Resources are processed with `.process("Resources")`.

## Interface
- `AppResources.bundle`: the resource bundle, `TurboSparkApp_TurboSparkAppResources.bundle` under `Bundle.main.resourceURL` when present, otherwise `Bundle.module` [AppResources.swift:AppResources].
- Bundled content under `Resources/` (not Swift, listed by what is present): localized `*.lproj` folders holding `Localizable.strings` and `Localizable.stringsdict` (ar, ca, cs, da, de, el, en, es, fi, fr, he, hi, id, it, ja, ko, nb, nl, pl, pt-BR, pt-PT, ru, sv, th, tr, uk, vi, zh-Hans, zh-Hant), `Fonts` with their OFL or license texts, `Icons`, `ToolIcons` (with `manifest.json` naming the Lucide revision), `SiteIcons` (`sites.json`, `site-icon-manifest.json`), `Logos`, `Licenses` (SQLCipher, FluidAudio), character and app icon images, and JSON or data files `app-prompts.json`, `greetings.json`, `image-styles.json`, `permission-gate-oracle.json`, `permission-gate.bin`, and `workflow-facade-v1.md`.
- Child directories: none with an EGM.md.

## Invariants
- The lookup tries `Bundle.main.resourceURL/<bundle name>` first and touches `Bundle.module` only as the fallback, so a toolchain whose generated accessor would crash is never asked. The fallback serves `swift run` and the test runner, where the bundle sits next to the build products [AppResources.swift:bundle].
- The bundle name string `TurboSparkApp_TurboSparkAppResources.bundle` must match SwiftPM's `<package>_<target>` naming and sit under `Bundle.main.resourceURL` (the `.app`'s `Contents/Resources`). The file explains that putting the bundle at the `.app` root is impossible because `codesign` refuses extra root content and symlinks [AppResources.swift].
- Do not add `resources:` to the `TurboSparkApp` target (it would produce a colliding `Bundle.module`) [Package.swift comment].
- Strings are looked up through `Text(..., bundle: .module)` and `AppStrings.localized` in app code, and CLAUDE.md requires every new UI string to keep the ten-locale parity gates passing. Adding a locale or key here is a change to those gates, not just to a file. This directory does not itself enforce parity.
- Several resources are data consumed by code elsewhere (`permission-gate.bin` with `permission-gate-oracle.json`, `image-styles.json`, `ToolIcons/manifest.json`). Their formats are owned by their readers, which are outside this directory and were not read here.

## Side Effects
- Constructs a `Bundle` from a URL once, lazily, on first access to `AppResources.bundle` (static initializer) [AppResources.swift:bundle].
- None for disk writes, network, Keychain, UserDefaults, process spawn or notifications.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
