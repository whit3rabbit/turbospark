---
uuid: "669ebe08-526f-4425-8b1d-8690f37bacb3"
title: "TurboSparkApp: build, UI, and code-quality gotchas"
summary: "A swift run build and the installed .app use different preferences domains and bundle identities, so settings appear to reset between them"
tags: ["swift", "app", "gotchas"]
depends_on: ["f872646f-6cbf-44b4-a2bd-85a13de114f8"]
source: "swift/CLAUDE.md"
created: "2026-09-05"
updated: "2026-09-05"
---

## What build, UI, and code-quality traps has this app already hit?

Smaller traps this app's history has already paid for once, cheap to
avoid and expensive to rediscover.

## Don't

- Don't expect `@AppStorage` settings to carry over between a `swift run`
  build and the installed `.app`. `swift run` writes
  `~/Library/Preferences/TurboSparkApp.plist`, the bundle writes
  `com.whit3rabbit.turbospark.plist`, so opening the installed app for the
  first time reads as a full settings reset. The three JSON stores under
  Application Support are unaffected (hardcoded paths, no bundle input). A
  `swift run` binary also has no bundle identifier, so it cannot be driven
  by UI automation even while frontmost.
- Don't bump `TurboSpark/Package.swift`'s or `TurboSparkApp/Package.swift`'s
  deployment target alone. `scripts/swift-lib.sh` pins a THIRD number
  (`MACOSX_DEPLOYMENT_TARGET=13.0`), and `LSMinimumSystemVersion` plus both
  Homebrew casks' `depends_on macos:` are a fourth and fifth copy of a
  related number. Change one, change all of them together.
- Don't assume a permission default is what one struct's initializer says.
  `AppProject.init` and its decode fallback both defaulted to the
  conservative `AppProjectPermissions.standard` (`terminal: .ask`), but the
  ONLY sheet that actually creates a project seeded the permissive `.auto`
  (`terminal: .allow`, `fileWrite: .allow`, `mcp: .allow`), so every real
  project ran model-proposed shell commands with no prompt. A default
  spelled at N call sites is only as safe as the one a user actually
  reaches.
- Don't assume a class initializer that throws part-way runs `deinit`.
  `TurboSparkSession.init` used to assign `handle` then read session info,
  and a failure on the read left an instance Swift never finished
  initializing, so `deinit` never ran and `ts_session_close` never freed
  the mapped weights, KV cache, or compiled Metal pipelines. Do fallible
  work BEFORE assigning a stored C handle, or close it by hand on failure.
- Don't trust a settings control just because it renders. Six
  `AppearanceManager` accessors had zero or one real caller, moving
  nothing else in the app against roughly 864 hardcoded `.font(...)`
  calls. Underneath sat a real bug too: `NSApp.effectiveAppearance` is
  application-level and isn't moved by `.preferredColorScheme`, which sets
  only the window's, so forcing Light on a dark system could still draw a
  dark accent from the wrong config. Audit a settings surface by grepping
  for the ACCESSOR, not the setting name.
- Don't give a `TextEditor` a height with a ZStack sizer or
  `.frame(minHeight:maxHeight:)`. A ZStack sizes to its largest child (the
  editor itself), and a flexible frame just clamps into range and lands on
  `maxHeight`. Both look correct in review and are only wrong in a
  screenshot. `PromptComposerView.editor` measures instead: a hidden
  `Text` with matching font/insets reports its ideal height through a
  `PreferenceKey`, and the real editor clamps to
  `min(max(measured, floor), ceiling)`.
