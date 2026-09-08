import AppKit
import SwiftUI
@testable import TurboSparkApp
import XCTest

@MainActor
final class AppearanceSettingsTests: XCTestCase {
    func testThemePresetsAvailable() {
        XCTAssertFalse(ThemePreset.presets.isEmpty)
        let codex = ThemePreset.presets.first(where: { $0.id == "codex" })
        XCTAssertNotNil(codex)
        XCTAssertEqual(codex?.light.preset, "Codex")
        XCTAssertEqual(codex?.dark.preset, "Codex")

        let emerald = ThemePreset.presets.first(where: { $0.id == "turbospark" })
        XCTAssertNotNil(emerald)
        XCTAssertEqual(emerald?.light.accentName, "Emerald")
    }

    /// A fresh install gets Spark Blue, through BOTH doors into the archive.
    ///
    /// Decoding `{}` is the honest fixture here: it exercises the memberwise
    /// defaults and `init(from:)`'s `decodeIfPresent ?? .default*` fallback,
    /// which is the pair a first launch actually takes. Calling
    /// `AppearanceFileStore.load()` instead would read (and on save, rewrite)
    /// the real user's `appearance.json` under `AppStorageRoot`.
    ///
    /// **A persisted config still wins**, which is the whole reason this is
    /// worth pinning: changing the default moves nobody who has ever opened
    /// the Appearance pane, so the constant below is the only thing that
    /// decides what a new install looks like.
    func testAFreshArchiveDefaultsToSparkBlue() throws {
        let archive = try JSONDecoder().decode(
            AppearanceArchive.self, from: Data("{}".utf8))

        for (mode, config) in [("light", archive.lightConfig), ("dark", archive.darkConfig)] {
            XCTAssertEqual(config.preset, "Spark Blue", "\(mode) default preset")
            XCTAssertEqual(config.accentName, "Spark", "\(mode) default accent name")
        }
        XCTAssertEqual(archive.lightConfig.accentHex, "#0E7C99")
        XCTAssertEqual(archive.darkConfig.accentHex, "#5FD8E8")
        // The blue cast is load-bearing: a neutral #181818 is what made the
        // cyan accent read as pasted on.
        XCTAssertEqual(archive.darkConfig.backgroundHex, "#12151B")

        XCTAssertEqual(archive.lightConfig, ThemeModeConfig.defaultLight)
        XCTAssertEqual(archive.darkConfig, ThemeModeConfig.defaultDark)
    }

    /// A preset names a set of colours; it never points at `default*`.
    ///
    /// The Codex entry used to BE `light: .defaultLight, dark: .defaultDark`,
    /// so it was not a preset at all, it was a second name for whatever the
    /// default happened to be. Changing the default would have silently
    /// rewritten Codex into Spark Blue and left two identical entries in the
    /// picker. This is the guard for that shape, not for the values.
    func testNoPresetAliasesTheDefaultExceptSparkBlue() {
        for preset in ThemePreset.presets where preset.id != "sparkblue" {
            XCTAssertNotEqual(
                preset.light, ThemeModeConfig.defaultLight,
                "\(preset.name) light tracks the default instead of naming its own colours")
            XCTAssertNotEqual(
                preset.dark, ThemeModeConfig.defaultDark,
                "\(preset.name) dark tracks the default instead of naming its own colours")
        }

        // Codex specifically, because it is the one that was aliased and the
        // one users may have persisted.
        let codex = ThemePreset.presets.first(where: { $0.id == "codex" })
        XCTAssertEqual(codex?.dark.accentHex, "#F3F4F6")
        XCTAssertEqual(codex?.dark.backgroundHex, "#181818")
        XCTAssertEqual(codex?.light.accentHex, "#111827")
    }

    func testColorHexParsingAndFormatting() {
        let hexWhite = "#FFFFFF"
        let colorWhite = Color(hex: hexWhite)
        XCTAssertNotNil(colorWhite)

        let hexBlue = "#2563EB"
        let colorBlue = Color(hex: hexBlue)
        XCTAssertNotNil(colorBlue)

        let reHex = colorBlue?.toHex()
        XCTAssertNotNil(reHex)
        XCTAssertEqual(reHex?.uppercased(), "#2563EB")
    }

    func testAppearanceManagerPresetApplication() {
        let manager = AppearanceManager.shared

        if let midnight = ThemePreset.presets.first(where: { $0.id == "midnight" }) {
            manager.applyPreset(midnight, forMode: false)
            XCTAssertEqual(manager.lightConfig.preset, "Midnight Blue")
            XCTAssertEqual(manager.lightConfig.accentName, "Blue")

            // Test Dark mode preset application
            manager.applyPreset(midnight, forMode: true)
            XCTAssertEqual(manager.darkConfig.preset, "Midnight Blue")
        }

        // Restore default
        if let codex = ThemePreset.presets.first(where: { $0.id == "codex" }) {
            manager.applyPreset(codex)
            XCTAssertEqual(manager.lightConfig.preset, "Codex")
            XCTAssertEqual(manager.darkConfig.preset, "Codex")
        }
    }

    func testFontWeightResolution() {
        XCTAssertEqual(Font.Weight.fromName("regular"), .regular)
        XCTAssertEqual(Font.Weight.fromName("medium"), .medium)
        XCTAssertEqual(Font.Weight.fromName("semibold"), .semibold)
        XCTAssertEqual(Font.Weight.fromName("bold"), .bold)
        XCTAssertEqual(Font.Weight.fromName("unknown"), .regular)
    }

    func testReduceMotionResolution() {
        let manager = AppearanceManager.shared
        manager.reduceMotion = .system
        XCTAssertFalse(manager.shouldReduceMotion(systemReduceMotion: false))
        XCTAssertTrue(manager.shouldReduceMotion(systemReduceMotion: true))

        manager.reduceMotion = .on
        XCTAssertTrue(manager.shouldReduceMotion(systemReduceMotion: false))
        XCTAssertTrue(manager.shouldReduceMotion(systemReduceMotion: true))

        manager.reduceMotion = .off
        XCTAssertFalse(manager.shouldReduceMotion(systemReduceMotion: false))
        XCTAssertFalse(manager.shouldReduceMotion(systemReduceMotion: true))

        // Reset to system
        manager.reduceMotion = .system
    }

    func testDockIconSelection() {
        let manager = AppearanceManager.shared
        manager.dockIcon = .emeraldSpark
        XCTAssertEqual(manager.dockIcon, .emeraldSpark)
        XCTAssertEqual(manager.dockIcon.label, "TurboSpark")

        manager.dockIcon = .codexDark
        XCTAssertEqual(manager.dockIcon, .codexDark)
        XCTAssertEqual(manager.dockIcon.label, "Codex")

        // Reset
        manager.dockIcon = .emeraldSpark
    }

    // MARK: - Accent picker

    /// A `Picker` whose selection matches no tag renders BLANK.
    ///
    /// The curated six accents did not cover the shipped presets, so 4 of the
    /// 10 preset-and-mode combinations drew an EMPTY Accent control -- Codex
    /// in both modes, and Codex is the default, so this is what a fresh
    /// install showed. It reads as a broken control, not as an unlisted color.
    func testEveryShippedPresetIsSelectableInTheAccentPicker() {
        for preset in ThemePreset.presets {
            for (isDark, config) in [(false, preset.light), (true, preset.dark)] {
                let options = AccentOption.options(
                    isDark: isDark,
                    selectedHex: config.accentHex,
                    selectedName: config.accentName)
                XCTAssertTrue(
                    options.contains(where: { $0.hex == config.accentHex }),
                    "\(preset.name) (\(isDark ? "dark" : "light")) accent \(config.accentHex) "
                        + "has no matching tag, so the picker renders blank")
            }
        }
    }

    func testACustomAccentIsCarriedRatherThanDropped() {
        let options = AccentOption.options(
            isDark: true, selectedHex: "#ABCDEF", selectedName: "Custom")
        XCTAssertEqual(options.count, AccentOption.curated(isDark: true).count + 1)
        XCTAssertEqual(options.last?.hex, "#ABCDEF")
        XCTAssertEqual(options.last?.name, "Custom")

        // A curated value must NOT be duplicated into the list.
        let emerald = AccentOption.curated(isDark: true)[0].hex
        XCTAssertEqual(
            AccentOption.options(isDark: true, selectedHex: emerald, selectedName: "Emerald").count,
            AccentOption.curated(isDark: true).count)
    }

    func testAccentNameFollowsTheHexAndFallsBackToCustom() {
        XCTAssertEqual(AccentOption.name(forHex: "#6ABA71", isDark: true), "Emerald")
        XCTAssertEqual(AccentOption.name(forHex: "#237D32", isDark: false), "Emerald")
        // Emerald's dark hex is not a light option, so in light mode it is custom.
        XCTAssertEqual(AccentOption.name(forHex: "#6ABA71", isDark: false), "Custom")
        XCTAssertEqual(AccentOption.name(forHex: "#ABCDEF", isDark: true), "Custom")
    }

    // MARK: - Mode resolution

    /// The bug this whole environment value exists to fix.
    ///
    /// `TurboSparkTheme` branched on `NSApp.effectiveAppearance`, which is
    /// APPLICATION-level and is not moved by SwiftUI's `.preferredColorScheme`.
    /// Forcing the app to Light on a dark system therefore drew light chrome
    /// out of `darkConfig`. The forced cases are the ones that carry the file:
    /// a test that only exercised `.system` would pass against the old code.
    func testAppearanceModeIgnoresTheSystemSchemeWhenItIsForced() {
        XCTAssertFalse(AppAppearance.light.isDark(systemColorScheme: .dark))
        XCTAssertTrue(AppAppearance.dark.isDark(systemColorScheme: .light))

        XCTAssertTrue(AppAppearance.system.isDark(systemColorScheme: .dark))
        XCTAssertFalse(AppAppearance.system.isDark(systemColorScheme: .light))
    }

    func testResolvedThemeTakesItsConfigFromTheForcedModeNotTheSystem() {
        let manager = AppearanceManager.shared
        let savedAppearance = manager.appearance
        let savedLight = manager.lightConfig
        let savedDark = manager.darkConfig
        defer {
            manager.appearance = savedAppearance
            manager.lightConfig = savedLight
            manager.darkConfig = savedDark
        }

        manager.lightConfig.foregroundHex = "#111111"
        manager.lightConfig.contrast = 40
        manager.darkConfig.foregroundHex = "#EEEEEE"
        manager.darkConfig.contrast = 80

        // App forced Light while the system renders Dark.
        manager.appearance = .light
        let forcedLight = ResolvedAppTheme.resolve(
            manager: manager, colorScheme: .dark, installedFamilies: [])
        XCTAssertFalse(forcedLight.isDark)
        XCTAssertEqual(forcedLight.contrast, 40)

        // And the mirror case, so neither arm can pass by always answering one way.
        manager.appearance = .dark
        let forcedDark = ResolvedAppTheme.resolve(
            manager: manager, colorScheme: .light, installedFamilies: [])
        XCTAssertTrue(forcedDark.isDark)
        XCTAssertEqual(forcedDark.contrast, 80)

        // Following the system still works.
        manager.appearance = .system
        XCTAssertTrue(ResolvedAppTheme.resolve(
            manager: manager, colorScheme: .dark, installedFamilies: []).isDark)
        XCTAssertFalse(ResolvedAppTheme.resolve(
            manager: manager, colorScheme: .light, installedFamilies: []).isDark)
    }

    func testResolvedThemeCarriesTheSizesTheUserSet() {
        let manager = AppearanceManager.shared
        let savedUI = manager.uiFontSize
        let savedCode = manager.codeFontSize
        defer {
            manager.uiFontSize = savedUI
            manager.codeFontSize = savedCode
        }

        manager.uiFontSize = 17
        manager.codeFontSize = 11
        let theme = ResolvedAppTheme.resolve(
            manager: manager, colorScheme: .light, installedFamilies: [])

        XCTAssertEqual(theme.uiFontDescriptor.size, 17)
        XCTAssertEqual(theme.codeFontDescriptor.size, 11)
        XCTAssertTrue(theme.codeFontDescriptor.isCode)
        XCTAssertFalse(theme.uiFontDescriptor.isCode)
    }

    func testAppTextSizeScaleAndStepProgression() {
        let manager = AppearanceManager.shared
        let savedSize = manager.textSize
        defer { manager.textSize = savedSize }

        manager.textSize = .standard
        XCTAssertEqual(AppTextSize.standard.scale, 1.0)
        XCTAssertEqual(AppTextSize.large.scale, 1.15)
        XCTAssertEqual(AppTextSize.extraLarge.scale, 1.30)

        manager.makeTextBigger()
        XCTAssertEqual(manager.textSize, .large)

        manager.makeTextBigger()
        XCTAssertEqual(manager.textSize, .extraLarge)

        manager.makeTextBigger()
        XCTAssertEqual(manager.textSize, .extraLarge)

        manager.makeTextSmaller()
        XCTAssertEqual(manager.textSize, .large)

        manager.makeTextSmaller()
        XCTAssertEqual(manager.textSize, .standard)

        manager.makeTextSmaller()
        XCTAssertEqual(manager.textSize, .standard)

        manager.textSize = .extraLarge
        manager.resetTextSize()
        XCTAssertEqual(manager.textSize, .standard)
    }

    func testResetTextSizeRestoresDefaults() {
        let manager = AppearanceManager.shared
        let savedUI = manager.uiFontSize
        let savedCode = manager.codeFontSize
        let savedText = manager.textSize
        defer {
            manager.uiFontSize = savedUI
            manager.codeFontSize = savedCode
            manager.textSize = savedText
        }

        manager.uiFontSize = 22.0
        manager.codeFontSize = 18.0
        manager.textSize = .large

        manager.resetTextSize()
        XCTAssertEqual(manager.uiFontSize, 16.0)
        XCTAssertEqual(manager.codeFontSize, 12.0)
        XCTAssertEqual(manager.textSize, .standard)
    }

    /// `resetToDefaults` restores EVERY published field, not just the ones a
    /// reset happens to mention: the pane's button claims the whole factory
    /// theme, so a field left behind would silently survive a reset. The
    /// comparison baseline is a zero-arg `AppearanceArchive()`, which IS the
    /// factory defaults, and every field is first moved OFF its default so a
    /// no-op reset cannot pass.
    func testResetToDefaultsRestoresEveryField() {
        let manager = AppearanceManager.shared
        let saved = (
            manager.appearance, manager.textSize, manager.lightConfig,
            manager.darkConfig, manager.statusBarViewMode, manager.usePointerCursors,
            manager.dockIcon, manager.reduceMotion, manager.uiFontSize,
            manager.codeFontSize, manager.diffMarkers)
        defer {
            manager.appearance = saved.0
            manager.textSize = saved.1
            manager.lightConfig = saved.2
            manager.darkConfig = saved.3
            manager.statusBarViewMode = saved.4
            manager.usePointerCursors = saved.5
            manager.dockIcon = saved.6
            manager.reduceMotion = saved.7
            manager.uiFontSize = saved.8
            manager.codeFontSize = saved.9
            manager.diffMarkers = saved.10
        }

        // Move every field away from the factory value. Two fields have
        // factory values their types cannot step away from with a raw
        // literal (statusBarViewMode .text, reduceMotion .system), so they
        // take their non-default cases directly.
        manager.appearance = .dark
        manager.textSize = .extraLarge
        manager.lightConfig = ThemePreset.presets[0].light
        manager.darkConfig = ThemePreset.presets[0].dark
        manager.statusBarViewMode = manager.statusBarViewMode == .graphs ? .text : .graphs
        manager.usePointerCursors = true
        manager.dockIcon = manager.dockIcon == .codexDark ? .emeraldSpark : .codexDark
        manager.reduceMotion = manager.reduceMotion == .off ? .on : .off
        manager.uiFontSize = 21.0
        manager.codeFontSize = 17.0
        manager.diffMarkers = manager.diffMarkers == .plusMinus ? .color : .plusMinus

        let factory = AppearanceArchive()

        manager.resetToDefaults()

        XCTAssertEqual(manager.appearance, AppAppearance.resolve(factory.appearance), "appearance")
        XCTAssertEqual(manager.textSize, AppTextSize.resolve(factory.textSize), "textSize")
        XCTAssertEqual(manager.lightConfig, factory.lightConfig, "lightConfig")
        XCTAssertEqual(manager.darkConfig, factory.darkConfig, "darkConfig")
        XCTAssertEqual(
            manager.statusBarViewMode, StatusBarViewMode(rawValue: factory.statusBarViewMode),
            "statusBarViewMode")
        XCTAssertEqual(manager.usePointerCursors, factory.usePointerCursors, "usePointerCursors")
        XCTAssertEqual(manager.dockIcon, AppDockIcon(rawValue: factory.dockIcon), "dockIcon")
        XCTAssertEqual(
            manager.reduceMotion, ReduceMotionPreference(rawValue: factory.reduceMotion),
            "reduceMotion")
        XCTAssertEqual(manager.uiFontSize, factory.uiFontSize, "uiFontSize")
        XCTAssertEqual(manager.codeFontSize, factory.codeFontSize, "codeFontSize")
        XCTAssertEqual(manager.diffMarkers, DiffMarkerPreference(rawValue: factory.diffMarkers), "diffMarkers")

        // The fonts ride inside the configs, so the two assertions above
        // already cover them -- but a reader of this test should not have to
        // derive that, and a future field split would land here first.
        XCTAssertEqual(manager.uiFontFamily, factory.lightConfig.uiFontFamily, "uiFontFamily")
        XCTAssertEqual(manager.codeFontFamily, factory.darkConfig.codeFontFamily, "codeFontFamily")
    }

    func testResolvedThemeScalesWithAppTextSize() {
        let manager = AppearanceManager.shared
        let savedUI = manager.uiFontSize
        let savedCode = manager.codeFontSize
        let savedTextSize = manager.textSize
        defer {
            manager.uiFontSize = savedUI
            manager.codeFontSize = savedCode
            manager.textSize = savedTextSize
        }

        manager.uiFontSize = 14
        manager.codeFontSize = 12

        manager.textSize = .standard
        let standardTheme = ResolvedAppTheme.resolve(
            manager: manager, colorScheme: .light, installedFamilies: [])
        XCTAssertEqual(standardTheme.uiFontDescriptor.size, 14)
        XCTAssertEqual(standardTheme.codeFontDescriptor.size, 12)
        XCTAssertEqual(standardTheme.textSize, .standard)

        manager.textSize = .large
        let largeTheme = ResolvedAppTheme.resolve(
            manager: manager, colorScheme: .light, installedFamilies: [])
        XCTAssertEqual(largeTheme.uiFontDescriptor.size, 16)
        XCTAssertEqual(largeTheme.codeFontDescriptor.size, 14)
        XCTAssertEqual(largeTheme.textSize, .large)

        manager.textSize = .extraLarge
        let extraLargeTheme = ResolvedAppTheme.resolve(
            manager: manager, colorScheme: .light, installedFamilies: [])
        XCTAssertEqual(extraLargeTheme.uiFontDescriptor.size, 18)
        XCTAssertEqual(extraLargeTheme.codeFontDescriptor.size, 16)
        XCTAssertEqual(extraLargeTheme.textSize, .extraLarge)
    }

    // MARK: - Font catalog

    func testCatalogOffersOnlyFamiliesThatCanRender() {
        // Nothing installed: only the families that name the system face survive.
        let bare = AppFontCatalog.availableCodeFamilies(installed: [])
        XCTAssertEqual(bare, ["System default", "SF Mono"])
        XCTAssertFalse(bare.contains("JetBrains Mono"))

        let withJetBrains = AppFontCatalog.availableCodeFamilies(installed: ["JetBrains Mono"])
        XCTAssertTrue(withJetBrains.contains("JetBrains Mono"))
        XCTAssertFalse(withJetBrains.contains("Fira Code"))

        // Menu order is preserved rather than being whatever the set iterates in.
        XCTAssertEqual(
            AppFontCatalog.availableUIFamilies(installed: ["Avenir", "Inter"]),
            ["System default", "SF Pro", "Inter", "Avenir"])

        let withSerifs = AppFontCatalog.availableUIFamilies(installed: ["Georgia", "Palatino", "Charter"])
        XCTAssertTrue(withSerifs.contains("Georgia"))
        XCTAssertTrue(withSerifs.contains("Palatino"))
        XCTAssertTrue(withSerifs.contains("Charter"))
    }

    /// A config naming a family that is not installed falls back BY NAME.
    /// Handing the missing name to `Font.custom` anyway draws the system face
    /// while the picker still claims the missing family, which is the silent
    /// substitution this whole change is about.
    func testAMissingFamilyFallsBackToTheSystemFace() {
        XCTAssertEqual(
            AppFontCatalog.resolveFamily(
                "Fira Code",
                offered: AppFontCatalog.offeredCodeFamilies,
                installed: []),
            "System default")

        XCTAssertEqual(
            AppFontCatalog.resolveFamily(
                "Fira Code",
                offered: AppFontCatalog.offeredCodeFamilies,
                installed: ["Fira Code"]),
            "Fira Code")

        // A family nobody offers is refused even when it IS installed, so the
        // renderer can never draw something the picker cannot show.
        XCTAssertEqual(
            AppFontCatalog.resolveFamily(
                "Comic Sans MS",
                offered: AppFontCatalog.offeredCodeFamilies,
                installed: ["Comic Sans MS"]),
            "System default")
    }

    func testResolvedThemeRefusesAFamilyThatIsNotInstalled() {
        let manager = AppearanceManager.shared
        let saved = manager.lightConfig
        defer { manager.lightConfig = saved }

        manager.lightConfig.codeFontFamily = "JetBrains Mono"

        let without = ResolvedAppTheme.resolve(
            manager: manager, colorScheme: .light, installedFamilies: [])
        XCTAssertEqual(without.codeFontDescriptor.family, "System default")
        XCTAssertNil(without.codeFontDescriptor.customFamilyName)

        let with = ResolvedAppTheme.resolve(
            manager: manager, colorScheme: .light, installedFamilies: ["JetBrains Mono"])
        XCTAssertEqual(with.codeFontDescriptor.family, "JetBrains Mono")
        XCTAssertEqual(with.codeFontDescriptor.customFamilyName, "JetBrains Mono")
    }

    // MARK: - Descriptors

    /// `System default`, `SF Pro` and `SF Mono` are the system face reached
    /// through `Font.system(design:)`, not files on disk. Sending them to
    /// `Font.custom` looks up a family that does not exist and falls back
    /// silently.
    func testSystemFamiliesResolveToTheSystemFaceAndOthersDoNot() {
        for family in ["System default", "SF Pro", "SF Mono"] {
            let descriptor = AppFontDescriptor(
                family: family, weight: .regular, size: 13, isCode: false)
            XCTAssertNil(descriptor.customFamilyName, family)
        }

        for family in ["Inter", "Menlo", "JetBrains Mono", "Fira Code", "Avenir"] {
            let descriptor = AppFontDescriptor(
                family: family, weight: .regular, size: 13, isCode: true)
            XCTAssertEqual(descriptor.customFamilyName, family)
        }
    }

    func testFontStepsScaleFromTheConfiguredBaseSize() {
        let theme = ResolvedAppTheme.fallback
        XCTAssertEqual(theme.codeFontDescriptor.size, 12)

        // The steps are relative, so every code surface moves together when
        // the base size does. Rounded, because a font size of 9.96 is not one.
        XCTAssertEqual(AppFontStep.base.factor, 1.0)
        XCTAssertLessThan(AppFontStep.tiny.factor, AppFontStep.small.factor)
        XCTAssertLessThan(AppFontStep.small.factor, AppFontStep.base.factor)
        XCTAssertLessThan(AppFontStep.base.factor, AppFontStep.large.factor)
    }

    func testContrastDrivesMetadataAndBorderTreatment() {
        var low = ResolvedAppTheme.fallback
        low.contrast = 10
        var high = ResolvedAppTheme.fallback
        high.contrast = 95

        XCTAssertEqual(low.metadataForeground, Color.secondary)
        XCTAssertNotEqual(high.metadataForeground, Color.secondary)

        // The floor keeps a border visible at contrast 0 rather than invisible.
        XCTAssertEqual(low.borderStrokeOpacity, 0.3)
        XCTAssertGreaterThan(high.borderStrokeOpacity, low.borderStrokeOpacity)
    }

    // MARK: - Bundled font registration

    /// Font files in the SOURCE tree, found without going through the bundle.
    ///
    /// The skip condition has to be independent of the code under test. Two
    /// earlier versions were not: keying it on `registerBundledFonts()`'s
    /// return value, and then on `bundledFontURLs`, both skipped when the
    /// bundle lookup was broken -- which is the failure, so the test passed
    /// against it. Mutating the lookup turned the case green (skipped) both
    /// times instead of red. `#filePath` reaches the checked-in files, which
    /// no bug in the lookup can move.
    private var sourceFontFiles: [String] {
        let testsDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        let fonts = testsDir
            .deletingLastPathComponent()      // Tests/
            .deletingLastPathComponent()      // TurboSparkApp/
            .appendingPathComponent("Sources/TurboSparkApp/Resources/Fonts")
        let names = (try? FileManager.default.contentsOfDirectory(atPath: fonts.path)) ?? []
        return names.filter { $0.hasSuffix(".ttf") }
    }

    func testRegisteringBundledFontsIsIdempotentAndReportsWhatItDid() throws {
        let onDisk = sourceFontFiles
        try XCTSkipIf(
            onDisk.isEmpty,
            "No .ttf files in Sources/TurboSparkApp/Resources/Fonts. The three bundled "
                + "families are offered by the pickers only once their files are in the tree.")

        // The lookup must find every file that is really there. This is what
        // reddens when the bundle lookup regresses, rather than skipping.
        XCTAssertEqual(
            AppFontRegistrar.bundledFontURLs.count, onDisk.count,
            "the bundle lookup found \(AppFontRegistrar.bundledFontURLs.count) of \(onDisk.count) "
                + "font files that are checked in; SwiftPM's .process rule flattens Fonts/ "
                + "into the bundle root")

        let first = AppFontRegistrar.registerBundledFonts()
        let second = AppFontRegistrar.registerBundledFonts()
        XCTAssertEqual(first, second, "registration must be idempotent per process")
        XCTAssertEqual(
            first, onDisk.count,
            "every bundled face must register; a shortfall is a file Core Text refused")

        // Every bundled family must be reachable once registered, or the
        // pickers would filter it out of its own menu.
        let installed = AppFontCatalog.installedFamilies()
        for family in AppFontCatalog.bundledFamilies {
            XCTAssertTrue(installed.contains(family), "\(family) registered but not reported")
        }
    }

    /// The bundled families must survive the filter that narrows the pickers.
    /// Bundling a font and then hiding it from its own menu is the failure
    /// this pairing exists to make impossible.
    func testBundledFamiliesSurviveTheInstalledFilter() throws {
        try XCTSkipIf(sourceFontFiles.isEmpty, "No bundled font files in the source tree.")
        AppFontRegistrar.registerBundledFonts()

        let installed = AppFontCatalog.installedFamilies()
        let ui = AppFontCatalog.availableUIFamilies(installed: installed)
        let code = AppFontCatalog.availableCodeFamilies(installed: installed)

        XCTAssertTrue(ui.contains("Inter"))
        XCTAssertTrue(code.contains("JetBrains Mono"))
        XCTAssertTrue(code.contains("Fira Code"))
    }

    func testColorContrastForeground() {
        // Light accents (like Codex dark mode's #F3F4F6 or pure white) must resolve a dark foreground
        let whiteColor = Color(hex: "#FFFFFF")!
        XCTAssertTrue(whiteColor.isLight)
        XCTAssertNotEqual(whiteColor.contrastForeground, Color.white)

        let codexWhite = Color(hex: "#F3F4F6")!
        XCTAssertTrue(codexWhite.isLight)
        XCTAssertNotEqual(codexWhite.contrastForeground, Color.white)

        // Dark accents (like Emerald #237D32, Blue #2563EB, Black #000000) must resolve white
        let emeraldColor = Color(hex: "#237D32")!
        XCTAssertFalse(emeraldColor.isLight)
        XCTAssertEqual(emeraldColor.contrastForeground, Color.white)

        let blackColor = Color(hex: "#000000")!
        XCTAssertFalse(blackColor.isLight)
        XCTAssertEqual(blackColor.contrastForeground, Color.white)
    }

    func testMarkdownRenderingWithCustomFont() {
        let text = "The quick brown fox jumps over the lazy dog"
        let descSystem = AppFontDescriptor(family: "System default", weight: .regular, size: 16, isCode: false)
        let descAvenir = AppFontDescriptor(family: "Avenir", weight: .regular, size: 16, isCode: false)
        let codeDesc = AppFontDescriptor(family: "SF Mono", weight: .regular, size: 12, isCode: true)

        let viewSystem = ChatMessageMarkdownView(text)
            .environment(\.appTheme, ResolvedAppTheme(
                isDark: false, accent: .black, foreground: .black, contrast: 50,
                uiFontDescriptor: descSystem, codeFontDescriptor: codeDesc))
        let viewAvenir = ChatMessageMarkdownView(text)
            .environment(\.appTheme, ResolvedAppTheme(
                isDark: false, accent: .black, foreground: .black, contrast: 50,
                uiFontDescriptor: descAvenir, codeFontDescriptor: codeDesc))

        let hostSystem = NSHostingView(rootView: viewSystem)
        hostSystem.layout()
        let hostAvenir = NSHostingView(rootView: viewAvenir)
        hostAvenir.layout()

        print("Markdown fitting width: System=\(hostSystem.fittingSize.width), Avenir=\(hostAvenir.fittingSize.width)")
        XCTAssertNotEqual(hostSystem.fittingSize.width, hostAvenir.fittingSize.width)
    }
}

