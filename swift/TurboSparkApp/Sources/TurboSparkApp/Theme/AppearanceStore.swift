import Foundation

/// The on-disk appearance record, written to `<root>/appearance.json`.
///
/// **MOVED OFF `UserDefaults.standard`** for the same reason
/// `DisabledItemStore` documents on its own header: a bundle-identity change
/// (a re-signed build, a different machine restoring a partial domain)
/// silently resets preferences stored under the host's defaults, while the
/// `AppStorageRoot` JSON stores survive it. `AppearanceManager` was the last
/// settings surface still on UserDefaults, carrying eleven keys, two of them
/// JSON-encoded blobs.
///
/// Migration is ONE-WAY and happens at load: when `appearance.json` is
/// absent, the archive is built from the legacy keys, and they are removed
/// only AFTER the first successful save through
/// `AppearanceManager.persist()` -- so a crash mid-migration loses nothing.
public struct AppearanceArchive: Codable, Equatable, Sendable {
    public var appearance: String
    public var textSize: String
    public var lightConfig: ThemeModeConfig
    public var darkConfig: ThemeModeConfig
    public var statusBarViewMode: String
    public var usePointerCursors: Bool
    public var dockIcon: String
    public var reduceMotion: String
    public var uiFontSize: Double
    public var codeFontSize: Double
    public var diffMarkers: String
    public var savedThemes: [SavedTheme]
    public var selectedThemeID: String?

    public init(
        appearance: String = AppAppearance.system.rawValue,
        textSize: String = AppTextSize.standard.rawValue,
        lightConfig: ThemeModeConfig = .defaultLight,
        darkConfig: ThemeModeConfig = .defaultDark,
        statusBarViewMode: String = StatusBarViewMode.text.rawValue,
        usePointerCursors: Bool = false,
        dockIcon: String = AppDockIcon.emeraldSpark.rawValue,
        reduceMotion: String = ReduceMotionPreference.system.rawValue,
        uiFontSize: Double = 16.0,
        codeFontSize: Double = 12.0,
        diffMarkers: String = DiffMarkerPreference.color.rawValue,
        savedThemes: [SavedTheme] = [], selectedThemeID: String? = "sparkblue"
    ) {
        self.appearance = appearance
        self.textSize = textSize
        self.lightConfig = lightConfig
        self.darkConfig = darkConfig
        self.statusBarViewMode = statusBarViewMode
        self.usePointerCursors = usePointerCursors
        self.dockIcon = dockIcon
        self.reduceMotion = reduceMotion
        self.uiFontSize = uiFontSize
        self.codeFontSize = codeFontSize
        self.diffMarkers = diffMarkers
        self.savedThemes = savedThemes
        self.selectedThemeID = selectedThemeID
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.appearance = try container.decodeIfPresent(String.self, forKey: .appearance) ?? AppAppearance.system.rawValue
        self.textSize = try container.decodeIfPresent(String.self, forKey: .textSize) ?? AppTextSize.standard.rawValue
        self.lightConfig = try container.decodeIfPresent(ThemeModeConfig.self, forKey: .lightConfig) ?? .defaultLight
        self.darkConfig = try container.decodeIfPresent(ThemeModeConfig.self, forKey: .darkConfig) ?? .defaultDark
        self.statusBarViewMode = try container.decodeIfPresent(String.self, forKey: .statusBarViewMode) ?? StatusBarViewMode.text.rawValue
        self.usePointerCursors = try container.decodeIfPresent(Bool.self, forKey: .usePointerCursors) ?? false
        self.dockIcon = try container.decodeIfPresent(String.self, forKey: .dockIcon) ?? AppDockIcon.emeraldSpark.rawValue
        self.reduceMotion = try container.decodeIfPresent(String.self, forKey: .reduceMotion) ?? ReduceMotionPreference.system.rawValue
        self.uiFontSize = try container.decodeIfPresent(Double.self, forKey: .uiFontSize) ?? 16.0
        self.codeFontSize = try container.decodeIfPresent(Double.self, forKey: .codeFontSize) ?? 12.0
        self.diffMarkers = try container.decodeIfPresent(String.self, forKey: .diffMarkers) ?? DiffMarkerPreference.color.rawValue
        self.savedThemes = (try container.decodeIfPresent([SavedTheme].self, forKey: .savedThemes) ?? []).compactMap { try? $0.validated() }
        if container.contains(.selectedThemeID) {
            self.selectedThemeID = try container.decodeIfPresent(String.self, forKey: .selectedThemeID)
        } else {
            self.selectedThemeID = (ThemePreset.presets + ThemePreset.additionalPresets).first {
                lightConfig.matchesPalette($0.light) && darkConfig.matchesPalette($0.dark)
            }?.id
        }
    }
}

public enum AppearanceFileStore {
    static var fileURL: URL {
        AppStorageRoot.file("appearance.json")
    }

    /// Loads the archive. When the JSON file is absent (or was quarantined),
    /// the archive is built from the legacy UserDefaults keys instead and
    /// `migrated` is true -- the caller must save successfully before the
    /// legacy keys may be removed.
    public static func load() -> (archive: AppearanceArchive, migrated: Bool) {
        if let loaded: AppearanceArchive = AppJSONStore.load(AppearanceArchive.self, from: fileURL, label: "appearance") {
            return (loaded, false)
        }
        return (legacyArchive(), legacyArchiveHasAnyValue)
    }

    /// Writes the archive atomically.
    @discardableResult
    public static func save(_ archive: AppearanceArchive) -> Bool {
        AppJSONStore.save(archive, to: fileURL, label: "Appearance")
    }

    /// Deletes the legacy UserDefaults keys. Only called after the JSON
    /// store has held a successful save, per the migration note above.
    public static func removeLegacyKeys() {
        for key in legacyKeys {
            UserDefaults.standard.removeObject(forKey: key)
        }
    }

    private static let legacyKeys = [
        "TurboSpark.appearance",
        AppTextSize.storageKey,
        "TurboSpark.theme.lightConfig",
        "TurboSpark.theme.darkConfig",
        "TurboSpark.prefs.usePointerCursors",
        "TurboSpark.prefs.dockIcon",
        "TurboSpark.prefs.reduceMotion",
        "TurboSpark.prefs.uiFontSize",
        "TurboSpark.prefs.codeFontSize",
        "TurboSpark.prefs.diffMarkers",
        "TurboSpark.prefs.statusBarViewMode"
    ]

    private static var legacyArchiveHasAnyValue: Bool {
        let defaults = UserDefaults.standard
        return legacyKeys.contains { defaults.object(forKey: $0) != nil }
    }

    /// Reads the eleven legacy keys with the same fallbacks
    /// `AppearanceManager`'s init used before the store existed.
    private static func legacyArchive() -> AppearanceArchive {
        let defaults = UserDefaults.standard
        let lightData = defaults.data(forKey: "TurboSpark.theme.lightConfig")
        let darkData = defaults.data(forKey: "TurboSpark.theme.darkConfig")

        return AppearanceArchive(
            appearance: defaults.string(forKey: "TurboSpark.appearance") ?? AppAppearance.system.rawValue,
            textSize: defaults.string(forKey: AppTextSize.storageKey) ?? AppTextSize.standard.rawValue,
            lightConfig: lightData.flatMap { try? JSONDecoder().decode(ThemeModeConfig.self, from: $0) }
                ?? .defaultLight,
            darkConfig: darkData.flatMap { try? JSONDecoder().decode(ThemeModeConfig.self, from: $0) }
                ?? .defaultDark,
            statusBarViewMode: defaults.string(forKey: "TurboSpark.prefs.statusBarViewMode")
                ?? StatusBarViewMode.text.rawValue,
            usePointerCursors: defaults.object(forKey: "TurboSpark.prefs.usePointerCursors") as? Bool ?? false,
            dockIcon: defaults.string(forKey: "TurboSpark.prefs.dockIcon") ?? AppDockIcon.emeraldSpark.rawValue,
            reduceMotion: defaults.string(forKey: "TurboSpark.prefs.reduceMotion")
                ?? ReduceMotionPreference.system.rawValue,
            uiFontSize: defaults.object(forKey: "TurboSpark.prefs.uiFontSize") as? Double ?? 16.0,
            codeFontSize: defaults.object(forKey: "TurboSpark.prefs.codeFontSize") as? Double ?? 12.0,
            diffMarkers: defaults.string(forKey: "TurboSpark.prefs.diffMarkers")
                ?? DiffMarkerPreference.color.rawValue
        )
    }
}
