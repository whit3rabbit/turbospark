import Foundation
import SwiftUI

/// A portable palette pair. Typography and accessibility preferences remain personal.
public struct SavedTheme: Codable, Equatable, Identifiable, Sendable {
    public var version: Int = 1
    public var id: String
    public var name: String
    public var light: ThemeModeConfig
    public var dark: ThemeModeConfig

    public var preset: ThemePreset { ThemePreset(id: id, name: name, light: light, dark: dark) }

    public func validated() throws -> SavedTheme {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard version == 1, !trimmed.isEmpty, trimmed.count <= 80,
              Self.valid(light), Self.valid(dark) else { throw ThemeLibraryError.invalidTheme }
        var result = self
        result.name = trimmed
        return result
    }

    private static func valid(_ config: ThemeModeConfig) -> Bool {
        [config.accentHex, config.backgroundHex, config.foregroundHex].allSatisfy {
            $0.range(of: "^#[0-9a-fA-F]{6}$", options: .regularExpression) != nil
        } && config.contrast.isFinite && (0...100).contains(config.contrast)
    }
}

public enum ThemeLibraryError: LocalizedError {
    case invalidTheme
    public var errorDescription: String? { "Enter a name and valid six-digit colors for both theme modes." }
}

extension ThemeModeConfig {
    /// A palette match ignores font choices and display labels.
    func matchesPalette(_ other: ThemeModeConfig) -> Bool {
        accentHex.lowercased() == other.accentHex.lowercased()
            && backgroundHex.lowercased() == other.backgroundHex.lowercased()
            && foregroundHex.lowercased() == other.foregroundHex.lowercased()
            && translucentSidebar == other.translucentSidebar && contrast == other.contrast
    }

    public var textContrastRatio: Double {
        func luminance(_ hex: String) -> Double {
            guard let n = UInt32(hex.dropFirst(), radix: 16) else { return 0 }
            let channels = [Double((n >> 16) & 255), Double((n >> 8) & 255), Double(n & 255)].map {
                let value = $0 / 255
                return value <= 0.04045 ? value / 12.92 : pow((value + 0.055) / 1.055, 2.4)
            }
            return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722
        }
        let a = luminance(backgroundHex), b = luminance(foregroundHex)
        return (max(a, b) + 0.05) / (min(a, b) + 0.05)
    }
}

extension ThemePreset {
    public static var additionalPresets: [ThemePreset] {
        [palette("sandstone", "Sandstone", "#98602B", "#FAF6EE", "#302A24", "#E6B77C", "#211C17", "#F3E9D8"),
         palette("orchid", "Orchid", "#7B3DA5", "#FAF5FC", "#2C2034", "#D3A6F0", "#201626", "#F4EAF9"),
         palette("forest", "Forest", "#176B5D", "#F0F8F4", "#18342C", "#76D3B3", "#10251F", "#E2F5EB")]
    }

    private static func palette(_ id: String, _ name: String, _ la: String, _ lb: String, _ lf: String,
                                _ da: String, _ db: String, _ df: String) -> ThemePreset {
        ThemePreset(id: id, name: name,
            light: ThemeModeConfig(preset: name, accentName: name, accentHex: la, backgroundHex: lb, foregroundHex: lf),
            dark: ThemeModeConfig(preset: name, accentName: name, accentHex: da, backgroundHex: db, foregroundHex: df, contrast: 65))
    }
}

@MainActor
extension AppearanceManager {
    public var availableThemes: [ThemePreset] { ThemePreset.presets + ThemePreset.additionalPresets + savedThemes.map(\.preset) }
    public var currentThemeID: String? {
        guard let selectedThemeID, let preset = availableThemes.first(where: { $0.id == selectedThemeID }),
              lightConfig.matchesPalette(preset.light), darkConfig.matchesPalette(preset.dark) else { return nil }
        return selectedThemeID
    }
    public var currentThemeName: String {
        availableThemes.first { $0.id == currentThemeID }?.name ?? "Custom"
    }

    @discardableResult
    public func saveTheme(named name: String) throws -> SavedTheme {
        let saved = try SavedTheme(id: UUID().uuidString, name: name, light: lightConfig, dark: darkConfig).validated()
        savedThemes.append(saved)
        selectedThemeID = saved.id
        return saved
    }

    public func renameTheme(id: String, name: String) throws {
        guard let index = savedThemes.firstIndex(where: { $0.id == id }) else { return }
        var theme = savedThemes[index]
        theme.name = name
        savedThemes[index] = try theme.validated()
    }

    public func deleteTheme(id: String) {
        if selectedThemeID == id { selectedThemeID = nil }
        savedThemes.removeAll { $0.id == id }
    }

    @discardableResult
    public func importTheme(data: Data) throws -> SavedTheme {
        var theme = try JSONDecoder().decode(SavedTheme.self, from: data).validated()
        // Imported identifiers cannot overwrite a built-in or another saved theme.
        theme.id = UUID().uuidString
        savedThemes.append(theme)
        applyPreset(theme.preset)
        return theme
    }

    public func exportTheme() throws -> Data {
        let theme = try SavedTheme(id: currentThemeID ?? UUID().uuidString, name: currentThemeName,
            light: lightConfig, dark: darkConfig).validated()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(theme)
    }
}
