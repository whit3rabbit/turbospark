import AppKit
import SwiftUI

/// Selectable application appearance mode (System default, forced Light, or forced Dark).
public enum AppAppearance: String, CaseIterable, Identifiable, Sendable {
    case system
    case light
    case dark

    /// UserDefaults key where appearance choice is persisted.
    public static let storageKey = "TurboSpark.appearance"

    public var id: String { rawValue }

    /// Human-readable label for appearance picker.
    public var label: String {
        switch self {
        case .system: "System"
        case .light: "Light"
        case .dark: "Dark"
        }
    }

    /// SF Symbol icon name representing the appearance state.
    public var systemImage: String {
        switch self {
        case .system: "circle.lefthalf.filled"
        case .light: "sun.max"
        case .dark: "moon"
        }
    }

    /// ColorScheme applied to SwiftUI views, or nil to follow system mode.
    public var preferredColorScheme: ColorScheme? {
        switch self {
        case .system: nil
        case .light: .light
        case .dark: .dark
        }
    }

    /// Resolves stored raw string value into an `AppAppearance` instance.
    public static func resolve(_ storedValue: String) -> Self {
        Self(rawValue: storedValue) ?? .system
    }

    /// Whether dark styling applies, given the scheme SwiftUI is rendering in.
    ///
    /// Take the argument from `@Environment(\.colorScheme)` and never from
    /// `NSApp.effectiveAppearance`. That property is APPLICATION-level and is
    /// not moved by `.preferredColorScheme`, so an app forced to Light on a
    /// dark system still reports dark and picks `darkConfig` -- light chrome
    /// drawn with the dark theme's accent, background and contrast.
    public func isDark(systemColorScheme: ColorScheme) -> Bool {
        switch self {
        case .system: systemColorScheme == .dark
        case .light: false
        case .dark: true
        }
    }
}
