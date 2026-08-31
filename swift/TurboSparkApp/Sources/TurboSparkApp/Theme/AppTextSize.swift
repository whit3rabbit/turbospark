import SwiftUI

/// User interface scaling options for readability and accessibility sizing.
public enum AppTextSize: String, CaseIterable, Identifiable, Sendable {
    case standard
    case large
    case extraLarge

    /// UserDefaults key where text size preference is persisted.
    public static let storageKey = "TurboSpark.textSize"

    public var id: String { rawValue }

    /// Human-readable label for settings selector.
    public var label: String {
        switch self {
        case .standard: return "Default"
        case .large: return "Large"
        case .extraLarge: return "Extra Large"
        }
    }

    /// Corresponding SwiftUI DynamicTypeSize applied to the environment.
    public var dynamicTypeSize: DynamicTypeSize {
        switch self {
        // One notch above the system baseline (.large): the app read small
        // at the true baseline, and this is the existing text-size knob
        // rather than a new one.
        case .standard: return .xLarge
        case .large: return .xxLarge
        case .extraLarge: return .xxxLarge
        }
    }

    /// Resolves stored raw string value into an `AppTextSize` instance.
    public static func resolve(_ storedValue: String) -> Self {
        Self(rawValue: storedValue) ?? .standard
    }
}
