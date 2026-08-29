import SwiftUI

public enum AppTextSize: String, CaseIterable, Identifiable, Sendable {
    case standard
    case large
    case extraLarge

    public static let storageKey = "TurboSpark.textSize"

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .standard: return "Default"
        case .large: return "Large"
        case .extraLarge: return "Extra Large"
        }
    }

    public var dynamicTypeSize: DynamicTypeSize {
        switch self {
        case .standard: return .large
        case .large: return .xLarge
        case .extraLarge: return .xxxLarge
        }
    }

    public static func resolve(_ storedValue: String) -> Self {
        Self(rawValue: storedValue) ?? .standard
    }
}
