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

    /// Linear font scale factor applied to base point sizes.
    public var scale: CGFloat {
        switch self {
        case .standard: return 1.0
        case .large: return 1.15
        case .extraLarge: return 1.30
        }
    }

    /// Applies `scale` to a base point size and rounds to a whole point,
    /// the one formula every font-size call site should share so a future
    /// change to the rounding or scaling rule needs one edit rather than
    /// hunting down each inlined copy.
    public func scaled(_ basePointSize: CGFloat) -> CGFloat {
        (basePointSize * scale).rounded()
    }

    /// Resolves stored raw string value into an `AppTextSize` instance.
    public static func resolve(_ storedValue: String) -> Self {
        Self(rawValue: storedValue) ?? .standard
    }
}
