import Foundation
import SwiftUI

/// Ephemeral in-app toast notification model.
public struct AppToast: Identifiable, Equatable, Sendable {
    /// Visual style indicating the severity or purpose of the toast.
    public enum Style: String, Sendable {
        case info
        case success
        case warning
        case error

        /// SF Symbol name for the style indicator.
        public var systemImage: String {
            switch self {
            case .info: return "info.circle.fill"
            case .success: return "checkmark.circle.fill"
            case .warning: return "exclamationmark.triangle.fill"
            case .error: return "xmark.octagon.fill"
            }
        }

        /// Tint color for the style indicator icon.
        public var tintColor: Color {
            switch self {
            case .info: return .accentColor
            case .success: return .green
            case .warning: return .orange
            case .error: return .red
            }
        }
    }

    /// Unique identifier for the toast instance.
    public let id: UUID
    /// Message body displayed to the user.
    public let message: String
    /// Toast visual severity style.
    public let style: Style
    /// Display duration in seconds before auto-dismissal.
    public let duration: TimeInterval

    public init(
        id: UUID = UUID(),
        message: String,
        style: Style = .info,
        duration: TimeInterval = 3.0
    ) {
        self.id = id
        self.message = message
        self.style = style
        self.duration = duration
    }
}
