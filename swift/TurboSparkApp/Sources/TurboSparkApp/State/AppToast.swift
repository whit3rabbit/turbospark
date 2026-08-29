import Foundation
import SwiftUI

public struct AppToast: Identifiable, Equatable, Sendable {
    public enum Style: String, Sendable {
        case info
        case success
        case warning
        case error

        public var systemImage: String {
            switch self {
            case .info: return "info.circle.fill"
            case .success: return "checkmark.circle.fill"
            case .warning: return "exclamationmark.triangle.fill"
            case .error: return "xmark.octagon.fill"
            }
        }

        public var tintColor: Color {
            switch self {
            case .info: return .accentColor
            case .success: return .green
            case .warning: return .orange
            case .error: return .red
            }
        }
    }

    public let id: UUID
    public let message: String
    public let style: Style
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
