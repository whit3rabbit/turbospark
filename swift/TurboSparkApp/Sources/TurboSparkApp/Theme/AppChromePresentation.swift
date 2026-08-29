import Foundation

public enum AppSidebarKind: CaseIterable, Equatable, Sendable {
    case chats
    case inspector
}

public struct AppSidebarControlPresentation: Equatable, Sendable {
    public let systemImage: String
    public let title: String
    public let help: String
    public let accessibilityValue: String

    public init(
        sidebar: AppSidebarKind,
        isVisible: Bool
    ) {
        accessibilityValue = isVisible ? "Visible" : "Hidden"
        switch sidebar {
        case .chats:
            systemImage = "sidebar.left"
            title = isVisible ? "Hide chats" : "Show chats"
            help = "\(title) (⌃⌘S)"
        case .inspector:
            systemImage = "sidebar.right"
            title = isVisible ? "Hide settings" : "Show settings"
            help = "\(title) (⇧⌘I)"
        }
    }
}

public enum AppChromeLayout {
    public static let primaryMinimumWidth: CGFloat = 560
    public static let chatSidebarWidth: CGFloat = 272
    public static let inspectorWidth: CGFloat = 320
    public static let dividerWidth: CGFloat = 1
    public static let minimumHeight: CGFloat = 520
    public static let headerHorizontalPadding: CGFloat = 20

    public static func minimumWindowWidth(
        isChatSidebarVisible: Bool,
        isInspectorVisible: Bool
    ) -> CGFloat {
        primaryMinimumWidth
            + (isChatSidebarVisible ? chatSidebarWidth + dividerWidth : 0)
            + (isInspectorVisible ? inspectorWidth + dividerWidth : 0)
    }
}
