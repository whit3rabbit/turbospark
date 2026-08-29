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
    public static let primaryMinimumWidth: CGFloat = 520
    public static let chatSidebarWidth: CGFloat = 260
    public static let inspectorWidth: CGFloat = 320
    public static let dividerWidth: CGFloat = 1
    public static let minimumHeight: CGFloat = 520

    /// Width of the always-visible icon rail holding the top-level sections.
    public static let navigationRailWidth: CGFloat = 52
    /// Height of the flat top bar carrying the model loader.
    public static let topBarHeight: CGFloat = 44
    /// Height of the bottom status strip carrying memory and throughput.
    public static let statusBarHeight: CGFloat = 26
    /// Leading inset the top bar needs so its first control clears the traffic
    /// lights, which a `.hiddenTitleBar` window still draws over the content.
    /// The zoom button's right edge sits near x = 66 and the rail is 52 wide,
    /// so 26 puts the first control at 78.
    public static let trafficLightClearance: CGFloat = 26

    public static func minimumWindowWidth(
        isChatSidebarVisible: Bool,
        isInspectorVisible: Bool
    ) -> CGFloat {
        navigationRailWidth
            + dividerWidth
            + primaryMinimumWidth
            + (isChatSidebarVisible ? chatSidebarWidth + dividerWidth : 0)
            + (isInspectorVisible ? inspectorWidth + dividerWidth : 0)
    }
}
