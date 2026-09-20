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
            // Collapse/expand, not show/hide: the column is always present,
            // because the sections live in it and navigation must never be
            // hideable. See `AppSidebarView`.
            title = isVisible ? "Collapse sidebar" : "Expand sidebar"
            help = "\(title) (Ctrl+Cmd+S)"
        case .inspector:
            systemImage = "sidebar.right"
            title = isVisible ? "Hide settings" : "Show settings"
            help = "\(title) (Shift+Cmd+I)"
        }
    }
}

public enum AppChromeLayout {
    public static let primaryMinimumWidth: CGFloat = 640
    public static let chatSidebarWidth: CGFloat = 260
    public static let inspectorWidth: CGFloat = 320
    public static let expandedInspectorWidth: CGFloat = 720
    public static let dividerWidth: CGFloat = 1
    public static let minimumHeight: CGFloat = 640

    public static func inspectorWidth(isExpanded: Bool) -> CGFloat {
        isExpanded ? expandedInspectorWidth : inspectorWidth
    }

    /// Width of the icon rail, which is the sidebar's COLLAPSED presentation
    /// rather than a band of its own since the two left columns merged.
    public static let navigationRailWidth: CGFloat = 52

    /// The single left column's width in each of its two presentations.
    ///
    /// There used to be two columns side by side (rail plus chat sidebar) and
    /// therefore two widths to add up. There is one now, so a caller sizing
    /// the window asks this rather than summing.
    public static func sidebarColumnWidth(isExpanded: Bool) -> CGFloat {
        isExpanded ? chatSidebarWidth : navigationRailWidth
    }
    /// Height of the flat top bar carrying the machine telemetry.
    public static let topBarHeight: CGFloat = 44
    /// Height of the bottom status strip carrying memory and throughput.
    public static let statusBarHeight: CGFloat = 26
    /// Leading inset the top bar needs so its first control clears the traffic
    /// lights, which a `.hiddenTitleBar` window still draws over the content.
    /// The zoom button's right edge sits near x = 66 and the rail is 52 wide,
    /// so 26 puts the first control at 78.
    public static let trafficLightClearance: CGFloat = 26
}

