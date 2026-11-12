import SwiftUI

/// Image Generation's two modes. Create shows the composer under the gallery;
/// Organize trades it for search, selection and a denser grid.
enum ImagesWorkspaceMode: String, SidebarPanelMode {
    case create, organize

    var id: String { rawValue }
    /// The English catalog key, rendered through `titleText`.
    var title: String {
        switch self {
        case .create: return "Create"
        case .organize: return "Organize"
        }
    }
    var titleText: Text { Text(LocalizedStringKey(title), bundle: .module) }
    var symbol: String {
        switch self {
        case .create: return "photo.badge.plus"
        case .organize: return "square.grid.2x2"
        }
    }
}

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The Image Generation panel beside the rail. It replaced the Create and
/// Organize segmented control in the pane header.
@MainActor
struct ImagesSidebarPanel: View {
    @ObservedObject var model: AppModel

    var body: some View {
        SidebarModePanel(
            title: Text("Image Generation", bundle: .module),
            groupLabel: Text("Image mode", bundle: .module),
            selection: $model.imagesMode)
    }
}
