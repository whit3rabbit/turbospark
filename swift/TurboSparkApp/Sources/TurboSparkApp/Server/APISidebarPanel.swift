import SwiftUI

extension APIWorkspaceTab: SidebarPanelMode {
    var symbol: String {
        switch self {
        case .text: return "text.alignleft"
        case .image: return "photo"
        case .audio: return "waveform"
        case .embeddings: return "point.3.connected.trianglepath.dotted"
        case .typeSafe: return "checkmark.shield"
        }
    }
}

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The API panel beside the rail. It replaced the row of workspace buttons
/// across the top of the API pane.
@MainActor
struct APISidebarPanel: View {
    @ObservedObject var model: AppModel

    var body: some View {
        SidebarModePanel(
            title: Text("API", bundle: .module),
            groupLabel: Text("API mode", bundle: .module),
            selection: $model.apiWorkspaceTab)
    }
}
