import SwiftUI

/// The installed model kinds the Installed section manages, each with the
/// same lifecycle: list and detail, load and unload, reveal, delete.
enum InstalledModelKind: String, SidebarPanelMode {
    case text, images, audio

    var id: String { rawValue }
    /// The English catalog key, rendered through `titleText`.
    var title: String {
        switch self {
        case .text: return "Text Models"
        case .images: return "Image Models"
        case .audio: return "Audio Models"
        }
    }
    var titleText: Text { Text(LocalizedStringKey(title), bundle: .module) }
    var symbol: String {
        switch self {
        case .text: return "text.alignleft"
        case .images: return "photo"
        case .audio: return "waveform"
        }
    }
}

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The Installed panel beside the rail. It replaced the model-type segmented
/// control that sat under the pane header.
@MainActor
struct InstalledSidebarPanel: View {
    @ObservedObject var model: AppModel

    var body: some View {
        SidebarModePanel(
            title: Text("Installed", bundle: .module),
            groupLabel: Text("Installed model type", bundle: .module),
            selection: $model.installedModelKind)
    }
}
