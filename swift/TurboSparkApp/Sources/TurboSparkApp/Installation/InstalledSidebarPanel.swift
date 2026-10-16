import SwiftUI

/// The installed model kinds the Installed section manages. Audio installs
/// live in the Audio workspace, which the panel's footnote points to.
enum InstalledModelKind: String, SidebarPanelMode {
    case text, images

    var id: String { rawValue }
    /// The English catalog key, rendered through `titleText`.
    var title: String {
        switch self {
        case .text: return "Text Models"
        case .images: return "Image Models"
        }
    }
    var titleText: Text { Text(LocalizedStringKey(title), bundle: .module) }
    var symbol: String {
        switch self {
        case .text: return "text.alignleft"
        case .images: return "photo"
        }
    }
}

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The Installed panel beside the rail. It replaced the model-type segmented
/// control and the scope line that sat under the pane header; the scope line
/// stays with the kinds it explains, so a missing third kind is not a mystery.
@MainActor
struct InstalledSidebarPanel: View {
    @ObservedObject var model: AppModel

    var body: some View {
        SidebarModePanel(
            title: Text("Installed", bundle: .module),
            groupLabel: Text("Installed model type", bundle: .module),
            selection: $model.installedModelKind
        ) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Audio models are managed in Audio.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                Button {
                    model.activeSection = .audio
                } label: {
                    Label { Text("Open Audio", bundle: .module) } icon: { Image(systemName: "waveform") }
                }
                .appAction(.quiet, size: .small)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 8)
            .padding(.top, 12)
        }
    }
}
