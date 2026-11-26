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
/// The Models panel beside the rail: the installed model kinds, then Discover.
/// Discover is a mode of Models rather than a sidebar row of its own, so one
/// row is selected at a time across both.
@MainActor
struct InstalledSidebarPanel: View {
    @ObservedObject var model: AppModel

    var body: some View {
        SidebarPanelList(title: Text("Models", bundle: .module)) {
            VStack(spacing: 2) {
                ForEach(InstalledModelKind.allCases) { kind in
                    SidebarPanelRow(
                        title: kind.titleText, systemImage: kind.symbol,
                        isSelected: model.activeSection == .modelManager
                            && model.installedModelKind == kind
                    ) {
                        model.installedModelKind = kind
                        model.activeSection = .modelManager
                    }
                }
            }
            .accessibilityElement(children: .contain)
            .accessibilityLabel(Text("Installed model type", bundle: .module))

            SidebarPanelRow(
                title: Text("Discover Models", bundle: .module), systemImage: "shippingbox",
                isSelected: model.activeSection == .modelHub
            ) { model.activeSection = .modelHub }
            .padding(.top, 6)
        }
        .tsAnimation(TSMotion.select, value: model.activeSection)
        .tsAnimation(TSMotion.select, value: model.installedModelKind)
    }
}
