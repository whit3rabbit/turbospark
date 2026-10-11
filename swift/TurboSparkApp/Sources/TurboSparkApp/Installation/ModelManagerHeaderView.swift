import AppKit
import SwiftUI
import TurboSpark

/// Header bar for ModelManagerView showing storage stats, rescan, add folder, and hub discovery buttons.
///
/// Shares its structure with the Discover header (`ModelWorkspaceTitle` plus
/// `AppAdaptiveHeader`): Discover Models is the one primary action, and when
/// the row is too narrow it stays visible while Rescan and Add Folder fold
/// into a labeled More menu.
@MainActor
struct ModelManagerHeaderView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        AppAdaptiveHeader {
            ModelWorkspaceTitle(
                systemImage: "internaldrive.fill",
                tint: .teal,
                title: Text("Installed Models", bundle: .module),
                badge: Text("LOCAL STORAGE", bundle: .module),
                summary: summaryText)
        } actions: {
            rescanButton.appAction(.secondary)
            addFolderButton.appAction(.secondary)
            discoverButton
        } compactActions: {
            discoverButton
            Menu {
                rescanButton
                addFolderButton
            } label: {
                Label { Text("More", bundle: .module) } icon: { Image(systemName: "ellipsis.circle") }
            }
            .appMenuAction()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    private var rescanButton: some View {
        Button {
            model.refreshModels()
            model.showToast(AppStrings.localized("Refreshed model libraries"), style: .info)
        } label: {
            Label { Text("Rescan", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
        }
        .help(Text("Rescan storage directories for models", bundle: .module))
    }

    private var addFolderButton: some View {
        Button {
            ModelLocationPicker.choose(for: model)
        } label: {
            Label { Text("Add Folder\u{2026}", bundle: .module) } icon: { Image(systemName: "folder.badge.plus") }
        }
        .help(Text("Add an external folder containing .gturbo bundles or .gguf files", bundle: .module))
    }

    private var discoverButton: some View {
        Button {
            model.activeSection = .modelHub
        } label: {
            Label { Text("Discover Models", bundle: .module) } icon: { Image(systemName: "shippingbox.fill") }
        }
        .appAction(.primary)
        .help(Text("Browse the curated catalog and download models", bundle: .module))
    }

    private var summaryText: Text {
        let count = model.installed.count
        let totalBytes = model.installed.reduce(UInt64(0)) { $0 + $1.installBytes }
        let separator = Text(verbatim: " \u{2022} ")
        var line = Text("Installed models: \(count)", bundle: .module)
        if totalBytes > 0 {
            line = line + separator + Text("On disk: \(MetricFormat.storage(totalBytes))", bundle: .module)
        }
        if let sel = model.selected, model.session != nil {
            line = line + separator + Text("Active: \(sel.alias)", bundle: .module)
        }
        return line
    }
}
