import AppKit
import SwiftUI
import TurboSpark

/// Header bar for ModelManagerView showing the Installed | Discover switcher, storage stats, rescan and add folder.
///
/// Shares its structure with the Discover header (`ModelWorkspaceTitle` plus
/// `AppAdaptiveHeader`): the Installed | Discover switcher leads, and when the
/// row is too narrow Rescan and Add Folder fold into a labeled More menu.
@MainActor
struct ModelManagerHeaderView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        AppAdaptiveHeader {
            // The switcher leads; the title block sits under it.
            VStack(alignment: .leading, spacing: 8) {
                ModelWorkspaceSwitcher(model: model)
                ModelWorkspaceTitle(
                    systemImage: "internaldrive.fill",
                    title: Text("Installed Models", bundle: .module),
                    summary: summaryText)
            }
        } actions: {
            rescanButton.appAction(.secondary)
            addFolderButton.appAction(.secondary)
        } compactActions: {
            Menu {
                rescanButton
                addFolderButton
            } label: {
                Label { Text("More", bundle: .module) } icon: { Image(systemName: "ellipsis.circle") }
            }
            .appMenuAction()
        }
        // ViewThatFits centers a header narrower than the pane; keep it leading.
        .frame(maxWidth: .infinity, alignment: .leading)
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

    private var summaryText: Text {
        let count = model.installed.count
        let totalBytes = model.installed.reduce(UInt64(0)) { $0 + $1.installBytes }
        let separator = Text(verbatim: " \u{2022} ")
        var line = Text("Installed models: \(count)", bundle: .module)
        if totalBytes > 0 {
            line = line + separator + Text("On disk: \(MetricFormat.storage(totalBytes))", bundle: .module)
        }
        return line
    }
}
