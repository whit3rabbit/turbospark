import AppKit
import SwiftUI
import TurboSpark

/// Settings section for managing additional external folders to scan for models.
// Isolated explicitly: its helpers call each other, and only `body` is
// isolated by the protocol on the macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
@MainActor
struct CustomModelFoldersSectionView: View {
    @ObservedObject var model: AppModel

    /// Model counts per folder, from the last `rescanCounts()`.
    ///
    /// `ModelStorageManager.scanModels` walks the whole directory tree with
    /// a `FileManager.enumerator`, stat-ing every entry -- real I/O, not a
    /// property read. It used to run inline in `body`, so SwiftUI re-ran the
    /// walk, for every configured folder, on every body evaluation this view
    /// received for any reason (`swift/docs/SWIFT_SETTINGS_AUDIT.md`). `.task(id:)`
    /// below reruns this exactly when the folder list changes.
    @State private var scannedCounts: [String: Int] = [:]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            AppAdaptiveHeader {
                Label { Text("Additional Model Folders", bundle: .module) } icon: { Image(systemName: "folder.badge.plus") }
                    .settingsControl("Additional Model Folders", pane: .models, timing: .action)
                    .themedFont(.base, weight: .semibold)
            } actions: {
                Button {
                    addCustomFolder()
                } label: {
                    Label {
                        Text("Add Folder\u{2026}", bundle: .module)
                    } icon: {
                        Image(systemName: "plus")
                    }
                }
                .appAction(.secondary, size: .small)
            }

            VStack(alignment: .leading, spacing: 10) {
                if model.customModelDirectories.isEmpty {
                    HStack {
                        Spacer()
                        VStack(spacing: 6) {
                            Image(systemName: "folder.badge.questionmark")
                                .themedFont(.title2)
                                .foregroundStyle(.appSecondary)
                                .accessibilityHidden(true)
                            Text("No additional model scan folders configured.", bundle: .module)
                                .themedFont(.small)
                                .foregroundStyle(.appSecondary)
                        }
                        .padding(.vertical, 16)
                        Spacer()
                    }
                } else {
                    ForEach(model.customModelDirectories, id: \.self) { dir in
                        folderRow(dir)
                    }
                }

                Text("Add folders on external drives or secondary locations to scan for .gturbo bundles and .gguf models.", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }
            .modelsSettingsCard()
        }
        .task(id: model.customModelDirectories) {
            await rescanCounts()
        }
    }

    private func folderRow(_ dir: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "folder")
                .foregroundStyle(.appSecondary)
                .accessibilityHidden(true)
            SettingsPathText(dir)
            Spacer()

            // A count label rather than a plural sentence: the catalog
            // carries no plural variants for this key.
            Text("Models: \(scannedCounts[dir] ?? 0)", bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
                .fixedSize()

            AppIconButton("arrow.up.right.square", label: Text("Reveal in Finder", bundle: .module)) {
                ModelStorageManager.revealInFinder(path: dir)
            }
            .appAction(.quiet, size: .small)

            AppIconButton("trash", label: Text("Remove this folder from scan list", bundle: .module), role: .destructive) {
                removeCustomFolder(dir)
            }
            .appAction(.quiet, size: .small)
            .foregroundStyle(.red)
        }
        .padding(8)
        .background(.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
    }

    /// Walks off the main actor: a folder on an external or network drive can
    /// take seconds, and the walk used to block the UI.
    private func rescanCounts() async {
        let folders = model.customModelDirectories
        let counts = await Task.detached(priority: .utility) {
            Self.folderCounts(for: folders)
        }.value
        guard !Task.isCancelled else { return }
        scannedCounts = counts
    }

    /// `customModelDirectories` is decoded leniently and never deduplicated,
    /// so a repeated path must not trap the dictionary build.
    nonisolated static func folderCounts(
        for folders: [String], count: (String) -> Int = {
            // Image installs live in the same folders and are found by the
            // image scan, not the text-model one, so the count adds both.
            ModelStorageManager.scanModels(in: $0, sourceTag: "Custom").count
                + ((try? TurboSparkCatalog.imageInstalled(in: ModelStorageManager.expandPath($0)))?.count ?? 0)
        }
    ) -> [String: Int] {
        var result: [String: Int] = [:]
        for folder in folders where result[folder] == nil {
            result[folder] = count(folder)
        }
        return result
    }

    private func addCustomFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = AppStrings.localized("Add Scan Folder")
        panel.message = AppStrings.localized("Select a folder containing models to scan")

        if panel.runModal() == .OK, let url = panel.url {
            let path = url.path
            if !model.customModelDirectories.contains(path) {
                model.customModelDirectories.append(path)
                model.persistSettings()
                model.refreshModels()
                model.showToast(AppStrings.localized("Added model scan folder: \(url.lastPathComponent)"), style: .success)
            }
        }
    }

    private func removeCustomFolder(_ folder: String) {
        model.customModelDirectories.removeAll { $0 == folder }
        model.persistSettings()
        model.refreshModels()
        model.showToast(AppStrings.localized("Removed model scan folder"), style: .info)
    }
}
