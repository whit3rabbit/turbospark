import AppKit
import SwiftUI

/// Settings section for managing additional external folders to scan for models.
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
            HStack {
                Label("Additional Model Folders", systemImage: "folder.badge.plus")
                    .themedFont(.base, weight: .semibold)
                Spacer()
                Button {
                    addCustomFolder()
                } label: {
                    Label {
                        Text("Add Folder…", bundle: .module)
                    } icon: {
                        Image(systemName: "plus")
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            VStack(alignment: .leading, spacing: 10) {
                if model.customModelDirectories.isEmpty {
                    HStack {
                        Spacer()
                        VStack(spacing: 6) {
                            Image(systemName: "folder.badge.questionmark")
                                .themedFont(.title2)
                                .foregroundStyle(.secondary)
                            Text("No additional model scan folders configured.", bundle: .module)
                                .themedFont(.small)
                                .foregroundStyle(.secondary)
                        }
                        .padding(.vertical, 16)
                        Spacer()
                    }
                } else {
                    ForEach(model.customModelDirectories, id: \.self) { dir in
                        HStack(spacing: 8) {
                            Image(systemName: "folder")
                                .foregroundStyle(.secondary)
                            Text(dir)
                                .themedCode(.small)
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Spacer()

                            let found = scannedCounts[dir] ?? 0
                            Text("\(found) model\(found == 1 ? "" : "s")", bundle: .module)
                                .themedFont(.tiny)
                                .foregroundStyle(.secondary)

                            Button {
                                ModelStorageManager.revealInFinder(path: dir)
                            } label: {
                                Image(systemName: "arrow.up.right.square")
                            }
                            .buttonStyle(.plain)
                            .help("Reveal in Finder")

                            Button {
                                removeCustomFolder(dir)
                            } label: {
                                Image(systemName: "trash")
                                    .foregroundStyle(.red)
                            }
                            .buttonStyle(.plain)
                            .help("Remove this folder from scan list")
                        }
                        .padding(8)
                        .background(Color(nsColor: .controlBackgroundColor))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                    }
                }

                Text("Add folders on external drives or secondary locations to scan for .gturbo bundles and .gguf models.", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.secondary)
            }
            .padding(14)
            .background(Color(nsColor: .windowBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color(nsColor: .separatorColor).opacity(0.4), lineWidth: 1))
        }
        .task(id: model.customModelDirectories) {
            rescanCounts()
        }
    }

    private func rescanCounts() {
        scannedCounts = Dictionary(
            uniqueKeysWithValues: model.customModelDirectories.map {
                ($0, ModelStorageManager.scanModels(in: $0, sourceTag: "Custom").count)
            })
    }

    private func addCustomFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Add Scan Folder"
        panel.message = "Select a folder containing models to scan"

        if panel.runModal() == .OK, let url = panel.url {
            let path = url.path
            if !model.customModelDirectories.contains(path) {
                model.customModelDirectories.append(path)
                model.persistSettings()
                model.refreshModels()
                model.showToast("Added model scan folder: \(url.lastPathComponent)", style: .success)
            }
        }
    }

    private func removeCustomFolder(_ folder: String) {
        model.customModelDirectories.removeAll { $0 == folder }
        model.persistSettings()
        model.refreshModels()
        model.showToast("Removed model scan folder", style: .info)
    }
}
