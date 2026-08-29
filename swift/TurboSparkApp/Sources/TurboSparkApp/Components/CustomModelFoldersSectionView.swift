import AppKit
import SwiftUI

/// Settings section for managing additional external folders to scan for models.
struct CustomModelFoldersSectionView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Additional Model Folders", systemImage: "folder.badge.plus")
                    .font(.headline)
                Spacer()
                Button {
                    addCustomFolder()
                } label: {
                    Label("Add Folder...", systemImage: "plus")
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
                                .font(.title2)
                                .foregroundStyle(.secondary)
                            Text("No additional model scan folders configured.")
                                .font(.caption)
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
                                .font(.system(.caption, design: .monospaced))
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Spacer()

                            let found = ModelStorageManager.scanModels(in: dir, sourceTag: "Custom").count
                            Text("\(found) model\(found == 1 ? "" : "s")")
                                .font(.caption2)
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

                Text("Add folders on external drives or secondary locations to scan for .gturbo bundles and .gguf models.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .padding(14)
            .background(Color(nsColor: .windowBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color(nsColor: .separatorColor).opacity(0.4), lineWidth: 1))
        }
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
