import AppKit
import SwiftUI
import TurboSpark

/// Models and storage preferences pane for managing primary TurboSpark download location,
/// automatic LM Studio library detection, and external model search folders.
public struct ModelsSettingsPaneView: View {
    @ObservedObject var model: AppModel

    @State private var showingCustomLmPath = false
    @State private var customLmPathInput: String = ""

    public init(model: AppModel) {
        self.model = model
    }

    private var activeLmStudioPath: String {
        model.lmStudioDirectory.isEmpty ? ModelStorageManager.defaultLMStudioModelsDirectory : model.lmStudioDirectory
    }

    private var isLmStudioPresent: Bool {
        ModelStorageManager.isLMStudioDirectoryPresent(customPath: model.lmStudioDirectory)
    }

    private var activeTurboSparkPath: String {
        model.modelsDirectory.isEmpty ? ModelStorageManager.defaultTurboSparkModelsDirectory : model.modelsDirectory
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Section 1: TurboSpark Primary Storage
                turboSparkStorageSection

                // Section 2: LM Studio Integration
                lmStudioIntegrationSection

                // Section 3: Additional Custom Folders
                additionalFoldersSection
            }
            .padding(20)
        }
        .onAppear {
            customLmPathInput = model.lmStudioDirectory
        }
    }

    // MARK: - TurboSpark Primary Storage
    private var turboSparkStorageSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("TurboSpark Models Storage", systemImage: "cylinder.split.1x2")
                    .font(.headline)
                Spacer()
                Text("Default Download Destination")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 8) {
                    Image(systemName: "folder.fill")
                        .foregroundStyle(Color.accentColor)
                    Text(activeTurboSparkPath)
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()

                    Button {
                        ModelStorageManager.revealInFinder(path: activeTurboSparkPath)
                    } label: {
                        Label("Reveal in Finder", systemImage: "arrow.up.right.square")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .help("Opens the TurboSpark models directory in Finder")

                    Button("Change...") {
                        selectTurboSparkFolder()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .help("Select a custom primary directory for TurboSpark models")

                    if !model.modelsDirectory.isEmpty {
                        Button("Reset") {
                            model.modelsDirectory = ""
                            model.persistSettings()
                            model.refreshModels()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .help("Reset to default ~/.turbospark/models")
                    }
                }
                .padding(10)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(nsColor: .separatorColor).opacity(0.3), lineWidth: 1))

                HStack {
                    Text("All new models downloaded via the catalog or HF pull are saved to this folder.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    let size = ModelStorageManager.directorySize(at: activeTurboSparkPath)
                    Text("Total space: \(MetricFormat.storage(size))")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            .padding(14)
            .background(Color(nsColor: .windowBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color(nsColor: .separatorColor).opacity(0.4), lineWidth: 1))
        }
    }

    // MARK: - LM Studio Integration
    private var lmStudioIntegrationSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("LM Studio Library Integration", systemImage: "shippingbox.fill")
                    .font(.headline)
                Spacer()

                if isLmStudioPresent {
                    HStack(spacing: 4) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                        Text("Detected")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.green)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.green.opacity(0.12), in: Capsule())
                } else {
                    HStack(spacing: 4) {
                        Image(systemName: "info.circle")
                            .foregroundStyle(.secondary)
                        Text("Not Detected")
                            .font(.caption.weight(.medium))
                            .foregroundStyle(.secondary)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color(nsColor: .controlBackgroundColor), in: Capsule())
                }
            }

            VStack(alignment: .leading, spacing: 12) {
                Toggle(isOn: Binding(
                    get: { model.enableLMStudioDetection },
                    set: { newVal in
                        model.enableLMStudioDetection = newVal
                        model.persistSettings()
                        model.refreshModels()
                    }
                )) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Include LM Studio models in TurboSpark library")
                            .font(.subheadline.weight(.medium))
                        Text("Automatically scan and run models stored in LM Studio without copying files.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .toggleStyle(.switch)

                Divider()

                HStack(spacing: 8) {
                    Image(systemName: "folder")
                        .foregroundStyle(.secondary)
                    Text(activeLmStudioPath)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()

                    if isLmStudioPresent {
                        Button {
                            ModelStorageManager.revealInFinder(path: activeLmStudioPath)
                        } label: {
                            Label("Reveal in Finder", systemImage: "arrow.up.right.square")
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }

                    Button("Change Path...") {
                        selectLmStudioFolder()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)

                    if !model.lmStudioDirectory.isEmpty {
                        Button("Default") {
                            model.lmStudioDirectory = ""
                            model.persistSettings()
                            model.refreshModels()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .help("Reset to default ~/.lmstudio/models")
                    }
                }
                .padding(8)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6))

                // Zero-copy explanation callout
                HStack(alignment: .top, spacing: 10) {
                    Image(systemName: "bolt.shield")
                        .font(.title3)
                        .foregroundStyle(Color.accentColor)
                        .frame(width: 24)

                    VStack(alignment: .leading, spacing: 3) {
                        Text("Zero-Copy In-Place Inference")
                            .font(.caption.weight(.semibold))
                        Text("Models discovered in LM Studio are indexed in-place. TurboSpark reads weights directly from your LM Studio folder without duplicating disk space or copying gigabytes of parameters.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .padding(10)
                .background(Color.accentColor.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 8))

                HStack {
                    if isLmStudioPresent && model.enableLMStudioDetection {
                        let models = ModelStorageManager.scanModels(in: activeLmStudioPath, sourceTag: "LM Studio")
                        Text("\(models.count) model\(models.count == 1 ? "" : "s") found in LM Studio folder")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button {
                        model.refreshModels()
                        model.showToast("Rescanned local and LM Studio models", style: .info)
                    } label: {
                        Label("Rescan Now", systemImage: "arrow.clockwise")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
            }
            .padding(14)
            .background(Color(nsColor: .windowBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color(nsColor: .separatorColor).opacity(0.4), lineWidth: 1))
        }
    }

    // MARK: - Additional Folders
    private var additionalFoldersSection: some View {
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

    // MARK: - Folder Picker Helpers
    private func selectTurboSparkFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        panel.prompt = "Choose Storage Folder"
        panel.message = "Select directory to store downloaded TurboSpark models"

        if panel.runModal() == .OK, let url = panel.url {
            model.modelsDirectory = url.path
            model.persistSettings()
            model.refreshModels()
            model.showToast("Updated TurboSpark models storage location", style: .success)
        }
    }

    private func selectLmStudioFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Select LM Studio Models Folder"
        panel.message = "Select your LM Studio models directory"

        if panel.runModal() == .OK, let url = panel.url {
            model.lmStudioDirectory = url.path
            model.persistSettings()
            model.refreshModels()
            model.showToast("Configured LM Studio folder: \(url.lastPathComponent)", style: .success)
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
