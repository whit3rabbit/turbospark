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
                CustomModelFoldersSectionView(model: model)
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

                Text("TurboSpark stores converted .gturbo models and direct GGUF downloads here.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
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
                Label("LM Studio Library Integration", systemImage: "arrow.triangle.2.circlepath")
                    .font(.headline)
                Spacer()
                Toggle("", isOn: Binding(
                    get: { model.enableLMStudioDetection },
                    set: { val in
                        model.enableLMStudioDetection = val
                        model.persistSettings()
                        model.refreshModels()
                    }
                ))
                .toggleStyle(.switch)
                .labelsHidden()
            }

            VStack(alignment: .leading, spacing: 10) {
                Text("Automatically detect and run GGUF models downloaded by LM Studio without copying or redownloading files.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                HStack(spacing: 8) {
                    Image(systemName: isLmStudioPresent ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                        .foregroundStyle(isLmStudioPresent ? Color.green : Color.orange)

                    Text(isLmStudioPresent ? "LM Studio folder detected" : "Folder not found")
                        .font(.subheadline.weight(.medium))

                    Spacer()

                    Button(showingCustomLmPath ? "Hide Path" : "Configure Custom Path") {
                        showingCustomLmPath.toggle()
                    }
                    .buttonStyle(.borderless)
                    .font(.caption)
                }

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
                            Image(systemName: "arrow.up.right.square")
                        }
                        .buttonStyle(.plain)
                        .help("Reveal in Finder")
                    }

                    Button("Select Folder...") {
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
}
