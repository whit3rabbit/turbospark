import AppKit
import SwiftUI
import TurboSpark

/// Models and storage preferences pane for managing primary TurboSpark download location,
/// automatic LM Studio library detection, and external model search folders.
public struct ModelsSettingsPaneView: View {
    @ObservedObject var model: AppModel

    /// From the last `rescanLmStudioCount()`. `ModelStorageManager.scanModels`
    /// walks the whole LM Studio directory tree with a `FileManager`
    /// enumerator, stat-ing every entry -- real I/O, not a property read. It
    /// used to run inline in `body`, so SwiftUI re-ran the walk on every body
    /// evaluation this view received for any reason
    /// (`swift/docs/SWIFT_SETTINGS_AUDIT.md`).
    @State private var lmStudioModelCount: Int = 0
    @State private var showingStoreMigration = false
    @State private var showingProviderDiscovery = false

    public init(model: AppModel) {
        self.model = model
    }

    private var activeLmStudioPath: String {
        model.lmStudioDirectory.isEmpty ? ModelStorageManager.defaultLMStudioModelsDirectory : model.lmStudioDirectory
    }

    private var isLmStudioPresent: Bool {
        ModelStorageManager.isLMStudioDirectoryPresent(customPath: model.lmStudioDirectory)
    }

    private var turboSparkStoreRoot: String {
        ModelStorageManager.defaultTurboSparkStoreRoot
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Hugging Face Authentication API token
                HfAuthTokenCardView(model: model)

                // Section 1: TurboSpark Primary Storage
                turboSparkStorageSection

                // Section 2: LM Studio Integration
                lmStudioIntegrationSection

                // Section 3: Additional Custom Folders
                CustomModelFoldersSectionView(model: model)

                // Section 3b: what "Remove from TurboSpark" actually did, and
                // the way back (state#88).
                ExcludedScanPathsSectionView(model: model)

                // Section 4: Local model defaults (AutoFit floor, memory
                // guardrails). Its own file; this one is near the 400-line
                // guideline.
                LocalModelDefaultsSectionView(model: model)
            }
            .padding(20)
        }
        .sheet(isPresented: $showingStoreMigration) {
            ModelStoreMigrationSheet(model: model)
        }
        .sheet(isPresented: $showingProviderDiscovery) {
            ModelProviderDiscoverySheet(model: model)
        }
    }

    // MARK: - TurboSpark Primary Storage
    private var turboSparkStorageSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label { Text("TurboSpark Models Storage", bundle: .module) } icon: { Image(systemName: "cylinder.split.1x2") }
                    .themedFont(.base, weight: .semibold)
                Spacer()
                Button {
                    showingStoreMigration = true
                } label: {
                    Label { Text("Move Store...", bundle: .module) } icon: { Image(systemName: "externaldrive") }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .settingsControl("Move Model Store", pane: .models, timing: .immediate)
            }

            VStack(alignment: .leading, spacing: 10) {
                Text(turboSparkStoreRoot)
                    .themedCode(.small)
                    .textSelection(.enabled)
                    .lineLimit(1)
                    .truncationMode(.middle)
                storagePathRow(label: "Text", path: ModelStorageManager.defaultTurboSparkModelsDirectory)
                storagePathRow(label: "Image", path: ModelStorageManager.defaultTurboSparkImageModelsDirectory)
                storagePathRow(label: "Audio", path: ModelStorageManager.defaultTurboSparkAudioModelsDirectory)

                HStack {
                    Button {
                        ModelStorageManager.revealInFinder(path: turboSparkStoreRoot)
                    } label: {
                        Label { Text("Reveal in Finder", bundle: .module) } icon: { Image(systemName: "arrow.up.right.square") }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .help(Text("Opens the TurboSpark store in Finder", bundle: .module))
                    .accessibilityLabel(Text("Reveal in Finder", bundle: .module))
                    Spacer()
                    Button {
                        showingProviderDiscovery = true
                    } label: {
                        Label { Text("Detect Model Libraries", bundle: .module) } icon: { Image(systemName: "magnifyingglass") }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .help(Text("Detect Model Libraries", bundle: .module))
                    .accessibilityLabel(Text("Detect Model Libraries", bundle: .module))
                }

                Text("Catalog installs and future text, image, and audio installs use this shared root. Other folders are scanned in place and are never copied automatically.", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }
            .padding(14)
            .background(.appPage)
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(.appBorder.opacity(0.4), lineWidth: 1))
        }
    }

    private func storagePathRow(label: String, path: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "folder.fill")
                .foregroundStyle(Color.accentColor)
            Text(label)
                .themedFont(.small, weight: .medium)
                .frame(width: 52, alignment: .leading)
            Text(path)
                .themedCode(.tiny)
                .textSelection(.enabled)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
        }
        .padding(8)
        .background(.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: 7))
    }

    // MARK: - LM Studio Integration
    private var lmStudioIntegrationSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label { Text("LM Studio Library Integration", bundle: .module) } icon: { Image(systemName: "arrow.triangle.2.circlepath") }
                .settingsControl("LM Studio Library Integration", pane: .models, timing: .immediate)
                    .themedFont(.base, weight: .semibold)
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
                Text("Automatically detect and run GGUF models downloaded by LM Studio without copying or redownloading files.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)

                HStack(spacing: 8) {
                    Image(systemName: isLmStudioPresent ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                        .foregroundStyle(isLmStudioPresent ? Color.green : Color.orange)

                    Text(isLmStudioPresent ? "LM Studio folder detected" : "Folder not found")
                        .themedFont(.small, weight: .medium)

                    Spacer()
                }

                HStack(spacing: 8) {
                    Image(systemName: "folder")
                        .foregroundStyle(.appSecondary)
                    Text(activeLmStudioPath)
                        .themedCode(.small)
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
                        .help(Text("Reveal in Finder", bundle: .module))
                    }

                    Button {
                        selectLmStudioFolder()
                    } label: { Text("Select Folder...", bundle: .module) }
                    .buttonStyle(.bordered)
                    .controlSize(.small)

                    if !model.lmStudioDirectory.isEmpty {
                        Button {
                            model.lmStudioDirectory = ""
                            model.persistSettings()
                            model.refreshModels()
                        } label: { Text("Default", bundle: .module) }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .help("Reset to default ~/.lmstudio/models")
                    }
                }
                .padding(8)
                .background(.appSurface)
                .clipShape(RoundedRectangle(cornerRadius: 6))

                // Zero-copy explanation callout
                HStack(alignment: .top, spacing: 10) {
                    Image(systemName: "bolt.shield")
                        .themedFont(.title3)
                        .foregroundStyle(Color.accentColor)
                        .frame(width: 24)

                    VStack(alignment: .leading, spacing: 3) {
                        Text("Zero-Copy In-Place Inference", bundle: .module)
                            .themedFont(.small, weight: .semibold)
                        Text("Models discovered in LM Studio are indexed in-place. TurboSpark reads weights directly from your LM Studio folder without duplicating disk space or copying gigabytes of parameters.", bundle: .module)
                            .themedFont(.tiny)
                            .foregroundStyle(.appSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .padding(10)
                .background(Color.accentColor.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 8))

                HStack {
                    if isLmStudioPresent && model.enableLMStudioDetection {
                        Text("\(lmStudioModelCount) model\(lmStudioModelCount == 1 ? "" : "s") found in LM Studio folder", bundle: .module)
                            .themedFont(.small)
                            .foregroundStyle(.appSecondary)
                    }
                    Spacer()
                    Button {
                        model.refreshModels()
                        rescanLmStudioCount()
                        model.showToast("Rescanned local and LM Studio models", style: .info)
                    } label: {
                        Label { Text("Rescan Now", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
            }
            .padding(14)
            .task(id: "\(activeLmStudioPath)|\(model.enableLMStudioDetection)") {
                rescanLmStudioCount()
            }
            .background(.appPage)
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(.appBorder.opacity(0.4), lineWidth: 1))
        }
    }

    // MARK: - Folder Picker Helpers
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

    private func rescanLmStudioCount() {
        lmStudioModelCount = ModelStorageManager.scanModels(in: activeLmStudioPath, sourceTag: "LM Studio").count
    }
}
