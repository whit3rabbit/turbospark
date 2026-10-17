import AppKit
import SwiftUI
import TurboSpark

/// Models and storage preferences pane for managing primary TurboSpark download location,
/// automatic LM Studio library detection, and external model search folders.
// Isolated explicitly: its helpers call each other, and only `body` is
// isolated by the protocol on the macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
@MainActor
public struct ModelsSettingsPaneView: View {
    @ObservedObject var model: AppModel

    /// From the last `rescanLmStudioCount()`. `ModelStorageManager.scanModels`
    /// walks the whole LM Studio directory tree with a `FileManager`
    /// enumerator, stat-ing every entry -- real I/O, not a property read. It
    /// used to run inline in `body`, so SwiftUI re-ran the walk on every body
    /// evaluation this view received for any reason
    /// (`swift/docs/SWIFT_SETTINGS_AUDIT.md`).
    @State private var lmStudioModelCount: Int = 0
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
                // The canonical home of Hugging Face authentication: General
                // used to carry a second copy, and two surfaces for one token
                // let them disagree about which token was in effect.
                HfAuthTokenCardView(model: model)
                    .settingsControl("Hugging Face Authentication", pane: .models, timing: .immediate)

                // Section 0: model libraries (drives that hold models). Above
                // the storage section on purpose: a library is where models
                // can live besides the home that section describes.
                ModelLibrariesSectionView(model: model)

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
            .padding(SettingsLayout.contentInsets)
        }
        .sheet(isPresented: $showingProviderDiscovery) {
            ModelProviderDiscoverySheet(model: model)
        }
    }

    // MARK: - TurboSpark Primary Storage
    private var turboSparkStorageSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            // A plain heading: the section had no action once the store move
            // was retired (requirement 9.1), and an adaptive header with an
            // empty actions slot still lays out a spacer and a second row.
            Label { Text("TurboSpark Models Storage", bundle: .module) } icon: { Image(systemName: "cylinder.split.1x2") }
                .themedFont(.base, weight: .semibold)

            VStack(alignment: .leading, spacing: 10) {
                SettingsPathText(turboSparkStoreRoot)
                storagePathRow(label: Text("Text", bundle: .module), path: ModelStorageManager.defaultTurboSparkModelsDirectory)
                storagePathRow(label: Text("Image", bundle: .module), path: ModelStorageManager.defaultTurboSparkImageModelsDirectory)
                storagePathRow(label: Text("Audio", bundle: .module), path: ModelStorageManager.defaultTurboSparkAudioModelsDirectory)

                HStack(spacing: AppMetrics.actionSpacing) {
                    Button {
                        ModelStorageManager.revealInFinder(path: turboSparkStoreRoot)
                    } label: {
                        Label { Text("Reveal in Finder", bundle: .module) } icon: { Image(systemName: "arrow.up.right.square") }
                    }
                    .appAction(.secondary, size: .small)
                    .help(Text("Opens the TurboSpark store in Finder", bundle: .module))
                    Spacer()
                    Button {
                        showingProviderDiscovery = true
                    } label: {
                        Label { Text("Detect Model Libraries", bundle: .module) } icon: { Image(systemName: "magnifyingglass") }
                    }
                    .appAction(.secondary, size: .small)
                }

                Text("Catalog installs and future text, image, and audio installs use this shared root. Other folders are scanned in place and are never copied automatically.", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }
            .modelsSettingsCard()
        }
    }

    private func storagePathRow(label: Text, path: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "folder.fill")
                .foregroundStyle(.appAccent)
                .accessibilityHidden(true)
            label
                .themedFont(.small, weight: .medium)
                .frame(minWidth: 52, alignment: .leading)
                .fixedSize()
            SettingsPathText(path, size: .tiny)
            Spacer()
        }
        .padding(8)
        .background(.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
    }

    // MARK: - LM Studio Integration
    private var lmStudioIntegrationSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label { Text("LM Studio Library Integration", bundle: .module) } icon: { Image(systemName: "arrow.triangle.2.circlepath") }
                    .settingsControl("LM Studio Library Integration", pane: .models, timing: .immediate)
                    .themedFont(.base, weight: .semibold)
                Spacer()
                Toggle(isOn: Binding(
                    get: { model.enableLMStudioDetection },
                    set: { val in
                        model.enableLMStudioDetection = val
                        model.persistSettings()
                        model.refreshModels()
                    }
                )) {
                    Text("LM Studio Library Integration", bundle: .module)
                }
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
                        .accessibilityHidden(true)
                    Group {
                        if isLmStudioPresent {
                            Text("LM Studio folder detected", bundle: .module)
                        } else {
                            Text("Folder not found", bundle: .module)
                        }
                    }
                    .themedFont(.small, weight: .medium)
                    Spacer()
                }

                lmStudioPathRow

                // Zero-copy explanation callout
                HStack(alignment: .top, spacing: 10) {
                    Image(systemName: "bolt.shield")
                        .themedFont(.title3)
                        .foregroundStyle(.appAccent)
                        .frame(width: 24)
                        .accessibilityHidden(true)

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
                .background(.appAccent.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))

                HStack {
                    if isLmStudioPresent && model.enableLMStudioDetection {
                        // A count label rather than a plural sentence: the
                        // catalog carries no plural variants for this key.
                        Text("Models found in LM Studio folder: \(lmStudioModelCount)", bundle: .module)
                            .themedFont(.small)
                            .foregroundStyle(.appSecondary)
                    }
                    Spacer()
                    Button {
                        model.refreshModels()
                        Task { await rescanLmStudioCount() }
                        model.showToast(AppStrings.localized("Rescanned local and LM Studio models"), style: .info)
                    } label: {
                        Label { Text("Rescan Now", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
                    }
                    .appAction(.secondary, size: .small)
                }
            }
            .task(id: "\(activeLmStudioPath)|\(model.enableLMStudioDetection)") {
                await rescanLmStudioCount()
            }
            .modelsSettingsCard()
        }
    }

    private var lmStudioPathRow: some View {
        HStack(spacing: 8) {
            Image(systemName: "folder")
                .foregroundStyle(.appSecondary)
                .accessibilityHidden(true)
            SettingsPathText(activeLmStudioPath)
            Spacer()

            if isLmStudioPresent {
                AppIconButton("arrow.up.right.square", label: Text("Reveal in Finder", bundle: .module)) {
                    ModelStorageManager.revealInFinder(path: activeLmStudioPath)
                }
                .appAction(.quiet, size: .small)
            }

            Button {
                selectLmStudioFolder()
            } label: { Text("Select Folder...", bundle: .module) }
            .appAction(.secondary, size: .small)

            if !model.lmStudioDirectory.isEmpty {
                Button {
                    model.lmStudioDirectory = ""
                    model.persistSettings()
                    model.refreshModels()
                } label: { Text("Default", bundle: .module) }
                .appAction(.secondary, size: .small)
                .help(Text("Reset to default ~/.lmstudio/models", bundle: .module))
            }
        }
        .padding(8)
        .background(.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
    }

    // MARK: - Folder Picker Helpers
    private func selectLmStudioFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = AppStrings.localized("Select LM Studio Models Folder")
        panel.message = AppStrings.localized("Select your LM Studio models directory")

        if panel.runModal() == .OK, let url = panel.url {
            model.lmStudioDirectory = url.path
            model.persistSettings()
            model.refreshModels()
            model.showToast(AppStrings.localized("Configured LM Studio folder: \(url.lastPathComponent)"), style: .success)
        }
    }

    /// Counts off the main actor: the walk can span hundreds of GB on an
    /// external or network volume, and ran inline on the main thread before.
    private func rescanLmStudioCount() async {
        let path = activeLmStudioPath
        let enabled = model.enableLMStudioDetection
        let count = await Task.detached(priority: .utility) {
            Self.lmStudioModelCount(path: path, detectionEnabled: enabled)
        }.value
        guard !Task.isCancelled else { return }
        lmStudioModelCount = count
    }

    /// The count is only displayed while detection is on, so a disabled
    /// detection never touches the disk.
    nonisolated static func lmStudioModelCount(path: String, detectionEnabled: Bool) -> Int {
        guard detectionEnabled else { return 0 }
        return ModelStorageManager.scanModels(in: path, sourceTag: "LM Studio").count
    }
}

extension View {
    /// The bordered page card every Models & Storage section sits in, so the
    /// sections (several in their own files) cannot drift apart.
    func modelsSettingsCard() -> some View {
        padding(14)
            .background(.appPage)
            .clipShape(RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                    .stroke(.appBorder.opacity(0.4), lineWidth: 1)
            )
    }
}
