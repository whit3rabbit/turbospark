import SwiftUI

/// Shows known local model libraries before adding any scan roots.
// Isolated explicitly: its helpers call each other, and only `body` is
// isolated by the protocol on the macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
@MainActor
struct ModelProviderDiscoverySheet: View {
    @ObservedObject var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var candidates: [ModelProviderCandidate] = []
    @State private var selectedIDs: Set<String> = []
    @State private var isScanning = true

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Detect Model Libraries", bundle: .module)
                .themedFont(.title3, weight: .semibold)
            Text("TurboSpark checks known Hugging Face, LM Studio, and Ollama locations. Only supported GGUF and GTurbo artifacts can be added to the runnable model list.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)

            if isScanning {
                AppProgressBar(value: nil, label: Text("Scanning known locations...", bundle: .module))
                    .padding(.vertical, 8)
            } else if candidates.isEmpty {
                AppEmptyState(
                    systemImage: "folder.badge.questionmark",
                    title: Text("No known provider folders were found.", bundle: .module)
                )
                .frame(maxHeight: 200)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(candidates) { candidate in
                            candidateRow(candidate)
                        }
                    }
                }
                .frame(maxHeight: 300)
            }

            HStack(spacing: AppMetrics.actionSpacing) {
                Spacer()
                Button(role: .cancel) { dismiss() } label: {
                    Text("Cancel", bundle: .module)
                }
                .keyboardShortcut(.cancelAction)
                Button { addSelected() } label: {
                    Text("Add Selected", bundle: .module)
                }
                .appAction(.primary)
                .keyboardShortcut(.defaultAction)
                .disabled(isScanning || selectedIDs.isEmpty)
            }
        }
        .padding(22)
        .frame(minWidth: 520, idealWidth: 620, maxWidth: 720)
        .task { scan() }
    }

    private func candidateRow(_ candidate: ModelProviderCandidate) -> some View {
        Toggle(isOn: Binding(
            get: { selectedIDs.contains(candidate.id) },
            set: { selected in
                if selected { selectedIDs.insert(candidate.id) }
                else { selectedIDs.remove(candidate.id) }
            })) {
            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    // Provider names are product names.
                    Text(verbatim: candidate.provider.displayName)
                        .themedFont(.small, weight: .semibold)
                    Spacer()
                    statusText(candidate)
                        .themedFont(.tiny)
                        .foregroundStyle(candidate.supportedModelCount > 0 ? AnyShapeStyle(.appAccent) : AnyShapeStyle(AppAttention.color))
                }
                SettingsPathText(candidate.path, size: .tiny)
                if candidate.provider == .ollama && candidate.supportedModelCount == 0 {
                    Text("Ollama's blob store is not directly importable; no files will be copied.", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
            }
        }
        .toggleStyle(.checkbox)
        .padding(9)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius))
        .disabled(candidate.supportedModelCount == 0 && candidate.provider != .lmStudio)
    }

    /// Mirrors `ModelProviderCandidate.statusText`, whose English string is
    /// built in the model layer; the sheet renders it from the catalog.
    private func statusText(_ candidate: ModelProviderCandidate) -> Text {
        if candidate.supportedModelCount > 0 {
            return Text("Supported models: \(candidate.supportedModelCount)", bundle: .module)
        }
        if candidate.unsupportedArtifactCount > 0 {
            return Text("Contains unsupported provider artifacts", bundle: .module)
        }
        return Text("No directly runnable models found", bundle: .module)
    }

    private func scan() {
        let lmPath = model.lmStudioDirectory.isEmpty ? nil : model.lmStudioDirectory
        Task.detached(priority: .utility) {
            let paths = ModelStorageManager.knownProviderPaths(lmStudioPath: lmPath)
            let result = paths.map { ModelStorageManager.inspectProvider($0.0, path: $0.1) }
            await MainActor.run {
                candidates = result
                selectedIDs = Set(result.filter { $0.supportedModelCount > 0 }.map(\.id))
                isScanning = false
            }
        }
    }

    private func addSelected() {
        for candidate in candidates where selectedIDs.contains(candidate.id) {
            switch candidate.provider {
            case .lmStudio:
                if candidate.path != ModelStorageManager.defaultLMStudioModelsDirectory {
                    model.lmStudioDirectory = candidate.path
                }
                model.enableLMStudioDetection = true
            case .huggingFace, .ollama:
                if !model.customModelDirectories.contains(candidate.path) {
                    model.customModelDirectories.append(candidate.path)
                }
            }
        }
        model.persistSettings()
        model.refreshModels()
        model.showToast(AppStrings.localized("Added detected model libraries"), style: .success)
        dismiss()
    }
}
