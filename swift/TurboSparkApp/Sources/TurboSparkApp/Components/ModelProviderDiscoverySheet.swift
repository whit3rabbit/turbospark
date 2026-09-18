import SwiftUI

/// Shows known local model libraries before adding any scan roots.
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
                HStack(spacing: 8) {
                    ProgressView()
                    Text("Scanning known locations...", bundle: .module)
                }
            } else if candidates.isEmpty {
                Text("No known provider folders were found.", bundle: .module)
                    .foregroundStyle(.appSecondary)
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

            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                Button("Add Selected") { addSelected() }
                    .buttonStyle(.borderedProminent)
                    .disabled(isScanning || selectedIDs.isEmpty)
            }
        }
        .padding(22)
        .frame(width: 620)
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
                    Text(candidate.provider.displayName)
                        .themedFont(.small, weight: .semibold)
                    Spacer()
                    Text(candidate.statusText)
                        .themedFont(.tiny)
                        .foregroundStyle(candidate.supportedModelCount > 0 ? .green : .orange)
                }
                Text(candidate.path)
                    .themedCode(.tiny)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if candidate.provider == .ollama && candidate.supportedModelCount == 0 {
                    Text("Ollama's blob store is not directly importable; no files will be copied.", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
            }
        }
        .toggleStyle(.checkbox)
        .padding(9)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: 7))
        .disabled(candidate.supportedModelCount == 0 && candidate.provider != .lmStudio)
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
        model.showToast("Added detected model libraries", style: .success)
        dismiss()
    }
}
