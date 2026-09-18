import AppKit
import SwiftUI
import TurboSpark

/// Moves the managed TurboSpark store to a new, empty root.
struct ModelStoreMigrationSheet: View {
    @ObservedObject var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var destination = ""
    @State private var isMoving = false
    @State private var completedBytes: UInt64 = 0
    @State private var totalBytes: UInt64 = 0
    @State private var errorText: String?

    private var currentRoot: String { ModelStorageManager.defaultTurboSparkStoreRoot }
    private var canMove: Bool {
        !isMoving && !model.generating && model.session == nil && !model.isInstallingModel
            && !destination.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Move TurboSpark Models", bundle: .module)
                .themedFont(.title3, weight: .semibold)

            Text("Move the managed text, image, and audio model store to an empty folder. Files owned by Hugging Face, LM Studio, or Ollama are not changed.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)

            locationRow(title: "Current store", path: currentRoot)
            locationRow(title: "New store", path: destination.isEmpty ? "Choose a folder" : destination)

            if model.session != nil || model.generating || model.isInstallingModel {
                Label {
                    Text("Stop generation and unload the current model before moving the store.", bundle: .module)
                } icon: {
                    Image(systemName: "exclamationmark.triangle.fill")
                }
                .foregroundStyle(.orange)
            }

            if isMoving {
                if totalBytes > 0 {
                    ProgressView(value: Double(completedBytes) / Double(totalBytes))
                } else {
                    ProgressView()
                }
                Text(totalBytes > 0 ? "\(MetricFormat.storage(completedBytes)) of \(MetricFormat.storage(totalBytes))" : "Preparing move...", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }

            if let errorText {
                Text(errorText)
                    .themedFont(.small)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                    .disabled(isMoving)
                Button("Choose Folder...") { chooseDestination() }
                    .disabled(isMoving)
                Button("Move Store") { moveStore() }
                    .buttonStyle(.borderedProminent)
                    .disabled(!canMove)
            }
        }
        .padding(22)
        .frame(width: 560)
    }

    private func locationRow(title: String, path: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .themedFont(.tiny, weight: .semibold)
                .foregroundStyle(.appSecondary)
            Text(path)
                .themedCode(.small)
                .textSelection(.enabled)
                .lineLimit(1)
                .truncationMode(.middle)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(9)
                .background(.appSurface, in: RoundedRectangle(cornerRadius: 7))
        }
    }

    private func chooseDestination() {
        let panel = NSOpenPanel()
        panel.title = "Choose TurboSpark Store Location"
        panel.message = "Choose an empty folder on this Mac or an external drive."
        panel.prompt = "Choose Store"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            destination = url.standardizedFileURL.path
            errorText = nil
        }
    }

    private func moveStore() {
        guard canMove else { return }
        isMoving = true
        errorText = nil
        completedBytes = 0
        totalBytes = 0
        let target = ModelStorageManager.expandPath(destination)
        Task {
            do {
                for try await event in TurboSparkCatalog.relocateStore(to: target) {
                    switch event {
                    case let .bytes(done, total):
                        completedBytes = done
                        totalBytes = total
                    case let .finished(result):
                        model.turboSparkStoreRoot = result.destination
                        model.persistSettings()
                        model.refreshModels()
                        model.showToast("Moved TurboSpark models to \(result.destination)", style: .success, duration: 5)
                        isMoving = false
                        dismiss()
                    }
                }
            } catch {
                errorText = error.localizedDescription
                isMoving = false
            }
        }
    }
}
