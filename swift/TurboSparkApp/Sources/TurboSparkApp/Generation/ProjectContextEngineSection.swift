import SwiftUI
import SyntextContext

/// The project sheet's context engine opt-in and its index status
/// (`swift/docs/SWIFT_CONTEXT_ENGINE.md`). The toggle is saved with the
/// sheet; Reindex and Remove act on the saved project at once.
struct ProjectContextEngineSection: View {
    @ObservedObject var model: AppModel
    /// The saved project, or nil while creating one.
    let project: AppProject?
    let rootDirectoryPath: String
    @Binding var enabled: Bool

    @State private var stats: SyntextStoreStats?
    @State private var confirmingRemove = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Context Engine", bundle: .module)
                    .themedFont(.small, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)
                Text("Retrieves relevant passages from this project's files for each message, and adds context search tools.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }

            let hasRoot = !rootDirectoryPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            Toggle(isOn: $enabled) {
                Text("Index this project for the context engine", bundle: .module)
            }
            .disabled(!model.contextEngineEnabled || !hasRoot)

            if !model.contextEngineEnabled {
                note(Text("The context engine is turned off in Settings > General.", bundle: .module))
            } else if !hasRoot {
                Text("Set a codebase root directory above to enable indexing.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .padding(8)
            } else if let project, project.contextEngineEnabled {
                status(for: project)
            }
        }
        .task(id: project?.id) { await refreshStats() }
        .onChange(of: syncState) { _, _ in Task { await refreshStats() } }
        .confirmationDialog(
            Text("Remove Context Index?", bundle: .module), isPresented: $confirmingRemove
        ) {
            Button(role: .destructive) {
                if let project { model.removeContextEngineIndex(for: project.id) }
                stats = nil
            } label: { Text("Remove Index", bundle: .module) }
        } message: {
            Text("This deletes this project's context index and the conversation turns its chats recorded. Your files are not changed.", bundle: .module)
        }
    }

    private var syncState: ContextEngineSyncState? {
        project.flatMap { model.contextEngineSyncStates[$0.id] }
    }

    @ViewBuilder
    private func status(for project: AppProject) -> some View {
        HStack(spacing: 12) {
            switch syncState {
            case .syncing(let observed, let total):
                ProgressView()
                    .controlSize(.small)
                Text("Reading project files: \(observed) of \(total)", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            case .failed(let message):
                Text("Context indexing failed: \(message)", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            default:
                if let stats {
                    Text("Documents indexed: \(stats.documents) (\(MetricFormat.storage(stats.bytes)))", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                } else {
                    Text("Not indexed yet. Indexing starts when the project is selected.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
            }
            Spacer()
            Button {
                model.rebuildContextEngineIndex(for: project)
            } label: {
                Text("Reindex", bundle: .module)
            }
            .appAction(.secondary, size: .small)
            .accessibilityLabel(Text("Reindex project for the context engine", bundle: .module))
            if stats != nil {
                Button(role: .destructive) {
                    confirmingRemove = true
                } label: { Text("Remove Index", bundle: .module) }
                .appAction(.destructive, size: .small)
                .accessibilityLabel(Text("Remove project context index", bundle: .module))
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(TurboSparkTheme.surfaceColor, in: RoundedRectangle(cornerRadius: 8))
    }

    private func note(_ text: Text) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "info.circle")
                .foregroundStyle(.appSecondary)
            text
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(TurboSparkTheme.surfaceColor, in: RoundedRectangle(cornerRadius: 8))
    }

    private func refreshStats() async {
        guard let id = project?.id else { stats = nil; return }
        stats = await ContextEngineManager.shared.stats(projectID: id)
    }
}
