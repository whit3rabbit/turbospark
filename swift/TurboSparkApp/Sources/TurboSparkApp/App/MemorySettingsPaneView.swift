import AppKit
import SwiftUI

/// Settings pane for the auto-memory feature: the enable toggle, the folder
/// buttons, and what the selected project currently remembers.
///
/// The toggle is the ONLY writer of `model.memoryEnabled`; its `didSet`
/// re-points `MemoryStore.shared.isModelEnabled`, which the tool catalog and
/// both prompt assemblers read.
struct MemorySettingsPaneView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    var body: some View {
        Form {
            enableSection
            profileSection
            locationSection
            contentsSection
        }
        .formStyle(.grouped)
        .padding(16)
    }

    private var enableSection: some View {
        Section(header: Text("Memory", bundle: .module)) {
            HStack(alignment: .center, spacing: 18) {
                TSIdlingMemorySparkView(size: 80)
                    .opacity(model.memoryEnabled ? 1.0 : 0.65)
                    .animation(TSMotion.select, value: model.memoryEnabled)

                VStack(alignment: .leading, spacing: 8) {
                    Toggle(isOn: Binding(
                        get: { model.memoryEnabled },
                        set: { newValue in
                            model.memoryEnabled = newValue
                            model.persistSettingsDebounced()
                        })) {
                        Text("Let the model remember across conversations", bundle: .module)
                    }
                    .settingsControl("Let the model remember across conversations", pane: .memory, timing: .nextTurn)

                    Text("Memory is profile-specific. Project memory remains separate. Type `#` or `/memory text` to save a dated entry, or `/memory` to open the profile memory folder.", bundle: .module)
                        .font(theme.ui(.small))
                        .foregroundStyle(.appSecondary)
                }
            }
            .padding(.vertical, 4)
        }
            .settingsControl("Memory", pane: .memory, timing: .nextTurn)
    }

    private var profileSection: some View {
        Section(header: Text("Profile Memory", bundle: .module)) {
            HStack {
                Text(ProfileMemoryStore.shared.fileURL.path)
                    .font(theme.code(.small))
                    .foregroundStyle(.appSecondary)
                    .lineLimit(1)
                    .truncationMode(.head)
                Spacer()
                Button {
                    NSWorkspace.shared.open(ProfileMemoryStore.shared.directory)
                } label: { Text("Open Folder", bundle: .module) }
            }
            TextField("Arctic embedding model path or alias", text: $model.memoryEmbeddingModel)
                .onSubmit { model.persistSettingsDebounced() }
            Text("Embeddings are optional. Without a configured model, profile memory uses bounded Markdown and lexical recall.", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
            HStack {
                Label(
                    ProfileMemoryStore.shared.hasIndex
                        ? "Index: \(ProfileMemoryStore.shared.indexedModel ?? "ready")"
                        : "Index: not built",
                    systemImage: "circle"
                )
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
                Spacer()
                Button {
                    Task { try? await ProfileMemoryStore.shared.rebuildIndex(modelPath: model.memoryEmbeddingModel) }
                } label: { Text("Rebuild Index", bundle: .module) }
                .disabled(model.memoryEmbeddingModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                Button {
                    ProfileMemoryStore.shared.clearIndex()
                } label: { Text("Clear Index", bundle: .module) }
                .disabled(!ProfileMemoryStore.shared.hasIndex)
            }
        }
            .settingsControl("Profile Memory", pane: .memory, timing: .nextTurn)
    }

    /// The selected chat's project root, by the same resolution the submit
    /// path makes. Nil with no attached project.
    private var selectedProjectRoot: URL? {
        let project = model.turnProject(chatID: model.selectedChatID)
            ?? (model.interactionMode == .projects ? model.selectedProject : nil)
        guard let root = project?.rootDirectoryURL, !root.path.isEmpty else {
            return nil
        }
        return root
    }

    private var selectedMemoryDirectory: URL? {
        selectedProjectRoot.map { MemoryStore.shared.directory(forProjectRoot: $0) }
    }

    private var locationSection: some View {
        Section(header: Text("Storage", bundle: .module)) {
            if let directory = selectedMemoryDirectory {
                HStack {
                    Text(directory.path)
                        .font(theme.code(.small))
                        .foregroundStyle(.appSecondary)
                        .lineLimit(1)
                        .truncationMode(.head)
                    Spacer()
                    Button {
                        NSWorkspace.shared.open(directory)
                    } label: { Text("Open Folder", bundle: .module) }
                }
            } else {
                HStack {
                    Text("No project attached. The shared memory root:", bundle: .module)
                        .font(theme.ui(.small))
                        .foregroundStyle(.appSecondary)
                    Spacer()
                    Button {
                        NSWorkspace.shared.open(MemoryStore.defaultBase())
                    } label: { Text("Open Folder", bundle: .module) }
                }
            }
        }
            .settingsControl("Storage", pane: .memory, timing: .nextTurn)
    }

    private var contentsSection: some View {
        Section(header: Text("What This Project Remembers", bundle: .module)) {
            if let root = selectedProjectRoot {
                let entries = MemoryStore.parseIndex(
                    MemoryStore.shared.loadIndex(forProjectRoot: root))
                if entries.isEmpty {
                    Text("Nothing yet. The model saves memories as it learns durable facts; you can save one with `#` followed by text.", bundle: .module)
                        .font(theme.ui(.small))
                        .foregroundStyle(.appSecondary)
                } else {
                    ForEach(entries, id: \.fileName) { entry in
                        VStack(alignment: .leading, spacing: 2) {
                            HStack {
                                Text(entry.title)
                                    .font(theme.ui(.small, weight: .semibold))
                                Spacer()
                                Text(entry.fileName)
                                    .font(theme.code(.small))
                                    .foregroundStyle(.appSecondary)
                            }
                            if !entry.hook.isEmpty {
                                Text(entry.hook)
                                    .font(theme.ui(.small))
                                    .foregroundStyle(.appSecondary)
                            }
                        }
                        .padding(.vertical, 2)
                    }
                }
            } else {
                Text("Attach a project to see its memories.", bundle: .module)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
            }
        }
            .settingsControl("What This Project Remembers", pane: .memory, timing: .nextTurn)
    }
}
