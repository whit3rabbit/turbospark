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
            locationSection
            contentsSection
        }
        .formStyle(.grouped)
        .padding(16)
    }

    private var enableSection: some View {
        Section("Memory") {
            Toggle("Let the model remember across conversations", isOn: Binding(
                get: { model.memoryEnabled },
                set: { newValue in
                    model.memoryEnabled = newValue
                    model.persistSettingsDebounced()
                }))
            .settingsControl("Let the model remember across conversations", pane: .memory, timing: .nextTurn)
            Text(
                "With a project attached, the model gets a persistent memory directory and a "
                    + "`memory` tool: it saves durable facts about you and the project as it "
                    + "learns them, and an index of what it remembers is always in its context. "
                    + "Memories live on disk, one folder per project. Type `#` followed by text "
                    + "to save one yourself, or /memory to open the folder."
            )
            .font(theme.ui(.small))
            .foregroundStyle(.appSecondary)
        }
            .settingsControl("Memory", pane: .memory, timing: .nextTurn)
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
        Section("Storage") {
            if let directory = selectedMemoryDirectory {
                HStack {
                    Text(directory.path)
                        .font(theme.code(.small))
                        .foregroundStyle(.appSecondary)
                        .lineLimit(1)
                        .truncationMode(.head)
                    Spacer()
                    Button("Open Folder") {
                        NSWorkspace.shared.open(directory)
                    }
                }
            } else {
                HStack {
                    Text("No project attached. The shared memory root:", bundle: .module)
                        .font(theme.ui(.small))
                        .foregroundStyle(.appSecondary)
                    Spacer()
                    Button("Open Folder") {
                        NSWorkspace.shared.open(MemoryStore.defaultBase())
                    }
                }
            }
        }
            .settingsControl("Storage", pane: .memory, timing: .nextTurn)
    }

    private var contentsSection: some View {
        Section("What This Project Remembers") {
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
