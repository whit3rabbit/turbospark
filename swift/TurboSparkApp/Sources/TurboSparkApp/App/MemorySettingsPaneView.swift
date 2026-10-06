import AppKit
import SwiftUI

/// Settings pane for approved, proposed, and legacy memory in the encrypted profile.
///
/// The toggle is the ONLY writer of `model.memoryEnabled`; its `didSet`
/// re-points `MemoryStore.shared.isModelEnabled`, which the tool catalog and
/// both prompt assemblers read.
struct MemorySettingsPaneView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @State private var ledgerState = MemoryLedgerState()
    @State private var scopeSelection = "profile"
    @State private var statusSelection = "all"
    @State private var searchText = ""
    @State private var draftText = ""
    @State private var editingClaim: MemoryClaim?
    @State private var managerError: String?

    var body: some View {
        Form {
            enableSection
            profileSection
            managerSection
            reflectionsSection
        }
        .formStyle(.grouped)
        .padding(16)
        .onAppear { reloadLedger() }
        .sheet(item: $editingClaim) { claim in
            VStack(alignment: .leading, spacing: 12) {
                Text("Edit Memory", bundle: .module).themedFont(.title3)
                TextEditor(text: $draftText)
                    .frame(minWidth: 520, minHeight: 220)
                HStack {
                    Button { editingClaim = nil } label: { Text("Cancel", bundle: .module) }
                    Spacer()
                    Button {
                        perform { try MemoryLedgerStore.shared.edit(claim.id, text: draftText) }
                        editingClaim = nil
                    } label: { Text("Save", bundle: .module) }
                    .disabled(draftText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
            .padding(20)
        }
    }

    private var selectedScope: String {
        if scopeSelection == "project", let root = selectedProjectRoot {
            return "project:\(MemoryStore.projectKey(forProjectRoot: root))"
        }
        return "profile"
    }

    private var visibleClaims: [MemoryClaim] {
        ledgerState.claims.filter { claim in
            claim.scope == selectedScope &&
            (statusSelection == "all" || claim.status.rawValue == statusSelection) &&
            (searchText.isEmpty || claim.text.localizedCaseInsensitiveContains(searchText))
        }.sorted { $0.updatedAt > $1.updatedAt }
    }

    private var visibleReflections: [MemoryReflection] {
        ledgerState.reflections.sorted { $0.date > $1.date }
    }

    private func reloadLedger() {
        try? MemoryLedgerStore.shared.importLegacyIfNeeded()
        ledgerState = MemoryLedgerStore.shared.snapshot()
    }

    private func perform(_ action: () throws -> Void) {
        do {
            try action()
            managerError = nil
            reloadLedger()
        } catch {
            managerError = error.localizedDescription
        }
    }

    private var managerSection: some View {
        Section(header: Text("Memory Manager (\(ledgerState.claims.filter { $0.status == .pending }.count) pending)", bundle: .module)) {
            HStack {
                Picker("Scope", selection: $scopeSelection) {
                    Text("Profile", bundle: .module).tag("profile")
                    if selectedProjectRoot != nil {
                        Text("This Project", bundle: .module).tag("project")
                    }
                }
                .pickerStyle(.segmented)
                Picker("Status", selection: $statusSelection) {
                    Text("All", bundle: .module).tag("all")
                    ForEach(MemoryClaimStatus.allCases, id: \.rawValue) { status in
                        Text(verbatim: status.rawValue.capitalized).tag(status.rawValue)
                    }
                }
                Button { reloadLedger() } label: { Text("Refresh", bundle: .module) }
            }
            TextField("Search memories", text: $searchText)
            if let managerError {
                Text(verbatim: managerError).foregroundStyle(.red)
            }
            if visibleClaims.isEmpty {
                Text("No memories match this view.", bundle: .module)
                    .foregroundStyle(.appSecondary)
            }
            ForEach(visibleClaims) { claim in
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text(verbatim: claim.kind.capitalized).font(theme.ui(.small, weight: .semibold))
                        Text(verbatim: claim.status.rawValue)
                            .font(theme.ui(.small)).foregroundStyle(.appSecondary)
                        Spacer()
                        Text(verbatim: String(claim.id.uuidString.prefix(8)))
                            .font(theme.code(.small)).foregroundStyle(.appSecondary)
                    }
                    Text(verbatim: claim.text.isEmpty ? "(forgotten)" : claim.text)
                        .textSelection(.enabled)
                    Text(verbatim: "Evidence: \(claim.confidence)")
                        .font(theme.ui(.small)).foregroundStyle(.appSecondary)
                    ForEach(claim.evidence) { evidence in
                        HStack {
                            Text(verbatim: "\"\(evidence.quote)\"")
                                .font(theme.ui(.small)).textSelection(.enabled)
                            Button {
                                model.selectChat(id: evidence.chatID)
                                Task { @MainActor in
                                    await Task.yield()
                                    model.turnNavigationTargetID = evidence.messageID
                                    model.turnNavigationToken += 1
                                }
                            } label: { Text("Open Source Message", bundle: .module) }
                            .disabled(!model.chats.contains(where: { $0.id == evidence.chatID }))
                        }
                    }
                    HStack {
                        if claim.status == .pending || claim.status == .legacy {
                            Button {
                                perform { try MemoryLedgerStore.shared.approve(claim.id) }
                            } label: { Text("Approve", bundle: .module) }
                            let replacements = ledgerState.claims.filter {
                                $0.scope == claim.scope && $0.status == .active && $0.id != claim.id
                            }
                            if !replacements.isEmpty {
                                Menu {
                                    ForEach(replacements) { existing in
                                        Button("Replace \(String(existing.text.prefix(48)))") {
                                            perform { try MemoryLedgerStore.shared.approve(
                                                claim.id, supersedes: existing.id) }
                                        }
                                        Button("Merge with \(String(existing.text.prefix(48)))") {
                                            perform { try MemoryLedgerStore.shared.approve(
                                                claim.id,
                                                text: existing.text + "\n" + claim.text,
                                                supersedes: existing.id) }
                                        }
                                    }
                                } label: { Text("Merge or Replace", bundle: .module) }
                            }
                        }
                        if claim.status == .pending {
                            Button {
                                perform { try MemoryLedgerStore.shared.dismiss(claim.id) }
                            } label: { Text("Dismiss", bundle: .module) }
                        }
                        if claim.status == .active {
                            Button {
                                draftText = claim.text
                                editingClaim = claim
                            } label: { Text("Edit", bundle: .module) }
                            Button(role: .destructive) {
                                perform { try MemoryLedgerStore.shared.forget(claim.id) }
                            } label: { Text("Forget", bundle: .module) }
                        }
                    }
                }
                .padding(.vertical, 5)
            }
            Text("Forgetting removes this profile's memory records and future exports. Existing external backups must be deleted separately. Original chat messages remain visible in chat history.", bundle: .module)
                .font(theme.ui(.small)).foregroundStyle(.appSecondary)
            Text(verbatim: "Base prompt preview: \(promptPreview)")
                .font(theme.code(.small)).textSelection(.enabled)
        }
    }

    private var promptPreview: String {
        guard model.memoryEnabled else { return "Model memory is disabled." }
        let profile = MemoryPromptBuilder.profileSection(approvedOnly: true)
        guard let root = selectedProjectRoot else { return profile }
        return profile + "\n\n" + MemoryPromptBuilder.section(
            store: .shared, projectRoot: root, approvedOnly: true)
    }

    private var reflectionsSection: some View {
        Section(header: Text("Nightly Reflections", bundle: .module)) {
            if ledgerState.reflections.isEmpty {
                Text("No reflections yet.", bundle: .module)
                    .foregroundStyle(.appSecondary)
            }
            ForEach(visibleReflections, id: \.id) { reflection in
                VStack(alignment: .leading, spacing: 5) {
                    Text(reflection.date, style: .date)
                    Text(verbatim: reflection.prose).textSelection(.enabled)
                    Text(verbatim: reflection.proposedGuidance)
                        .font(theme.ui(.small)).foregroundStyle(.appSecondary)
                    if !reflection.approved {
                        Button {
                            perform { try MemoryLedgerStore.shared.approveReflection(reflection.id) }
                        } label: { Text("Activate Guidance", bundle: .module) }
                    }
                }
            }
        }
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

                    Toggle(isOn: Binding(
                        get: { model.memoryAutoCaptureEnabled },
                        set: { value in
                            model.memoryAutoCaptureEnabled = value
                            model.persistSettingsDebounced()
                        })) {
                        Text("Suggest memories from idle conversations", bundle: .module)
                    }
                    .disabled(!model.memoryEnabled)

                    Text("Use # or /memory text for profile memory. Use /memory project text for this project's memory. Review model suggestions below.", bundle: .module)
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
            Text("Claims and evidence are stored in the encrypted profile. Profile export includes the ledger and readable Markdown projections.", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
            TextField("Arctic embedding model path or alias", text: $model.memoryEmbeddingModel)
                .onSubmit { model.persistSettingsDebounced() }
            Text("Embeddings are optional. Approved claims remain searchable without a model.", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
            HStack {
                Label(
                    MemoryLedgerStore.shared.hasEmbeddingIndex(modelPath: model.memoryEmbeddingModel)
                        ? "Index: \(model.memoryEmbeddingModel)"
                        : "Index: not built",
                    systemImage: "circle"
                )
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
                Spacer()
                Button {
                    Task {
                        do {
                            try await MemoryLedgerStore.shared.rebuildEmbeddings(
                                modelPath: model.memoryEmbeddingModel)
                            reloadLedger()
                        } catch { managerError = error.localizedDescription }
                    }
                } label: { Text("Rebuild Index", bundle: .module) }
                .disabled(model.memoryEmbeddingModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                Button {
                    perform { try MemoryLedgerStore.shared.clearEmbeddings() }
                } label: { Text("Clear Index", bundle: .module) }
                .disabled(!MemoryLedgerStore.shared.hasEmbeddingIndex(modelPath: model.memoryEmbeddingModel))
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

}
