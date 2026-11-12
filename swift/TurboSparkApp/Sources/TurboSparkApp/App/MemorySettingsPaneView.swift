import AppKit
import SwiftUI

/// Settings pane for approved, proposed, and legacy memory in the encrypted profile.
///
/// The toggle is the ONLY writer of `model.memoryEnabled`; its `didSet`
/// re-points `MemoryStore.shared.isModelEnabled`, which the tool catalog and
/// both prompt assemblers read.
// Isolated explicitly: its helpers call each other, and only `body` is
// isolated by the protocol on the macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
@MainActor
struct MemorySettingsPaneView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @State private var ledgerState = MemoryLedgerState()
    @State private var scopeSelection = "profile"
    @State private var statusSelection = "all"
    @State private var searchText = ""
    @State private var draftText = ""
    @State private var editingClaim: MemoryClaim?
    @State private var claimToForget: MemoryClaim?
    @State private var managerError: String?

    var body: some View {
        Form {
            Text("Let the model remember across conversations", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)

            enableSection
            profileSection
            managerSection
            reflectionsSection
        }
        .formStyle(.grouped)
        // The grouped Form paints its own system gray, which ignores the theme.
        .scrollContentBackground(.hidden)
        .background(.appPage)
        .padding(SettingsLayout.formInsets)
        .onAppear { reloadLedger() }
        .sheet(item: $editingClaim) { claim in
            MemoryEditSheet(draftText: $draftText) {
                // Keep the sheet (and the draft) open when the edit fails.
                let saved = perform({ try MemoryLedgerStore.shared.edit(claim.id, text: draftText) })
                if saved { editingClaim = nil }
                return saved
            } onCancel: {
                editingClaim = nil
            }
        }
        .confirmationDialog(
            Text("Forget", bundle: .module),
            isPresented: Binding(
                get: { claimToForget != nil },
                set: { if !$0 { claimToForget = nil } }
            ),
            titleVisibility: .visible
        ) {
            Button(role: .destructive) {
                if let target = claimToForget {
                    perform { try MemoryLedgerStore.shared.forget(target.id) }
                    claimToForget = nil
                }
            } label: {
                Text("Forget", bundle: .module)
            }
            Button(role: .cancel) {
                claimToForget = nil
            } label: {
                Text("Cancel", bundle: .module)
            }
        } message: {
            Text("Forgetting removes this profile's memory records and future exports. Existing external backups must be deleted separately. Original chat messages remain visible in chat history.", bundle: .module)
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

    @discardableResult
    private func perform(_ action: () throws -> Void) -> Bool {
        do {
            try action()
            managerError = nil
            reloadLedger()
            return true
        } catch {
            managerError = error.localizedDescription
            return false
        }
    }

    private var managerSection: some View {
        Section(header: Text("Memory Manager (\(ledgerState.claims.filter { $0.status == .pending }.count) pending)", bundle: .module)) {
            HStack(spacing: AppMetrics.actionSpacing) {
                Picker(selection: $scopeSelection) {
                    Text("Profile", bundle: .module).tag("profile")
                    if selectedProjectRoot != nil {
                        Text("This Project", bundle: .module).tag("project")
                    }
                } label: { Text("Scope", bundle: .module) }
                .pickerStyle(.segmented)
                Picker(selection: $statusSelection) {
                    Text("All", bundle: .module).tag("all")
                    ForEach(MemoryClaimStatus.allCases, id: \.rawValue) { status in
                        MemoryClaimText.status(status).tag(status.rawValue)
                    }
                } label: { Text("Status", bundle: .module) }
                Button { reloadLedger() } label: { Text("Refresh", bundle: .module) }
                    .appAction(.secondary, size: .small)
            }
            AppSearchField(text: $searchText, prompt: Text("Search memories...", bundle: .module), maxWidth: .infinity)
            if let managerError {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(AppAttention.color)
                    Text(verbatim: managerError)
                        .font(theme.ui(.small))
                        .foregroundStyle(.appSecondary)
                }
            }
            if visibleClaims.isEmpty {
                AppEmptyState(
                    systemImage: "brain",
                    title: Text("No memories match this view.", bundle: .module)
                )
            }
            ForEach(visibleClaims) { claim in
                claimRow(claim)
            }
            Text("Forgetting removes this profile's memory records and future exports. Existing external backups must be deleted separately. Original chat messages remain visible in chat history.", bundle: .module)
                .font(theme.ui(.small)).foregroundStyle(.appSecondary)
            // The label is UI; the preview after it is the prompt text the
            // model receives, which is never translated.
            Text("Base prompt preview: \(promptPreview)", bundle: .module)
                .font(theme.code(.small)).textSelection(.enabled)
        }
    }

    private func claimRow(_ claim: MemoryClaim) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                MemoryClaimText.kind(claim.kind).font(theme.ui(.small, weight: .semibold))
                MemoryClaimText.status(claim.status)
                    .font(theme.ui(.small)).foregroundStyle(.appSecondary)
                Spacer()
                Text(verbatim: String(claim.id.uuidString.prefix(8)))
                    .font(theme.code(.small)).foregroundStyle(.appSecondary)
            }
            Group {
                if claim.text.isEmpty {
                    Text("(forgotten)", bundle: .module)
                } else {
                    Text(verbatim: claim.text)
                }
            }
            .textSelection(.enabled)
            Text("Evidence: \(MemoryClaimText.confidence(claim.confidence))", bundle: .module)
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
                    .appAction(.quiet, size: .small)
                    .disabled(!model.chats.contains(where: { $0.id == evidence.chatID }))
                }
            }
            claimActions(claim)
        }
        .padding(.vertical, 5)
    }

    private func claimActions(_ claim: MemoryClaim) -> some View {
        HStack(spacing: AppMetrics.actionSpacing) {
            if claim.status == .pending || claim.status == .legacy {
                Button {
                    perform { try MemoryLedgerStore.shared.approve(claim.id) }
                } label: { Text("Approve", bundle: .module) }
                .appAction(.secondary, size: .small)
                let replacements = ledgerState.claims.filter {
                    $0.scope == claim.scope && $0.status == .active && $0.id != claim.id
                }
                if !replacements.isEmpty {
                    Menu {
                        ForEach(replacements) { existing in
                            let excerpt = String(existing.text.prefix(48))
                            Button {
                                perform { try MemoryLedgerStore.shared.approve(
                                    claim.id, supersedes: existing.id) }
                            } label: { Text("Replace \(excerpt)", bundle: .module) }
                            Button {
                                perform { try MemoryLedgerStore.shared.approve(
                                    claim.id,
                                    text: existing.text + "\n" + claim.text,
                                    supersedes: existing.id) }
                            } label: { Text("Merge with \(excerpt)", bundle: .module) }
                        }
                    } label: { Text("Merge or Replace", bundle: .module) }
                    .appMenuAction(size: .small)
                }
            }
            if claim.status == .pending {
                Button {
                    perform { try MemoryLedgerStore.shared.dismiss(claim.id) }
                } label: { Text("Dismiss", bundle: .module) }
                .appAction(.quiet, size: .small)
            }
            if claim.status == .active {
                Button {
                    draftText = claim.text
                    editingClaim = claim
                } label: { Text("Edit", bundle: .module) }
                .appAction(.secondary, size: .small)
                Button(role: .destructive) {
                    claimToForget = claim
                } label: { Text("Forget", bundle: .module) }
                .appAction(.destructive, size: .small)
            }
        }
    }

    private var promptPreview: String {
        guard model.memoryEnabled else { return AppStrings.localized("Model memory is disabled.") }
        let profile = MemoryPromptBuilder.profileSection(approvedOnly: true)
        guard let root = selectedProjectRoot else { return profile }
        return profile + "\n\n" + MemoryPromptBuilder.section(
            store: .shared, projectRoot: root, approvedOnly: true)
    }

    private var reflectionsSection: some View {
        Section(header: Text("Nightly Reflections", bundle: .module)) {
            if ledgerState.reflections.isEmpty {
                AppEmptyState(
                    systemImage: "moon.stars",
                    title: Text("No reflections yet.", bundle: .module)
                )
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
                        .appAction(.secondary, size: .small)
                    }
                }
            }
        }
    }

    private var enableSection: some View {
        Section {
            HStack(alignment: .center, spacing: 18) {
                TSIdlingMemorySparkView(size: 80)
                    .opacity(model.memoryEnabled ? 1.0 : 0.65)
                    .tsAnimation(TSMotion.select, value: model.memoryEnabled)

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
                    .help(Text("Model memory is disabled.", bundle: .module))

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
            EmbeddingModelPicker(
                model: model, selection: $model.memoryEmbeddingModel,
                onCommit: { model.persistSettingsDebounced() })
            Text("Embeddings are optional. Approved claims remain searchable without a model.", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
            HStack {
                Label {
                    if MemoryLedgerStore.shared.hasEmbeddingIndex(modelPath: model.memoryEmbeddingModel) {
                        Text("Index: \(model.memoryEmbeddingModel)", bundle: .module)
                    } else {
                        Text("Index: not built", bundle: .module)
                    }
                } icon: {
                    Image(systemName: "circle")
                }
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
                .appAction(.secondary, size: .small)
                .disabled(model.memoryEmbeddingModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                Button {
                    perform { try MemoryLedgerStore.shared.clearEmbeddings() }
                } label: { Text("Clear Index", bundle: .module) }
                .appAction(.secondary, size: .small)
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
