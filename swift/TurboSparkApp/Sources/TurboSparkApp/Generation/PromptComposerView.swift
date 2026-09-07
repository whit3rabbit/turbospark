import SwiftUI
import UniformTypeIdentifiers

struct PromptComposerView: View {
    @ObservedObject var model: AppModel
    @Environment(\.appTheme) private var theme
    @FocusState private var promptFocused: Bool
    @StateObject private var autocomplete = ComposerAutocompleteController()
    @State private var showingPromptTips = false
    @State private var isImportingDocuments = false
    @State private var isImportingFolder = false
    @State private var isExtractingDocuments = false
    @State private var documentImportError: String?
    @State private var showingProjectSettingsSheet = false
    @State private var showingAddMcpSheet = false
    @State private var isCreatingSkill = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !model.promptAttachments.isEmpty {
                PromptAttachmentsView(model: model)
            }
            if let documentImportError {
                Text(documentImportError)
                    .themedFont(.small)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if autocomplete.isVisible {
                ComposerSuggestionPopup(controller: autocomplete) {
                    acceptAutocomplete()
                }
            }
            QueuedPromptsSection(model: model)
            PromptComposerEditor(
                model: model, promptFocused: $promptFocused, autocomplete: autocomplete)
            if model.isInGhostChat {
                // Under the text box, before sending: the one place the user
                // is certain to look as they compose. Persistent rather than
                // first-send-only, so a return to the chat re-warns too.
                HStack(spacing: 4) {
                    Image(systemName: "ghost")
                        .accessibilityHidden(true)
                    Text("Temporary chat: this conversation can't be recovered.", bundle: .module)
                }
                .font(theme.ui(points: 11, weight: .medium))
                .foregroundStyle(.secondary)
                .accessibilityElement(children: .combine)
            }
            PromptComposerFooter(
                model: model,
                showingPromptTips: $showingPromptTips,
                isExtractingDocuments: isExtractingDocuments,
                onAttachFiles: {
                    documentImportError = nil
                    isImportingDocuments = true
                },
                onAttachFolder: {
                    documentImportError = nil
                    isImportingFolder = true
                },
                onNewProject: {
                    showingProjectSettingsSheet = true
                },
                onAddMcpServer: {
                    showingAddMcpSheet = true
                },
                onCreateSkill: {
                    isCreatingSkill = true
                },
                onInsertPromptText: { text in
                    insertPromptText(text)
                },
                promptFocused: $promptFocused
            )
        }
        .padding(.horizontal, 14)
        .padding(.top, 12)
        .padding(.bottom, 10)
        .background {
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .fill(theme.composerBackground)
                .overlay {
                    RoundedRectangle(cornerRadius: 20, style: .continuous)
                        .stroke(
                            promptFocused
                                ? theme.accent.opacity(0.55)
                                : Color.primary.opacity(theme.isDark ? 0.18 : 0.14),
                            lineWidth: promptFocused ? 1.5 : 1.0
                        )
                }
                .shadow(
                    color: Color.black.opacity(theme.isDark ? 0.28 : 0.08),
                    radius: promptFocused ? 10 : 6,
                    x: 0,
                    y: 2
                )
        }
        .animation(.easeInOut(duration: 0.15), value: promptFocused)
        .fileImporter(
            isPresented: $isImportingDocuments,
            allowedContentTypes: model.attachmentContentTypes,
            allowsMultipleSelection: true,
            onCompletion: handleDocumentSelection)
        .fileImporter(
            isPresented: $isImportingFolder,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: true,
            onCompletion: handleFolderSelection)
        .sheet(isPresented: $showingProjectSettingsSheet) {
            ProjectSettingsSheet(
                model: model,
                editingProject: nil,
                onDismiss: { showingProjectSettingsSheet = false }
            )
        }
        .sheet(isPresented: $showingAddMcpSheet) {
            McpServerEditorSheet(
                existingConfig: nil,
                workingDirectory: model.selectedProject?.rootDirectoryURL,
                existingNames: (model.globalMcpServers + (model.selectedProject?.mcpServers ?? [])).map(\.name),
                onSave: { newConfig in
                    if let project = model.selectedProject {
                        model.addProjectMcpServer(projectID: project.id, config: newConfig)
                    } else {
                        model.addGlobalMcpServer(newConfig)
                    }
                    showingAddMcpSheet = false
                },
                onDismiss: { showingAddMcpSheet = false }
            )
        }
        .sheet(isPresented: $isCreatingSkill) {
            SkillEditorSheet(
                model: model,
                skillToEdit: nil,
                defaultScope: (model.selectedProject?.rootDirectoryPath).map { .projectLocal(projectPath: $0) } ?? .userGlobal
            )
        }
        .onReceive(NotificationCenter.default.publisher(for: .focusPrompt)) { _ in
            // Cmd+L from the menu moves focus to the prompt editor so a
            // keyboard-only user can start typing without reaching for the
            // mouse.
            promptFocused = true
        }
        .onChange(of: model.promptText) { _, newValue in
            // The `/` and `@` popup rides the draft text: every keystroke
            // re-derives the trigger from the trailing token.
            autocomplete.textChanged(
                text: newValue,
                skills: model.effectiveSkills,
                projectRoot: model.selectedProject?.rootDirectoryURL)
        }
        .onChange(of: promptFocused) { _, isFocused in
            autocomplete.focusChanged(isFocused: isFocused)
        }
    }

    private func acceptAutocomplete() {
        if let newText = autocomplete.accept(in: model.promptText) {
            model.promptText = newText
        }
    }

    private func handleDocumentSelection(_ result: Result<[URL], any Error>) {
        switch result {
        case .success(let urls):
            importDocuments(urls)
        case .failure(let error):
            documentImportError = error.localizedDescription
        }
    }

    private func handleFolderSelection(_ result: Result<[URL], any Error>) {
        switch result {
        case .success(let urls):
            importFolders(urls)
        case .failure(let error):
            documentImportError = error.localizedDescription
        }
    }

    private func importDocuments(_ urls: [URL]) {
        guard !urls.isEmpty else { return }
        let targetChatID = model.selectedChatID
        isExtractingDocuments = true
        documentImportError = nil

        Task {
            let outcome = await AttachmentImporter.importDocuments(
                urls,
                into: model,
                chatID: targetChatID)
            documentImportError = outcome.errorText
            isExtractingDocuments = false
        }
    }

    private func importFolders(_ urls: [URL]) {
        guard !urls.isEmpty else { return }
        let targetChatID = model.selectedChatID
        isExtractingDocuments = true
        documentImportError = nil

        Task {
            var totalCount = 0
            var allFailures: [String] = []
            for url in urls {
                let outcome = await AttachmentImporter.importFolder(
                    url,
                    into: model,
                    chatID: targetChatID)
                totalCount += outcome.importedCount
                allFailures.append(contentsOf: outcome.failures)
            }
            documentImportError = allFailures.isEmpty ? nil : allFailures.joined(separator: "\n")
            if totalCount > 0 {
                model.showToast("Attached \(totalCount) files from folder.", style: .info)
            }
            isExtractingDocuments = false
        }
    }

    private func insertPromptText(_ text: String) {
        if model.promptText.isEmpty {
            model.promptText = text
        } else if model.promptText.hasSuffix("\n") || model.promptText.hasSuffix(" ") {
            model.promptText += text
        } else {
            model.promptText += " " + text
        }
        promptFocused = true
    }
}

/// The chat's queued prompts, shown between the composer's attachments and
/// its editor while a turn is running.
///
/// Claude Code reference: the REPL's queued-input row (`messageQueueManager`
/// consumers). Each pill is a parked draft; clicking one PULLS IT BACK into
/// the composer for editing (`restoreQueuedMessage` removes it from the
/// queue as it restores), because a queued prompt the user can no longer
/// correct is a trap -- the drain will send it verbatim.
private struct QueuedPromptsSection: View {
    @ObservedObject var model: AppModel
    @Environment(\.appTheme) private var theme

    var body: some View {
        let queued = model.queuedMessages(for: model.selectedChatID)
        if !queued.isEmpty {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(queued) { entry in
                    Button {
                        model.restoreQueuedMessage(id: entry.id)
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "text.badge.clock")
                                .accessibilityHidden(true)
                            Text(entry.text.isEmpty
                                ? "(attachment only)"
                                : String(entry.text.prefix(80)))
                                .lineLimit(1)
                                .truncationMode(.tail)
                            Spacer(minLength: 0)
                            Image(systemName: "pencil")
                                .accessibilityHidden(true)
                        }
                        .themedFont(.small)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 5)
                        .background(
                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .fill(theme.composerBackground.opacity(0.6))
                        )
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Queued message. Click to edit it before it sends.")
                    .help("Queued: sends after the current turn. Click to edit it now.")
                }
                Text("Queued behind the running turn; drains when it finishes.",
                     bundle: .module)
                    .themedFont(points: 10)
                    .foregroundStyle(.tertiary)
            }
        }
    }
}
