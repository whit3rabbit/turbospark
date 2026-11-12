import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The Files section: every document attached to any chat, in one list.
///
/// Attachments were previously reachable only as chips on the composer of the
/// chat that owned them, so a document attached three conversations ago was
/// invisible and could not be removed without finding that chat first.
@MainActor
struct FilesSectionView: View {
    @ObservedObject var model: AppModel

    @State private var searchText = ""
    @State private var isImporting = false
    @State private var isExtracting = false
    @State private var importError: String?

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if let importError {
                errorRow(importError)
                Divider()
            }
            if rows.isEmpty {
                emptyState
            } else {
                list
            }
        }
        .fileImporter(
            isPresented: $isImporting,
            allowedContentTypes: model.attachmentContentTypes,
            allowsMultipleSelection: true,
            onCompletion: handleSelection)
    }

    // MARK: - Header

    /// The search field and Add files action only appear once something is
    /// attached: with nothing to filter, the empty state below carries the
    /// one Add files action, so the pane never shows two primary buttons.
    private var header: some View {
        AppAdaptiveHeader {
            VStack(alignment: .leading, spacing: 1) {
                Text("Files", bundle: .module)
                    .themedFont(.callout, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)
                summaryText
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }
        } actions: {
            if !model.allAttachments.isEmpty {
                AppSearchField(text: $searchText, prompt: Text("Filter files...", bundle: .module))
                addFilesButton
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    /// The one Add files action, used by the header and the empty state.
    /// The icon turns into a spinner while documents are extracted, and the
    /// action stays disabled until extraction ends or the model finishes.
    private var addFilesButton: some View {
        Button {
            importError = nil
            isImporting = true
        } label: {
            Label {
                Text("Add files", bundle: .module)
            } icon: {
                if isExtracting {
                    ProgressView().controlSize(.mini)
                } else {
                    Image(systemName: "plus")
                }
            }
        }
        .appAction(.primary)
        .disabled(isExtracting || model.isRunning)
        .help(Text("Attach documents to the active chat", bundle: .module))
    }

    private var summaryText: Text {
        let all = model.allAttachments
        guard !all.isEmpty else { return Text("No documents attached", bundle: .module) }
        let count = Text("Documents: \(all.count)", bundle: .module)
        guard let size = MetricFormat.fileSize(model.allAttachmentsByteSize) else { return count }
        return count + Text(verbatim: " \u{2022} \(size)")
    }

    private func errorRow(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.appWarning)
                .accessibilityHidden(true)
            Text(text)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
            Button { importError = nil } label: { Text("Dismiss", bundle: .module) }
                .appAction(.quiet, size: .small)
        }
        .themedFont(.small)
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    // MARK: - List

    private var rows: [AppAttachmentReference] {
        let all = model.allAttachments
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return all }
        return all.filter {
            $0.attachment.fileName.localizedCaseInsensitiveContains(query)
                || $0.chatTitle.localizedCaseInsensitiveContains(query)
        }
    }

    private var list: some View {
        ScrollView {
            LazyVStack(spacing: 2) {
                ForEach(rows) { row in
                    FileRowView(
                        model: model,
                        reference: row,
                        isSelected: model.previewAttachmentID == row.id)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 10)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// Keyed on whether anything is attached at all, not on the search
    /// text: detaching the last file while a filter is typed must still land
    /// on the Add files state, since the header hides the field by then.
    @ViewBuilder
    private var emptyState: some View {
        if model.allAttachments.isEmpty {
            AppEmptyState(
                systemImage: "folder",
                title: Text("No files attached", bundle: .module),
                message: Text(
                    "Attach PDFs, Office documents, or source files to give a chat something to read.",
                    bundle: .module)
            ) {
                addFilesButton
            }
        } else {
            AppEmptyState(
                systemImage: "magnifyingglass",
                title: Text("No matching files", bundle: .module),
                message: Text("Nothing matches \u{201C}\(searchText)\u{201D}.", bundle: .module)
            ) {
                Button { searchText = "" } label: { Text("Clear search", bundle: .module) }
                    .appAction(.secondary)
            }
        }
    }

    // MARK: - Import

    private func handleSelection(_ result: Result<[URL], any Error>) {
        switch result {
        case .success(let urls):
            runImport(urls)
        case .failure(let error):
            importError = error.localizedDescription
        }
    }

    private func runImport(_ urls: [URL]) {
        guard !urls.isEmpty else { return }
        let chatID = model.selectedChatID
        isExtracting = true
        importError = nil
        Task {
            let outcome = await AttachmentImporter.importDocuments(urls, into: model, chatID: chatID)
            importError = outcome.errorText
            isExtracting = false
            if outcome.importedCount > 0 {
                model.showToast(
                    AppStrings.localized("Documents attached to this chat: \(outcome.importedCount)"),
                    style: .success)
            }
        }
    }
}

/// One row of the Files list.
@MainActor
private struct FileRowView: View {
    @ObservedObject var model: AppModel
    let reference: AppAttachmentReference
    let isSelected: Bool

    @State private var isHovering = false

    private var attachment: AppPromptAttachment { reference.attachment }

    var body: some View {
        HStack(spacing: 10) {
            AttachmentThumbnailView(
                url: attachment.thumbnailSourceURL,
                pixelSize: 28,
                fallbackSymbol: attachment.symbolName,
                symbolStep: .callout,
                symbolTint: isSelected ? TurboSparkTheme.accentColor : Color.secondary
            )
            .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 2) {
                Text(attachment.fileName)
                    .themedFont(.small, weight: isSelected ? .semibold : .regular)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(detailText)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .lineLimit(1)
            }

            Spacer(minLength: 4)

            if !attachment.sourceExists {
                Image(systemName: "questionmark.folder")
                    .themedFont(.tiny)
                    .foregroundStyle(.appWarning)
                    .help(Text("The source file has moved; only the extracted text remains.", bundle: .module))
            }

            if isHovering || isSelected {
                rowActions
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(
            isSelected ? TurboSparkTheme.accentColor.opacity(0.12)
                : (isHovering ? Color.primary.opacity(0.04) : Color.clear),
            in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .onHover { isHovering = $0 }
        .onTapGesture { model.showPreview(attachmentID: attachment.id) }
        .contextMenu { actionMenuItems }
        .help(Text("Preview \(attachment.fileName)", bundle: .module))
        .accessibilityElement(children: .contain)
        .accessibilityLabel(attachment.fileName)
        .accessibilityValue(detailText)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .accessibilityHint(Text("Opens this document in the preview pane", bundle: .module))
    }

    private var detailText: String { reference.listDetailText }

    /// The row's frequent actions as quiet, named icon buttons, the same
    /// treatment every list row uses. Preview is the row tap itself, and the
    /// full set (Open with default app included) stays in the context menu.
    private var rowActions: some View {
        HStack(spacing: 2) {
            AppIconButton("bubble.left", label: Text("Go to chat", bundle: .module)) {
                goToChat()
            }
            if attachment.sourceExists {
                AppIconButton("folder", label: Text("Reveal in Finder", bundle: .module)) {
                    model.revealAttachmentInFinder(attachment)
                }
            }
            AppIconButton("trash", label: Text("Detach", bundle: .module), role: .destructive) {
                model.removeAttachment(reference: reference)
            }
            .disabled(model.isRunning)
        }
        .appAction(.quiet, size: .small)
        .foregroundStyle(.appSecondary)
    }

    @ViewBuilder
    private var actionMenuItems: some View {
        Button {
            model.showPreview(attachmentID: attachment.id)
        } label: {
            Label { Text("Preview", bundle: .module) } icon: { Image(systemName: "eye") }
        }
        Button {
            goToChat()
        } label: {
            Label { Text("Go to chat", bundle: .module) } icon: { Image(systemName: "bubble.left") }
        }
        if attachment.sourceExists {
            Divider()
            Button {
                model.revealAttachmentInFinder(attachment)
            } label: {
                Label { Text("Reveal in Finder", bundle: .module) } icon: { Image(systemName: "folder") }
            }
            Button {
                model.openAttachmentExternally(attachment)
            } label: {
                Label { Text("Open with default app", bundle: .module) } icon: { Image(systemName: "arrow.up.forward.app") }
            }
        }
        Divider()
        Button(role: .destructive) {
            model.removeAttachment(reference: reference)
        } label: {
            Label { Text("Detach", bundle: .module) } icon: { Image(systemName: "trash") }
        }
        .disabled(model.isRunning)
    }

    private func goToChat() {
        model.selectChat(id: reference.chatID)
        model.activeSection = .chat
    }
}
