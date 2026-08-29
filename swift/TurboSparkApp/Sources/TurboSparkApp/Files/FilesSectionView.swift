import SwiftUI

/// The Files section: every document attached to any chat, in one list.
///
/// Attachments were previously reachable only as chips on the composer of the
/// chat that owned them, so a document attached three conversations ago was
/// invisible and could not be removed without finding that chat first.
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
            allowedContentTypes: DocumentTextExtractor.supportedContentTypes,
            allowsMultipleSelection: true,
            onCompletion: handleSelection)
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 1) {
                Text("Files")
                    .font(.system(size: 14, weight: .semibold))
                    .accessibilityAddTraits(.isHeader)
                Text(summaryText)
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 8)

            searchField

            Button {
                importError = nil
                isImporting = true
            } label: {
                Group {
                    if isExtracting {
                        ProgressView().controlSize(.small)
                    } else {
                        Label("Add files", systemImage: "plus")
                            .font(.system(size: 11, weight: .medium))
                    }
                }
                .frame(height: 22)
                .padding(.horizontal, 9)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .background(TurboSparkTheme.accentColor.opacity(0.14), in: Capsule())
            .foregroundStyle(TurboSparkTheme.accentColor)
            .disabled(isExtracting || model.isRunning)
            .help("Attach documents to the active chat")
            .accessibilityLabel("Add files to the active chat")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private var searchField: some View {
        HStack(spacing: 5) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 10))
                .foregroundStyle(.tertiary)
                .accessibilityHidden(true)
            TextField("Filter", text: $searchText)
                .textFieldStyle(.plain)
                .font(.system(size: 11))
                .frame(width: 120)
            if !searchText.isEmpty {
                Button {
                    searchText = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 10))
                }
                .buttonStyle(.plain)
                .foregroundStyle(.tertiary)
                .help("Clear filter")
                .accessibilityLabel("Clear filter")
            }
        }
        .padding(.horizontal, 8)
        .frame(height: 22)
        .background(TurboSparkTheme.surfaceColor, in: Capsule())
        .overlay { Capsule().stroke(TurboSparkTheme.hairlineColor, lineWidth: 0.5) }
        .accessibilityLabel("Filter files by name")
    }

    private var summaryText: String {
        let all = model.allAttachments
        guard !all.isEmpty else { return "No documents attached" }
        let sizePart = MetricFormat.fileSize(model.allAttachmentsByteSize).map { " • \($0)" } ?? ""
        return "\(all.count) document\(all.count == 1 ? "" : "s")\(sizePart)"
    }

    private func errorRow(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
                .accessibilityHidden(true)
            Text(text)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
            Button("Dismiss") { importError = nil }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
        }
        .font(.caption)
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

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "folder")
                .font(.system(size: 30))
                .foregroundStyle(.quaternary)
                .accessibilityHidden(true)
            Text(searchText.isEmpty ? "No files attached" : "No matching files")
                .font(.callout.weight(.medium))
            Text(searchText.isEmpty
                 ? "Attach PDFs, Office documents, or source files to give a chat something to read."
                 : "Nothing matches \u{201C}\(searchText)\u{201D}.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 320)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .accessibilityElement(children: .combine)
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
                    "Attached \(outcome.importedCount) document\(outcome.importedCount == 1 ? "" : "s") to this chat.",
                    style: .success)
            }
        }
    }
}

/// One row of the Files list.
private struct FileRowView: View {
    @ObservedObject var model: AppModel
    let reference: AppAttachmentReference
    let isSelected: Bool

    @State private var isHovering = false

    private var attachment: AppPromptAttachment { reference.attachment }

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: attachment.symbolName)
                .font(.system(size: 14))
                .foregroundStyle(isSelected ? TurboSparkTheme.accentColor : Color.secondary)
                .frame(width: 20)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 2) {
                Text(attachment.fileName)
                    .font(.system(size: 12, weight: isSelected ? .semibold : .regular))
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(detailText)
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Spacer(minLength: 4)

            if !attachment.sourceExists {
                Image(systemName: "questionmark.folder")
                    .font(.system(size: 10))
                    .foregroundStyle(.orange)
                    .help("The source file has moved; only the extracted text remains.")
            }

            if isHovering || isSelected {
                actionMenu
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
        .help("Preview \(attachment.fileName)")
        .accessibilityElement(children: .contain)
        .accessibilityLabel(attachment.fileName)
        .accessibilityValue(detailText)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .accessibilityHint("Opens this document in the preview pane")
    }

    private var detailText: String {
        var parts = [attachment.formatLabel]
        if let size = MetricFormat.fileSize(attachment.sourceByteSize) {
            parts.append(size)
        }
        parts.append("\(attachment.characterCount.formatted(.number.notation(.compactName))) chars")
        parts.append(reference.chatTitle)
        return parts.joined(separator: " • ")
    }

    private var actionMenu: some View {
        Menu {
            Button("Preview", systemImage: "eye") {
                model.showPreview(attachmentID: attachment.id)
            }
            Button("Go to chat", systemImage: "bubble.left") {
                model.selectChat(id: reference.chatID)
                model.activeSection = .chat
            }
            if attachment.sourceExists {
                Divider()
                Button("Reveal in Finder", systemImage: "folder") {
                    model.revealAttachmentInFinder(attachment)
                }
                Button("Open with default app", systemImage: "arrow.up.forward.app") {
                    model.openAttachmentExternally(attachment)
                }
            }
            Divider()
            Button("Detach", systemImage: "trash", role: .destructive) {
                model.removeAttachment(reference: reference)
            }
            .disabled(model.isRunning)
        } label: {
            Image(systemName: "ellipsis")
                .font(.system(size: 11, weight: .semibold))
                .frame(width: 22, height: 22)
                .contentShape(Rectangle())
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .foregroundStyle(.secondary)
        .help("Actions for \(attachment.fileName)")
        .accessibilityLabel("Actions for \(attachment.fileName)")
    }
}
