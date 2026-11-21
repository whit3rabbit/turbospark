import SwiftUI

/// Horizontal scroll of attachment chips shown above the prompt editor.
struct PromptAttachmentsView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ScrollView(.horizontal) {
                HStack(spacing: 6) {
                    ForEach(model.promptAttachments) { attachment in
                        PromptAttachmentChipView(
                            attachment: attachment,
                            isPreviewing: model.previewAttachmentID == attachment.id,
                            isRunning: model.isRunning,
                            onPreview: { model.showPreview(attachmentID: attachment.id) },
                            onRemove: { model.removePromptAttachment(id: attachment.id) }
                        )
                    }
                }
                .padding(.bottom, 2)
            }
            .scrollIndicators(.hidden)
            if let budget {
                PromptAttachmentsBudgetLine(
                    attachmentTokens: budget.tokens, reading: budget.reading)
            }
        }
    }

    /// The tray's budget, local chats with a loaded session only: before a
    /// load the window is a placeholder (`resolvedContextTokens` falls back
    /// to 4096), and a remote chat's window is the provider's. The same
    /// gate as the context strip (`ResourceWarningStripView.contextFraction`).
    private var budget: (tokens: Int, reading: AttachmentContextBudget.Reading)? {
        guard model.session != nil,
              model.remoteContextReading(chatID: model.selectedChatID) == nil,
              let tokens = AttachmentContextBudget.totalTokens(model.promptAttachments),
              let reading = AttachmentContextBudget.reading(
                  attachmentTokens: tokens,
                  otherPromptTokens: model.estimatedPromptTokens,
                  windowTokens: model.resolvedContextTokens,
                  reservedForReplyTokens: model.maxNewTokens)
        else { return nil }
        return (tokens, reading)
    }
}

/// One line under the chips: what the attachments weigh, or a warning when
/// they bring the prompt near or past the usable window. Glyph plus words,
/// never the color alone (StatusLabel guide).
struct PromptAttachmentsBudgetLine: View {
    let attachmentTokens: Int
    let reading: AttachmentContextBudget.Reading

    var body: some View {
        let tokens = attachmentTokens.formatted(.number.notation(.compactName))
        switch reading.level {
        case .fits:
            Text("Attachments use about \(tokens) tokens, \(reading.attachmentPercent)% of the context window", bundle: .module)
                .themedFont(.micro)
                .foregroundStyle(.appSecondary)
        case .nearLimit:
            Label {
                Text("Attachments bring this prompt to \(reading.totalPercent)% of the context window", bundle: .module)
            } icon: {
                Image(systemName: "exclamationmark.triangle.fill")
                    .accessibilityHidden(true)
            }
            .themedFont(.micro, weight: .medium)
            .foregroundStyle(.appWarning)
        case .overLimit:
            Label {
                Text("Attachments push this prompt past the context window. Remove one to make room.", bundle: .module)
            } icon: {
                Image(systemName: "exclamationmark.octagon.fill")
                    .accessibilityHidden(true)
            }
            .themedFont(.micro, weight: .medium)
            .foregroundStyle(.appDanger)
        }
    }
}

/// An individual attachment chip showing icon, name, character count, and preview/remove actions.
struct PromptAttachmentChipView: View {
    @Environment(\.appTheme) private var theme
    let attachment: AppPromptAttachment
    let isPreviewing: Bool
    let isRunning: Bool
    let onPreview: () -> Void
    let onRemove: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            Button(action: onPreview) {
                HStack(spacing: 6) {
                    AttachmentThumbnailView(
                        url: attachment.thumbnailSourceURL,
                        pixelSize: 16,
                        fallbackSymbol: attachment.symbolName,
                        symbolStep: .tiny,
                        symbolTint: isPreviewing ? theme.accent : theme.secondaryText
                    )
                    .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 0) {
                        Text(attachment.fileName)
                            .themedFont(.tiny, weight: .medium)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        HStack(spacing: 6) {
                            Text(detailText)
                            if let tokenText {
                                // Kept whole: the char count truncates first.
                                Text("~\(tokenText) tokens", bundle: .module)
                                    .layoutPriority(1)
                            }
                        }
                        .lineLimit(1)
                        .themedFont(.micro)
                        .foregroundStyle(.appSecondary)
                    }
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help(Text("Preview \(attachment.fileName)", bundle: .module))
            .accessibilityLabel(Text("Preview \(attachment.fileName)", bundle: .module))
            .accessibilityValue(accessibilityDetail)
            .accessibilityHint(Text("Opens this document in the preview pane", bundle: .module))

            AppIconButton("xmark", label: Text("Remove \(attachment.fileName)", bundle: .module)) {
                onRemove()
            }
            .foregroundStyle(.appSecondary)
            .disabled(isRunning)
            .help(
                isRunning
                    ? Text("Cannot remove attachments while generation is active", bundle: .module)
                    : Text("Remove \(attachment.fileName)", bundle: .module)
            )
            .accessibilityHint(Text("Detaches this document from the prompt", bundle: .module))
        }
        .frame(maxWidth: (220 * AppMetrics.textScale(theme)).rounded(), alignment: .leading)
        .padding(.leading, 8)
        .padding(.trailing, 6)
        .padding(.vertical, 5)
        .background(
            isPreviewing ? AnyShapeStyle(theme.accent.opacity(0.12)) : AnyShapeStyle(.appElevated),
            in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
        )
        .overlay {
            RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                .stroke(
                    isPreviewing ? AnyShapeStyle(theme.accent) : AnyShapeStyle(theme.border),
                    lineWidth: 1
                )
        }
    }

    // The subtitle is a VALUE on the attachment, not assembled here, so it
    // can be tested without a view (`swift/CLAUDE.md` Gotcha 26).
    private var detailText: String { attachment.detailText }

    /// Nil for images: no known vision token cost, so no figure at all.
    private var tokenText: String? {
        AttachmentContextBudget.tokens(for: attachment)
            .map { $0.formatted(.number.notation(.compactName)) }
    }

    /// "about" rather than "~", which VoiceOver reads as "tilde".
    private var accessibilityDetail: Text {
        guard let tokenText else { return Text(verbatim: detailText) }
        return Text("\(detailText), about \(tokenText) tokens", bundle: .module)
    }
}
