import SwiftUI

/// Horizontal scroll of attachment chips shown above the prompt editor.
struct PromptAttachmentsView: View {
    @ObservedObject var model: AppModel

    var body: some View {
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
    }
}

/// An individual attachment chip showing icon, name, character count, and preview/remove actions.
struct PromptAttachmentChipView: View {
    let attachment: AppPromptAttachment
    let isPreviewing: Bool
    let isRunning: Bool
    let onPreview: () -> Void
    let onRemove: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            Button(action: onPreview) {
                HStack(spacing: 6) {
                    Image(systemName: attachment.symbolName)
                        .themedFont(.tiny)
                        .foregroundStyle(isPreviewing ? TurboSparkTheme.accentColor : Color.secondary)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 0) {
                        Text(attachment.fileName)
                            .themedFont(.tiny, weight: .medium)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Text(detailText)
                            .themedFont(.micro)
                            .foregroundStyle(.secondary)
                    }
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Preview \(attachment.fileName)")
            .accessibilityLabel("Preview \(attachment.fileName)")
            .accessibilityValue(detailText)
            .accessibilityHint("Opens this document in the preview pane")

            Button(action: onRemove) {
                Image(systemName: "xmark")
                    .themedFont(.micro, weight: .bold)
                    .frame(width: 14, height: 14)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .foregroundStyle(.tertiary)
            .disabled(isRunning)
            .help("Remove \(attachment.fileName)")
            .accessibilityLabel("Remove attachment \(attachment.fileName)")
            .accessibilityHint("Detaches this document from the prompt")
        }
        .frame(maxWidth: 220, alignment: .leading)
        .padding(.leading, 8)
        .padding(.trailing, 6)
        .padding(.vertical, 5)
        .background(
            isPreviewing ? TurboSparkTheme.accentColor.opacity(0.12) : Color.primary.opacity(0.05),
            in: .rect(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .stroke(
                    isPreviewing ? TurboSparkTheme.accentColor.opacity(0.4) : TurboSparkTheme.hairlineColor,
                    lineWidth: 0.5)
        }
    }

    // The subtitle is a VALUE on the attachment, not assembled here, so it
    // can be tested without a view (`swift/CLAUDE.md` Gotcha 26).
    private var detailText: String { attachment.detailText }
}
