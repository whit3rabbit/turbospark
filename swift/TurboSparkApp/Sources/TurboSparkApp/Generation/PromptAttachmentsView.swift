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
                        symbolTint: isPreviewing ? theme.accent : .secondary
                    )
                    .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 0) {
                        Text(attachment.fileName)
                            .themedFont(.tiny, weight: .medium)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Text(detailText)
                            .themedFont(.micro)
                            .foregroundStyle(.appSecondary)
                    }
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help(Text("Preview \(attachment.fileName)", bundle: .module))
            .accessibilityLabel(Text("Preview \(attachment.fileName)", bundle: .module))
            .accessibilityValue(detailText)
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
        .frame(maxWidth: 220, alignment: .leading)
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
}
