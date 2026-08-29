import AppKit
import Foundation

/// One attachment together with the chat it belongs to.
///
/// The Files section lists documents across every conversation, so a row has
/// to carry its origin: an attachment id alone cannot be traced back to a chat
/// once it leaves `selectedChat`.
public struct AppAttachmentReference: Identifiable, Equatable, Sendable {
    public var attachment: AppPromptAttachment
    public var chatID: UUID
    public var chatTitle: String

    public var id: UUID { attachment.id }

    public init(attachment: AppPromptAttachment, chatID: UUID, chatTitle: String) {
        self.attachment = attachment
        self.chatID = chatID
        self.chatTitle = chatTitle
    }
}

extension AppModel {
    /// Every attachment across every stored chat, newest chat first.
    public var allAttachments: [AppAttachmentReference] {
        var rows: [AppAttachmentReference] = []
        for chat in chats.sorted(by: { $0.updatedAt > $1.updatedAt }) {
            for attachment in chat.draftAttachments {
                rows.append(
                    AppAttachmentReference(
                        attachment: attachment,
                        chatID: chat.id,
                        chatTitle: chat.title))
            }
        }
        // The active draft chat is not in `chats` until it has a message, so
        // its attachments would otherwise be invisible in the Files section
        // precisely while the user is assembling them.
        if !chats.contains(where: { $0.id == selectedChatID }) {
            let chat = selectedChat
            for attachment in chat.draftAttachments {
                rows.append(
                    AppAttachmentReference(
                        attachment: attachment,
                        chatID: chat.id,
                        chatTitle: chat.title))
            }
        }
        return rows
    }

    /// Total bytes of the source files behind every known attachment.
    public var allAttachmentsByteSize: Int {
        allAttachments.reduce(0) { $0 + ($1.attachment.sourceByteSize ?? 0) }
    }

    /// Looks an attachment up by identifier across every chat.
    public func attachmentReference(id: UUID) -> AppAttachmentReference? {
        allAttachments.first { $0.id == id }
    }

    /// The attachment the preview pane is currently showing, if any.
    public var previewAttachment: AppPromptAttachment? {
        guard let previewAttachmentID else { return nil }
        return attachmentReference(id: previewAttachmentID)?.attachment
    }

    /// Shows the given attachment in the right-hand preview pane.
    public func showPreview(attachmentID: UUID) {
        previewAttachmentID = attachmentID
    }

    /// Closes the preview pane.
    public func dismissPreview() {
        previewAttachmentID = nil
    }

    /// Removes an attachment from whichever chat holds it.
    public func removeAttachment(reference: AppAttachmentReference) {
        if previewAttachmentID == reference.id {
            previewAttachmentID = nil
        }
        if let index = chats.firstIndex(where: { $0.id == reference.chatID }) {
            chats[index].draftAttachments.removeAll { $0.id == reference.id }
            chats[index].updatedAt = Date()
            persistChats()
        } else if reference.chatID == selectedChatID {
            removePromptAttachment(id: reference.id)
        }
    }

    /// Reveals the attachment's source file in Finder when the path still resolves.
    public func revealAttachmentInFinder(_ attachment: AppPromptAttachment) {
        guard let url = attachment.sourceURL, attachment.sourceExists else {
            showToast("\(attachment.fileName) is no longer at its original path.", style: .warning)
            return
        }
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    /// Opens the attachment's source file in its default application.
    public func openAttachmentExternally(_ attachment: AppPromptAttachment) {
        guard let url = attachment.sourceURL, attachment.sourceExists else {
            showToast("\(attachment.fileName) is no longer at its original path.", style: .warning)
            return
        }
        NSWorkspace.shared.open(url)
    }
}
