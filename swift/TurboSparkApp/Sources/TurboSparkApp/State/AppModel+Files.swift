import AppKit
import Foundation
import UniformTypeIdentifiers

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

    /// The subtitle the Files list shows under the file name.
    ///
    /// Composed from `attachment.detailText` rather than reassembled, because
    /// `FileRowView` used to spell its own and the copy had drifted: it
    /// appended a character count UNCONDITIONALLY, so an image row advertised
    /// "0 chars" -- the exact zero-from-an-absent-measurement that
    /// `detailText` exists to avoid, on the one surface
    /// `testAnImageChipDoesNotAdvertiseAZeroCharacterCount` cannot reach.
    /// A VALUE here so this surface is testable too (Gotcha 26).
    public var listDetailText: String {
        "\(attachment.detailText) • \(chatTitle)"
    }

    public init(attachment: AppPromptAttachment, chatID: UUID, chatTitle: String) {
        self.attachment = attachment
        self.chatID = chatID
        self.chatTitle = chatTitle
    }
}

extension AppModel {
    /// Whether the loaded session would actually SERVE an image.
    ///
    /// Read off `session.info.vision.active`, which the engine resolved at
    /// open, rather than inferred from the family name: an install can carry
    /// a tower and still refuse every image (the pixel budget comes from the
    /// checkpoint's own sidecar and has no safe default). Nil session means
    /// no, so nothing offers an attachment before a model is loaded.
    public var visionIsActive: Bool { session?.info.vision.active ?? false }

    /// Why images are refused, when this install has a tower and cannot use
    /// it. Nil whenever there is nothing a user could act on.
    public var visionRefusalReason: String? { session?.info.vision.reason }

    /// The types both attachment pickers accept.
    ///
    /// ONE accessor for the same reason `activeLoadGuard` is one: two views
    /// assembling their own list would drift, and the one that drifted would
    /// accept a file the turn then refuses.
    public var attachmentContentTypes: [UTType] {
        DocumentTextExtractor.supportedContentTypes
            + (visionIsActive ? AppPromptAttachment.imageContentTypes : [])
    }

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
    ///
    /// **NOT THROUGH `allAttachments`.** That sorts every chat by date and
    /// walks all of their draft attachments to build an array, and this is
    /// reached from `previewAttachment`, which a SwiftUI body reads -- so it
    /// ran once per streamed token during a generation. Scanning for the one
    /// id directly is the same answer without the sort or the array.
    public func attachmentReference(id: UUID) -> AppAttachmentReference? {
        for chat in chats {
            if let attachment = chat.draftAttachments.first(where: { $0.id == id }) {
                return AppAttachmentReference(
                    attachment: attachment, chatID: chat.id, chatTitle: chat.title)
            }
        }
        // The active draft chat is not in `chats` until it has a message.
        if !chats.contains(where: { $0.id == selectedChatID }) {
            let chat = selectedChat
            if let attachment = chat.draftAttachments.first(where: { $0.id == id }) {
                return AppAttachmentReference(
                    attachment: attachment, chatID: chat.id, chatTitle: chat.title)
            }
        }
        return nil
    }

    /// The attachment the preview pane is currently showing, if any.
    public var previewAttachment: AppPromptAttachment? {
        guard let previewAttachmentID else { return nil }
        return attachmentReference(id: previewAttachmentID)?.attachment
    }

    /// Shows the given attachment in the right-hand preview pane.
    ///
    /// An explicit click, so it takes the column from every turn-driven
    /// panel: both other claimants clear here, and they clear this one in
    /// theirs (`AppModel+ArtifactPanel`). `AppRightColumnClaimant.resolve`
    /// is only the belt for a writer that forgets.
    public func showPreview(attachmentID: UUID) {
        previewAttachmentID = attachmentID
        openArtifactID = nil
        htmlPreview = nil
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
