import CryptoKit
import Foundation

/// Inline HTML content behind the right column's `.htmlPreview` claimant.
///
/// A fence is not a file, so this is NOT an `AppArtifact` and never joins
/// `chat.artifacts`: the archive stores paths and metadata rather than bytes,
/// and there is no path to store. The bytes live here, in memory, for as long
/// as the panel shows them. A new `id` per distinct html is what makes a
/// network grant content-bound -- the grant is keyed on the id-bearing
/// content key, so editing the fence and clicking Preview again re-asks.
public struct ArtifactHTMLPreview: Identifiable, Equatable, Sendable {
    public let id: UUID
    public let title: String
    public let html: String

    public init(id: UUID = UUID(), title: String, html: String) {
        self.id = id
        self.title = title
        self.html = html
    }
}

extension AppModel {
    // MARK: - Claimants
    //
    // The three right-column claimants clear each other HERE and in
    // `showPreview(attachmentID:)` (`AppModel+Files`), so at most one is ever
    // set. `AppRightColumnClaimant.resolve` is the belt for a writer that
    // forgets, not the mechanism.

    /// Opens the artifact panel on a registered artifact.
    public func openArtifact(id: UUID) {
        openArtifactID = id
        previewAttachmentID = nil
        htmlPreview = nil
    }

    /// Closes the artifact panel.
    public func dismissArtifact() {
        openArtifactID = nil
    }

    /// Opens the panel on inline html (a chat fence's Preview click).
    ///
    /// Clicking Preview twice on the SAME fence keeps the identity, so the
    /// panel does not reset its own scroll or re-ask a granted network
    /// permission; a fence whose html changed mints a new id and therefore a
    /// fresh grant question.
    public func openHTMLPreview(title: String, html: String) {
        if let existing = htmlPreview, existing.html == html {
            htmlPreview = existing
        } else {
            htmlPreview = ArtifactHTMLPreview(title: title, html: html)
        }
        previewAttachmentID = nil
        openArtifactID = nil
    }

    /// Closes the inline html preview.
    public func dismissHTMLPreview() {
        htmlPreview = nil
    }

    /// The id `AppRightColumnClaimant.resolve` reads for the inline preview.
    public var htmlPreviewID: UUID? { htmlPreview?.id }

    /// Looks an artifact up by id across every chat.
    ///
    /// Across, not "in the selected chat": an artifact carries the chat that
    /// produced it, and the panel must survive a selection switch rather
    /// than vanish or show a neighbour's row. Linear scan, like
    /// `attachmentReference(id:)`, because a SwiftUI body reads it.
    public func artifact(id: UUID) -> AppArtifact? {
        for chat in chats {
            if let artifact = chat.artifacts.first(where: { $0.id == id }) {
                return artifact
            }
        }
        if !chats.contains(where: { $0.id == selectedChatID }) {
            return activeDraftChat.artifacts.first(where: { $0.id == id })
        }
        return nil
    }

    // MARK: - Network grants
    //
    // Offline by default. A grant is keyed to the EXACT content: a rewritten
    // artifact bumps `revision` (new contentKey), an edited fence mints a new
    // preview id, and both re-ask. Keyed on the artifact id alone would let a
    // page the user never approved ride in on a file the model overwrote.

    /// Namespaced grant key for a file-backed artifact. Moves on rewrite, so
    /// the grant cannot survive new bytes.
    public func networkGrantKey(for artifact: AppArtifact) -> String {
        "artifact:\(artifact.contentKey)"
    }

    /// Namespaced grant key for an inline preview. The hash binds the grant
    /// to the bytes, not to the title or the fence it came from.
    public func networkGrantKey(for preview: ArtifactHTMLPreview) -> String {
        "html:\(SHA256.hash(data: Data(preview.html.utf8)).map { String(format: "%02x", $0) }.joined())"
    }

    /// Whether the panel may load remote resources for the given content.
    public func isNetworkAllowed(for artifact: AppArtifact) -> Bool {
        artifactNetworkGrants[networkGrantKey(for: artifact)] ?? false
    }

    /// Whether the panel may load remote resources for an inline preview.
    public func isNetworkAllowed(for preview: ArtifactHTMLPreview) -> Bool {
        artifactNetworkGrants[networkGrantKey(for: preview)] ?? false
    }

    /// Grants network for whichever html content is currently shown.
    ///
    /// The panel calls this from its banner; the reload itself is driven by
    /// the view (the web view's identity includes the grant).
    public func allowNetworkForCurrentPreview() {
        if let preview = htmlPreview, openArtifactID == nil {
            artifactNetworkGrants[networkGrantKey(for: preview)] = true
        } else if let id = openArtifactID, let artifact = artifact(id: id),
                  artifact.renderKind == .html {
            artifactNetworkGrants[networkGrantKey(for: artifact)] = true
        }
    }

    /// Revokes the grant for the currently shown content.
    public func revokeNetworkForCurrentPreview() {
        if let preview = htmlPreview, openArtifactID == nil {
            artifactNetworkGrants.removeValue(forKey: networkGrantKey(for: preview))
        } else if let id = openArtifactID, let artifact = artifact(id: id),
                  artifact.renderKind == .html {
            artifactNetworkGrants.removeValue(forKey: networkGrantKey(for: artifact))
        }
    }

    // MARK: - Auto-open
    //
    // The upsert doc records why this can work at all: ids are STABLE across
    // rewrites, so "already opened" keyed on the id fires once per artifact,
    // not once per write. A panel the user closed stays closed.

    /// Auto-opens the panel when a produced artifact is HTML and has never
    /// been shown.
    ///
    /// Called from `registerProducedArtifacts` with the row AS UPSERTED, so
    /// the id tested is the surviving one and a rewrite of an already-shown
    /// file no-ops.
    func maybeAutoOpenArtifact(_ artifact: AppArtifact) {
        guard artifact.renderKind == .html else { return }
        guard !autoOpenedArtifactIDs.contains(artifact.id) else { return }
        autoOpenedArtifactIDs.insert(artifact.id)
        openArtifact(id: artifact.id)
    }
}
