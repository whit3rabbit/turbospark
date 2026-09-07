import Foundation

/// The ONE funnel every produced file goes through on its way to a chat's
/// artifact list.
///
/// Four executors and two tools can produce a file the user should be able to
/// reopen, and registering from each of them separately is how six copies of
/// one policy come to disagree. The hook is shaped like
/// `SendUserFileExecutor.onFileSent` for the same reason that one is:
/// `AppToolRegistry.execute` is `nonisolated async`, so it cannot touch
/// `AppModel` and has to hand the fact out.
public enum ArtifactRegistrar {
    /// A file a tool call just wrote or presented.
    public struct ProducedFile: Sendable, Equatable {
        public var url: URL
        /// The transcript anchor. Every producing arm of `execute` has
        /// `call.id` in scope, which is what lets the inline card sit beside
        /// the call that made the file.
        public var toolCallID: UUID
        public var toolName: String
        /// A model-supplied label, when the tool carried one (SendUserFile's
        /// `message`, a plan's title). Nil falls back to the file name.
        public var title: String?
        public var origin: AppArtifact.Origin

        public init(
            url: URL,
            toolCallID: UUID,
            toolName: String,
            title: String? = nil,
            origin: AppArtifact.Origin
        ) {
            self.url = url
            self.toolCallID = toolCallID
            self.toolName = toolName
            self.title = title
            self.origin = origin
        }
    }

    /// Fired once per tool call, with the files that survived the policy.
    ///
    /// The `UUID?` is the TURN'S chat id as `execute` received it, never the
    /// selection. A nil means the caller had no chat, and the handler must
    /// register nothing rather than fall back to `selectedChatID` -- that
    /// fallback is the `TodoWriteExecutor` shape `AppToolRegistry.execute`'s
    /// own doc comment records as the reason the parameter exists (state#9,
    /// state#30, state#67).
    public static var onArtifactsProduced: (@Sendable (UUID?, [ProducedFile]) -> Void)?

    /// The policy, as a PURE function so it can be tested with fixtures
    /// rather than through a live tool call (Gotcha 26's `ServerStatusRows`
    /// lesson).
    ///
    /// **Markdown and HTML from anywhere, everything else only when the
    /// model presented it deliberately.** A `write_file` on `src/lib.rs` is
    /// not an artifact: auto-popping a panel per source edit fights the
    /// worktree pane and trains the user to close it. A `.docx` reaches the
    /// panel through SendUserFile, which is the tool that means "look at
    /// this". HTML joins markdown because both are documents the panel
    /// RENDERS rather than shows as source, and because the panel is where
    /// an html write becomes worth anything at all.
    ///
    /// Both tests read the panel's own extension sets
    /// (`AppArtifactRenderKind.markdownExtensions` / `.htmlExtensions`), so a
    /// format cannot become registrable without becoming renderable in the
    /// same edit.
    public static func artifactCandidates(from files: [ProducedFile]) -> [ProducedFile] {
        files.filter { file in
            switch file.origin {
            case .sentToUser, .plan:
                return true
            case .fileWrite:
                let ext = file.url.pathExtension.lowercased()
                return AppArtifactRenderKind.markdownExtensions.contains(ext)
                    || AppArtifactRenderKind.htmlExtensions.contains(ext)
            }
        }
    }

    /// Reports the surviving files, and does nothing at all when none
    /// survive.
    ///
    /// Called from inside `execute`'s `do`, never its `catch`: a `write_file`
    /// that threw wrote nothing, and an artifact registered on that path
    /// names a file that does not exist.
    public static func report(chatID: UUID?, produced: [ProducedFile]) {
        let candidates = artifactCandidates(from: produced)
        guard !candidates.isEmpty else { return }
        onArtifactsProduced?(chatID, candidates)
    }
}
