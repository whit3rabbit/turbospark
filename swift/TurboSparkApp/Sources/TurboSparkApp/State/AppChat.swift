import Foundation
import TurboSpark
import UniformTypeIdentifiers

/// A document attachment associated with a prompt draft.
public struct AppPromptAttachment: Identifiable, Codable, Equatable, Sendable {
    /// How the preview pane should render this attachment.
    public enum PreviewKind: Equatable, Sendable {
        /// A PDF rendered page by page.
        case pdf
        /// A raster image rendered at its natural aspect ratio.
        case image
        /// Anything else, rendered as the extracted text.
        case text
    }

    /// Unique identifier for the attachment.
    public var id = UUID()
    /// Original file name of the attached document.
    public var fileName: String
    /// Display format label (e.g. PDF, Word, Code).
    public var formatLabel: String
    /// Normalized extracted text content.
    public var extractedText: String
    /// Whether the extracted text was truncated to fit safety limits.
    public var wasTruncatedDuringExtraction: Bool
    /// Filesystem path the document was imported from, when it is still known.
    ///
    /// Optional so an archive written before previews existed still decodes:
    /// the synthesized `init(from:)` uses `decodeIfPresent` for an Optional,
    /// and a non-optional field here would discard every saved chat
    /// (`swift/CLAUDE.md` Gotcha 13).
    public var sourcePath: String?
    /// Size of the source file on disk in bytes, when it was readable.
    public var sourceByteSize: Int?

    /// Total character count of the extracted text.
    public var characterCount: Int { extractedText.count }

    /// The source file location, when the import recorded one.
    public var sourceURL: URL? {
        guard let sourcePath, !sourcePath.isEmpty else { return nil }
        return URL(fileURLWithPath: sourcePath)
    }

    /// Whether the source file is still present at the recorded path.
    public var sourceExists: Bool {
        guard let sourceURL else { return false }
        return FileManager.default.fileExists(atPath: sourceURL.path)
    }

    /// Lowercased file extension of the attachment, or the empty string.
    public var fileExtension: String {
        (fileName as NSString).pathExtension.lowercased()
    }

    /// The raster formats this app treats as pictures.
    ///
    /// ONE list, read by `previewKind`, `symbolName` and `isImage` alike. It
    /// was spelled out twice before a third reader arrived, and a display
    /// value restated per branch is correct on the day it is written and
    /// silently wrong at the next addition (`swift/CLAUDE.md` Gotcha 22).
    public static let imageFileExtensions: Set<String> = [
        "png", "jpg", "jpeg", "gif", "heic", "tiff", "bmp", "webp",
    ]

    /// The picker types for [`imageFileExtensions`], derived from the same
    /// list rather than restated as a second one.
    ///
    /// Offered ONLY when the loaded session reports `vision.active`: an
    /// install can carry a tower and still refuse every image, and a picker
    /// that accepted one anyway would promise work the engine then declines
    /// (`swift/CLAUDE.md` Gotcha 23).
    public static var imageContentTypes: [UTType] {
        imageFileExtensions.sorted().compactMap { UTType(filenameExtension: $0) }
    }

    /// Whether this attachment is a picture rather than a document.
    ///
    /// A pure function of the FILE NAME, deliberately: a stored flag would be
    /// a new non-optional key on a `Codable` the archive already holds, which
    /// is the decode hazard that silently emptied a user's chat list once
    /// (`swift/CLAUDE.md` Gotcha 13).
    public var isImage: Bool {
        Self.imageFileExtensions.contains(fileExtension)
    }

    /// Whether this picture can actually be sent to the engine.
    ///
    /// The engine reads the file by PATH, so an attachment whose source has
    /// moved or was never recorded cannot be encoded however well it renders
    /// from a preview. Checked before the turn rather than discovered inside
    /// it.
    public var isSendableImage: Bool { isImage && sourceExists }

    /// How the preview pane should render this attachment.
    public var previewKind: PreviewKind {
        if fileExtension == "pdf" { return sourceExists ? .pdf : .text }
        if isImage { return sourceExists ? .image : .text }
        return .text
    }

    /// The subtitle a chip or row shows under the file name.
    ///
    /// A VALUE rather than inline view code, which is the only reason it can
    /// be tested at all -- the same reason `ServerStatusRows` was extracted
    /// (`swift/CLAUDE.md` Gotcha 26).
    ///
    /// **An image extracts no text, so a character count on one is a zero
    /// from an absent measurement rather than a measurement of zero.**
    /// "Image - 0 chars" reads as a failed import; what is actually known
    /// about a picture is its size and whether it is still there.
    public var detailText: String {
        if isImage {
            let size = MetricFormat.fileSize(sourceByteSize).map { " • \($0)" } ?? ""
            let missing = sourceExists ? "" : " • file missing"
            return "\(formatLabel)\(size)\(missing)"
        }
        let count = characterCount.formatted(.number.notation(.compactName))
        let suffix = wasTruncatedDuringExtraction ? " • truncated" : ""
        return "\(formatLabel) • \(count) chars\(suffix)"
    }

    /// SF Symbol representing the document type in lists and chips.
    public var symbolName: String {
        if isImage { return "photo" }
        switch fileExtension {
        case "pdf": return "doc.richtext"
        case "docx", "doc": return "doc.text"
        case "xlsx", "xls", "csv": return "tablecells"
        case "pptx", "ppt": return "rectangle.on.rectangle"
        case "json", "yaml", "yml", "toml": return "curlybraces"
        case "swift", "rs", "py", "c", "cpp", "h", "js", "ts", "html", "css":
            return "chevron.left.forwardslash.chevron.right"
        case "md", "txt": return "doc.plaintext"
        default: return "doc"
        }
    }

    /// Creates a prompt attachment.
    public init(
        id: UUID = UUID(),
        fileName: String,
        formatLabel: String,
        extractedText: String,
        wasTruncatedDuringExtraction: Bool,
        sourcePath: String? = nil,
        sourceByteSize: Int? = nil
    ) {
        self.id = id
        self.fileName = fileName
        self.formatLabel = formatLabel
        self.extractedText = extractedText
        self.wasTruncatedDuringExtraction = wasTruncatedDuringExtraction
        self.sourcePath = sourcePath
        self.sourceByteSize = sourceByteSize
    }
}

/// A single message turn within an application chat conversation.
public struct AppChatMessage: Identifiable, Codable, Equatable, Sendable {
    /// Unique identifier for the message.
    public var id = UUID()
    /// The sender role of the message.
    public var role: ChatMessage.Role
    /// Text content of the message.
    public var content: String
    /// Optional thinking or reasoning output preceding the response.
    public var reasoning: String
    /// Reason why generation stopped for this message turn.
    public var stopReason: String?
    /// Parsed tool calls requested in this message turn.
    public var toolCalls: [AppToolCall]
    /// Results of executed tool calls for this message turn.
    public var toolResults: [AppToolResult]
    /// Paths of images sent with this turn, in order.
    ///
    /// **Stored on the MESSAGE rather than only on the draft, because the
    /// prompt is rebuilt from the transcript on every step.** An agent loop
    /// calls `executeGenerationTurn` repeatedly and each pass reconstructs
    /// `[ChatMessage]` from these turns, so a picture held only in the
    /// composer would be sent on the first step and silently dropped on the
    /// second -- with the model then answering about an image it can no
    /// longer see.
    ///
    /// Paths rather than bytes: the engine reads the file itself, and the
    /// archive is rewritten whole on every keystroke of the draft.
    public var imagePaths: [String]
    /// Inactive earlier versions of this message, oldest first.
    ///
    /// The struct's own fields ARE the active version, so every existing
    /// reader (prompt assembly, the transcript, the sidebar preview) keeps
    /// reading the row it always read; edit and retry push the replaced
    /// fields here and variant navigation swaps them back. A pushed copy is
    /// FLATTENED (its own `alternates` dropped) so the archive stays one
    /// level deep, and every version keeps its own `createdAt`, which is
    /// what gives the switcher its stable oldest-first order without a
    /// stored position field. The transcript row's own `id` never changes
    /// across a swap: it is the ForEach identity, and following the active
    /// version's id would rebuild the row mid-navigation.
    public var alternates: [AppChatMessage]
    /// Timestamp when this message turn was created.
    public var createdAt: Date

    /// Creates a chat message turn.
    public init(
        id: UUID = UUID(),
        role: ChatMessage.Role,
        content: String,
        reasoning: String = "",
        stopReason: String? = nil,
        toolCalls: [AppToolCall] = [],
        toolResults: [AppToolResult] = [],
        imagePaths: [String] = [],
        alternates: [AppChatMessage] = [],
        createdAt: Date = Date()
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.reasoning = reasoning
        self.stopReason = stopReason
        self.toolCalls = toolCalls
        self.toolResults = toolResults
        self.imagePaths = imagePaths
        self.alternates = alternates
        self.createdAt = createdAt
    }

    /// Tolerant decode: every field added after the first release is read with
    /// `decodeIfPresent` and a default.
    ///
    /// The synthesized decoder was NOT tolerant, and the failure is total and
    /// silent: `AppChatFileStore.load()` swallows the error and returns the
    /// empty archive, so ONE message written before `toolCalls` existed
    /// discards the user's entire chat history -- and the next `persistChats()`
    /// writes that emptiness back over the file. Measured on a real archive
    /// here, where a four-message chat was invisible in the app while sitting
    /// intact on disk. Any field added to this struct from now on gets the
    /// same treatment (`swift/CLAUDE.md` Gotcha 13).
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decodeIfPresent(UUID.self, forKey: .id) ?? UUID()
        // **AN ENUM DECODED BY ITS OWN CONFORMANCE IS INTOLERANT** (state#45).
        // `decode(Role.self)` throws `dataCorrupted` on any raw value this
        // build does not know -- a role a NEWER release wrote, or one this
        // one dropped -- and `AppJSONStore.load` then quarantines the whole
        // archive over a single message. That is the same total, silent loss
        // `swift/CLAUDE.md` Gotcha 13 records, reached through the one field
        // the tolerant decoder still read strictly. Decoded through
        // `rawValue`, an unknown role reads as `.assistant`: it is DISPLAY
        // text either way, and keeping the other 400 messages beats being
        // precise about one.
        role = container.decodeTolerant(ChatMessage.Role.self, forKey: .role, fallback: .assistant)
        content = try container.decodeIfPresent(String.self, forKey: .content) ?? ""
        reasoning = try container.decodeIfPresent(String.self, forKey: .reasoning) ?? ""
        stopReason = try container.decodeIfPresent(String.self, forKey: .stopReason)
        // Lossy: one malformed call must not take the whole archive down.
        toolCalls = try container.decodeLossyArray(AppToolCall.self, forKey: .toolCalls)
        toolResults = try container.decodeLossyArray(AppToolResult.self, forKey: .toolResults)
        imagePaths = try container.decodeIfPresent([String].self, forKey: .imagePaths) ?? []
        alternates = try container.decodeLossyArray(AppChatMessage.self, forKey: .alternates)
        createdAt = try container.decodeIfPresent(Date.self, forKey: .createdAt) ?? Date()
    }
}

/// A persistent multi-turn chat conversation session.
public struct AppChat: Identifiable, Codable, Equatable, Sendable {
    /// Unique identifier for the chat session.
    public var id = UUID()
    /// Optional project to which this chat belongs.
    public var projectID: UUID?
    /// Display title of the conversation.
    public var title: String
    /// Uncommitted draft prompt text.
    public var draft: String
    /// Uncommitted draft document attachments.
    public var draftAttachments: [AppPromptAttachment]
    /// Committed conversation message history.
    public var messages: [AppChatMessage]
    /// Active task checklist for this session.
    public var todos: [TodoItem]
    /// Files a tool call in this chat produced (`ArtifactRegistrar`), so the
    /// panel can reopen one after the turn that made it scrolls out of view.
    public var artifacts: [AppArtifact]
    /// Summary of the conversation's older turns, written by compaction.
    ///
    /// Long unwritten: compaction did not exist, so this stayed nil and the
    /// only reader was `clearOutput`. It now carries the structured summary
    /// `AppChatCompaction` produces; `compactedMessageCount` says which rows
    /// it covers.
    public var contextSummary: String?
    /// How many leading `messages` rows the summary in `contextSummary`
    /// replaces in the prompt. The rows stay in the transcript and on disk;
    /// only history assembly skips them, so the transcript and the prompt can
    /// disagree by exactly this prefix. 0 = nothing compacted.
    public var compactedMessageCount: Int
    /// System prompt for THIS conversation, overriding the app-wide default.
    ///
    /// `nil` means "use the default" and an empty string means the same, so a
    /// user who clears the editor gets the default back rather than a silently
    /// promptless chat. OPTIONAL so the synthesized `Codable` decodes a
    /// `chats.json` written before this field existed; the store has no
    /// migration step and relies on that.
    public var systemPrompt: String?
    /// THIS conversation's own sampling knobs, overriding the app-wide
    /// Generation Sampling settings.
    ///
    /// `nil` means "use the app-wide settings"; a snapshot is COMPLETE
    /// rather than per-key optional, so resolution is "this chat or the app
    /// defaults" and never a mix. OPTIONAL so the decoder accepts a
    /// `chats_archive.json` written before this field existed; the store has
    /// no migration step and relies on that, exactly as `systemPrompt` does.
    public var samplingOverride: AppSamplingSettings?
    /// Bounded execution state, when the project runs in SKILL.state mode.
    /// Per chat rather than per project: it describes one agent run.
    public var skillState: AppSkillState?
    /// Timestamp when the chat was created.
    public var createdAt: Date
    /// Timestamp when the chat was last modified.
    public var updatedAt: Date
    /// Temporary (Ghost Mode) chat marker.
    ///
    /// A ghost chat lives only in memory: the persist path filters `isGhost`
    /// rows out of every `AppChatArchive` it builds, so nothing about one
    /// ever reaches `chats_archive.json` -- not on quit, not on a profile
    /// switch, not on a crash. Its conversation contents (messages, todos,
    /// draft, skill state) do not even sit on the row; they live
    /// AES-GCM-sealed in `GhostChatVault` and those row fields stay empty.
    public var isGhost: Bool
    /// Sidebar pin. Pinned chats sort above unpinned ones (recency breaks
    /// ties inside each group). Allowed on a ghost row harmlessly: the row
    /// dies with the process either way.
    public var isPinned: Bool
    /// Cumulative token ledger recorded from the engine's own per-turn
    /// counts. `nil` on chats with nothing recorded yet (every chat written
    /// before recording existed, and ghost chats, which never record).
    public var usage: AppChatUsageLedger?
    /// This chat's active `/goal` (swift/docs/SWIFT_GOALS.md). `nil` when
    /// no goal is set. OPTIONAL so the decoder accepts a
    /// `chats_archive.json` written before this field existed. A ghost
    /// chat's goal lives in the vault payload, not here, like the rest of
    /// its conversation contents.
    public var goal: ChatGoalState?
    /// Archived marker (qwen-code session parity): an archived chat leaves
    /// the sidebar lists but is not deleted, and `/unarchive` or the
    /// sidebar's Archived section brings it back. OPTIONAL for the decoder,
    /// like every field after the first release.
    public var isArchived: Bool

    /// Creates a new chat session.
    public init(
        id: UUID = UUID(),
        projectID: UUID? = nil,
        title: String = "New Chat",
        draft: String = "",
        draftAttachments: [AppPromptAttachment] = [],
        messages: [AppChatMessage] = [],
        todos: [TodoItem] = [],
        artifacts: [AppArtifact] = [],
        contextSummary: String? = nil,
        compactedMessageCount: Int = 0,
        systemPrompt: String? = nil,
        samplingOverride: AppSamplingSettings? = nil,
        skillState: AppSkillState? = nil,
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        isGhost: Bool = false,
        isPinned: Bool = false,
        usage: AppChatUsageLedger? = nil,
        goal: ChatGoalState? = nil,
        isArchived: Bool = false
    ) {
        self.id = id
        self.projectID = projectID
        self.title = title
        self.draft = draft
        self.draftAttachments = draftAttachments
        self.messages = messages
        self.todos = todos
        self.artifacts = artifacts
        self.contextSummary = contextSummary
        self.compactedMessageCount = compactedMessageCount
        self.systemPrompt = systemPrompt
        self.samplingOverride = samplingOverride
        self.skillState = skillState
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.isGhost = isGhost
        self.isPinned = isPinned
        self.usage = usage
        self.goal = goal
        self.isArchived = isArchived
    }

    /// Tolerant decode, for the reason given on `AppChatMessage.init(from:)`:
    /// a chat saved before `draftAttachments` or `todos` existed must not take the whole
    /// archive down with it.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decodeIfPresent(UUID.self, forKey: .id) ?? UUID()
        projectID = try container.decodeIfPresent(UUID.self, forKey: .projectID)
        title = try container.decodeIfPresent(String.self, forKey: .title) ?? "New Chat"
        draft = try container.decodeIfPresent(String.self, forKey: .draft) ?? ""
        draftAttachments = try container.decodeIfPresent(
            [AppPromptAttachment].self, forKey: .draftAttachments) ?? []
        messages = try container.decodeLossyArray(AppChatMessage.self, forKey: .messages)
        todos = try container.decodeIfPresent([TodoItem].self, forKey: .todos) ?? []
        artifacts = try container.decodeLossyArray(AppArtifact.self, forKey: .artifacts)
        contextSummary = try container.decodeIfPresent(String.self, forKey: .contextSummary)
        compactedMessageCount = try container.decodeIfPresent(
            Int.self, forKey: .compactedMessageCount) ?? 0
        // **THIS LINE WAS MISSING, AND THE FAILURE WAS THE ONE THIS DECODER
        // EXISTS TO PREVENT.** Every other optional field was read with
        // `decodeIfPresent` except this one, so the synthesized ENCODER wrote
        // `systemPrompt` to the archive and the decoder never read it back: a
        // per-chat system prompt survived until relaunch and then silently
        // reverted to the app-wide default.
        systemPrompt = try container.decodeIfPresent(String.self, forKey: .systemPrompt)
        // **THE DECODER LINE IS THE FIELD.** `systemPrompt` above shipped
        // without its read-back line once, and every per-chat prompt silently
        // reverted to the default at relaunch. The lenient form (not plain
        // `decodeIfPresent`) because a wrong-typed value must cost THIS
        // override, not the chat: a throw here fails the row, and
        // `decodeLossyArray` at the archive level would drop the whole
        // conversation over one hand-edited key.
        samplingOverride = ((try? container.decodeIfPresent(
            AppSamplingSettings.self, forKey: .samplingOverride)) ?? nil)
        skillState = try container.decodeIfPresent(AppSkillState.self, forKey: .skillState)
        createdAt = try container.decodeIfPresent(Date.self, forKey: .createdAt) ?? Date()
        updatedAt = try container.decodeIfPresent(Date.self, forKey: .updatedAt) ?? Date()
        isGhost = try container.decodeIfPresent(Bool.self, forKey: .isGhost) ?? false
        isPinned = try container.decodeIfPresent(Bool.self, forKey: .isPinned) ?? false
        usage = ((try? container.decodeIfPresent(
            AppChatUsageLedger.self, forKey: .usage)) ?? nil)
        // The lenient form (not plain `decodeIfPresent`) for the same
        // reason `samplingOverride` uses it: a wrong-typed value must cost
        // THIS field, not the chat.
        goal = ((try? container.decodeIfPresent(ChatGoalState.self, forKey: .goal)) ?? nil)
        isArchived = try container.decodeIfPresent(Bool.self, forKey: .isArchived) ?? false
    }

    /// Single-line preview text for sidebar display.
    public var preview: String {
        if let last = messages.last {
            return String(last.content.prefix(60)).replacingOccurrences(of: "\n", with: " ")
        }
        if !draft.isEmpty {
            return String(draft.prefix(60)).replacingOccurrences(of: "\n", with: " ")
        }
        return ""
    }

    /// THE ordering for every chat list (sidebar, grouped projects,
    /// prev/next navigation): pinned chats first, recency inside each
    /// group. One function rather than a sort closure restated per view,
    /// so a list can never disagree with another about what pinned means.
    public static func sortedForSidebar(_ chats: [AppChat]) -> [AppChat] {
        chats.sorted { lhs, rhs in
            if lhs.isPinned != rhs.isPinned { return lhs.isPinned }
            return lhs.updatedAt > rhs.updatedAt
        }
    }

    /// A deep copy under fresh identities (qwen-code's duplicate session):
    /// new chat id, a new id on every message, alternate and artifact row,
    /// and a `Title (copy)` title. ForEach identity is the message id, so
    /// sharing ids with the original would make the two transcripts fight
    /// over rows. Draft state (draft text, attachments) does NOT carry:
    /// the copy is of the conversation, not of a half-typed thought.
    public func duplicated() -> AppChat {
        var copy = AppChat(
            id: UUID(),
            projectID: projectID,
            title: title + " (copy)",
            messages: [],
            todos: todos,
            artifacts: [],
            contextSummary: contextSummary,
            compactedMessageCount: compactedMessageCount,
            systemPrompt: systemPrompt,
            samplingOverride: samplingOverride,
            skillState: skillState,
            createdAt: Date(),
            updatedAt: Date(),
            isGhost: false,
            isPinned: false,
            usage: usage,
            goal: goal,
            isArchived: false)
        copy.messages = messages.map { message in
            var copied = message
            copied.id = UUID()
            copied.alternates = message.alternates.map { alternate in
                var copiedAlternate = alternate
                copiedAlternate.id = UUID()
                return copiedAlternate
            }
            return copied
        }
        copy.artifacts = artifacts.map { artifact in
            var copied = artifact
            copied.id = UUID()
            return copied
        }
        return copy
    }
}

/// Archive container for persisting all application chats and the active selection.
public struct AppChatArchive: Codable, Sendable {
    /// Identifier of the active chat conversation.
    public var selectedChatID: UUID
    /// All saved chat conversations.
    public var chats: [AppChat]

    /// Creates a chat archive.
    public init(selectedChatID: UUID, chats: [AppChat]) {
        self.selectedChatID = selectedChatID
        self.chats = chats
    }

    /// Creates a default empty archive containing no chats.
    public static func empty() -> AppChatArchive {
        AppChatArchive(selectedChatID: UUID(), chats: [])
    }

    /// Tolerant decode, same reasoning as `AppChatMessage.init(from:)`
    /// above: a future field added to the top-level archive should not be
    /// able to fail this decode any more than a field added to a nested
    /// message can.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        selectedChatID = try container.decodeIfPresent(UUID.self, forKey: .selectedChatID) ?? UUID()
        chats = try container.decodeLossyArray(AppChat.self, forKey: .chats)
    }
}

/// Filesystem storage utilities for saving and loading chat archives.
public enum AppChatFileStore {
    private static var storageDirectory: URL {
        AppStorageRoot.directory
    }

    private static var archiveFileURL: URL {
        storageDirectory.appendingPathComponent("chats_archive.json")
    }

    /// Loads the saved chat archive from disk or returns an empty default.
    ///
    /// An unreadable archive is QUARANTINED rather than merely reported: the
    /// empty fallback is still what the app comes up with, but the next
    /// `save()` overwrites this file whole and atomically, so reporting alone
    /// left the user's conversations gone by the time anyone read the log.
    public static func load() -> AppChatArchive {
        AppJSONStore.load(AppChatArchive.self, from: archiveFileURL, label: "chat archive")
            ?? AppChatArchive.empty()
    }

    /// Persists the chat archive to disk atomically.
    ///
    /// A failure is RECORDED (`AppJSONStore.lastWriteError`) rather than
    /// swallowed: this used to be two `try?`s, so a full disk or an
    /// uncreatable directory lost the whole session on quit with no sign.
    public static func save(_ archive: AppChatArchive) {
        AppJSONStore.save(archive, to: archiveFileURL, label: "Chat archive")
    }
}
