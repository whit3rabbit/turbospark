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
        role = try container.decode(ChatMessage.Role.self, forKey: .role)
        content = try container.decodeIfPresent(String.self, forKey: .content) ?? ""
        reasoning = try container.decodeIfPresent(String.self, forKey: .reasoning) ?? ""
        stopReason = try container.decodeIfPresent(String.self, forKey: .stopReason)
        toolCalls = try container.decodeIfPresent([AppToolCall].self, forKey: .toolCalls) ?? []
        toolResults = try container.decodeIfPresent([AppToolResult].self, forKey: .toolResults) ?? []
        imagePaths = try container.decodeIfPresent([String].self, forKey: .imagePaths) ?? []
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
    /// Optional context summary or metadata.
    public var contextSummary: String?
    /// Bounded execution state, when the project runs in SKILL.state mode.
    /// Per chat rather than per project: it describes one agent run.
    public var skillState: AppSkillState?
    /// Timestamp when the chat was created.
    public var createdAt: Date
    /// Timestamp when the chat was last modified.
    public var updatedAt: Date

    /// Creates a new chat session.
    public init(
        id: UUID = UUID(),
        projectID: UUID? = nil,
        title: String = "New Chat",
        draft: String = "",
        draftAttachments: [AppPromptAttachment] = [],
        messages: [AppChatMessage] = [],
        todos: [TodoItem] = [],
        contextSummary: String? = nil,
        skillState: AppSkillState? = nil,
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.projectID = projectID
        self.title = title
        self.draft = draft
        self.draftAttachments = draftAttachments
        self.messages = messages
        self.todos = todos
        self.contextSummary = contextSummary
        self.skillState = skillState
        self.createdAt = createdAt
        self.updatedAt = updatedAt
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
        messages = try container.decodeIfPresent([AppChatMessage].self, forKey: .messages) ?? []
        todos = try container.decodeIfPresent([TodoItem].self, forKey: .todos) ?? []
        contextSummary = try container.decodeIfPresent(String.self, forKey: .contextSummary)
        skillState = try container.decodeIfPresent(AppSkillState.self, forKey: .skillState)
        createdAt = try container.decodeIfPresent(Date.self, forKey: .createdAt) ?? Date()
        updatedAt = try container.decodeIfPresent(Date.self, forKey: .updatedAt) ?? Date()
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
        chats = try container.decodeIfPresent([AppChat].self, forKey: .chats) ?? []
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
    public static func load() -> AppChatArchive {
        guard let data = try? Data(contentsOf: archiveFileURL) else {
            return AppChatArchive.empty()
        }
        do {
            return try JSONDecoder().decode(AppChatArchive.self, from: data)
        } catch {
            // Reported rather than swallowed. The empty archive is still the
            // fallback (there is nothing better to return), but a silent one
            // presents as "my chats are gone" with no way to tell a corrupt
            // file from a schema drift. This line is how the missing
            // `toolCalls` key above was found at all.
            FileHandle.standardError.write(
                "TurboSpark: chat archive failed to decode, starting empty: \(error)\n"
                    .data(using: .utf8)!)
            return AppChatArchive.empty()
        }
    }

    /// Persists the chat archive to disk atomically.
    public static func save(_ archive: AppChatArchive) {
        if let data = try? JSONEncoder().encode(archive) {
            try? data.write(to: archiveFileURL, options: .atomic)
        }
    }
}
