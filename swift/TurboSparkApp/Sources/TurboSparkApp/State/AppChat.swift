import Foundation
import TurboSpark

/// A document attachment associated with a prompt draft.
public struct AppPromptAttachment: Identifiable, Codable, Equatable, Sendable {
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

    /// Total character count of the extracted text.
    public var characterCount: Int { extractedText.count }

    /// Creates a prompt attachment.
    public init(
        id: UUID = UUID(),
        fileName: String,
        formatLabel: String,
        extractedText: String,
        wasTruncatedDuringExtraction: Bool
    ) {
        self.id = id
        self.fileName = fileName
        self.formatLabel = formatLabel
        self.extractedText = extractedText
        self.wasTruncatedDuringExtraction = wasTruncatedDuringExtraction
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

    /// Creates a chat message turn.
    public init(
        id: UUID = UUID(),
        role: ChatMessage.Role,
        content: String,
        reasoning: String = "",
        stopReason: String? = nil,
        toolCalls: [AppToolCall] = [],
        toolResults: [AppToolResult] = []
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.reasoning = reasoning
        self.stopReason = stopReason
        self.toolCalls = toolCalls
        self.toolResults = toolResults
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
    /// Optional context summary or metadata.
    public var contextSummary: String?
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
        contextSummary: String? = nil,
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.projectID = projectID
        self.title = title
        self.draft = draft
        self.draftAttachments = draftAttachments
        self.messages = messages
        self.contextSummary = contextSummary
        self.createdAt = createdAt
        self.updatedAt = updatedAt
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
}

/// Filesystem storage utilities for saving and loading chat archives.
public enum AppChatFileStore {
    private static var storageDirectory: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let directory = appSupport.appendingPathComponent("TurboSpark", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private static var archiveFileURL: URL {
        storageDirectory.appendingPathComponent("chats_archive.json")
    }

    /// Loads the saved chat archive from disk or returns an empty default.
    public static func load() -> AppChatArchive {
        guard let data = try? Data(contentsOf: archiveFileURL),
              let archive = try? JSONDecoder().decode(AppChatArchive.self, from: data) else {
            return AppChatArchive.empty()
        }
        return archive
    }

    /// Persists the chat archive to disk atomically.
    public static func save(_ archive: AppChatArchive) {
        if let data = try? JSONEncoder().encode(archive) {
            try? data.write(to: archiveFileURL, options: .atomic)
        }
    }
}
