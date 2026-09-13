import Foundation

// MARK: - FileRead Tool

/// Input payload for reading files from disk, with optional pagination, limits, modes, and search/diff parameters.
public struct FileReadInput: Codable, Sendable, Equatable {
    /// File path to read (relative to project root or absolute).
    public var filePath: String
    /// Starting line offset (1-indexed).
    public var offset: Int?
    /// Maximum line count to read.
    public var limit: Int?
    /// Page range for PDF documents (e.g. "1-5").
    public var pages: String?
    /// Advanced reading mode: 'lines' (default), 'stats', 'diff', 'time_machine', 'search', 'preview'.
    public var mode: String?
    /// Search regex pattern when mode is 'search'.
    public var searchPattern: String?
    /// Context lines before and after match when mode is 'search'.
    public var contextLines: Int?
    /// Secondary file path when mode is 'diff'.
    public var comparisonPath: String?
    /// Number of git commit revisions to inspect when mode is 'time_machine'.
    public var numRevisions: Int?

    enum CodingKeys: String, CodingKey {
        case filePath = "file_path"
        case offset
        case limit
        case pages
        case mode
        case searchPattern = "search_pattern"
        case contextLines = "context_lines"
        case comparisonPath = "comparison_path"
        case numRevisions = "num_revisions"
    }

    public init(
        filePath: String,
        offset: Int? = nil,
        limit: Int? = nil,
        pages: String? = nil,
        mode: String? = nil,
        searchPattern: String? = nil,
        contextLines: Int? = nil,
        comparisonPath: String? = nil,
        numRevisions: Int? = nil
    ) {
        self.filePath = filePath
        self.offset = offset
        self.limit = limit
        self.pages = pages
        self.mode = mode
        self.searchPattern = searchPattern
        self.contextLines = contextLines
        self.comparisonPath = comparisonPath
        self.numRevisions = numRevisions
    }
}

/// Output payload from reading file content across supported document formats.
public enum FileReadOutput: Codable, Sendable, Equatable {
    case text(TextPayload)
    case image(ImagePayload)
    case notebook(NotebookPayload)
    case pdf(PDFPayload)
    case parts(PartsPayload)
    case fileUnchanged(UnchangedPayload)

    /// Text file content payload with line metadata.
    public struct TextPayload: Codable, Sendable, Equatable {
        public var filePath: String
        public var content: String
        public var numLines: Int
        public var startLine: Int
        public var totalLines: Int
        public var truncatedByTokenCap: Bool?
        public var artifactRead: ArtifactReadRef?

        public init(
            filePath: String,
            content: String,
            numLines: Int,
            startLine: Int,
            totalLines: Int,
            truncatedByTokenCap: Bool? = nil,
            artifactRead: ArtifactReadRef? = nil
        ) {
            self.filePath = filePath
            self.content = content
            self.numLines = numLines
            self.startLine = startLine
            self.totalLines = totalLines
            self.truncatedByTokenCap = truncatedByTokenCap
            self.artifactRead = artifactRead
        }
    }

    /// Base64 encoded image content payload.
    public struct ImagePayload: Codable, Sendable, Equatable {
        public var base64: String
        public var type: String
        public var originalSize: Int

        public init(base64: String, type: String, originalSize: Int) {
            self.base64 = base64
            self.type = type
            self.originalSize = originalSize
        }
    }

    /// Jupyter notebook summary payload.
    public struct NotebookPayload: Codable, Sendable, Equatable {
        public var filePath: String
        public var cellCount: Int

        public init(filePath: String, cellCount: Int) {
            self.filePath = filePath
            self.cellCount = cellCount
        }
    }

    /// Base64 encoded PDF payload.
    public struct PDFPayload: Codable, Sendable, Equatable {
        public var filePath: String
        public var base64: String
        public var originalSize: Int

        public init(filePath: String, base64: String, originalSize: Int) {
            self.filePath = filePath
            self.base64 = base64
            self.originalSize = originalSize
        }
    }

    /// Multi-part or paginated document payload.
    public struct PartsPayload: Codable, Sendable, Equatable {
        public var filePath: String
        public var originalSize: Int
        public var count: Int
        public var outputDir: String
        public var firstPage: Int?

        public init(filePath: String, originalSize: Int, count: Int, outputDir: String, firstPage: Int? = nil) {
            self.filePath = filePath
            self.originalSize = originalSize
            self.count = count
            self.outputDir = outputDir
            self.firstPage = firstPage
        }
    }

    /// Unchanged status payload when file cache matches disk.
    public struct UnchangedPayload: Codable, Sendable, Equatable {
        public var filePath: String
        public var source: String?

        public init(filePath: String, source: String? = nil) {
            self.filePath = filePath
            self.source = source
        }
    }

    /// Artifact version reference payload.
    public struct ArtifactReadRef: Codable, Sendable, Equatable {
        public var slug: String
        public var ver: String

        public init(slug: String, ver: String) {
            self.slug = slug
            self.ver = ver
        }
    }
}

// MARK: - FileWrite Tool

/// Input payload for writing entire file contents to a path.
public struct FileWriteInput: Codable, Sendable, Equatable {
    /// Destination file path.
    public var filePath: String
    /// New file content text.
    public var content: String
    /// Optional command run only after the write completed and its target was
    /// fingerprinted. Permission and hook checks cover both legs up front.
    public var validateCommand: String?
    /// Optional validation timeout in milliseconds.
    public var validateTimeoutMs: Int?

    enum CodingKeys: String, CodingKey {
        case filePath = "file_path"
        case content
        case validateCommand = "validate_command"
        case validateTimeoutMs = "validate_timeout_ms"
    }

    public init(
        filePath: String, content: String, validateCommand: String? = nil,
        validateTimeoutMs: Int? = nil
    ) {
        self.filePath = filePath
        self.content = content
        self.validateCommand = validateCommand
        self.validateTimeoutMs = validateTimeoutMs
    }
}

/// Structured diff hunk description.
public struct StructuredDiffPatch: Codable, Sendable, Equatable {
    public var oldStart: Int
    public var oldLines: Int
    public var newStart: Int
    public var newLines: Int
    public var lines: [String]

    public init(oldStart: Int, oldLines: Int, newStart: Int, newLines: Int, lines: [String]) {
        self.oldStart = oldStart
        self.oldLines = oldLines
        self.newStart = newStart
        self.newLines = newLines
        self.lines = lines
    }
}

/// Git diff statistics summary for a modified file.
public struct GitDiffSummary: Codable, Sendable, Equatable {
    public var filename: String
    public var status: String
    public var additions: Int
    public var deletions: Int
    public var changes: Int
    public var patch: String
    public var repository: String?

    public init(
        filename: String,
        status: String,
        additions: Int,
        deletions: Int,
        changes: Int,
        patch: String,
        repository: String? = nil
    ) {
        self.filename = filename
        self.status = status
        self.additions = additions
        self.deletions = deletions
        self.changes = changes
        self.patch = patch
        self.repository = repository
    }
}

/// Output payload returned after writing a file.
public struct FileWriteOutput: Codable, Sendable, Equatable {
    public var type: String
    public var filePath: String
    public var content: String
    public var structuredPatch: [StructuredDiffPatch]
    public var originalFile: String?
    public var gitDiff: GitDiffSummary?
    public var userModified: Bool?

    public init(
        type: String,
        filePath: String,
        content: String,
        structuredPatch: [StructuredDiffPatch] = [],
        originalFile: String? = nil,
        gitDiff: GitDiffSummary? = nil,
        userModified: Bool? = nil
    ) {
        self.type = type
        self.filePath = filePath
        self.content = content
        self.structuredPatch = structuredPatch
        self.originalFile = originalFile
        self.gitDiff = gitDiff
        self.userModified = userModified
    }
}

// MARK: - FileEdit Tool

/// Input payload for editing an existing file via string replacement, line insertion, regex replacement, or rollback.
public struct FileEditInput: Codable, Sendable, Equatable {
    /// File path to edit.
    public var filePath: String
    /// Edit command: 'str_replace' (default), 'insert', 'pattern_replace', 'undo_edit'.
    public var command: String?
    /// Exact text sequence to find and replace (for str_replace).
    public var oldString: String?
    /// Replacement or insertion text sequence.
    public var newString: String?
    /// Whether to replace all occurrences or just the first (for str_replace).
    public var replaceAll: Bool?
    /// Line number (1-indexed) or text target to find and insert relative to (for insert).
    public var insertLine: String?
    /// Insertion position: 'before' or 'after' (default: 'after').
    public var position: String?
    /// Regular expression pattern to replace (for pattern_replace).
    public var regexPattern: String?
    public var validateCommand: String?
    public var validateTimeoutMs: Int?

    enum CodingKeys: String, CodingKey {
        case filePath = "file_path"
        case command
        case oldString = "old_string"
        case newString = "new_string"
        case replaceAll = "replace_all"
        case insertLine = "insert_line"
        case position
        case regexPattern = "regex_pattern"
        case validateCommand = "validate_command"
        case validateTimeoutMs = "validate_timeout_ms"
    }

    public init(
        filePath: String,
        command: String? = nil,
        oldString: String? = nil,
        newString: String? = nil,
        replaceAll: Bool? = nil,
        insertLine: String? = nil,
        position: String? = nil,
        regexPattern: String? = nil,
        validateCommand: String? = nil,
        validateTimeoutMs: Int? = nil
    ) {
        self.filePath = filePath
        self.command = command
        self.oldString = oldString
        self.newString = newString
        self.replaceAll = replaceAll
        self.insertLine = insertLine
        self.position = position
        self.regexPattern = regexPattern
        self.validateCommand = validateCommand
        self.validateTimeoutMs = validateTimeoutMs
    }
}

/// Output payload returned after performing string replacement on a file.
public struct FileEditOutput: Codable, Sendable, Equatable {
    public var filePath: String
    public var oldString: String
    public var newString: String
    public var originalFile: String?
    public var structuredPatch: [StructuredDiffPatch]
    public var userModified: Bool
    public var replaceAll: Bool
    public var gitDiff: GitDiffSummary?

    public init(
        filePath: String,
        oldString: String,
        newString: String,
        originalFile: String? = nil,
        structuredPatch: [StructuredDiffPatch] = [],
        userModified: Bool = false,
        replaceAll: Bool = false,
        gitDiff: GitDiffSummary? = nil
    ) {
        self.filePath = filePath
        self.oldString = oldString
        self.newString = newString
        self.originalFile = originalFile
        self.structuredPatch = structuredPatch
        self.userModified = userModified
        self.replaceAll = replaceAll
        self.gitDiff = gitDiff
    }
}

// MARK: - OpenAI Tool Definitions for File Read/Write Operations

/// OpenAI tool definition schemas for FileRead, FileWrite, and FileEdit operations.
public enum FileReadWriteToolDefinitions {
    public static let fileRead = OpenAITool.function(
        name: "FileRead",
        description: "Read file contents from the codebase with optional line offset, limit, or advanced modes (stats, diff, time_machine, search, preview).",
        parameters: .object(
            properties: [
                "file_path": .string(description: "The path to the file to read (relative to workspace or absolute)."),
                "offset": .integer(description: "Line number to start reading from (1-indexed)."),
                "limit": .integer(description: "Maximum number of lines to read."),
                "pages": .string(description: "Page range for PDF files (e.g. '1-5')."),
                "mode": .string(description: "Reading mode: 'lines' (default), 'stats' (metrics), 'diff' (compare with comparison_path), 'time_machine' (git revision history), 'search' (regex search with context), 'preview' (first N lines)."),
                "search_pattern": .string(description: "Regex pattern to search for in file when mode is 'search'."),
                "context_lines": .integer(description: "Context lines before/after match when mode is 'search' (default: 3)."),
                "comparison_path": .string(description: "Path of file to compare against when mode is 'diff'."),
                "num_revisions": .integer(description: "Number of git revisions to display when mode is 'time_machine' (default: 5).")
            ],
            required: ["file_path"]
        )
    )

    public static let fileWrite = OpenAITool.function(
        name: "FileWrite",
        description: "Create or replace the contents of a file at the specified path.",
        parameters: .object(
            properties: [
                "file_path": .string(description: "The path of the file to write."),
                "content": .string(description: "The text content to write into the file."),
                "validate_command": .string(description: "Optional terminal command to run after the write and target fingerprint verification."),
                "validate_timeout_ms": .integer(description: "Optional validation timeout in milliseconds.")
            ],
            required: ["file_path", "content"]
        )
    )

    public static let fileEdit = OpenAITool.function(
        name: "FileEdit",
        description: "Edit an existing file via string replacement (str_replace), line insertion (insert), regex pattern replacement (pattern_replace), or rollback (undo_edit).",
        parameters: .object(
            properties: [
                "file_path": .string(description: "The path of the file to edit."),
                "command": .string(description: "Edit command: 'str_replace' (default), 'insert', 'pattern_replace', 'undo_edit'."),
                "old_string": .string(description: "The exact existing text to replace (for str_replace)."),
                "new_string": .string(description: "The replacement or insertion text (for str_replace, insert, pattern_replace)."),
                "replace_all": .boolean(description: "Whether to replace all occurrences (for str_replace)."),
                "insert_line": .string(description: "Line number (1-indexed) or text target to find and insert relative to (for insert)."),
                "position": .string(description: "Insertion position: 'before' or 'after' (default: 'after')."),
                "regex_pattern": .string(description: "Regex pattern to replace (for pattern_replace)."),
                "validate_command": .string(description: "Optional terminal command to run after the edit and target fingerprint verification."),
                "validate_timeout_ms": .integer(description: "Optional validation timeout in milliseconds.")
            ],
            required: ["file_path"]
        )
    )

    public static let snip = OpenAITool.function(
        name: "Snip",
        description: "Extract a specific range of lines from a file with line numbers and context.",
        parameters: .object(
            properties: [
                "path": .string(description: "The path of the file to extract from."),
                "start_line": .integer(description: "Starting line number (1-indexed)."),
                "end_line": .integer(description: "Ending line number (1-indexed).")
            ],
            required: ["path", "start_line", "end_line"]
        )
    )

    public static let sendUserFile = OpenAITool.function(
        name: "SendUserFile",
        description: "Present, reveal, or send a local workspace file to the user with an optional message.",
        parameters: .object(
            properties: [
                "path": .string(description: "The path of the file to send/present."),
                "message": .string(description: "Optional description or note accompanying the file.")
            ],
            required: ["path"]
        )
    )

    public static let all: [OpenAITool] = [
        fileRead, fileWrite, fileEdit, snip, sendUserFile
    ]
}
