import Foundation

// MARK: - FileRead Tool

public struct FileReadInput: Codable, Sendable, Equatable {
    public var filePath: String
    public var offset: Int?
    public var limit: Int?
    public var pages: String?

    enum CodingKeys: String, CodingKey {
        case filePath = "file_path"
        case offset
        case limit
        case pages
    }

    public init(filePath: String, offset: Int? = nil, limit: Int? = nil, pages: String? = nil) {
        self.filePath = filePath
        self.offset = offset
        self.limit = limit
        self.pages = pages
    }
}

public enum FileReadOutput: Codable, Sendable, Equatable {
    case text(TextPayload)
    case image(ImagePayload)
    case notebook(NotebookPayload)
    case pdf(PDFPayload)
    case parts(PartsPayload)
    case fileUnchanged(UnchangedPayload)

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

    public struct NotebookPayload: Codable, Sendable, Equatable {
        public var filePath: String
        public var cellCount: Int

        public init(filePath: String, cellCount: Int) {
            self.filePath = filePath
            self.cellCount = cellCount
        }
    }

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

    public struct UnchangedPayload: Codable, Sendable, Equatable {
        public var filePath: String
        public var source: String?

        public init(filePath: String, source: String? = nil) {
            self.filePath = filePath
            self.source = source
        }
    }

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

public struct FileWriteInput: Codable, Sendable, Equatable {
    public var filePath: String
    public var content: String

    enum CodingKeys: String, CodingKey {
        case filePath = "file_path"
        case content
    }

    public init(filePath: String, content: String) {
        self.filePath = filePath
        self.content = content
    }
}

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

public struct FileEditInput: Codable, Sendable, Equatable {
    public var filePath: String
    public var oldString: String
    public var newString: String
    public var replaceAll: Bool?

    enum CodingKeys: String, CodingKey {
        case filePath = "file_path"
        case oldString = "old_string"
        case newString = "new_string"
        case replaceAll = "replace_all"
    }

    public init(filePath: String, oldString: String, newString: String, replaceAll: Bool? = nil) {
        self.filePath = filePath
        self.oldString = oldString
        self.newString = newString
        self.replaceAll = replaceAll
    }
}

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

public enum FileReadWriteToolDefinitions {
    public static let fileRead = OpenAITool.function(
        name: "FileRead",
        description: "Read the contents of a file from the codebase with optional line offset and limit.",
        parameters: .object(
            properties: [
                "file_path": .string(description: "The path to the file to read (relative to workspace or absolute)."),
                "offset": .integer(description: "Line number to start reading from (1-indexed)."),
                "limit": .integer(description: "Maximum number of lines to read."),
                "pages": .string(description: "Page range for PDF files (e.g. '1-5').")
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
                "content": .string(description: "The text content to write into the file.")
            ],
            required: ["file_path", "content"]
        )
    )

    public static let fileEdit = OpenAITool.function(
        name: "FileEdit",
        description: "Replace a specific target string within an existing file with new content.",
        parameters: .object(
            properties: [
                "file_path": .string(description: "The path of the file to edit."),
                "old_string": .string(description: "The exact existing text to replace."),
                "new_string": .string(description: "The replacement text."),
                "replace_all": .boolean(description: "Whether to replace all occurrences.")
            ],
            required: ["file_path", "old_string", "new_string"]
        )
    )

    public static let all: [OpenAITool] = [
        fileRead, fileWrite, fileEdit
    ]
}
