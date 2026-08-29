import Foundation

// MARK: - Glob Tool

public struct GlobInput: Codable, Sendable, Equatable {
    public var pattern: String
    public var path: String?

    public init(pattern: String, path: String? = nil) {
        self.pattern = pattern
        self.path = path
    }
}

public struct GlobOutput: Codable, Sendable, Equatable {
    public var durationMs: Double
    public var numFiles: Int
    public var filenames: [String]
    public var truncated: Bool
    public var totalMatches: Int?
    public var countIsComplete: Bool?

    public init(
        durationMs: Double,
        numFiles: Int,
        filenames: [String],
        truncated: Bool,
        totalMatches: Int? = nil,
        countIsComplete: Bool? = nil
    ) {
        self.durationMs = durationMs
        self.numFiles = numFiles
        self.filenames = filenames
        self.truncated = truncated
        self.totalMatches = totalMatches
        self.countIsComplete = countIsComplete
    }
}

// MARK: - Grep Tool

public struct GrepInput: Codable, Sendable, Equatable {
    public var pattern: String
    public var path: String?
    public var glob: String?
    public var outputMode: String?
    public var context: Int?
    public var caseInsensitive: Bool?
    public var type: String?
    public var headLimit: Int?
    public var offset: Int?
    public var multiline: Bool?

    enum CodingKeys: String, CodingKey {
        case pattern
        case path
        case glob
        case outputMode = "output_mode"
        case context
        case caseInsensitive = "-i"
        case type
        case headLimit = "head_limit"
        case offset
        case multiline
    }

    public init(
        pattern: String,
        path: String? = nil,
        glob: String? = nil,
        outputMode: String? = nil,
        context: Int? = nil,
        caseInsensitive: Bool? = nil,
        type: String? = nil,
        headLimit: Int? = nil,
        offset: Int? = nil,
        multiline: Bool? = nil
    ) {
        self.pattern = pattern
        self.path = path
        self.glob = glob
        self.outputMode = outputMode
        self.context = context
        self.caseInsensitive = caseInsensitive
        self.type = type
        self.headLimit = headLimit
        self.offset = offset
        self.multiline = multiline
    }
}

public struct GrepOutput: Codable, Sendable, Equatable {
    public var mode: String?
    public var numFiles: Int
    public var filenames: [String]
    public var content: String?
    public var numLines: Int?
    public var numMatches: Int?
    public var totalFiles: Int?
    public var totalLines: Int?
    public var appliedLimit: Int?
    public var appliedOffset: Int?

    public init(
        mode: String? = nil,
        numFiles: Int = 0,
        filenames: [String] = [],
        content: String? = nil,
        numLines: Int? = nil,
        numMatches: Int? = nil,
        totalFiles: Int? = nil,
        totalLines: Int? = nil,
        appliedLimit: Int? = nil,
        appliedOffset: Int? = nil
    ) {
        self.mode = mode
        self.numFiles = numFiles
        self.filenames = filenames
        self.content = content
        self.numLines = numLines
        self.numMatches = numMatches
        self.totalFiles = totalFiles
        self.totalLines = totalLines
        self.appliedLimit = appliedLimit
        self.appliedOffset = appliedOffset
    }
}

// MARK: - NotebookEdit Tool

public struct NotebookEditInput: Codable, Sendable, Equatable {
    public var notebookPath: String
    public var cellId: String?
    public var newSource: String
    public var cellType: String?
    public var editMode: String?

    enum CodingKeys: String, CodingKey {
        case notebookPath = "notebook_path"
        case cellId = "cell_id"
        case newSource = "new_source"
        case cellType = "cell_type"
        case editMode = "edit_mode"
    }

    public init(
        notebookPath: String,
        cellId: String? = nil,
        newSource: String,
        cellType: String? = nil,
        editMode: String? = nil
    ) {
        self.notebookPath = notebookPath
        self.cellId = cellId
        self.newSource = newSource
        self.cellType = cellType
        self.editMode = editMode
    }
}

public struct NotebookEditOutput: Codable, Sendable, Equatable {
    public var newSource: String
    public var oldSource: String?
    public var cellId: String?
    public var cellType: String
    public var language: String
    public var editMode: String
    public var error: String?
    public var notebookPath: String

    enum CodingKeys: String, CodingKey {
        case newSource = "new_source"
        case oldSource = "old_source"
        case cellId = "cell_id"
        case cellType = "cell_type"
        case language
        case editMode = "edit_mode"
        case error
        case notebookPath = "notebook_path"
    }

    public init(
        newSource: String,
        oldSource: String? = nil,
        cellId: String? = nil,
        cellType: String = "code",
        language: String = "python",
        editMode: String = "replace",
        error: String? = nil,
        notebookPath: String
    ) {
        self.newSource = newSource
        self.oldSource = oldSource
        self.cellId = cellId
        self.cellType = cellType
        self.language = language
        self.editMode = editMode
        self.error = error
        self.notebookPath = notebookPath
    }
}

// MARK: - OpenAI Tool Definitions for File Search Operations

public enum FileSearchToolDefinitions {
    public static let glob = OpenAITool.function(
        name: "Glob",
        description: "Search for files matching a glob pattern across the workspace.",
        parameters: .object(
            properties: [
                "pattern": .string(description: "The glob pattern to match against filenames."),
                "path": .string(description: "Directory to search within (defaults to workspace root).")
            ],
            required: ["pattern"]
        )
    )

    public static let grep = OpenAITool.function(
        name: "Grep",
        description: "Search for regular expression patterns across files in the codebase.",
        parameters: .object(
            properties: [
                "pattern": .string(description: "Regular expression or literal string pattern to search."),
                "path": .string(description: "File or directory path to search."),
                "glob": .string(description: "Glob filter (e.g. '*.swift', '*.rs')."),
                "output_mode": .string(description: "Output format: 'content', 'files_with_matches', 'count'."),
                "context": .integer(description: "Number of context lines before and after match."),
                "-i": .boolean(description: "Case-insensitive matching."),
                "type": .string(description: "File type filter (e.g. swift, rust, json)."),
                "head_limit": .integer(description: "Maximum number of lines or files to return.")
            ],
            required: ["pattern"]
        )
    )

    public static let notebookEdit = OpenAITool.function(
        name: "NotebookEdit",
        description: "Edit, insert, or delete cells in a Jupyter notebook (.ipynb) file.",
        parameters: .object(
            properties: [
                "notebook_path": .string(description: "Path to the notebook file."),
                "cell_id": .string(description: "Identifier of the cell to modify."),
                "new_source": .string(description: "Source code or markdown content for the cell."),
                "cell_type": .string(description: "Cell type: 'code' or 'markdown'."),
                "edit_mode": .string(description: "Edit operation: 'replace', 'insert', or 'delete'.")
            ],
            required: ["notebook_path", "new_source"]
        )
    )

    public static let all: [OpenAITool] = [
        glob, grep, notebookEdit
    ]
}
