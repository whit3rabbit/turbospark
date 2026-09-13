import Foundation

// MARK: - Glob Tool

/// Input payload for searching workspace files using a filename pattern.
public struct GlobInput: Codable, Sendable, Equatable {
    /// Glob matching pattern (e.g. `**/*.swift` or `src/*.rs`).
    public var pattern: String
    /// Base directory path to search from (defaults to workspace root).
    public var path: String?

    public init(pattern: String, path: String? = nil) {
        self.pattern = pattern
        self.path = path
    }
}

/// Output payload containing matching file paths.
public struct GlobOutput: Codable, Sendable, Equatable {
    /// Execution duration in milliseconds.
    public var durationMs: Double
    /// Count of matched files.
    public var numFiles: Int
    /// List of matched relative file paths.
    public var filenames: [String]
    /// Whether results were truncated by a limit.
    public var truncated: Bool
    /// Total matches found before truncation.
    public var totalMatches: Int?
    /// Whether the total match count is definitive.
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

/// Input payload for regular expression pattern searching across files.
public struct GrepInput: Codable, Sendable, Equatable {
    /// Regular expression or exact query pattern.
    public var pattern: String
    /// File or directory path to search.
    public var path: String?
    /// Glob filter pattern restricting searched file paths.
    public var glob: String?
    /// Output mode ("content", "files_with_matches", "count").
    public var outputMode: String?
    /// Number of context lines before and after match.
    public var context: Int?
    /// Case-insensitive search flag.
    public var caseInsensitive: Bool?
    /// File type filter name.
    public var type: String?
    /// Maximum count of results returned.
    public var headLimit: Int?
    /// Pagination offset.
    public var offset: Int?
    /// Multiline match support.
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

/// Output payload containing regex match lines or filenames.
public struct GrepOutput: Codable, Sendable, Equatable {
    /// Applied output formatting mode.
    public var mode: String?
    /// Number of matching files.
    public var numFiles: Int
    /// Matched file paths.
    public var filenames: [String]
    /// Formatted content text with line numbers and snippets.
    public var content: String?
    /// Number of matched lines.
    public var numLines: Int?
    /// Total match occurrences.
    public var numMatches: Int?
    /// Total files searched.
    public var totalFiles: Int?
    /// Total lines examined.
    public var totalLines: Int?
    /// Applied result limit.
    public var appliedLimit: Int?
    /// Applied result offset.
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

/// Input payload for reading and mutating Jupyter notebook cells.
public struct NotebookEditInput: Codable, Sendable, Equatable {
    /// File path to `.ipynb` notebook.
    public var notebookPath: String
    /// Cell ID to target.
    public var cellId: String?
    /// Replacement or new cell content text.
    public var newSource: String
    /// Type of cell ("code" or "markdown").
    public var cellType: String?
    /// Edit operation ("replace", "insert", "delete").
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

/// Output payload describing result of notebook modification.
public struct NotebookEditOutput: Codable, Sendable, Equatable {
    /// Updated cell source.
    public var newSource: String
    /// Previous cell source prior to edit.
    public var oldSource: String?
    /// Cell identifier.
    public var cellId: String?
    /// Cell type.
    public var cellType: String
    /// Programming language of the notebook kernel.
    public var language: String
    /// Operation mode applied.
    public var editMode: String
    /// Error message if modification failed.
    public var error: String?
    /// Notebook path.
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

/// OpenAI tool definition schemas for Glob, Grep, and NotebookEdit.
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
        description: "Search for regular expression patterns across files in the codebase. Uses the project's Syntext index when enabled and falls back to unindexed scanning.",
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

    public static let grepSearch = OpenAITool.function(
        name: "grep_search",
        description: "Preferred code-discovery tool: fast indexed regex and literal search when the project enables Syntext, with an automatic unindexed fallback. Use it instead of shelling out to grep or ripgrep. Returns paths, line numbers, and context in ripgrep format.",
        parameters: .object(
            properties: [
                "query": .string(description: "The search pattern (regular expression or literal string)."),
                "literal_search": .boolean(description: "Set to true if searching for exact literal characters like `func()`, `array[0]`, `$var`, or `Map<K,V>`."),
                "path_filter": .string(description: "Optional glob filter to restrict files (e.g. '*.swift', 'Sources/**')."),
                "file_types": .array(items: .string(description: "File extension"), description: "Optional list of file extensions (e.g. ['swift', 'rs'])."),
                "case_sensitive": .boolean(description: "Case-sensitive search flag (defaults to false)."),
                "context_lines": .integer(description: "Number of lines of code to show before and after each match (default: 2)."),
                "max_results": .integer(description: "Maximum number of matches to return (default: 50).")
            ],
            required: ["query"]
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
        glob, grep, grepSearch, notebookEdit
    ]
}
