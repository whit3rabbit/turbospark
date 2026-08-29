import Foundation

// MARK: - Artifact Tool

/// Input payload for publishing, viewing, or managing interactive visual HTML artifacts and media assets.
public struct ArtifactInput: Codable, Sendable, Equatable {
    /// Action method ("publish", "list", "read", "watch", "unwatch", "status", "upload_asset", "list_assets", "read_asset", "delete_asset").
    public var action: String?
    /// Path to file or asset to publish.
    public var filePath: String?
    /// Favicon icon symbol or URL.
    public var favicon: String?
    /// Result limit for listing artifacts.
    public var limit: Int?
    /// Scope identifier.
    public var scope: String?
    /// Artifact header title.
    public var title: String?
    /// Artifact description text.
    public var description: String?
    /// Badge label.
    public var label: String?
    /// Additional note text.
    public var note: String?
    /// Target artifact URL link.
    public var url: String?
    /// Contextual prompt for artifact generation.
    public var prompt: String?
    /// Force overwrite flag.
    public var force: Bool?
    /// Output directory path.
    public var outDir: String?
    /// Target asset identifier.
    public var assetId: String?
    /// Timestamp filter.
    public var after: String?
    /// Artifact contract schema.
    public var contract: String?

    enum CodingKeys: String, CodingKey {
        case action
        case filePath = "file_path"
        case favicon
        case limit
        case scope
        case title
        case description
        case label
        case note
        case url
        case prompt
        case force
        case outDir = "out_dir"
        case assetId = "asset_id"
        case after
        case contract
    }

    public init(
        action: String? = "publish",
        filePath: String? = nil,
        favicon: String? = nil,
        limit: Int? = nil,
        scope: String? = nil,
        title: String? = nil,
        description: String? = nil,
        label: String? = nil,
        note: String? = nil,
        url: String? = nil,
        prompt: String? = nil,
        force: Bool? = nil,
        outDir: String? = nil,
        assetId: String? = nil,
        after: String? = nil,
        contract: String? = nil
    ) {
        self.action = action
        self.filePath = filePath
        self.favicon = favicon
        self.limit = limit
        self.scope = scope
        self.title = title
        self.description = description
        self.label = label
        self.note = note
        self.url = url
        self.prompt = prompt
        self.force = force
        self.outDir = outDir
        self.assetId = assetId
        self.after = after
        self.contract = contract
    }
}

/// Summary item representing an artifact in list queries.
public struct ArtifactItemSummary: Codable, Sendable, Equatable {
    /// Artifact display title.
    public var title: String
    /// Artifact web URL.
    public var url: String
    /// Favicon icon image or symbol.
    public var favicon: String?
    /// ISO timestamp when last updated.
    public var updatedAt: String?
    /// Relative path or link relationship.
    public var rel: String?

    public init(
        title: String,
        url: String,
        favicon: String? = nil,
        updatedAt: String? = nil,
        rel: String? = nil
    ) {
        self.title = title
        self.url = url
        self.favicon = favicon
        self.updatedAt = updatedAt
        self.rel = rel
    }
}

/// Output payload from artifact management commands.
public struct ArtifactOutput: Codable, Sendable, Equatable {
    /// Rendered artifact URL.
    public var url: String?
    /// On-disk path to artifact.
    public var path: String?
    /// Unique artifact identifier.
    public var artifactId: String?
    /// Artifact title.
    public var title: String?
    /// Version string.
    public var version: String?
    /// Discovered artifacts list.
    public var artifacts: [ArtifactItemSummary]?
    /// Whether output was truncated.
    public var truncated: Bool?
    /// Scope identifier.
    public var scope: String?

    enum CodingKeys: String, CodingKey {
        case url
        case path
        case artifactId = "artifact_id"
        case title
        case version
        case artifacts
        case truncated
        case scope
    }

    public init(
        url: String? = nil,
        path: String? = nil,
        artifactId: String? = nil,
        title: String? = nil,
        version: String? = nil,
        artifacts: [ArtifactItemSummary]? = nil,
        truncated: Bool? = nil,
        scope: String? = nil
    ) {
        self.url = url
        self.path = path
        self.artifactId = artifactId
        self.title = title
        self.version = version
        self.artifacts = artifacts
        self.truncated = truncated
        self.scope = scope
    }
}

// MARK: - Git Worktree Tools

/// Input payload for creating or switching into an isolated git worktree branch.
public struct EnterWorktreeInput: Codable, Sendable, Equatable {
    /// Worktree branch name.
    public var name: String?
    /// Destination worktree folder path.
    public var path: String?

    public init(name: String? = nil, path: String? = nil) {
        self.name = name
        self.path = path
    }
}

/// Output returned when switching into a git worktree.
public struct EnterWorktreeOutput: Codable, Sendable, Equatable {
    /// Local directory path of the active worktree.
    public var worktreePath: String
    /// Checked out git branch name in worktree.
    public var worktreeBranch: String?
    /// Status message.
    public var message: String

    public init(worktreePath: String, worktreeBranch: String? = nil, message: String) {
        self.worktreePath = worktreePath
        self.worktreeBranch = worktreeBranch
        self.message = message
    }
}

/// Input payload for leaving a git worktree and returning to main repository root.
public struct ExitWorktreeInput: Codable, Sendable, Equatable {
    /// Action ("keep" to preserve worktree, "remove" to delete branch/worktree).
    public var action: String
    /// Whether uncommitted changes should be discarded if removing.
    public var discardChanges: Bool?

    enum CodingKeys: String, CodingKey {
        case action
        case discardChanges = "discard_changes"
    }

    public init(action: String, discardChanges: Bool? = nil) {
        self.action = action
        self.discardChanges = discardChanges
    }
}

/// Output returned upon exiting a git worktree.
public struct ExitWorktreeOutput: Codable, Sendable, Equatable {
    /// Executed exit action.
    public var action: String
    /// Restored working directory path.
    public var originalCwd: String
    /// Path of worktree that was exited.
    public var worktreePath: String
    /// Branch name of the exited worktree.
    public var worktreeBranch: String?
    /// Number of modified files discarded if applicable.
    public var discardedFiles: Int?
    /// Status message.
    public var message: String

    public init(
        action: String,
        originalCwd: String,
        worktreePath: String,
        worktreeBranch: String? = nil,
        discardedFiles: Int? = nil,
        message: String
    ) {
        self.action = action
        self.originalCwd = originalCwd
        self.worktreePath = worktreePath
        self.worktreeBranch = worktreeBranch
        self.discardedFiles = discardedFiles
        self.message = message
    }
}

// MARK: - OpenAI Tool Definitions for Artifacts and Worktrees

/// OpenAI tool definition schemas for Artifact publishing and Git Worktree branching.
public enum ArtifactWorktreeDefinitions {
    public static let artifact = OpenAITool.function(
        name: "Artifact",
        description: "Publish, update, view, or manage rich HTML/visual interactive artifacts.",
        parameters: .object(
            properties: [
                "action": .string(
                    description: "Action: 'publish', 'list', 'read', 'watch', 'unwatch', 'status', 'upload_asset', 'list_assets', 'read_asset', 'delete_asset'."
                ),
                "file_path": .string(description: "Path to the .html file or asset to publish/upload."),
                "title": .string(description: "Title of the artifact shown in headers."),
                "description": .string(description: "Subtitle or description of the artifact."),
                "url": .string(description: "Existing artifact URL to update or read."),
                "force": .boolean(description: "Force overwrite of existing artifact version.")
            ],
            required: []
        )
    )

    public static let enterWorktree = OpenAITool.function(
        name: "EnterWorktree",
        description: "Create or switch to an isolated git worktree branch for safe modifications.",
        parameters: .object(
            properties: [
                "name": .string(description: "Optional name for the worktree branch."),
                "path": .string(description: "Path to an existing git worktree.")
            ],
            required: []
        )
    )

    public static let exitWorktree = OpenAITool.function(
        name: "ExitWorktree",
        description: "Leave current git worktree and return to main repository root.",
        parameters: .object(
            properties: [
                "action": .string(description: "'keep' to retain worktree, 'remove' to clean up."),
                "discard_changes": .boolean(description: "Discard uncommitted changes if removing.")
            ],
            required: ["action"]
        )
    )

    public static let all: [OpenAITool] = [
        artifact, enterWorktree, exitWorktree
    ]
}
