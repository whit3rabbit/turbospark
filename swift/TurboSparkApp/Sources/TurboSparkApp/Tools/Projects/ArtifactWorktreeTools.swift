import Foundation

// MARK: - Artifact Tool

public struct ArtifactInput: Codable, Sendable, Equatable {
    public var action: String?
    public var filePath: String?
    public var favicon: String?
    public var limit: Int?
    public var scope: String?
    public var title: String?
    public var description: String?
    public var label: String?
    public var note: String?
    public var url: String?
    public var prompt: String?
    public var force: Bool?
    public var outDir: String?
    public var assetId: String?
    public var after: String?
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

public struct ArtifactItemSummary: Codable, Sendable, Equatable {
    public var title: String
    public var url: String
    public var favicon: String?
    public var updatedAt: String?
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

public struct ArtifactOutput: Codable, Sendable, Equatable {
    public var url: String?
    public var path: String?
    public var artifactId: String?
    public var title: String?
    public var version: String?
    public var artifacts: [ArtifactItemSummary]?
    public var truncated: Bool?
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

public struct EnterWorktreeInput: Codable, Sendable, Equatable {
    public var name: String?
    public var path: String?

    public init(name: String? = nil, path: String? = nil) {
        self.name = name
        self.path = path
    }
}

public struct EnterWorktreeOutput: Codable, Sendable, Equatable {
    public var worktreePath: String
    public var worktreeBranch: String?
    public var message: String

    public init(worktreePath: String, worktreeBranch: String? = nil, message: String) {
        self.worktreePath = worktreePath
        self.worktreeBranch = worktreeBranch
        self.message = message
    }
}

public struct ExitWorktreeInput: Codable, Sendable, Equatable {
    public var action: String
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

public struct ExitWorktreeOutput: Codable, Sendable, Equatable {
    public var action: String
    public var originalCwd: String
    public var worktreePath: String
    public var worktreeBranch: String?
    public var discardedFiles: Int?
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
