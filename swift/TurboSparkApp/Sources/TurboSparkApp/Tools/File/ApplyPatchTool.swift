import Foundation

// MARK: - ApplyPatch Tool (OpenCode Compatible)

/// Input payload for applying unified diff or multi-file patches.
public struct ApplyPatchInput: Codable, Sendable, Equatable {
    /// Raw patch text containing diff headers and chunk operations.
    public var patchText: String
    public var validateCommand: String?
    public var validateTimeoutMs: Int?

    enum CodingKeys: String, CodingKey {
        case patchText = "patch_text"
        case validateCommand = "validate_command"
        case validateTimeoutMs = "validate_timeout_ms"
    }

    public init(
        patchText: String, validateCommand: String? = nil, validateTimeoutMs: Int? = nil
    ) {
        self.patchText = patchText
        self.validateCommand = validateCommand
        self.validateTimeoutMs = validateTimeoutMs
    }

    /// Custom decoder supporting `patch_text`, `patchText`, or wrapped dictionary structures.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        if let text = try container.decodeIfPresent(String.self, forKey: .patchText) {
            self.patchText = text
        } else {
            let dynamic = try decoder.singleValueContainer()
            if let dict = try? dynamic.decode([String: String].self),
               let text = dict["patchText"] ?? dict["patch_text"] ?? dict["patch"] {
                self.patchText = text
            } else {
                self.patchText = ""
            }
        }
        self.validateCommand = try container.decodeIfPresent(String.self, forKey: .validateCommand)
        self.validateTimeoutMs = try container.decodeIfPresent(Int.self, forKey: .validateTimeoutMs)
    }
}

/// Description of a single file operation executed as part of a patch.
public struct AppliedPatchOperation: Codable, Sendable, Equatable {
    /// Operation kind ("add", "update", "delete").
    public var type: String
    /// Type of resource affected.
    public var resource: String
    /// Target file path modified by the operation.
    public var target: String

    public init(type: String, resource: String, target: String) {
        self.type = type
        self.resource = resource
        self.target = target
    }
}

/// Output result summarizing applied patch modifications.
public struct ApplyPatchOutput: Codable, Sendable, Equatable {
    /// Individual patch operations applied.
    public var applied: [AppliedPatchOperation]
    /// Diff statistics per modified file.
    public var files: [GitDiffSummary]
    /// Human-readable summary of applied edits.
    public var summary: String

    public init(
        applied: [AppliedPatchOperation] = [],
        files: [GitDiffSummary] = [],
        summary: String = ""
    ) {
        self.applied = applied
        self.files = files
        self.summary = summary
    }
}

// MARK: - OpenAI Tool Definition for ApplyPatch

/// OpenAI tool definition schema for workspace patching.
public enum ApplyPatchToolDefinitions {
    public static let applyPatch = OpenAITool.function(
        name: "apply_patch",
        description: "Apply a patch containing unified diff, add, update, or delete file operations sequentially across the workspace.",
        parameters: .object(
            properties: [
                "patch_text": .string(
                    description: "The full patch or unified diff text describing file additions, modifications, and deletions."
                ),
                "validate_command": .string(
                    description: "Optional terminal command to run after all patch targets pass fingerprint verification."
                ),
                "validate_timeout_ms": .integer(
                    description: "Optional validation timeout in milliseconds."
                )
            ],
            required: ["patch_text"]
        )
    )

    public static let all: [OpenAITool] = [applyPatch]
}
