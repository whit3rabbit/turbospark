import Foundation

// MARK: - ApplyPatch Tool (OpenCode Compatible)

public struct ApplyPatchInput: Codable, Sendable, Equatable {
    public var patchText: String

    enum CodingKeys: String, CodingKey {
        case patchText = "patch_text"
    }

    public init(patchText: String) {
        self.patchText = patchText
    }

    // Support both patchText and patch_text decodings
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
    }
}

public struct AppliedPatchOperation: Codable, Sendable, Equatable {
    public var type: String // "add", "update", "delete"
    public var resource: String
    public var target: String

    public init(type: String, resource: String, target: String) {
        self.type = type
        self.resource = resource
        self.target = target
    }
}

public struct ApplyPatchOutput: Codable, Sendable, Equatable {
    public var applied: [AppliedPatchOperation]
    public var files: [GitDiffSummary]
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

public enum ApplyPatchToolDefinitions {
    public static let applyPatch = OpenAITool.function(
        name: "apply_patch",
        description: "Apply a patch containing unified diff, add, update, or delete file operations sequentially across the workspace.",
        parameters: .object(
            properties: [
                "patch_text": .string(
                    description: "The full patch or unified diff text describing file additions, modifications, and deletions."
                )
            ],
            required: ["patch_text"]
        )
    )

    public static let all: [OpenAITool] = [applyPatch]
}
