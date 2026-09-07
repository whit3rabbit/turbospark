import Foundation

/// The `memory` tool: save, read, or remove a persistent project memory.
///
/// The tool takes a NAME, never a path. Every filesystem decision happens
/// inside `MemoryStore` from that name (kebab-case stems only, no
/// separators), so the write surface is contained by construction and
/// neither `PathContainment` nor the permission engine needed a new carve
/// out -- the alternative shape, letting the general `write_file` tool reach
/// outside the workspace root into the memory directory, is exactly the
/// hole Claude Code plugs with its `isAutoMemPath` permission exception
/// instead.
public enum MemoryToolExecutor {
    /// Executes one `memory` call. Throws with a model-readable message for
    /// a bad name, an unknown memory, or a malformed action; `AppToolRegistry`
    /// wraps the throw into an error result. The store is a parameter with a
    /// `.shared` default so tests can point the executor at a scratch
    /// directory instead of the process-wide one.
    public static func execute(
        arguments: [String: String], project: AppProject?, store: MemoryStore = .shared
    ) throws -> String {
        guard let root = project?.rootDirectoryURL else {
            throw NSError(domain: "TurboSparkMemory", code: 3, userInfo: [
                NSLocalizedDescriptionKey:
                    "The memory tool needs a project: memories are stored per project directory, "
                    + "and this chat has none to key them on."
            ])
        }
        let action = (arguments["action"] ?? "read").lowercased()
        let rawName = arguments["name"] ?? arguments["memory_name"] ?? ""
        // The slug is the containment: whatever the model sent, the stem the
        // store ever sees is kebab-case with no separators.
        let name = MemoryStore.slug(from: rawName, dated: false)

        switch action {
        case "save", "write", "remember":
            guard !rawName.trimmingCharacters(in: .whitespaces).isEmpty else {
                throw NSError(domain: "TurboSparkMemory", code: 4, userInfo: [
                    NSLocalizedDescriptionKey: "Missing 'name': give the memory a short kebab-case name."
                ])
            }
            let body = arguments["content"] ?? arguments["memory"] ?? arguments["text"] ?? ""
            guard !body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw NSError(domain: "TurboSparkMemory", code: 5, userInfo: [
                    NSLocalizedDescriptionKey: "Missing 'content': the memory body is empty."
                ])
            }
            let type = MemoryEntryType(rawValue: (arguments["type"] ?? "project").lowercased())
                ?? .project
            // A missing description falls back to the body's first line, cut
            // to a hook length: the index stays one informative line even
            // when the model skips the field.
            let description = arguments["description"] ?? {
                let firstLine = SkillParser.normalizedLines(body).first { !$0.trimmingCharacters(in: .whitespaces).isEmpty } ?? ""
                let trimmed = firstLine.trimmingCharacters(in: .whitespaces)
                return trimmed.count > 120 ? String(trimmed.prefix(120)) + "..." : trimmed
            }()
            let saved = try store.saveTopic(
                projectRoot: root, name: name, type: type, description: description, body: body)
            return (saved.created ? "Saved a new memory" : "Updated the existing memory")
                + " '\(name)' in this project's memory. The MEMORY.md index now lists it."

        case "read", "list", "index":
            if rawName.trimmingCharacters(in: .whitespaces).isEmpty {
                let index = store.loadIndex(forProjectRoot: root)
                return index.isEmpty
                    ? "MEMORY.md is empty: nothing is remembered for this project yet."
                    : "MEMORY.md:\n\n\(index)"
            }
            return try store.readTopic(projectRoot: root, name: name)

        case "forget", "delete", "remove":
            guard !rawName.trimmingCharacters(in: .whitespaces).isEmpty else {
                throw NSError(domain: "TurboSparkMemory", code: 4, userInfo: [
                    NSLocalizedDescriptionKey: "Missing 'name': name the memory to forget."
                ])
            }
            let removed = try store.forgetTopic(projectRoot: root, name: name)
            return removed
                ? "Forgot the memory '\(name)' and removed its index row."
                : "No memory named '\(name)' exists; nothing was removed."

        default:
            throw NSError(domain: "TurboSparkMemory", code: 6, userInfo: [
                NSLocalizedDescriptionKey:
                    "Unknown action '\(action)': use \"save\", \"read\", or \"forget\"."
            ])
        }
    }
}

/// OpenAI tool definitions for the memory subsystem. `all` is computed so
/// the `isModelEnabled` gate is read at ADVERTISING time: a disabled memory
/// feature never reaches the model's tool list, rather than being offered
/// and refused.
public enum MemoryToolDefinitions {
    public static let toolName = "memory"

    public static var all: [OpenAITool] {
        guard MemoryStore.shared.isModelEnabled else { return [] }
        return [definition]
    }

    public static let definition = OpenAITool.function(
        name: "memory",
        description: "Save, read, or remove a persistent memory for this project. Memories survive across conversations; the MEMORY.md index of what exists is already in your context. Use it when you learn something durable about the user or the project that the repository itself does not record.",
        parameters: .object(
            properties: [
                "action": .string(description: "What to do: \"save\" (write or update), \"read\" (show one memory, or the index when name is omitted), or \"forget\" (remove)."),
                "name": .string(description: "Short kebab-case name of the memory, e.g. `deploy-workflow`. Required for save and forget."),
                "description": .string(description: "One-line summary deciding relevance later. Used by save; falls back to the content's first line."),
                "type": .string(description: "One of user, feedback, project, reference. Used by save; defaults to project."),
                "content": .string(description: "Markdown body of the memory. Used by save; for feedback memories, include why it matters and how to apply it.")
            ],
            required: ["action"]
        )
    )
}
