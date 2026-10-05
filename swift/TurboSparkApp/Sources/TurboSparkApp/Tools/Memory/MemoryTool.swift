import Foundation

/// The `memory` tool reads approved claims and stages model-issued changes.
/// Project names are slugs, never paths; the ledger writes their Markdown
/// projections inside the encrypted profile after review.
public enum MemoryToolExecutor {
    public static func searchClaims(arguments: [String: String], project: AppProject?) throws -> String {
        guard MemoryStore.shared.isModelEnabled else {
            throw NSError(domain: "TurboSparkMemory", code: 7, userInfo: [
                NSLocalizedDescriptionKey: "Memory is disabled in Settings."])
        }
        let query = arguments["query"] ?? ""
        guard !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return "Provide a search query."
        }
        let requestedScope = (arguments["scope"] ?? "profile").lowercased()
        guard requestedScope == "profile" || requestedScope == "project" else {
            return "Scope must be profile or project."
        }
        if requestedScope == "project" && project?.rootDirectoryURL == nil {
            return "No project is attached to this chat."
        }
        let scope = requestedScope == "project"
            ? project?.rootDirectoryURL.map { "project:\(MemoryStore.projectKey(forProjectRoot: $0))" }
            : nil
        let limit = min(max(Int(arguments["limit"] ?? "5") ?? 5, 1), 10)
        let claims = MemoryLedgerStore.shared.search(query, scope: scope, limit: limit)
        return claims.isEmpty ? "No approved memory matched." : claims.map {
            "[\($0.id.uuidString)] \($0.text)"
        }.joined(separator: "\n")
    }

    public static func explainClaim(arguments: [String: String], project: AppProject?) throws -> String {
        guard MemoryStore.shared.isModelEnabled else {
            throw NSError(domain: "TurboSparkMemory", code: 7, userInfo: [
                NSLocalizedDescriptionKey: "Memory is disabled in Settings."])
        }
        guard let id = UUID(uuidString: arguments["claim_id"] ?? ""),
              let claim = MemoryLedgerStore.shared.explain(id), claim.status == .active,
              claim.scope == "profile" || project?.rootDirectoryURL.map({
                  claim.scope == "project:\(MemoryStore.projectKey(forProjectRoot: $0))"
              }) == true
        else { return "No approved claim with that ID exists." }
        let evidence = claim.evidence.map {
            "chat \($0.chatID.uuidString), message \($0.messageID.uuidString): \($0.quote)"
        }.joined(separator: "\n")
        let claims = MemoryLedgerStore.shared.snapshot().claims
        var chain: [String] = []
        var next = claim.supersedesClaimID
        var seen: Set<UUID> = []
        while let id = next, seen.insert(id).inserted,
              let prior = claims.first(where: { $0.id == id && $0.scope == claim.scope }) {
            chain.append("\(prior.id.uuidString) (\(prior.status.rawValue))")
            next = prior.supersedesClaimID
        }
        return "[\(id.uuidString)] \(claim.text)\nStatus: \(claim.status.rawValue)\n"
            + "Supersession chain: \(chain.isEmpty ? "none" : chain.joined(separator: " <- "))\n\(evidence)"
    }

    /// Executes one `memory` call. Throws with a model-readable message for
    /// a bad name, an unknown memory, or a malformed action; `AppToolRegistry`
    /// wraps the throw into an error result. The store is a parameter with a
    /// `.shared` default so tests can point the executor at a scratch
    /// directory instead of the process-wide one.
    public static func execute(
        arguments: [String: String], project: AppProject?, store: MemoryStore = .shared,
        ledger: MemoryLedgerStore = .shared
    ) throws -> String {
        guard store.isModelEnabled else {
            throw NSError(domain: "TurboSparkMemory", code: 7, userInfo: [
                NSLocalizedDescriptionKey:
                    "The memory tool is disabled in settings."
            ])
        }
        let scope = (arguments["scope"] ?? "project").lowercased()
        if scope == "profile" {
            return try executeProfile(arguments: arguments, ledger: ledger)
        }
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
            let scope = "project:\(MemoryStore.projectKey(forProjectRoot: root))"
            var proposal = MemoryClaim(scope: scope, kind: type.rawValue,
                                       text: "\(description)\n\(body)", projectTopicName: name)
            proposal.confidence = "agent-proposed"
            try ledger.propose(proposal)
            return "Proposed memory '\(name)' for review in Settings. It is not active yet."

        case "read", "list", "index":
            if rawName.trimmingCharacters(in: .whitespaces).isEmpty {
                let index = ledger.snapshot().claims.filter {
                    $0.scope == "project:\(MemoryStore.projectKey(forProjectRoot: root))" && $0.status == .active
                }.map { "[\($0.id.uuidString)] \($0.text)" }.joined(separator: "\n")
                return index.isEmpty
                    ? "MEMORY.md is empty: nothing is remembered for this project yet."
                    : "MEMORY.md:\n\n\(index)"
            }
            let scope = "project:\(MemoryStore.projectKey(forProjectRoot: root))"
            guard let claim = ledger.snapshot().claims.first(where: {
                $0.scope == scope && $0.projectTopicName == name && $0.status == .active
            }) else {
                throw NSError(domain: "TurboSparkMemory", code: 2, userInfo: [
                    NSLocalizedDescriptionKey: "No approved memory named '\(name)' exists."])
            }
            return claim.text

        case "forget", "delete", "remove":
            guard !rawName.trimmingCharacters(in: .whitespaces).isEmpty else {
                throw NSError(domain: "TurboSparkMemory", code: 4, userInfo: [
                    NSLocalizedDescriptionKey: "Missing 'name': name the memory to forget."
                ])
            }
            let scope = "project:\(MemoryStore.projectKey(forProjectRoot: root))"
            guard let target = ledger.snapshot().claims.first(where: {
                $0.scope == scope && $0.projectTopicName == name && $0.status == .active
            }) else { return "No approved memory named '\(name)' exists." }
            var proposal = MemoryClaim(scope: scope, kind: "forget", text: "Forget \(name)")
            proposal.requestedForgetID = target.id
            try ledger.propose(proposal)
            return "Proposed forgetting '\(name)' for review in Settings."

        default:
            throw NSError(domain: "TurboSparkMemory", code: 6, userInfo: [
                NSLocalizedDescriptionKey:
                    "Unknown action '\(action)': use \"save\", \"read\", or \"forget\"."
            ])
        }
    }

    private static func executeProfile(
        arguments: [String: String], ledger: MemoryLedgerStore
    ) throws -> String {
        let action = (arguments["action"] ?? "read").lowercased()
        switch action {
        case "save", "write", "remember":
            let text = arguments["content"] ?? arguments["memory"] ?? arguments["text"] ?? ""
            guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw NSError(domain: "TurboSparkMemory", code: 5, userInfo: [NSLocalizedDescriptionKey: "Missing 'content': the profile memory is empty."])
            }
            var proposal = MemoryClaim(scope: "profile", kind: "preference", text: text)
            proposal.confidence = "agent-proposed"
            try ledger.propose(proposal)
            return "Proposed a profile memory for review in Settings. It is not active yet."
        case "read", "list", "index":
            let text = ledger.snapshot().claims.filter {
                $0.scope == "profile" && $0.status == .active
            }.map { "[\($0.id.uuidString)] \($0.text)" }.joined(separator: "\n")
            return text.isEmpty ? "The profile MEMORY.md is empty." : text
        case "search":
            let query = arguments["query"] ?? arguments["text"] ?? arguments["content"] ?? ""
            let results = ledger.search(query, scope: nil)
            return results.isEmpty ? "No matching approved memories." : results.map {
                "[\($0.id.uuidString)] \($0.text)"
            }.joined(separator: "\n\n")
        case "explain":
            guard let id = UUID(uuidString: arguments["claim_id"] ?? ""),
                  let claim = ledger.explain(id), claim.status == .active,
                  claim.scope == "profile"
            else { return "No approved claim with that ID exists." }
            let evidence = claim.evidence.map {
                "\($0.chatID.uuidString)/\($0.messageID.uuidString): \($0.quote)"
            }.joined(separator: "\n")
            return "[\(id.uuidString)] \(claim.text)\n\(evidence)"
        default:
            throw NSError(domain: "TurboSparkMemory", code: 6, userInfo: [NSLocalizedDescriptionKey: "Unknown profile memory action '\(action)'. Use save, read, or search."])
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
        return [definition, searchDefinition, explainDefinition]
    }

    public static let definition = OpenAITool.function(
        name: "memory",
        description: "Save, read, search, or remove persistent memory. Use scope=profile for durable user preferences shared across projects, or scope=project for project-specific facts.",
        parameters: .object(
            properties: [
                "action": .string(description: "What to do: \"save\" (write or update), \"read\" (show one memory, or the index when name is omitted), or \"forget\" (remove)."),
                "name": .string(description: "Short kebab-case name of the memory, e.g. `deploy-workflow`. Required for save and forget."),
                "description": .string(description: "One-line summary deciding relevance later. Used by save; falls back to the content's first line."),
                "type": .string(description: "One of user, feedback, project, reference. Used by save; defaults to project."),
                "scope": .string(description: "Memory scope: profile for this user across projects, or project for the current project."),
                "query": .string(description: "Search text when action is search and scope is profile."),
                "content": .string(description: "Markdown body of the memory. Used by save; for feedback memories, include why it matters and how to apply it.")
            ],
            required: ["action"]
        )
    )

    public static let searchDefinition = OpenAITool.function(
        name: "memory_search",
        description: "Search approved profile and current-project memory. Results include claim IDs.",
        parameters: .object(properties: [
            "query": .string(description: "Text to search."),
            "scope": .string(description: "profile or project."),
            "limit": .string(description: "Maximum result count, 1 to 10.")
        ], required: ["query"])
    )

    public static let explainDefinition = OpenAITool.function(
        name: "memory_explain",
        description: "Show the source evidence and status of one approved memory claim.",
        parameters: .object(properties: [
            "claim_id": .string(description: "Claim UUID from memory_search.")
        ], required: ["claim_id"])
    )
}
