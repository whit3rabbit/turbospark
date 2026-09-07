import Foundation

/// The built-in slash commands every composer surface offers.
///
/// ONE table for four readers that used to restate the names as string
/// literals: the submit-time meta-command gate (`AppModel+Submission.swift`),
/// the agent parser (`AppModel+Agents.swift`), the plus menu, and the typing
/// autocomplete. The menu and the parsers had already drifted (the agent
/// parser accepts `/review` and `/reviewer`; the menu listed neither), which
/// is the failure a shared table exists to stop.
struct BuiltInSlashCommand: Identifiable, Equatable, Hashable {
    /// Where the draft carrying this command is dispatched.
    enum Kind: Equatable {
        /// Maintenance performed on the session, never prompt content: the
        /// gate in `run()` handles it before hooks or agents see anything.
        case metaCommand
        /// A prompt routed to an agent (`AppModel+Agents.swift`).
        case agentCommand
    }

    /// Primary spelling; what accepting the suggestion inserts.
    let name: String
    /// Additional accepted spellings at submit time. Aliases are recognized
    /// everywhere but offered as no separate row.
    let aliases: [String]
    /// One-line summary shown beside the command.
    let summary: String
    /// SF Symbol shown in the suggestion row and menu item.
    let iconName: String
    let kind: Kind
    /// The agent this command dispatches to, when the target is fixed.
    /// nil for `agent`, whose target is the command's own first argument.
    let fixedAgentName: String?

    var id: String { name }

    static let all: [BuiltInSlashCommand] = [
        BuiltInSlashCommand(
            name: "compact", aliases: [],
            summary: "Summarize older turns to free context",
            iconName: "rectangle.compress.vertical",
            kind: .metaCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "memory", aliases: [],
            summary: "Open this project's memory folder",
            iconName: "brain",
            kind: .metaCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "explore", aliases: [],
            summary: "Search the codebase with a subagent",
            iconName: "magnifyingglass",
            kind: .agentCommand, fixedAgentName: "explore"),
        BuiltInSlashCommand(
            name: "plan", aliases: [],
            summary: "Draft a multi-step implementation plan",
            iconName: "list.bullet.clipboard",
            kind: .agentCommand, fixedAgentName: "plan"),
        BuiltInSlashCommand(
            name: "review", aliases: ["reviewer"],
            summary: "Review code with the reviewer agent",
            iconName: "eye",
            kind: .agentCommand, fixedAgentName: "reviewer"),
        BuiltInSlashCommand(
            name: "agent", aliases: [],
            summary: "Invoke an agent by name: /agent <name> <task>",
            iconName: "person.crop.square",
            kind: .agentCommand, fixedAgentName: nil),
    ]

    /// The agent command whose primary name or alias is `name`,
    /// case-insensitive. Meta commands are outside this lookup on purpose:
    /// they are dispatched by `run()`'s gate before an agent parser runs.
    static func matching(_ name: String) -> BuiltInSlashCommand? {
        let clean = name.lowercased()
        return all.first {
            $0.kind == .agentCommand && ($0.name == clean || $0.aliases.contains(clean))
        }
    }

    /// Whether the draft IS a meta command (`/name` exactly, or `/name` with
    /// a space and arguments). Two today (`compact`, `memory`); the drift
    /// test pins the COUNT and each dispatcher arm, so a third one is a
    /// conscious edit of `run()`.
    static func isMetaCommand(_ draft: String) -> Bool {
        all.contains { command in
            guard command.kind == .metaCommand else { return false }
            return draft == "/\(command.name)" || draft.hasPrefix("/\(command.name) ")
        }
    }

    /// Whether the draft is the `/memory` meta command specifically, keyed
    /// off the table so the row and `run()`'s dispatch cannot drift apart.
    /// It lives ABOVE `canRun`'s guard: memory commands never generate, so
    /// they work with no model loaded, unlike `/compact`.
    static func isMemoryCommand(_ draft: String) -> Bool {
        all.contains { command in
            guard command.kind == .metaCommand, command.name == "memory" else { return false }
            return draft == "/\(command.name)" || draft.hasPrefix("/\(command.name) ")
        }
    }

    /// Every spelling the submit-time agent parser must recognize. The drift
    /// test walks this against `handleAgentSlashCommand` so a table row the
    /// parser lost (or a parser case the table never recorded) reddens a
    /// test instead of offering a command that falls through as prose.
    static var recognizedAgentNames: [String] {
        all.filter { $0.kind == .agentCommand }.flatMap { [$0.name] + $0.aliases }
    }

    /// Every spelling the menu and the popup offer. Nothing forbids a name
    /// from appearing in both kinds' lists, but the drift test pins today's
    /// disjointness so a collision is a decision rather than an accident.
    static var recognizedNames: [String] {
        all.flatMap { [$0.name] + $0.aliases }
    }
}
