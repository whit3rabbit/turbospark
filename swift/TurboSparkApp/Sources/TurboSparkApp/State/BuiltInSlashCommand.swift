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
        /// A local UI command (the qwen-code `localCommands.ts` parity:
        /// `/copy`, `/theme`, `/tasks`, ...). Never generates and never
        /// reaches a hook; dispatched in `run()`'s session-less band, where
        /// each handler enforces its own preconditions (a `/fork` with no
        /// model toasts, it does not queue).
        case localCommand
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
            name: "goal", aliases: [],
            summary: "Keep working toward a condition until it is met: /goal <condition>",
            iconName: "target",
            kind: .metaCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "stats", aliases: [],
            summary: "Show session stats for this chat",
            iconName: "chart.bar",
            kind: .metaCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "export", aliases: [],
            summary: "Export this conversation: /export [markdown|json|html]",
            iconName: "square.and.arrow.up",
            kind: .metaCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "help", aliases: [],
            summary: "Show commands and keyboard shortcuts",
            iconName: "questionmark.circle",
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
        BuiltInSlashCommand(
            name: "copy", aliases: [],
            summary: "Copy a code block or LaTeX from the last answer: /copy [code [lang] [index] | latex | inline-latex [index]]",
            iconName: "doc.on.doc",
            kind: .localCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "model", aliases: [],
            summary: "Switch the loaded model: /model [name]",
            iconName: "cpu",
            kind: .localCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "context", aliases: [],
            summary: "Show this chat's context usage breakdown",
            iconName: "gauge.with.needle",
            kind: .localCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "tasks", aliases: [],
            summary: "Show background agents, shells and tasks",
            iconName: "square.stack.3d.up",
            kind: .localCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "tools", aliases: [],
            summary: "List the tools the model can call",
            iconName: "wrench.and.screwdriver",
            kind: .localCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "status", aliases: [],
            summary: "Show app, engine and session status",
            iconName: "info.circle",
            kind: .localCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "mcp", aliases: [],
            summary: "Open MCP server settings",
            iconName: "server.rack",
            kind: .localCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "agents", aliases: [],
            summary: "Open agent settings",
            iconName: "person.crop.square.stack",
            kind: .localCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "skills", aliases: [],
            summary: "Open skill settings",
            iconName: "wand.and.stars",
            kind: .localCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "settings", aliases: [],
            summary: "Open settings",
            iconName: "gearshape",
            kind: .localCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "schedule", aliases: [],
            summary: "Open scheduled tasks settings",
            iconName: "clock.badge.checkmark",
            kind: .localCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "theme", aliases: [],
            summary: "Switch appearance: /theme light|dark|system",
            iconName: "circle.lefthalf.filled",
            kind: .localCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "language", aliases: [],
            summary: "Set the UI language: /language <code> (en, de, ja, ...)",
            iconName: "globe",
            kind: .localCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "approval-mode", aliases: [],
            summary: "Set the permission mode: /approval-mode ask|auto|agent",
            iconName: "checkmark.shield",
            kind: .localCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "new", aliases: [],
            summary: "Start a new chat",
            iconName: "plus.bubble",
            kind: .localCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "clear", aliases: [],
            summary: "Clear this conversation",
            iconName: "eraser",
            kind: .localCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "rename", aliases: [],
            summary: "Rename this chat: /rename <title>",
            iconName: "pencil",
            kind: .localCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "delete", aliases: [],
            summary: "Delete this chat (asks first)",
            iconName: "trash",
            kind: .localCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "archive", aliases: ["unarchive"],
            summary: "Archive this chat (or /unarchive to restore)",
            iconName: "archivebox",
            kind: .localCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "rewind", aliases: [],
            summary: "Rewind this chat to an earlier prompt",
            iconName: "arrow.counterclockwise",
            kind: .localCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "branch", aliases: [],
            summary: "Fork this conversation into a new chat: /branch [title]",
            iconName: "arrow.triangle.branch",
            kind: .localCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "btw", aliases: [],
            summary: "Ask a side question in the background: /btw <question>",
            iconName: "questionmark.bubble",
            kind: .localCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "fork", aliases: [],
            summary: "Continue this conversation in a background agent: /fork <directive>",
            iconName: "arrow.triangle.branch",
            kind: .localCommand, fixedAgentName: nil),
        BuiltInSlashCommand(
            name: "recap", aliases: [],
            summary: "Summarize this conversation so far",
            iconName: "doc.text.magnifyingglass",
            kind: .localCommand, fixedAgentName: nil),
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

    /// Whether the draft is one of the LOCAL meta commands (`stats`,
    /// `export`, `help`): like `/memory` they never generate and work
    /// without a model, and unlike `/compact` they are not about the
    /// session's context, so they get their own predicate and their own
    /// dispatcher arm in `run()`.
    static var localMetaNames: [String] { ["stats", "export", "help"] }

    static func isLocalMetaCommand(_ draft: String) -> Bool {
        localMetaNames.contains { name in
            draft == "/\(name)" || draft.hasPrefix("/\(name) ")
        }
    }

    /// Whether the draft is the `/goal` meta command specifically, keyed
    /// off the table like `isMemoryCommand`. It also lives ABOVE `canRun`:
    /// status and clear work with no model, and setting a goal while the
    /// chat is busy wants the QUEUE path (the directive turn parks behind
    /// the running work), which this placement gives it for free.
    static func isGoalCommand(_ draft: String) -> Bool {
        all.contains { command in
            guard command.kind == .metaCommand, command.name == "goal" else { return false }
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

    /// Whether the draft is a LOCAL command (`/copy`, `/theme`, `/tasks`,
    /// ...). Like `isMemoryCommand` these sit ABOVE `canRun`: they never
    /// generate, and several of them exist precisely to run WHILE a turn is
    /// generating (qwen-code's `/btw` semantics). Each handler enforces its
    /// own preconditions instead.
    static func isLocalCommand(_ draft: String) -> Bool {
        all.contains { command in
            guard command.kind == .localCommand else { return false }
            let spellings = [command.name] + command.aliases
            return spellings.contains { spelling in
                draft == "/\(spelling)" || draft.hasPrefix("/\(spelling) ")
            }
        }
    }

    /// Every spelling the local dispatcher (`handleLocalCommand`) must
    /// recognize, for the drift test.
    static var recognizedLocalNames: [String] {
        all.filter { $0.kind == .localCommand }.flatMap { [$0.name] + $0.aliases }
    }

    /// Every spelling the menu and the popup offer. Nothing forbids a name
    /// from appearing in both kinds' lists, but the drift test pins today's
    /// disjointness so a collision is a decision rather than an accident.
    static var recognizedNames: [String] {
        all.flatMap { [$0.name] + $0.aliases }
    }
}
