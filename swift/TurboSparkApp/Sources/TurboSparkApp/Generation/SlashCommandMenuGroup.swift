import SwiftUI

/// The task groups the composer's Add menu files built-in slash commands
/// under.
///
/// `BuiltInSlashCommand.Kind` says how a command is DISPATCHED (meta, agent,
/// local), which is not how anyone looks for one, and the flat submenu had
/// grown to 39 rows. This table names the task each command serves. It
/// lives beside the menu rather than in the shared command table because
/// only the menu reads it: the typing popup ranks by name and needs no
/// groups. A command no group names lands in `.other`, so a new table row
/// is never dropped from the menu, only filed generically until it is
/// placed here.
enum SlashCommandMenuGroup: String, CaseIterable, Identifiable {
    case chat
    case context
    case agents
    case project
    case status
    case settings
    case other

    var id: String { rawValue }

    /// The submenu title. A `Text` with the literal spelled here, so every
    /// title stays visible to the localization source scan.
    var title: Text {
        switch self {
        case .chat: Text("Chat", bundle: .module)
        case .context: Text("Context and Memory", bundle: .module)
        case .agents: Text("Agents and Tasks", bundle: .module)
        case .project: Text("Project and Git", bundle: .module)
        case .status: Text("Status and Help", bundle: .module)
        case .settings: Text("Settings", bundle: .module)
        case .other: Text("More", bundle: .module)
        }
    }

    var systemImage: String {
        switch self {
        case .chat: "bubble.left.and.bubble.right"
        case .context: "brain"
        case .agents: "person.2"
        case .project: "arrow.triangle.branch"
        case .status: "info.circle"
        case .settings: "gearshape"
        case .other: "ellipsis.circle"
        }
    }

    /// Command names per group, in the order the submenu lists them (most
    /// frequent task first, consequential ones such as `delete` after the
    /// harmless ones).
    private var commandNames: [String] {
        switch self {
        case .chat:
            ["new", "rename", "clear", "reset", "rewind", "branch", "resume", "export", "copy", "archive", "delete"]
        case .context:
            ["compact", "recap", "context", "goal", "memory"]
        case .agents:
            ["explore", "plan", "review", "agent", "fork", "btw", "tasks"]
        case .project:
            ["diff", "log", "prs"]
        case .status:
            ["stats", "status", "tools", "help"]
        case .settings:
            ["model", "approval-mode", "theme", "language", "mcp", "agents", "skills", "schedule", "settings"]
        case .other:
            []
        }
    }

    /// `commands` split into non-empty groups, in `allCases` order. Names a
    /// group lists that the table no longer has are skipped; table rows no
    /// group lists go to `.other` in table order.
    static func grouped(_ commands: [BuiltInSlashCommand]) -> [(group: SlashCommandMenuGroup, commands: [BuiltInSlashCommand])] {
        let byName = Dictionary(commands.map { ($0.name, $0) }, uniquingKeysWith: { first, _ in first })
        var placed = Set<String>()
        var result: [(group: SlashCommandMenuGroup, commands: [BuiltInSlashCommand])] = []
        for group in allCases where group != .other {
            let members = group.commandNames.compactMap { byName[$0] }
            placed.formUnion(members.map(\.name))
            if !members.isEmpty { result.append((group, members)) }
        }
        let leftovers = commands.filter { !placed.contains($0.name) }
        if !leftovers.isEmpty { result.append((.other, leftovers)) }
        return result
    }
}

extension BuiltInSlashCommand {
    /// The summary in the UI language. Every summary in `all` is a catalog
    /// key; the `/name` syntax inside one stays as written in translations.
    var localizedSummary: Text {
        Text(LocalizedStringKey(summary), bundle: .module)
    }
}
