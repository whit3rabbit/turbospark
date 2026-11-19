import Foundation

// The Cmd+K command center's engine: prefix parsing, the three providers
// (chats, commands, files), the history store and the action routing. Pure
// and free of SwiftUI so each rule is testable without a view (the
// `ChatSearch` and `ComposerAutocompleteEngine` precedent). Providers are
// SYNCHRONOUS reads of in-memory or already-cached state: nothing here
// touches the disk or the network on a keystroke.

/// The palette's scope tabs. `all` blends the three provider scopes.
enum PaletteScope: String, CaseIterable, Identifiable, Sendable {
    case all
    case commands
    case chats
    case files

    var id: String { rawValue }

    /// Catalog key for the tab title and the section header.
    var titleKey: String {
        switch self {
        case .all: return "All"
        case .commands: return "Commands"
        case .chats: return "Chats"
        case .files: return "Files"
        }
    }

    /// The character that selects this scope when it is the FIRST character
    /// of the query.
    var prefix: Character? {
        switch self {
        case .all: return nil
        case .commands: return ">"
        case .chats: return "#"
        case .files: return "@"
        }
    }

    /// The next tab, wrapping at both ends.
    func advanced(by delta: Int) -> PaletteScope {
        let all = PaletteScope.allCases
        let index = all.firstIndex(of: self) ?? 0
        return all[((index + delta) % all.count + all.count) % all.count]
    }
}

/// A raw query split into the effective scope and the search text.
///
/// **Prefix rule.** Only the FIRST character of the raw input is a prefix
/// (`>`, `#`, `@`); it overrides the selected tab for as long as it stays
/// there, so deleting it returns to the tab's scope. A `#` or `@` anywhere
/// else is literal text. The cost: a query that must START with one of the
/// three characters cannot be typed in the All tab (type it after a space,
/// or search from the Chats or Files tab, where the same first character
/// only re-selects that scope).
struct PaletteQuery: Equatable, Sendable {
    let scope: PaletteScope
    /// The search text with the prefix and surrounding whitespace removed.
    let text: String
    let isPrefixed: Bool

    static func parse(_ raw: String, tab: PaletteScope = .all) -> PaletteQuery {
        if let first = raw.first,
            let scope = PaletteScope.allCases.first(where: { $0.prefix == first })
        {
            return PaletteQuery(
                scope: scope,
                text: String(raw.dropFirst()).trimmingCharacters(in: .whitespacesAndNewlines),
                isPrefixed: true)
        }
        return PaletteQuery(
            scope: tab,
            text: raw.trimmingCharacters(in: .whitespacesAndNewlines),
            isPrefixed: false)
    }
}

// MARK: - Commands

/// One entry of the Commands scope.
struct PaletteCommand: Identifiable, Equatable, Sendable {
    enum Action: Equatable, Sendable {
        /// A built-in or plugin slash command, by name (no leading `/`).
        case slash(String)
        case settings(AppSettingsView.SettingsTab)
        case navigate(AppModel.AppNavigationSection)
        case checkForUpdates
    }

    let id: String
    /// A catalog key, except for slash commands whose title is the literal
    /// `/name` (author text for plugins, never translated).
    let title: String
    let titleIsCatalogKey: Bool
    let subtitle: String?
    let subtitleIsCatalogKey: Bool
    let iconName: String
    /// The chord that does the same thing, shown at the row's trailing edge.
    let binding: String?
    let action: Action
}

/// Builds the Commands scope from the tables the app already owns, so the
/// palette cannot drift from the menus: `BuiltInSlashCommand.all`, the
/// plugin-contributed skills, `SettingsTab` and `AppNavigationSection`.
enum PaletteCommandCatalog {
    /// Everything the Commands scope can list right now. `canCheckForUpdates`
    /// is the menu item's own enabled predicate: a command the menu would
    /// show disabled is left out instead of listed and ignored.
    /// `navigationBindings` carries the chord each section's View-menu item
    /// holds RIGHT NOW, resolved by the caller from `KeybindOverrideStore`;
    /// a section missing from the map lists with no chord rather than a
    /// stale default.
    static func commands(
        skills: [AppSkill],
        canCheckForUpdates: Bool,
        builtIns: [BuiltInSlashCommand] = BuiltInSlashCommand.all,
        navigationBindings: [AppModel.AppNavigationSection: String] = [:]
    ) -> [PaletteCommand] {
        var rows: [PaletteCommand] = []

        for command in builtIns {
            rows.append(PaletteCommand(
                id: "slash:\(command.name)", title: "/\(command.name)", titleIsCatalogKey: false,
                subtitle: command.summary, subtitleIsCatalogKey: true,
                iconName: command.iconName, binding: nil, action: .slash(command.name)))
        }

        // Plugin commands only: the plugin namespace is what makes a name
        // safe to list beside the built-ins. User and project skills stay
        // on the `/` popup. A skill the slash handler would refuse (disabled,
        // model-only, shadowed by a built-in) is dropped by the same filter
        // the popup uses.
        for skill in ComposerAutocompleteEngine.invocableSkills(skills, builtIns: builtIns) {
            guard case .plugin = skill.scope else { continue }
            rows.append(PaletteCommand(
                id: "slash:\(skill.name)", title: "/\(skill.name)", titleIsCatalogKey: false,
                subtitle: skill.skillDescription.isEmpty ? nil : skill.skillDescription,
                subtitleIsCatalogKey: false,
                iconName: "wand.and.stars", binding: nil, action: .slash(skill.name)))
        }

        for section in AppModel.AppNavigationSection.allCases {
            rows.append(PaletteCommand(
                id: "nav:\(section.rawValue)", title: section.title, titleIsCatalogKey: true,
                subtitle: "Navigate", subtitleIsCatalogKey: true,
                iconName: section.systemImage, binding: navigationBindings[section],
                action: .navigate(section)))
        }

        for tab in AppSettingsView.SettingsTab.allCases {
            rows.append(PaletteCommand(
                id: "settings:\(tab.rawValue)", title: tab.title, titleIsCatalogKey: true,
                subtitle: "Settings", subtitleIsCatalogKey: true,
                iconName: tab.systemImage, binding: nil, action: .settings(tab)))
        }

        if canCheckForUpdates {
            rows.append(PaletteCommand(
                id: "updates", title: "Check for Updates\u{2026}", titleIsCatalogKey: true,
                subtitle: nil, subtitleIsCatalogKey: false,
                iconName: "arrow.triangle.2.circlepath", binding: nil, action: .checkForUpdates))
        }
        return rows
    }
}

// MARK: - Rows and sections

/// One result row, whichever provider produced it.
struct PaletteRow: Identifiable, Equatable {
    enum Payload: Equatable {
        case chat(ChatSearch.Hit)
        case command(PaletteCommand)
        case file(ProjectFileEntry)
        /// A past query; choosing it refills the field.
        case history(String)
    }

    let payload: Payload

    var id: String {
        switch payload {
        case .chat(let hit): return "chat:\(hit.id.uuidString)"
        case .command(let command): return "cmd:\(command.id)"
        case .file(let entry): return "file:\(entry.relativePath)"
        case .history(let query): return "history:\(query)"
        }
    }
}

struct PaletteSection: Identifiable, Equatable {
    enum Kind: String, Equatable {
        case chats, commands, files, history
    }

    let kind: Kind
    let rows: [PaletteRow]
    var id: String { kind.rawValue }

    /// Catalog key of the header.
    var titleKey: String {
        switch kind {
        case .chats: return "Chats"
        case .commands: return "Commands"
        case .files: return "Files"
        case .history: return "Recent searches"
        }
    }
}

/// Everything the providers read, gathered by the view so the engine stays
/// pure.
struct PaletteContext {
    var documents: [ChatSearch.Document] = []
    var commands: [PaletteCommand] = []
    /// nil: no project is open, or its listing has not been read yet.
    var files: [ProjectFileEntry]?
    var hasProject = false
    var history: [String] = []
}

/// Ranking and merging. No scoring: each provider keeps its own native
/// order (chats by recency, commands by name bucket then table order, files
/// by name bucket then path), and All stacks capped sections.
enum PaletteEngine {
    /// Rows per section in All, so one provider cannot push the others off
    /// the screen.
    static let allScopeSectionCap = 6
    /// Rows in the Files scope; the index holds at most 2000 entries.
    static let filesScopeCap = 50
    /// Recent searches offered on the home view (the store keeps 20).
    static let homeHistoryCap = 5

    static func sections(query: PaletteQuery, context: PaletteContext) -> [PaletteSection] {
        switch query.scope {
        case .chats:
            return nonEmpty([chatSection(query, context, cap: nil)])
        case .commands:
            return nonEmpty([commandSection(query, context, cap: nil)])
        case .files:
            return nonEmpty([fileSection(query, context, cap: filesScopeCap)])
        case .all:
            // Chats FIRST: the first row stays the most recent (or best)
            // chat, so the old Cmd+K then Enter muscle memory still opens
            // that chat. With an empty query this is also the home view:
            // recent chats, then commands, then recent searches.
            var sections = [
                chatSection(query, context, cap: allScopeSectionCap),
                commandSection(query, context, cap: allScopeSectionCap),
                fileSection(query, context, cap: allScopeSectionCap),
            ]
            if query.text.isEmpty {
                // The home view lists no files: an alphabetical slice of the
                // tree is noise until the user types.
                sections.removeAll { $0.kind == .files }
                let history = context.history.prefix(homeHistoryCap).map {
                    PaletteRow(payload: .history($0))
                }
                if !history.isEmpty {
                    sections.append(PaletteSection(kind: .history, rows: Array(history)))
                }
            }
            return nonEmpty(sections)
        }
    }

    private static func nonEmpty(_ sections: [PaletteSection]) -> [PaletteSection] {
        sections.filter { !$0.rows.isEmpty }
    }

    // MARK: Chats

    private static func chatSection(
        _ query: PaletteQuery, _ context: PaletteContext, cap: Int?
    ) -> PaletteSection {
        // `ChatSearch` is the engine, untouched: AND-of-substring tokens,
        // recency order, snippets. The palette only wraps its hits.
        var hits = ChatSearch.hits(query: .parse(query.text), documents: context.documents)
        if let cap { hits = Array(hits.prefix(cap)) }
        return PaletteSection(kind: .chats, rows: hits.map { PaletteRow(payload: .chat($0)) })
    }

    // MARK: Commands

    private static func commandSection(
        _ query: PaletteQuery, _ context: PaletteContext, cap: Int?
    ) -> PaletteSection {
        let clean = query.text.lowercased()
        var ranked: [(bucket: Int, order: Int, command: PaletteCommand)] = []
        for (order, command) in context.commands.enumerated() {
            guard let bucket = commandBucket(clean, command) else { continue }
            ranked.append((bucket, order, command))
        }
        // The empty query keeps table order (bucket 0 for all).
        ranked.sort { $0.bucket != $1.bucket ? $0.bucket < $1.bucket : $0.order < $1.order }
        var rows = ranked.map { PaletteRow(payload: .command($0.command)) }
        if let cap { rows = Array(rows.prefix(cap)) }
        return PaletteSection(kind: .commands, rows: rows)
    }

    /// 0 = name prefix, 1 = name contains, 2 = summary contains (three or
    /// more characters), nil = no match. Same buckets as the `/` popup.
    /// Matching runs on the bare name: a typed `export` is a prefix of
    /// `/export`.
    static func commandBucket(_ clean: String, _ command: PaletteCommand) -> Int? {
        guard !clean.isEmpty else { return 0 }
        var name = command.title.lowercased()
        if !command.titleIsCatalogKey, name.hasPrefix("/") { name.removeFirst() }
        if name.hasPrefix(clean) { return 0 }
        if name.contains(clean) { return 1 }
        // The subtitle matches on its English source for built-ins and on
        // the author's own text for plugins; a translated summary is not
        // searched (the popup behaves the same way).
        if clean.count >= 3, let subtitle = command.subtitle?.lowercased(),
            subtitle.contains(clean)
        {
            return 2
        }
        return nil
    }

    // MARK: Files

    private static func fileSection(
        _ query: PaletteQuery, _ context: PaletteContext, cap: Int
    ) -> PaletteSection {
        guard let entries = context.files else {
            return PaletteSection(kind: .files, rows: [])
        }
        let clean = query.text.lowercased()
        var ranked: [(bucket: Int, entry: ProjectFileEntry)] = []
        for entry in entries {
            // The mention grammar has no escape for a quote inside a quoted
            // path, so such a path is never offered (the `@` popup skips it
            // for the same reason).
            guard !entry.relativePath.contains("\"") else { continue }
            let path = entry.relativePath.lowercased()
            let fileName = (path as NSString).lastPathComponent
            let bucket: Int
            if clean.isEmpty || fileName.hasPrefix(clean) {
                bucket = 0
            } else if fileName.contains(clean) {
                bucket = 1
            } else if path.contains(clean) {
                bucket = 2
            } else {
                continue
            }
            ranked.append((bucket, entry))
        }
        ranked.sort {
            if $0.bucket != $1.bucket { return $0.bucket < $1.bucket }
            return $0.entry.relativePath.lowercased() < $1.entry.relativePath.lowercased()
        }
        return PaletteSection(
            kind: .files,
            rows: ranked.prefix(cap).map { PaletteRow(payload: .file($0.entry)) })
    }
}

// MARK: - Actions

/// What choosing a row does, as a value so the routing is testable without
/// a model. `PaletteActionRouter.perform` (in the view file) executes it.
enum PaletteAction: Equatable {
    case openChat(UUID)
    /// A local command the model dispatches directly (`handleLocalCommand`).
    case runLocalCommand(String)
    /// `/stats`, `/help`, `/export`: `handleLocalMetaCommand`.
    case runLocalMetaCommand(String)
    /// Appends text to the composer draft and focuses it.
    case insertIntoComposer(String)
    case openSettings(AppSettingsView.SettingsTab)
    case navigate(AppModel.AppNavigationSection)
    case checkForUpdates
    /// Refills the field with a past query.
    case setQuery(String)
}

enum PaletteActionRouter {
    /// Slash commands that take no argument and cannot lose work, so Enter
    /// runs them. Everything else (commands that need an argument, and the
    /// destructive `/clear`, `/reset`, `/delete`, `/archive`) is INSERTED
    /// into the composer instead, where the user finishes or reviews it and
    /// the existing submit-time gates apply. A drift test pins every name
    /// here to the command table.
    static let runImmediately: Set<String> = [
        "context", "tasks", "tools", "status", "mcp", "agents", "skills", "settings",
        "schedule", "new", "rewind", "resume", "copy", "recap", "diff", "log", "prs",
    ]

    static func action(for row: PaletteRow) -> PaletteAction {
        switch row.payload {
        case .chat(let hit):
            return .openChat(hit.id)
        case .history(let query):
            return .setQuery(query)
        case .file(let entry):
            return .insertIntoComposer(
                ComposerSuggestion(entry: entry).replacementToken + " ")
        case .command(let command):
            switch command.action {
            case .settings(let tab): return .openSettings(tab)
            case .navigate(let section): return .navigate(section)
            case .checkForUpdates: return .checkForUpdates
            case .slash(let name): return slashAction(name)
            }
        }
    }

    static func slashAction(_ name: String) -> PaletteAction {
        if BuiltInSlashCommand.localMetaNames.contains(name) {
            // `/export` with no argument is the Markdown export, the same
            // as typing it bare.
            return .runLocalMetaCommand("/\(name)")
        }
        if runImmediately.contains(name) {
            return .runLocalCommand("/\(name)")
        }
        return .insertIntoComposer("/\(name) ")
    }
}

/// How a palette insertion joins the draft: the same rule as the composer's
/// Add menu (a separating space unless the draft is empty or already ends
/// in whitespace), kept pure so the handoff payload is testable.
enum PaletteDraft {
    static func appending(_ text: String, to draft: String) -> String {
        if draft.isEmpty || draft.hasSuffix("\n") || draft.hasSuffix(" ") {
            return draft + text
        }
        return draft + " " + text
    }
}

// MARK: - History

/// Per-workspace search history: most recent first, deduplicated without
/// regard to case, capped. The workspace key is the selected project's id,
/// or `global` with no project.
struct PaletteHistory: Codable, Equatable {
    static let cap = 20
    static let globalKey = "global"

    var byWorkspace: [String: [String]] = [:]

    func entries(workspace: String) -> [String] {
        byWorkspace[workspace] ?? []
    }

    mutating func record(_ query: String, workspace: String) {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        var list = byWorkspace[workspace] ?? []
        list.removeAll { $0.caseInsensitiveCompare(trimmed) == .orderedSame }
        list.insert(trimmed, at: 0)
        if list.count > Self.cap { list.removeLast(list.count - Self.cap) }
        byWorkspace[workspace] = list
    }

    mutating func clear(workspace: String) {
        byWorkspace[workspace] = nil
    }
}

/// The history's file, behind `AppJSONStore` like every other app store, so
/// a corrupt file is quarantined rather than overwritten. Read once on first
/// use and then served from memory: opening the palette costs no further
/// I/O. The history is local to this machine and profile and is never
/// uploaded; it holds only what the user typed into the palette.
@MainActor
final class PaletteHistoryStore {
    static let shared = PaletteHistoryStore()

    private let url: URL
    private var loaded: PaletteHistory?

    init(url: URL = AppStorageRoot.file("command_palette_history.json")) {
        self.url = url
    }

    private var history: PaletteHistory {
        if let loaded { return loaded }
        let value = AppJSONStore.load(
            PaletteHistory.self, from: url, label: "Command center history") ?? PaletteHistory()
        loaded = value
        return value
    }

    func entries(workspace: String) -> [String] {
        history.entries(workspace: workspace)
    }

    func record(_ query: String, workspace: String) {
        var updated = history
        updated.record(query, workspace: workspace)
        loaded = updated
        AppJSONStore.save(updated, to: url, label: "Command center history")
    }

    func clear(workspace: String) {
        var updated = history
        updated.clear(workspace: workspace)
        loaded = updated
        AppJSONStore.save(updated, to: url, label: "Command center history")
    }
}

extension PaletteQuery {
    /// The notification payload key that seeds the field (Cmd+Shift+P
    /// sends `>`).
    static let seedUserInfoKey = "TurboSpark.commandPalette.seed"
}
