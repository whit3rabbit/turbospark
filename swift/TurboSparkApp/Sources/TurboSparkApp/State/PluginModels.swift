import Foundation

// The plugin model, ported from Claude Code's plugin system
// (claude-code-build1/src/plugins). A plugin is a directory whose manifest
// lives at `.claude-plugin/plugin.json` -- or nowhere, in which case one is
// synthesized from the convention directories. It contributes skills,
// slash commands, agents, hooks and MCP servers; swift/docs/SWIFT_PLUGINS.md is
// the map of what this port supports versus parses-and-reports.

/// Where one installed plugin came from. The order these are listed in
/// `PluginManager` discovery is also the precedence order: first match on a
/// lowercased name wins, and a plugin id is `<name>@<originKey>` where
/// originKey is the marketplace name for `.marketplace` and the raw value
/// otherwise.
public enum PluginOriginKind: String, Codable, Sendable, CaseIterable {
    /// A flat directory under ~/.turbospark/plugins/<name>/ (hand-placed or
    /// written by an older discovery path).
    case turboSpark
    /// The versioned install cache: <root>/cache/<marketplace>/<plugin>/<version>/
    case marketplace
    /// A folder the user registered from Settings, the `--plugin-dir` analog.
    case local
    /// Claude Code's own install under ~/.claude/plugins/, read-only. This
    /// app never writes there, exactly as it never writes ~/.claude/skills.
    case claudeInterop
}

public struct PluginAuthor: Sendable, Equatable {
    public var name: String
    public var email: String?
    public var url: String?

    public init(name: String, email: String? = nil, url: String? = nil) {
        self.name = name
        self.email = email
        self.url = url
    }
}

/// One `userConfig` entry from the manifest. Values are stored per plugin by
/// the hooks options store (sourceID `plugin_<name>`) and reach hooks and MCP
/// servers as `CLAUDE_PLUGIN_OPTION_<KEY>` env vars plus `${user_config.KEY}`
/// substitution.
public struct PluginUserConfigOption: Sendable, Equatable {
    public var key: String
    public var type: AppHookOptionType
    public var title: String
    public var descriptionText: String
    public var isRequired: Bool
    public var defaultValue: String?
    public var isSensitive: Bool

    public init(
        key: String,
        type: AppHookOptionType,
        title: String,
        descriptionText: String,
        isRequired: Bool = false,
        defaultValue: String? = nil,
        isSensitive: Bool = false
    ) {
        self.key = key
        self.type = type
        self.title = title
        self.descriptionText = descriptionText
        self.isRequired = isRequired
        self.defaultValue = defaultValue
        self.isSensitive = isSensitive
    }

    /// The hooks-system option spec this maps onto, so the settings UI and
    /// the persistence store need no second option model.
    public var optionSpec: AppHookOptionSpec {
        AppHookOptionSpec(
            key: key,
            type: type,
            title: title,
            description: descriptionText,
            defaultValue: defaultValue,
            isRequired: isRequired,
            isSensitive: isSensitive)
    }
}

/// One entry of the manifest's `commands` field after its three accepted
/// shapes (single path, array of paths, object map) are normalized. The
/// object map's key becomes the command name; `inlineContent` carries the
/// `content:` form, which has no file behind it.
public struct PluginCommandSpec: Sendable, Equatable {
    public var name: String?
    public var sourcePath: String?
    public var inlineContent: String?
    public var commandDescription: String?
    public var argumentHint: String?

    public init(
        name: String? = nil,
        sourcePath: String? = nil,
        inlineContent: String? = nil,
        commandDescription: String? = nil,
        argumentHint: String? = nil
    ) {
        self.name = name
        self.sourcePath = sourcePath
        self.inlineContent = inlineContent
        self.commandDescription = commandDescription
        self.argumentHint = argumentHint
    }
}

/// The parsed `.claude-plugin/plugin.json`.
///
/// Unknown TOP-LEVEL keys are ignored (Claude Code's own rule: one unknown
/// field must not take down a whole plugin), and a field whose VALUE has the
/// wrong shape is dropped with a diagnostic rather than failing the plugin --
/// the same split `TolerantDecoding` draws between a file that will not
/// parse and a field that will not convert. Unparseable JSON is fatal for
/// that plugin alone.
public struct PluginManifest: Sendable, Equatable {
    public var name: String
    public var version: String?
    public var descriptionText: String?
    public var author: PluginAuthor?
    public var homepage: String?
    public var repository: String?
    public var license: String?
    public var keywords: [String]
    public var dependencies: [String]
    public var commandSpecs: [PluginCommandSpec]
    /// `agents` entries, each a path relative to the plugin root.
    public var agentPaths: [String]
    /// `skills` entries, each a directory path relative to the plugin root.
    public var skillPaths: [String]
    /// `hooks` entries that name a `./file.json` path.
    public var hookFilePaths: [String]
    /// The inline form of `hooks`, kept as JSON data because its shape is the
    /// hooks.json schema, which `AppHookStore` already knows how to parse.
    public var inlineHooksJSON: Data?
    /// `mcpServers` entries that name a `./file.json` path.
    public var mcpServerFilePaths: [String]
    /// The inline form of `mcpServers`, a `Record<name, server config>`.
    public var inlineMcpServersJSON: Data?
    public var userConfig: [PluginUserConfigOption]
    /// Surfaces this client does not implement (LSP servers, output styles,
    /// channels, .mcpb bundles, the `settings` contribution). One line each;
    /// loading continues.
    public var unsupportedNotes: [String]

    public init(
        name: String,
        version: String? = nil,
        descriptionText: String? = nil,
        author: PluginAuthor? = nil,
        homepage: String? = nil,
        repository: String? = nil,
        license: String? = nil,
        keywords: [String] = [],
        dependencies: [String] = [],
        commandSpecs: [PluginCommandSpec] = [],
        agentPaths: [String] = [],
        skillPaths: [String] = [],
        hookFilePaths: [String] = [],
        inlineHooksJSON: Data? = nil,
        mcpServerFilePaths: [String] = [],
        inlineMcpServersJSON: Data? = nil,
        userConfig: [PluginUserConfigOption] = [],
        unsupportedNotes: [String] = []
    ) {
        self.name = name
        self.version = version
        self.descriptionText = descriptionText
        self.author = author
        self.homepage = homepage
        self.repository = repository
        self.license = license
        self.keywords = keywords
        self.dependencies = dependencies
        self.commandSpecs = commandSpecs
        self.agentPaths = agentPaths
        self.skillPaths = skillPaths
        self.hookFilePaths = hookFilePaths
        self.inlineHooksJSON = inlineHooksJSON
        self.mcpServerFilePaths = mcpServerFilePaths
        self.inlineMcpServersJSON = inlineMcpServersJSON
        self.userConfig = userConfig
        self.unsupportedNotes = unsupportedNotes
    }
}

/// One plugin as resolved from disk, with its effective id, origin and
/// diagnostics. `isEnabled` is resolved by `PluginManager`'s cascade and is
/// a copy on this struct, so a stale one cannot leak: resolution is
/// memoized and invalidated on every toggle.
public struct LoadedPlugin: Identifiable, Sendable, Equatable {
    public var name: String
    public var manifest: PluginManifest
    public var directoryURL: URL
    public var origin: PluginOriginKind
    /// The marketplace a `.marketplace` plugin was installed from; nil
    /// otherwise.
    public var marketplaceName: String?
    /// Highest-priority version known: the manifest's, else the ledger's,
    /// else "unknown". The marketplace manager writes the manifest value
    /// into the ledger at install time, so both agreeing is normal.
    public var version: String
    public var diagnostics: [String]

    public init(
        name: String,
        manifest: PluginManifest,
        directoryURL: URL,
        origin: PluginOriginKind,
        marketplaceName: String? = nil,
        version: String = "unknown",
        diagnostics: [String] = []
    ) {
        self.name = name
        self.manifest = manifest
        self.directoryURL = directoryURL
        self.origin = origin
        self.marketplaceName = marketplaceName
        self.version = version
        self.diagnostics = diagnostics
    }

    /// The origin segment of the id. Claude Code's synthetic marketplace
    /// names: `inline` is reserved for session-only plugins and `builtin`
    /// for built-ins; this port's equivalents are `local` and (not yet
    /// existing) bundled content.
    public var originKey: String {
        switch origin {
        case .marketplace:
            return marketplaceName ?? "marketplace"
        case .local:
            return "local"
        case .turboSpark:
            return "turbospark"
        case .claudeInterop:
            return "claude"
        }
    }

    /// `<name>@<origin>` -- the `enabledPlugins` key and the ledger key.
    public var id: String {
        "\(name)@\(originKey)"
    }

    public var displayName: String {
        name
    }

    public var pluginDescription: String {
        manifest.descriptionText ?? ""
    }

    /// Namespaced identifier prefix used for skills, commands and agents
    /// contributed by this plugin: `pluginName:` first segment, as Claude
    /// Code does it.
    public var namespace: String {
        name.lowercased()
    }
}

/// An error that fails ONE plugin's load while others continue. Claude
/// Code's rule: a missing manifest is fine, a corrupt one is fatal for that
/// plugin, and neither touches its siblings.
public struct PluginLoadError: Error, LocalizedError, Sendable {
    public var pluginName: String?
    public var reason: String

    public init(pluginName: String?, reason: String) {
        self.pluginName = pluginName
        self.reason = reason
    }

    public var errorDescription: String? {
        if let pluginName {
            return "Plugin '\(pluginName)' failed to load: \(reason)"
        }
        return "A plugin failed to load: \(reason)"
    }
}
