import Foundation

/// Parses `.claude-plugin/plugin.json` and `.claude-plugin/marketplace.json`.
///
/// Lenient top-level, dropped-with-diagnostic field values, fatal only on
/// JSON that does not parse at all. Reserved names match Claude Code's:
/// `inline` is the synthetic marketplace for session-only plugins and
/// `builtin` for built-ins, so a real plugin or marketplace must not take
/// either.
public enum PluginManifestParser {
    public static let manifestRelativePath = ".claude-plugin/plugin.json"
    public static let marketplaceRelativePath = ".claude-plugin/marketplace.json"

    static let reservedPluginNames: Set<String> = ["inline", "builtin"]

    /// MARK: - Name validation

    /// Claude Code requires a non-empty name with no spaces. A name is the
    /// namespacing identifier (`plugin:skill`, `mcp:plugin:server`), so a
    /// space would make one plugin's contributions unaddressable.
    public static func validatePluginName(_ name: String) -> String? {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return "name is empty" }
        if trimmed.contains(where: { $0 == " " || $0 == "\t" }) {
            return "name '\(trimmed)' contains whitespace"
        }
        if reservedPluginNames.contains(trimmed.lowercased()) {
            return "'\(trimmed)' is a reserved name"
        }
        return nil
    }

    /// Marketplace names are directory names too, so path metacharacters and
    /// non-ASCII (homograph defense, as Claude Code does it) are refused on
    /// top of the plugin rules.
    public static func validateMarketplaceName(_ name: String) -> String? {
        if let pluginReason = validatePluginName(name) { return pluginReason }
        let forbidden = ["/", "\\", ".."]
        for token in forbidden where name.contains(token) {
            return "marketplace name '\(name)' must not contain '\(token)'"
        }
        if name == "." {
            return "marketplace name must not be '.'"
        }
        if !name.canBeConverted(to: .ascii) {
            return "marketplace name '\(name)' must be ASCII"
        }
        return nil
    }

    /// MARK: - plugin.json

    /// Parses manifest JSON. Throws `PluginLoadError` when the data is not
    /// JSON at all or the name is missing or unusable; everything softer
    /// becomes a diagnostic on the returned manifest.
    public static func parseManifest(
        data: Data,
        fallbackName: String,
        sourceDescription: String
    ) throws -> PluginManifest {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw PluginLoadError(
                pluginName: fallbackName,
                reason: "\(sourceDescription) is not a JSON object")
        }
        return try manifest(from: json, fallbackName: fallbackName, sourceDescription: sourceDescription)
    }

    /// What a plugin with NO manifest gets (Claude Code synthesizes the same
    /// shape): the directory name for the name and a description naming the
    /// source. Convention directories still make it a plugin.
    public static func synthesizedManifest(
        name: String,
        sourceDescription: String
    ) -> PluginManifest {
        PluginManifest(
            name: name,
            descriptionText: "Plugin from \(sourceDescription)")
    }

    public static func manifest(
        from json: [String: Any],
        fallbackName: String,
        sourceDescription: String
    ) throws -> PluginManifest {
        try manifest(
            from: json, fallbackName: fallbackName, sourceDescription: sourceDescription,
            diagnostics: [])
    }

    private static func manifest(
        from json: [String: Any],
        fallbackName: String,
        sourceDescription: String,
        diagnostics: [String]
    ) throws -> PluginManifest {
        var notes = diagnostics

        // `name` is required by Claude Code's schema, but a manifest without
        // one still describes a real plugin; the directory name is the
        // honest fallback and the diagnostic says so.
        let rawName = (json["name"] as? String) ?? fallbackName
        if let nameReason = validatePluginName(rawName) {
            throw PluginLoadError(
                pluginName: fallbackName,
                reason: "\(sourceDescription): \(nameReason)")
        }

        var manifest = PluginManifest(
            name: rawName,
            version: json["version"] as? String,
            descriptionText: json["description"] as? String)

        if let authorDict = json["author"] as? [String: Any] {
            if let authorName = authorDict["name"] as? String, !authorName.isEmpty {
                manifest.author = PluginAuthor(
                    name: authorName,
                    email: authorDict["email"] as? String,
                    url: authorDict["url"] as? String)
            }
        }
        manifest.homepage = json["homepage"] as? String
        manifest.repository = json["repository"] as? String
        manifest.license = json["license"] as? String
        manifest.keywords = stringArray(json["keywords"])
        manifest.dependencies = stringArray(json["dependencies"])

        manifest.commandSpecs = parseCommands(json["commands"], notes: &notes)
        manifest.agentPaths = relativePathArray(json["agents"], notes: &notes, field: "agents")
        manifest.skillPaths = relativePathArray(json["skills"], notes: &notes, field: "skills")

        parseHooks(json["hooks"], into: &manifest, notes: &notes)
        parseMcpServers(json["mcpServers"], into: &manifest, notes: &notes)
        manifest.userConfig = parseUserConfig(json["userConfig"], notes: &notes)
        manifest.uiSurfaces = parseUiSurfaces(json["ui"], notes: &notes)
        manifest.unsupportedNotes = collectUnsupportedNotes(json)

        // Record what was ignored so a plugin author can see why a surface
        // did not take effect, instead of debugging silence.
        if !notes.isEmpty {
            manifest.unsupportedNotes.append(contentsOf: notes)
        }
        return manifest
    }

    /// MARK: - Field shapes

    /// `commands` accepts a single path, an array of paths, or an object map
    /// whose key is the command name and whose value names `source` or
    /// `content` (exactly one of the two, as Claude Code refines it).
    private static func parseCommands(
        _ value: Any?, notes: inout [String]
    ) -> [PluginCommandSpec] {
        var specs: [PluginCommandSpec] = []
        if let single = value as? String {
            specs.append(PluginCommandSpec(sourcePath: single))
            return specs
        }
        if let array = value as? [Any] {
            for element in array {
                if let path = element as? String {
                    specs.append(PluginCommandSpec(sourcePath: path))
                } else if let dict = element as? [String: Any] {
                    specs.append(contentsOf: parseCommands(dict, notes: &notes))
                } else {
                    notes.append("commands: dropped a non-string, non-object entry")
                }
            }
            return specs
        }
        if let dict = value as? [String: Any] {
            for (commandName, entry) in dict {
                if let path = entry as? String {
                    specs.append(PluginCommandSpec(name: commandName, sourcePath: path))
                    continue
                }
                guard let entryDict = entry as? [String: Any] else {
                    notes.append("commands.\(commandName): dropped an entry with an unusable shape")
                    continue
                }
                let source = entryDict["source"] as? String
                let content = entryDict["content"] as? String
                if source != nil, content != nil {
                    notes.append(
                        "commands.\(commandName): has both source and content; kept source")
                    // Kept SOURCE means dropped content: setting both would
                    // make the inline text win over the file at load time.
                }
                if source == nil, content == nil {
                    notes.append(
                        "commands.\(commandName): needs one of source or content; entry dropped")
                    continue
                }
                specs.append(PluginCommandSpec(
                    name: commandName,
                    sourcePath: source,
                    inlineContent: (source == nil) ? content : nil,
                    commandDescription: entryDict["description"] as? String,
                    argumentHint: entryDict["argumentHint"] as? String))
            }
            return specs
        }
        if value != nil {
            notes.append("commands: unusable shape; field dropped")
        }
        return specs
    }

    /// `agents` and `skills` accept a single relative path or an array.
    private static func relativePathArray(
        _ value: Any?, notes: inout [String], field: String
    ) -> [String] {
        if let single = value as? String {
            if let usable = usableRelativePath(single, field: field, notes: &notes) {
                return [usable]
            }
            return []
        }
        if let array = value as? [Any] {
            return array.compactMap { element in
                guard let path = element as? String else {
                    notes.append("\(field): dropped a non-string entry")
                    return nil
                }
                return usableRelativePath(path, field: field, notes: &notes)
            }
        }
        if value != nil {
            notes.append("\(field): unusable shape; field dropped")
        }
        return []
    }

    /// `hooks` accepts a `./path.json`, an inline hooks schema, or an array
    /// of either. The inline form is kept as JSON data; `AppHookStore` parses
    /// it with the same code path as hooks.json.
    private static func parseHooks(_ value: Any?, into manifest: inout PluginManifest, notes: inout [String]) {
        var elements: [Any] = []
        if value is String || value is [String: Any] {
            elements = [value!]
        } else if let array = value as? [Any] {
            elements = array
        } else if value != nil {
            notes.append("hooks: unusable shape; field dropped")
            return
        } else {
            return
        }

        for element in elements {
            if let path = element as? String {
                guard path.hasPrefix("./"), path.hasSuffix(".json") else {
                    notes.append(
                        "hooks: '\(path)' is not a relative .json path; entry dropped")
                    continue
                }
                if let usable = usableRelativePath(path, field: "hooks", notes: &notes) {
                    manifest.hookFilePaths.append(usable)
                }
            } else if let dict = element as? [String: Any] {
                if let data = try? JSONSerialization.data(withJSONObject: dict) {
                    manifest.inlineHooksJSON = mergeJSON(manifest.inlineHooksJSON, data)
                } else {
                    notes.append("hooks: an inline entry is not serializable; dropped")
                }
            } else {
                notes.append("hooks: dropped a non-string, non-object entry")
            }
        }
    }

    /// `mcpServers` accepts a `./path.json`, an inline `Record<name, config>`,
    /// or an array of either.
    private static func parseMcpServers(_ value: Any?, into manifest: inout PluginManifest, notes: inout [String]) {
        var elements: [Any] = []
        if value is String || value is [String: Any] {
            elements = [value!]
        } else if let array = value as? [Any] {
            elements = array
        } else if value != nil {
            notes.append("mcpServers: unusable shape; field dropped")
            return
        } else {
            return
        }

        for element in elements {
            if let path = element as? String {
                guard path.hasPrefix("./"), path.hasSuffix(".json") else {
                    notes.append(
                        "mcpServers: '\(path)' is not a relative .json path; entry dropped")
                    continue
                }
                if let usable = usableRelativePath(path, field: "mcpServers", notes: &notes) {
                    manifest.mcpServerFilePaths.append(usable)
                }
            } else if let dict = element as? [String: Any] {
                if let data = try? JSONSerialization.data(withJSONObject: dict) {
                    manifest.inlineMcpServersJSON = mergeJSON(manifest.inlineMcpServersJSON, data)
                } else {
                    notes.append("mcpServers: an inline entry is not serializable; dropped")
                }
            } else {
                notes.append("mcpServers: dropped a non-string, non-object entry")
            }
        }
    }

    /// `userConfig`: keyed by option name, each with type/title/description
    /// and the optional flags. An entry that cannot be read is dropped with
    /// a diagnostic rather than failing the plugin.
    private static func parseUserConfig(
        _ value: Any?, notes: inout [String]
    ) -> [PluginUserConfigOption] {
        guard let dict = value as? [String: Any] else {
            if value != nil {
                notes.append("userConfig: unusable shape; field dropped")
            }
            return []
        }
        var options: [PluginUserConfigOption] = []
        for (key, entry) in dict {
            // Claude Code requires /^[A-Za-z_]\w*$/ -- the key becomes an env
            // var suffix.
            guard key.range(of: "^[A-Za-z_][A-Za-z0-9_]*$", options: .regularExpression) != nil else {
                notes.append("userConfig.\(key): key is not a valid identifier; dropped")
                continue
            }
            guard let entryDict = entry as? [String: Any] else {
                notes.append("userConfig.\(key): dropped an entry with an unusable shape")
                continue
            }
            let typeRaw = (entryDict["type"] as? String) ?? "string"
            guard let type = AppHookOptionType(rawValue: typeRaw) else {
                notes.append("userConfig.\(key): unknown type '\(typeRaw)'; dropped")
                continue
            }
            guard let title = entryDict["title"] as? String, !title.isEmpty else {
                notes.append("userConfig.\(key): requires a title; dropped")
                continue
            }
            options.append(PluginUserConfigOption(
                key: key,
                type: type,
                title: title,
                descriptionText: (entryDict["description"] as? String) ?? "",
                isRequired: (entryDict["required"] as? Bool) ?? false,
                defaultValue: stringify(entryDict["default"]),
                isSensitive: (entryDict["sensitive"] as? Bool) ?? false))
        }
        return options
    }

    /// `ui.surfaces` (ZP-06). A malformed surface is dropped with a named
    /// diagnostic and its siblings load; an unknown capability is dropped
    /// from its surface alone (fail closed: the page simply cannot do it).
    /// Every path goes through `PluginPageResolver.normalizeRelativePath`,
    /// the same function the scheme handler uses, so the parser and the
    /// server cannot disagree about what a safe path is.
    static func parseUiSurfaces(_ value: Any?, notes: inout [String]) -> [PluginUiSurface] {
        guard let value else { return [] }
        guard let ui = value as? [String: Any] else {
            notes.append("ui: unusable shape; field dropped")
            return []
        }
        guard let rawSurfaces = ui["surfaces"] else { return [] }
        guard let entries = rawSurfaces as? [Any] else {
            notes.append("ui.surfaces: not an array; field dropped")
            return []
        }
        var surfaces: [PluginUiSurface] = []
        var seen = Set<String>()
        for (index, entry) in entries.enumerated() {
            let label = "ui.surfaces[\(index)]"
            if surfaces.count >= PluginPageLimits.maxSurfacesPerPlugin {
                notes.append("\(label): more than \(PluginPageLimits.maxSurfacesPerPlugin) surfaces; dropped")
                continue
            }
            guard let dict = entry as? [String: Any] else {
                notes.append("\(label): not an object; dropped")
                continue
            }
            guard let id = dict["id"] as? String,
                id.range(of: "^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$", options: .regularExpression) != nil
            else {
                notes.append("\(label): id must match [A-Za-z0-9][A-Za-z0-9_-]{0,63}; dropped")
                continue
            }
            guard seen.insert(id).inserted else {
                notes.append("\(label): duplicate surface id '\(id)'; dropped")
                continue
            }
            guard let rawPage = dict["resourceUri"] as? String,
                let page = PluginPageResolver.normalizeRelativePath(rawPage)
            else {
                notes.append("\(label) '\(id)': resourceUri is missing or not a safe bundle-relative path; dropped")
                continue
            }
            guard page.lowercased().hasSuffix(".html") || page.lowercased().hasSuffix(".htm") else {
                notes.append("\(label) '\(id)': resourceUri must be an .html file; dropped")
                continue
            }

            var resources: [String] = []
            if let rawResources = dict["resources"] {
                if let array = rawResources as? [Any] {
                    for item in array {
                        guard resources.count < PluginPageLimits.maxResourcesPerSurface else {
                            notes.append("\(label) '\(id)': more than \(PluginPageLimits.maxResourcesPerSurface) resources; extras dropped")
                            break
                        }
                        if let raw = item as? String,
                            let normalized = PluginPageResolver.normalizeRelativePath(raw),
                            PluginPageResolver.mimeType(forPath: normalized) != nil
                        {
                            resources.append(normalized)
                        } else {
                            notes.append("\(label) '\(id)': a resource is not a safe path to a served file type; dropped")
                        }
                    }
                } else {
                    notes.append("\(label) '\(id)': resources is not an array; dropped")
                }
            }

            var capabilities: [PluginUiCapability] = []
            if let rawCaps = dict["capabilities"] {
                if let array = rawCaps as? [Any] {
                    for item in array {
                        if let name = item as? String, let cap = PluginUiCapability(rawValue: name) {
                            if !capabilities.contains(cap) { capabilities.append(cap) }
                        } else {
                            notes.append("\(label) '\(id)': unknown capability \(item); dropped")
                        }
                    }
                } else {
                    notes.append("\(label) '\(id)': capabilities is not an array; dropped")
                }
            }

            var placement = PluginUiPlacement.sidePane
            if let rawPlacement = dict["placement"] {
                if let name = rawPlacement as? String, let parsed = PluginUiPlacement(rawValue: name) {
                    placement = parsed
                } else {
                    notes.append("\(label) '\(id)': unknown placement \(rawPlacement); using sidePane")
                }
            }

            var tools: [String] = []
            if let rawTools = dict["tools"] {
                if let array = rawTools as? [Any] {
                    for item in array {
                        if let name = item as? String,
                            name.range(
                                of: "^[A-Za-z0-9_.-]{1,64}/[A-Za-z0-9_.-]{1,64}$",
                                options: .regularExpression) != nil
                        {
                            tools.append(name)
                        } else {
                            notes.append("\(label) '\(id)': tools entries must be \"server/tool\"; dropped")
                        }
                    }
                } else {
                    notes.append("\(label) '\(id)': tools is not an array; dropped")
                }
            }
            if !tools.isEmpty, !capabilities.contains(.invokeTool) {
                notes.append("\(label) '\(id)': tools declared without the invokeTool capability; the page cannot call them")
            }

            let title = (dict["title"] as? String).flatMap { $0.isEmpty ? nil : String($0.prefix(80)) }
            surfaces.append(PluginUiSurface(
                id: id, title: title, resourceUri: page, resources: resources,
                capabilities: capabilities, placement: placement, tools: tools))
        }
        return surfaces
    }

    private static func collectUnsupportedNotes(_ json: [String: Any]) -> [String] {
        var notes: [String] = []
        // Claude Code's LSP servers, output styles, channels, MCPB bundles
        // and the plugin-contributed `settings` layer have no client here.
        // Parse, report, continue.
        if json["lspServers"] != nil {
            notes.append("lspServers: this client has no LSP integration; field ignored")
        }
        if json["outputStyles"] != nil {
            notes.append("outputStyles: this client has no output styles; field ignored")
        }
        if json["channels"] != nil {
            notes.append("channels: this client has no MCP message channels; field ignored")
        }
        if json["settings"] != nil {
            notes.append("settings: plugin-contributed settings are not applied; field ignored")
        }
        if let mcpServers = json["mcpServers"] as? String, mcpServers.hasSuffix(".mcpb") {
            notes.append("mcpServers: .mcpb bundles are not supported; field ignored")
        } else if let array = json["mcpServers"] as? [Any],
            array.contains(where: { ($0 as? String)?.hasSuffix(".mcpb") == true }) {
            notes.append("mcpServers: .mcpb bundles are not supported; entries ignored")
        }
        return notes
    }

    // MARK: - Marketplace merge (strict mode)

    /// Applies a marketplace entry's fields onto a plugin, per the `strict`
    /// flag. Strict (the default) means the plugin's own manifest must exist
    /// and the entry only fills gaps; non-strict synthesizes from the entry;
    /// both defining the same components is a conflict, per Claude Code.
    public static func merging(
        pluginManifest: PluginManifest?,
        entryDict: [String: Any],
        pluginName: String,
        sourceDescription: String
    ) throws -> PluginManifest {
        let strict = (entryDict["strict"] as? Bool) ?? true
        let entryManifest = try manifest(
            from: entryDict, fallbackName: pluginName,
            sourceDescription: sourceDescription, diagnostics: [])

        guard let base = pluginManifest else {
            if strict {
                throw PluginLoadError(
                    pluginName: pluginName,
                    reason: "\(sourceDescription): strict marketplace entries require the plugin's own \(manifestRelativePath)")
            }
            return entryManifest
        }

        var merged = base
        if base.version == nil { merged.version = entryManifest.version }
        if base.descriptionText == nil { merged.descriptionText = entryManifest.descriptionText }
        if base.author == nil { merged.author = entryManifest.author }

        // A component defined on BOTH sides is ambiguous about intent, so it
        // is an error rather than a union.
        func conflict(_ field: String) throws {
            throw PluginLoadError(
                pluginName: pluginName,
                reason: "\(sourceDescription): both the plugin manifest and the marketplace entry define \(field)")
        }
        if !base.commandSpecs.isEmpty, !entryManifest.commandSpecs.isEmpty { try conflict("commands") }
        if !base.agentPaths.isEmpty, !entryManifest.agentPaths.isEmpty { try conflict("agents") }
        if !base.skillPaths.isEmpty, !entryManifest.skillPaths.isEmpty { try conflict("skills") }
        if base.hookFilePaths.isEmpty == false || base.inlineHooksJSON != nil,
            entryManifest.hookFilePaths.isEmpty == false || entryManifest.inlineHooksJSON != nil {
            try conflict("hooks")
        }
        if base.mcpServerFilePaths.isEmpty == false || base.inlineMcpServersJSON != nil,
            entryManifest.mcpServerFilePaths.isEmpty == false || entryManifest.inlineMcpServersJSON != nil {
            try conflict("mcpServers")
        }

        if merged.commandSpecs.isEmpty { merged.commandSpecs = entryManifest.commandSpecs }
        if merged.agentPaths.isEmpty { merged.agentPaths = entryManifest.agentPaths }
        if merged.skillPaths.isEmpty { merged.skillPaths = entryManifest.skillPaths }
        if merged.hookFilePaths.isEmpty { merged.hookFilePaths = entryManifest.hookFilePaths }
        if merged.inlineHooksJSON == nil { merged.inlineHooksJSON = entryManifest.inlineHooksJSON }
        if merged.mcpServerFilePaths.isEmpty { merged.mcpServerFilePaths = entryManifest.mcpServerFilePaths }
        if merged.inlineMcpServersJSON == nil { merged.inlineMcpServersJSON = entryManifest.inlineMcpServersJSON }
        if merged.userConfig.isEmpty { merged.userConfig = entryManifest.userConfig }
        return merged
    }

    // MARK: - Helpers

    private static func stringArray(_ value: Any?) -> [String] {
        (value as? [Any])?.compactMap { $0 as? String } ?? []
    }

    /// Paths must stay inside the plugin, so anything absolute or climbing
    /// out is refused here at the parse layer rather than at each consumer.
    static func usableRelativePath(
        _ path: String, field: String, notes: inout [String]
    ) -> String? {
        if path.hasPrefix("/") || path.contains("..") {
            notes.append("\(field): '\(path)' resolves outside the plugin; entry dropped")
            return nil
        }
        return path
    }

    private static func stringify(_ value: Any?) -> String? {
        switch value {
        case let string as String: return string
        // JSON booleans decode as NSNumber, so the BOOL check must come
        // first (via CFBoolean) or `true` reads as "1".
        case let number as NSNumber:
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                return number.boolValue ? "true" : "false"
            }
            return number.stringValue
        default: return nil
        }
    }

    /// Merges two inline JSON objects of the same shape (both are keyed
    /// records), with the second winning on key collisions.
    private static func mergeJSON(_ base: Data?, _ addition: Data) -> Data? {
        guard let base else { return addition }
        guard
            var baseDict = (try? JSONSerialization.jsonObject(with: base)) as? [String: Any],
            let addDict = (try? JSONSerialization.jsonObject(with: addition)) as? [String: Any]
        else { return base }
        for (key, value) in addDict { baseDict[key] = value }
        return try? JSONSerialization.data(withJSONObject: baseDict)
    }

    /// MARK: - Marketplace manifest

    /// Holds raw JSON (the entry's `source` is a union shape parsed at
    /// install time), so it carries neither `Equatable` nor `Sendable`; it
    /// lives on the main actor's sheet state and in async install calls
    /// where Swift 5 checking is fine with it.
    public struct MarketplaceEntry {
        public var name: String
        public var sourceValue: Any?
        public var strict: Bool
        public var raw: [String: Any]

        public init(name: String, sourceValue: Any?, strict: Bool, raw: [String: Any]) {
            self.name = name
            self.sourceValue = sourceValue
            self.strict = strict
            self.raw = raw
        }

        public var descriptionText: String? { raw["description"] as? String }
        public var version: String? { raw["version"] as? String }
        public var category: String? { raw["category"] as? String }
        public var tags: [String] { stringArray(raw["tags"]) }
    }

    /// Not `Sendable`/`Equatable`: it holds raw-JSON entries (same reason
    /// as `MarketplaceEntry`).
    public struct MarketplaceManifest {
        public var name: String
        public var descriptionText: String?
        public var ownerName: String?
        public var pluginRoot: String?
        public var entries: [MarketplaceEntry]
    }

    public static func parseMarketplace(data: Data, sourceDescription: String) throws -> MarketplaceManifest {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw PluginLoadError(
                pluginName: nil,
                reason: "\(sourceDescription) is not a JSON object")
        }
        return try marketplace(from: json, sourceDescription: sourceDescription)
    }

    public static func marketplace(
        from json: [String: Any],
        sourceDescription: String
    ) throws -> MarketplaceManifest {
        guard let name = json["name"] as? String else {
            throw PluginLoadError(
                pluginName: nil,
                reason: "\(sourceDescription) has no marketplace name")
        }
        if let nameReason = validateMarketplaceName(name) {
            throw PluginLoadError(
                pluginName: nil,
                reason: "\(sourceDescription): \(nameReason)")
        }
        var entries: [MarketplaceEntry] = []
        if let list = json["plugins"] as? [[String: Any]] {
            for entry in list {
                guard let entryName = entry["name"] as? String, !entryName.isEmpty else {
                    // One bad entry must not take down the marketplace's
                    // availability (Claude Code strips rather than rejects).
                    continue
                }
                entries.append(MarketplaceEntry(
                    name: entryName,
                    sourceValue: entry["source"],
                    strict: (entry["strict"] as? Bool) ?? true,
                    raw: entry))
            }
        }
        let owner = json["owner"] as? [String: Any]
        let metadata = json["metadata"] as? [String: Any]
        return MarketplaceManifest(
            name: name,
            descriptionText: json["description"] as? String,
            ownerName: owner?["name"] as? String,
            pluginRoot: metadata?["pluginRoot"] as? String,
            entries: entries)
    }
}
