import Foundation

/// OC-10: one language server the USER chose. This is app-level configuration
/// kept in the user's own defaults. A repository can never add, edit or enable
/// a row: nothing here is read from a project folder.
struct LSPServerConfig: Codable, Equatable, Identifiable, Sendable {
    var id: String
    var name: String
    /// File extensions this server covers, lowercase, without the dot.
    var extensions: [String]
    /// The LSP `languageId` sent in didOpen.
    var languageID: String
    /// An absolute path or a bare command name looked up on the login-shell
    /// PATH. Empty on the discovered row, which resolves at launch.
    var command: String
    var args: [String]
    var enabled: Bool
    /// True for the discovered sourcekit-lsp row. Its command is never stored.
    var isDiscovered: Bool
    /// Variable NAMES the user typed for this row. At launch each one is
    /// copied from the app's own environment on top of the scrubbed allowlist
    /// (never a value typed here, never anything from a project). Names that
    /// fail `LSPEnvironmentPolicy` are dropped on load and again at launch.
    var envPassthrough: [String] = []

    static let sourcekitID = "sourcekit-lsp"

    /// The discovered default. Off until the user turns it on, even when the
    /// binary exists on this Mac.
    static var sourcekitDefault: LSPServerConfig {
        LSPServerConfig(
            id: sourcekitID, name: "sourcekit-lsp", extensions: ["swift"], languageID: "swift",
            command: "", args: [], enabled: false, isDiscovered: true)
    }

    func covers(extension ext: String) -> Bool {
        extensions.contains(ext.lowercased())
    }
}

extension LSPServerConfig {
    enum CodingKeys: String, CodingKey {
        case id, name, extensions, languageID, command, args, enabled, isDiscovered, envPassthrough
    }

    /// A row stored before `envPassthrough` existed has no such key and loads
    /// with none.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        name = try c.decode(String.self, forKey: .name)
        extensions = try c.decode([String].self, forKey: .extensions)
        languageID = try c.decode(String.self, forKey: .languageID)
        command = try c.decode(String.self, forKey: .command)
        args = try c.decode([String].self, forKey: .args)
        enabled = try c.decode(Bool.self, forKey: .enabled)
        isDiscovered = try c.decode(Bool.self, forKey: .isDiscovered)
        envPassthrough = (try? c.decodeIfPresent([String].self, forKey: .envPassthrough)) ?? []
    }
}

/// Which variable names a server row may pass through from the app's own
/// environment. The launch environment is an allowlist (PATH, HOME, LANG,
/// TMPDIR, see `McpClientEngine.childEnvironment`); this is the only way to
/// add to it and it is a DENY list on top of an explicit, user-typed name, so
/// a typo or a hand-edited defaults file cannot hand a language server a
/// credential or an injection variable.
enum LSPEnvironmentPolicy {
    static let maximumNames = 8
    static let maximumNameLength = 64

    /// Substrings (upper-cased name) that mark a secret.
    static let deniedSubstrings = [
        "TOKEN", "KEY", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH", "PRIVATE", "PROXY", "COOKIE",
    ]
    /// Prefixes: provider credentials, ssh and git (both can run commands),
    /// and the dynamic loader's injection variables.
    static let deniedPrefixes = [
        "AWS_", "ANTHROPIC_", "OPENAI_", "GITHUB_", "SSH_", "GIT_", "DYLD_", "LD_",
    ]
    /// Interpreter and shell variables that run code at startup.
    static let deniedNames: Set<String> = [
        "NODE_OPTIONS", "NODE_PATH", "PYTHONSTARTUP", "PYTHONPATH", "PYTHONHOME", "PERL5OPT", "PERL5LIB",
        "RUBYOPT", "RUBYLIB", "BASH_ENV", "ENV", "ZDOTDIR", "IFS", "SHELLOPTS", "PS4", "PROMPT_COMMAND",
        "JAVA_TOOL_OPTIONS", "_JAVA_OPTIONS", "JDK_JAVA_OPTIONS", "CLASSPATH",
    ]

    /// Why `name` is refused, or nil when it may be passed through.
    static func refusal(for name: String) -> String? {
        guard !name.isEmpty, name.count <= maximumNameLength,
              let first = name.unicodeScalars.first,
              first == "_" || (first.isASCII && CharacterSet.letters.contains(first)),
              name.unicodeScalars.allSatisfy({ $0.isASCII && (CharacterSet.alphanumerics.contains($0) || $0 == "_") })
        else { return "not a valid variable name" }
        let upper = name.uppercased()
        if McpClientEngine.baselineEnvironmentKeys.contains(upper) { return "already provided" }
        if deniedNames.contains(upper) { return "can inject code" }
        if let prefix = deniedPrefixes.first(where: { upper.hasPrefix($0) }) { return "\(prefix)* is never passed" }
        if let part = deniedSubstrings.first(where: { upper.contains($0) }) { return "looks like a secret (\(part))" }
        return nil
    }

    /// The names that pass, deduplicated, in order, bounded.
    static func accepted(_ names: [String]) -> [String] {
        var out: [String] = []
        for raw in names {
            let name = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if refusal(for: name) == nil, !out.contains(name) { out.append(name) }
            if out.count == maximumNames { break }
        }
        return out
    }

    /// Splits typed text (commas, spaces, newlines) into accepted and refused
    /// names, for the settings field.
    static func parse(_ text: String) -> (accepted: [String], refused: [String]) {
        let pieces = text.split(whereSeparator: { $0 == "," || $0 == " " || $0 == "\n" || $0 == "\t" }).map(String.init)
        var refused: [String] = []
        for piece in pieces where refusal(for: piece) != nil && !refused.contains(piece) { refused.append(piece) }
        return (accepted(pieces), refused)
    }
}

/// The app-level LSP settings. Two keys: the master switch (default off) and
/// the user's rows. Reading is a plain `UserDefaults` lookup so the catalog's
/// tool filter can consult it on every turn, the way `CodemodeSettings` does.
enum LSPSettings {
    static let enabledKey = "lsp.enabled"
    static let serversKey = "lsp.servers"
    /// Hard bound on rows, so a hand-edited defaults file cannot fan out.
    static let maximumServers = 8
    static let maximumArguments = 32
    static let maximumArgumentLength = 512

    static func isEnabled(defaults: UserDefaults = .standard) -> Bool {
        defaults.bool(forKey: enabledKey)
    }

    static func setEnabled(_ value: Bool, defaults: UserDefaults = .standard) {
        defaults.set(value, forKey: enabledKey)
    }

    /// The rows, with the discovered sourcekit-lsp row always present.
    static func servers(defaults: UserDefaults = .standard) -> [LSPServerConfig] {
        var rows: [LSPServerConfig] = []
        if let data = defaults.data(forKey: serversKey),
           let stored = try? JSONDecoder().decode([LSPServerConfig].self, from: data) {
            rows = Array(stored.prefix(maximumServers))
        }
        if !rows.contains(where: { $0.id == LSPServerConfig.sourcekitID }) {
            rows.insert(.sourcekitDefault, at: 0)
        }
        // The discovered row never carries a stored command or args: a
        // hand-edited defaults file must not turn it into an arbitrary launch.
        return rows.map { row in
            guard row.id == LSPServerConfig.sourcekitID else {
                var custom = row
                custom.isDiscovered = false
                custom.args = Self.boundedArguments(custom.args)
                custom.envPassthrough = LSPEnvironmentPolicy.accepted(custom.envPassthrough)
                return custom
            }
            var discovered = LSPServerConfig.sourcekitDefault
            discovered.enabled = row.enabled
            return discovered
        }
    }

    static func save(_ servers: [LSPServerConfig], defaults: UserDefaults = .standard) {
        if let data = try? JSONEncoder().encode(Array(servers.prefix(maximumServers))) {
            defaults.set(data, forKey: serversKey)
        }
    }

    /// One argument per line, as typed. Arguments are an array handed to the
    /// process as is: never a shell string, never expanded.
    static func parseArguments(_ text: String) -> [String] {
        boundedArguments(text.split(separator: "\n", omittingEmptySubsequences: true).map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines)
        })
    }

    static func boundedArguments(_ args: [String]) -> [String] {
        Array(args.filter { !$0.isEmpty && $0.count <= maximumArgumentLength && !$0.contains("\0") }
            .prefix(maximumArguments))
    }

    /// Normalizes an extension list typed by the user: "swift, .H ,h" becomes
    /// ["swift", "h"].
    static func parseExtensions(_ text: String) -> [String] {
        var seen: [String] = []
        for piece in text.split(whereSeparator: { $0 == "," || $0 == " " }) {
            let cleaned = piece.trimmingCharacters(in: CharacterSet(charactersIn: ". \t")).lowercased()
            if !cleaned.isEmpty, cleaned.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "_" || $0 == "+" || $0 == "-" }),
               !seen.contains(cleaned) { seen.append(cleaned) }
        }
        return seen
    }

    /// The enabled row covering `path`'s extension, in row order, or nil when
    /// no enabled server covers it (the CUJ 3 passthrough).
    static func server(forPath path: String, in servers: [LSPServerConfig]) -> LSPServerConfig? {
        let ext = (path as NSString).pathExtension.lowercased()
        guard !ext.isEmpty else { return nil }
        return servers.first { $0.enabled && $0.covers(extension: ext) }
    }
}

/// Finds the toolchain's sourcekit-lsp WITHOUT executing anything: a file
/// existence check in fixed system locations. No `xcrun`, no PATH search, no
/// workspace lookup, so discovery cannot run a repository's binary.
enum LSPDiscovery {
    static let sourcekitCandidates = [
        "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/sourcekit-lsp",
        "/Applications/Xcode-beta.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/sourcekit-lsp",
        "/Library/Developer/CommandLineTools/usr/bin/sourcekit-lsp",
    ]

    static func sourcekitLSPPath(
        candidates: [String] = sourcekitCandidates,
        isExecutable: (String) -> Bool = { FileManager.default.isExecutableFile(atPath: $0) }
    ) -> String? {
        candidates.first(where: isExecutable)
    }
}
