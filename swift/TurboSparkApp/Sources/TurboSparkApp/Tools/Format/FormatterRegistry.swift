import Foundation

/// OC-11: one formatter the USER chose. App-level configuration kept in the
/// user's own defaults. A repository can never add, edit or enable a row:
/// nothing here is read from a project folder.
struct FormatterRow: Codable, Equatable, Identifiable, Sendable {
    var id: String
    var name: String
    /// File extensions this formatter covers, lowercase, without the dot.
    var extensions: [String]
    /// An absolute path or a bare command name looked up on the login-shell
    /// PATH. Empty on the built-in row, which resolves from fixed locations.
    var command: String
    /// The argument vector, no shell. `{file}` is replaced by the path of the
    /// temporary copy; without a `{file}` the path is appended.
    var args: [String]
    var enabled: Bool
    var isBuiltin: Bool

    static let swiftFormatID = "swift-format"
    static let filePlaceholder = "{file}"

    /// The one shipped row. The command line is `swift-format format
    /// --in-place <temp copy>`.
    static var swiftFormat: FormatterRow {
        FormatterRow(
            id: swiftFormatID, name: "swift-format", extensions: ["swift"], command: "",
            args: ["format", "--in-place", filePlaceholder], enabled: true, isBuiltin: true)
    }

    func covers(extension ext: String) -> Bool {
        extensions.contains(ext.lowercased())
    }

    /// The argument vector for a temporary copy at `path`.
    func arguments(forFile path: String) -> [String] {
        guard args.contains(where: { $0.contains(Self.filePlaceholder) }) else { return args + [path] }
        return args.map { $0.replacingOccurrences(of: Self.filePlaceholder, with: path) }
    }
}

/// The app-level formatter settings: the master switch (default OFF, the
/// user's statement that a formatter may rewrite what the agent wrote) and
/// the rows. Plain `UserDefaults` lookups, like `LSPSettings`.
enum FormatterSettings {
    static let enabledKey = "formatter.enabled"
    static let rowsKey = "formatter.rows"
    /// Hard bound on rows, so a hand-edited defaults file cannot fan out.
    static let maximumRows = 8

    static func isEnabled(defaults: UserDefaults = .standard) -> Bool {
        defaults.bool(forKey: enabledKey)
    }

    static func setEnabled(_ value: Bool, defaults: UserDefaults = .standard) {
        defaults.set(value, forKey: enabledKey)
    }

    /// The rows, with the built-in swift-format row always present. An
    /// unreadable value decodes to just the built-in row.
    static func rows(defaults: UserDefaults = .standard) -> [FormatterRow] {
        var rows: [FormatterRow] = []
        if let data = defaults.data(forKey: rowsKey),
           let stored = try? JSONDecoder().decode([FormatterRow].self, from: data) {
            rows = Array(stored.prefix(maximumRows))
        }
        if !rows.contains(where: { $0.id == FormatterRow.swiftFormatID }) {
            rows.insert(.swiftFormat, at: 0)
        }
        // The built-in row never carries a stored command, arguments or
        // extensions: a hand-edited defaults file must not turn it into an
        // arbitrary launch.
        return rows.map { row in
            guard row.id == FormatterRow.swiftFormatID else {
                var custom = row
                custom.isBuiltin = false
                return custom
            }
            var builtin = FormatterRow.swiftFormat
            builtin.enabled = row.enabled
            return builtin
        }
    }

    static func save(_ rows: [FormatterRow], defaults: UserDefaults = .standard) {
        if let data = try? JSONEncoder().encode(Array(rows.prefix(maximumRows))) {
            defaults.set(data, forKey: rowsKey)
        }
    }

    /// "--write {file}" becomes ["--write", "{file}"]. Whitespace splitting
    /// only: there is no shell, so quotes are literal characters.
    static func parseArguments(_ text: String) -> [String] {
        text.split(whereSeparator: { $0 == " " || $0 == "\t" }).map(String.init)
    }
}

enum FormatterRegistry {
    /// The first enabled row covering `path`'s extension, or nil (no pass).
    static func row(forPath path: String, in rows: [FormatterRow]) -> FormatterRow? {
        let ext = (path as NSString).pathExtension.lowercased()
        guard !ext.isEmpty else { return nil }
        return rows.first { $0.enabled && $0.covers(extension: ext) }
    }

    /// Fixed absolute locations for the built-in swift-format. A file
    /// existence check only: no `xcrun`, no PATH search, no workspace lookup.
    static let swiftFormatCandidates = [
        "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/swift-format",
        "/Applications/Xcode-beta.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/swift-format",
        "/Library/Developer/CommandLineTools/usr/bin/swift-format",
        "/opt/homebrew/bin/swift-format",
        "/usr/local/bin/swift-format",
    ]

    static func swiftFormatPath(
        candidates: [String] = swiftFormatCandidates,
        isExecutable: (String) -> Bool = { FileManager.default.isExecutableFile(atPath: $0) }
    ) -> String? {
        candidates.first(where: isExecutable)
    }
}

enum FormatterLaunchError: Error, Equatable {
    case notInstalled
    case emptyCommand
    case relativeCommand(String)
    case notFound(String)
    case insideWorkspace(String)
    case badArgument
}

/// What may be launched as a formatter. The command is the USER's
/// configuration; this policy refuses what would let a cloned repository
/// supply the binary. It mirrors `LSPLaunchPolicy`.
enum FormatterLaunchPolicy {
    struct Launch: Equatable {
        var executable: String
        var arguments: [String]
    }

    static func resolve(
        row: FormatterRow, rootURL: URL, searchPath: String?, filePath: String,
        builtinPath: String? = FormatterRegistry.swiftFormatPath()
    ) throws -> Launch {
        let command: String
        if row.isBuiltin {
            guard let path = builtinPath else { throw FormatterLaunchError.notInstalled }
            command = path
        } else {
            command = row.command.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard !command.isEmpty else { throw FormatterLaunchError.emptyCommand }
        guard !command.contains("\0"), !command.contains("\n") else { throw FormatterLaunchError.badArgument }
        guard row.args.allSatisfy({ !$0.contains("\0") }) else { throw FormatterLaunchError.badArgument }

        // "./tools/fmt" and "node_modules/.bin/prettier" are workspace
        // binaries by construction. Bare names go to the PATH, absolute and
        // "~" paths are taken as written.
        if command.contains("/"), !command.hasPrefix("/"), !command.hasPrefix("~") {
            throw FormatterLaunchError.relativeCommand(command)
        }
        guard let resolved = McpClientEngine.resolveExecutablePath(command, relativeTo: nil, searchPath: searchPath),
              FileManager.default.isExecutableFile(atPath: resolved)
        else { throw FormatterLaunchError.notFound(command) }
        // Symlinks resolved on both sides: a PATH entry or a symlink that
        // lands inside the project is still a workspace binary.
        if PathContainment.isContained(URL(fileURLWithPath: resolved), in: rootURL) {
            throw FormatterLaunchError.insideWorkspace(resolved)
        }
        return Launch(executable: resolved, arguments: row.arguments(forFile: filePath))
    }

    /// PATH, HOME, LANG and TMPDIR from the allowlist in
    /// `McpClientEngine.childEnvironment`, nothing else. No credentials, no
    /// proxy settings, no passthrough.
    static func environment(
        parent: [String: String] = ProcessInfo.processInfo.environment,
        executable: String, searchPath: String?
    ) -> [String: String] {
        McpClientEngine.childEnvironment(
            parent: parent, passthrough: [], declared: [:],
            executableDirectory: (executable as NSString).deletingLastPathComponent,
            searchPath: searchPath)
    }
}
