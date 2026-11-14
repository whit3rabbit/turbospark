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

/// The app-level LSP settings. Two keys: the master switch (default off) and
/// the user's rows. Reading is a plain `UserDefaults` lookup so the catalog's
/// tool filter can consult it on every turn, the way `CodemodeSettings` does.
enum LSPSettings {
    static let enabledKey = "lsp.enabled"
    static let serversKey = "lsp.servers"
    /// Hard bound on rows, so a hand-edited defaults file cannot fan out.
    static let maximumServers = 8

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
