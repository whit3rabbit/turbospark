import Foundation

/// Central manager for discovering, loading, and persisting user-defined custom tools.
public final class CustomToolManager: @unchecked Sendable {
    public static let shared = CustomToolManager()

    public private(set) var globalTools: [CustomToolDefinition] = []
    private let lock = NSLock()

    private init() {
        reloadGlobalTools()
    }

    /// Global storage directory in Application Support.
    public var globalToolsDirectory: URL {
        AppStorageRoot.subdirectory("tools")
    }

    /// User home directory tools path: `~/.turbospark/tools` for the Default
    /// profile, inside that profile's own folder for anyone else. For a
    /// non-default profile this is the SAME directory as
    /// `globalToolsDirectory`, which `reloadGlobalTools` de-duplicates.
    public var userHomeToolsDirectory: URL {
        UserProfileStore.userScopeSubdirectory("tools")
    }

    /// Scans and reloads global custom tools from Application Support and ~/.turbospark/tools.
    public func reloadGlobalTools() {
        lock.lock()
        defer { lock.unlock() }

        var toolsByName: [String: CustomToolDefinition] = [:]
        let fm = FileManager.default

        // Scan Application Support first. For a non-default profile both
        // directories are the profile's own `tools/` folder, so scanning
        // twice would only re-parse the same files.
        var dirs = [globalToolsDirectory]
        if userHomeToolsDirectory.standardizedFileURL.path
            != dirs[0].standardizedFileURL.path {
            dirs.append(userHomeToolsDirectory)
        }
        for dir in dirs {
            guard fm.fileExists(atPath: dir.path) else { continue }
            if let enumerator = fm.enumerator(at: dir, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles]) {
                for case let fileURL as URL in enumerator {
                    let ext = fileURL.pathExtension.lowercased()
                    guard ext == "json" || ext == "yaml" || ext == "yml" else { continue }
                    if let tool = try? CustomToolParser.parse(fileURL: fileURL, scope: .userGlobal),
                       !AppToolCatalog.isBuiltInToolName(tool.name) {
                        toolsByName[tool.name.lowercased()] = tool
                    }
                }
            }
        }

        self.globalTools = Array(toolsByName.values).sorted(by: { $0.name < $1.name })
    }

    /// Resolves the effective list of active custom tools for an optional project root.
    public func resolveEffectiveTools(for projectURL: URL?) -> [CustomToolDefinition] {
        lock.lock()
        defer { lock.unlock() }

        var effectiveByName: [String: CustomToolDefinition] = [:]
        for tool in globalTools where tool.isEnabled {
            effectiveByName[tool.name.lowercased()] = tool
        }

        guard let projectURL else {
            return Array(effectiveByName.values).sorted(by: { $0.name < $1.name })
        }

        // Untrusted project tools are skipped entirely, including a "disabled"
        // definition that would otherwise suppress a global tool of that name.
        for tool in scanProjectTools(at: projectURL)
        where CustomToolTrustStore.shared.isTrusted(tool) {
            if tool.isEnabled {
                // Project tools override global tools
                effectiveByName[tool.name.lowercased()] = tool
            } else {
                effectiveByName.removeValue(forKey: tool.name.lowercased())
            }
        }

        return Array(effectiveByName.values).sorted(by: { $0.name < $1.name })
    }

    /// Every parseable, non-colliding tool file in the project's tool
    /// folders, trusted or not.
    private func scanProjectTools(at projectURL: URL) -> [CustomToolDefinition] {
        let projectToolDir = projectURL.appendingPathComponent(".turbospark/tools", isDirectory: true)
        let agentToolDir = projectURL.appendingPathComponent(".agents/tools", isDirectory: true)
        let fm = FileManager.default
        var found: [CustomToolDefinition] = []
        for dir in [projectToolDir, agentToolDir] {
            guard fm.fileExists(atPath: dir.path) else { continue }
            guard let enumerator = fm.enumerator(at: dir, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles]) else { continue }
            for case let fileURL as URL in enumerator {
                let ext = fileURL.pathExtension.lowercased()
                guard ext == "json" || ext == "yaml" || ext == "yml" else { continue }
                guard let tool = try? CustomToolParser.parse(
                    fileURL: fileURL, scope: .projectLocal(projectPath: projectURL.path))
                else { continue }
                // A custom tool may never take a shipped tool's name, or a
                // repo file named `bash.json` could stand in for the shell.
                guard !AppToolCatalog.isBuiltInToolName(tool.name) else { continue }
                found.append(tool)
            }
        }
        return found
    }

    /// Project tools that exist on disk but are not trusted yet, so they are
    /// neither offered nor executed. The approval UI lists these.
    public func untrustedProjectTools(for projectURL: URL) -> [CustomToolDefinition] {
        lock.lock()
        defer { lock.unlock() }
        return scanProjectTools(at: projectURL)
            .filter { $0.isEnabled && !CustomToolTrustStore.shared.isTrusted($0) }
            .sorted { $0.name < $1.name }
    }

    /// The approval entry point a UI calls once the user has reviewed the
    /// tool's command: trusts the CURRENT on-disk definition of `name`.
    /// Returns false when no pending project tool has that name.
    @discardableResult
    public func trustProjectTool(named name: String, projectURL: URL) -> Bool {
        guard let tool = untrustedProjectTools(for: projectURL)
            .first(where: { $0.name.lowercased() == name.lowercased() }) else { return false }
        CustomToolTrustStore.shared.trust(tool)
        return true
    }

    /// Persists a custom tool definition to disk.
    public func saveTool(_ tool: CustomToolDefinition, projectURL: URL? = nil) throws {
        let targetDir: URL
        switch tool.scope {
        case .userGlobal, .bundled:
            targetDir = globalToolsDirectory
        case .projectLocal(let projectPath):
            targetDir = URL(fileURLWithPath: projectPath).appendingPathComponent(".turbospark/tools", isDirectory: true)
        case .plugin:
            throw NSError(domain: "TurboSparkCustomTool", code: 3, userInfo: [NSLocalizedDescriptionKey: "A custom tool cannot be saved into plugin scope."])
        }

        try FileManager.default.createDirectory(at: targetDir, withIntermediateDirectories: true)
        let fileURL = targetDir.appendingPathComponent("\(tool.name).json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(tool)
        try data.write(to: fileURL, options: .atomic)
        // A tool the user just authored in the app is trusted; the trust key
        // includes the parsed source path, so re-read the file to compute it.
        if tool.scope.isProjectScope,
           let saved = try? CustomToolParser.parse(fileURL: fileURL, scope: tool.scope) {
            CustomToolTrustStore.shared.trust(saved)
        }
        reloadGlobalTools()
    }

    /// Removes a custom tool from disk.
    public func deleteTool(_ tool: CustomToolDefinition) throws {
        if let path = tool.sourcePath, FileManager.default.fileExists(atPath: path) {
            try FileManager.default.removeItem(atPath: path)
        } else {
            let globalFile = globalToolsDirectory.appendingPathComponent("\(tool.name).json")
            if FileManager.default.fileExists(atPath: globalFile.path) {
                try FileManager.default.removeItem(at: globalFile)
            }
        }
        reloadGlobalTools()
    }
}
