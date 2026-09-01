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

    /// User home directory tools path (~/.turbospark/tools).
    public var userHomeToolsDirectory: URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let dir = home.appendingPathComponent(".turbospark/tools", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Scans and reloads global custom tools from Application Support and ~/.turbospark/tools.
    public func reloadGlobalTools() {
        lock.lock()
        defer { lock.unlock() }

        var toolsByName: [String: CustomToolDefinition] = [:]
        let fm = FileManager.default

        // Scan Application Support first
        for dir in [globalToolsDirectory, userHomeToolsDirectory] {
            guard fm.fileExists(atPath: dir.path) else { continue }
            if let enumerator = fm.enumerator(at: dir, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles]) {
                for case let fileURL as URL in enumerator {
                    let ext = fileURL.pathExtension.lowercased()
                    guard ext == "json" || ext == "yaml" || ext == "yml" else { continue }
                    if let tool = try? CustomToolParser.parse(fileURL: fileURL, scope: .userGlobal) {
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

        let projectToolDir = projectURL.appendingPathComponent(".turbospark/tools", isDirectory: true)
        let agentToolDir = projectURL.appendingPathComponent(".agents/tools", isDirectory: true)
        let fm = FileManager.default

        for dir in [projectToolDir, agentToolDir] {
            guard fm.fileExists(atPath: dir.path) else { continue }
            if let enumerator = fm.enumerator(at: dir, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles]) {
                for case let fileURL as URL in enumerator {
                    let ext = fileURL.pathExtension.lowercased()
                    guard ext == "json" || ext == "yaml" || ext == "yml" else { continue }
                    if let tool = try? CustomToolParser.parse(fileURL: fileURL, scope: .projectLocal(projectPath: projectURL.path)) {
                        if tool.isEnabled {
                            // Project tools override global tools
                            effectiveByName[tool.name.lowercased()] = tool
                        } else {
                            effectiveByName.removeValue(forKey: tool.name.lowercased())
                        }
                    }
                }
            }
        }

        return Array(effectiveByName.values).sorted(by: { $0.name < $1.name })
    }

    /// Persists a custom tool definition to disk.
    public func saveTool(_ tool: CustomToolDefinition, projectURL: URL? = nil) throws {
        let targetDir: URL
        switch tool.scope {
        case .userGlobal, .bundled:
            targetDir = globalToolsDirectory
        case .projectLocal(let projectPath):
            targetDir = URL(fileURLWithPath: projectPath).appendingPathComponent(".turbospark/tools", isDirectory: true)
        }

        try FileManager.default.createDirectory(at: targetDir, withIntermediateDirectories: true)
        let fileURL = targetDir.appendingPathComponent("\(tool.name).json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(tool)
        try data.write(to: fileURL, options: .atomic)
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
