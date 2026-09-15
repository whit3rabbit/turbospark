import Foundation

/// The source selected for the app-wide SOUL prompt.
public enum SoulPromptSource: String, Sendable {
    case hermes
    case turboSpark
}

/// A known external harness that can provide a SOUL.md for import.
public enum SoulPromptImportKind: String, CaseIterable, Identifiable, Sendable {
    case hermes
    case openClaw

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .hermes: "Hermes"
        case .openClaw: "OpenClaw"
        }
    }
}

/// A detected external SOUL.md, never an export destination.
public struct SoulPromptImportSource: Equatable, Identifiable, Sendable {
    public let kind: SoulPromptImportKind
    public let fileURL: URL

    public var id: String { kind.id }
    public var displayName: String { kind.displayName }

    public init(kind: SoulPromptImportKind, fileURL: URL) {
        self.kind = kind
        self.fileURL = fileURL
    }
}

/// The resolved SOUL prompt and the source used to obtain it.
public struct SoulPromptResolution: Equatable, Sendable {
    public let content: String
    public let source: SoulPromptSource
    public let hermesFileURL: URL
    public let hermesFileExists: Bool
    public let readError: String?

    public init(
        content: String,
        source: SoulPromptSource,
        hermesFileURL: URL,
        hermesFileExists: Bool,
        readError: String? = nil
    ) {
        self.content = content
        self.source = source
        self.hermesFileURL = hermesFileURL
        self.hermesFileExists = hermesFileExists
        self.readError = readError
    }
}

/// Reads Hermes' global SOUL.md and provides the TurboSpark fallback path.
public enum SoulPromptStore {
    public static let fileName = "SOUL.md"

    private static func configuredPath(
        _ value: String?, homeDirectory: URL
    ) -> URL? {
        guard let value else { return nil }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if trimmed == "~" {
            return homeDirectory
        }
        if trimmed.hasPrefix("~/") {
            return homeDirectory.appendingPathComponent(
                String(trimmed.dropFirst(2)), isDirectory: true)
        }
        return URL(fileURLWithPath: trimmed, isDirectory: true)
    }

    /// Resolves the home path without consulting process-global state, so the
    /// path contract can be tested without changing the test runner's env.
    public static func hermesHomeURL(
        environment: [String: String], homeDirectory: URL
    ) -> URL {
        if let configured = configuredPath(environment["HERMES_HOME"], homeDirectory: homeDirectory) {
            return configured
        }
        return homeDirectory.appendingPathComponent(".hermes", isDirectory: true)
    }

    /// Resolves OpenClaw's default workspace and its supported path overrides.
    public static func openClawWorkspaceURL(
        environment: [String: String], homeDirectory: URL
    ) -> URL {
        if let configured = configuredPath(
            environment["OPENCLAW_WORKSPACE_DIR"], homeDirectory: homeDirectory) {
            return configured
        }
        if let stateDirectory = configuredPath(
            environment["OPENCLAW_STATE_DIR"], homeDirectory: homeDirectory) {
            return stateDirectory.appendingPathComponent("workspace", isDirectory: true)
        }
        let openClawHome = configuredPath(
            environment["OPENCLAW_HOME"], homeDirectory: homeDirectory) ?? homeDirectory
        let profile = environment["OPENCLAW_PROFILE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let workspaceName = profile.map { $0.isEmpty || $0 == "default" ? "workspace" : "workspace-\($0)" }
            ?? "workspace"
        return openClawHome
            .appendingPathComponent(".openclaw", isDirectory: true)
            .appendingPathComponent(workspaceName, isDirectory: true)
    }

    /// Resolves Hermes' configurable home before falling back to its default.
    public static var hermesHomeURL: URL {
        // A test process must not read the developer's real Hermes identity.
        // The injected resolver above still covers the production path rules.
        if AppStorageRoot.isRunningTests {
            return AppStorageRoot.machineRoot.appendingPathComponent("hermes", isDirectory: true)
        }
        return hermesHomeURL(
            environment: ProcessInfo.processInfo.environment,
            homeDirectory: FileManager.default.homeDirectoryForCurrentUser)
    }

    public static var hermesFileURL: URL {
        hermesHomeURL.appendingPathComponent(fileName)
    }

    public static var detectedImportSources: [SoulPromptImportSource] {
        if AppStorageRoot.isRunningTests {
            return detectedImportSources(
                environment: [
                    "HERMES_HOME": AppStorageRoot.machineRoot
                        .appendingPathComponent("hermes", isDirectory: true).path
                ],
                homeDirectory: AppStorageRoot.machineRoot)
        }
        return detectedImportSources(
            environment: ProcessInfo.processInfo.environment,
            homeDirectory: FileManager.default.homeDirectoryForCurrentUser)
    }

    /// Returns only known external files that currently exist. A zero-byte
    /// file is still a valid source and remains available to import.
    public static func detectedImportSources(
        environment: [String: String],
        homeDirectory: URL,
        fileManager: FileManager = .default
    ) -> [SoulPromptImportSource] {
        let locations: [(SoulPromptImportKind, URL)] = [
            (.hermes, hermesHomeURL(environment: environment, homeDirectory: homeDirectory)
                .appendingPathComponent(fileName)),
            (.openClaw, openClawWorkspaceURL(environment: environment, homeDirectory: homeDirectory)
                .appendingPathComponent(fileName))
        ]
        return locations.compactMap { kind, fileURL in
            fileManager.fileExists(atPath: fileURL.path)
                ? SoulPromptImportSource(kind: kind, fileURL: fileURL)
                : nil
        }
    }

    /// Uses Hermes content whenever the file exists, including an empty file.
    /// A missing file falls back to the per-profile TurboSpark setting.
    public static func resolve(
        fallback: String,
        hermesHome: URL? = nil,
        fileManager: FileManager = .default
    ) -> SoulPromptResolution {
        let fileURL = (hermesHome ?? hermesHomeURL).appendingPathComponent(fileName)
        guard fileManager.fileExists(atPath: fileURL.path) else {
            return SoulPromptResolution(
                content: fallback,
                source: .turboSpark,
                hermesFileURL: fileURL,
                hermesFileExists: false)
        }

        do {
            let content = try String(contentsOf: fileURL, encoding: .utf8)
            return SoulPromptResolution(
                content: content,
                source: .hermes,
                hermesFileURL: fileURL,
                hermesFileExists: true)
        } catch {
            return SoulPromptResolution(
                content: "",
                source: .hermes,
                hermesFileURL: fileURL,
                hermesFileExists: true,
                readError: error.localizedDescription)
        }
    }

    /// Reads an existing Hermes file for import without creating one.
    public static func readHermes(
        hermesHome: URL? = nil
    ) throws -> String {
        try read(fileURL: (hermesHome ?? hermesHomeURL).appendingPathComponent(fileName))
    }

    /// Reads a user-selected or detected SOUL.md as UTF-8 Markdown.
    public static func read(fileURL: URL) throws -> String {
        try String(contentsOf: fileURL, encoding: .utf8)
    }

    /// Creates Hermes' SOUL.md only when it does not already exist.
    /// Refusing an existing path prevents an export action from clobbering
    /// content owned by another Hermes installation.
    public static func createHermes(
        content: String,
        hermesHome: URL? = nil,
        fileManager: FileManager = .default
    ) throws {
        let home = hermesHome ?? hermesHomeURL
        let fileURL = home.appendingPathComponent(fileName)
        if fileManager.fileExists(atPath: fileURL.path) {
            throw CocoaError(.fileWriteFileExists)
        }
        try fileManager.createDirectory(at: home, withIntermediateDirectories: true)
        try Data(content.utf8).write(to: fileURL, options: .atomic)
    }

    /// Updates an existing Hermes file atomically.
    public static func writeHermes(
        content: String,
        hermesHome: URL? = nil
    ) throws {
        let fileURL = (hermesHome ?? hermesHomeURL).appendingPathComponent(fileName)
        try Data(content.utf8).write(to: fileURL, options: .atomic)
    }
}
