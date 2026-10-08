import CryptoKit
import Foundation

/// Which PROJECT-scoped custom tools the user has reviewed and trusted.
///
/// A tool definition under `<project>/.turbospark/tools` or
/// `<project>/.agents/tools` is authored by whoever wrote the repository and
/// runs `/bin/zsh -c` (or an HTTP request) with the user's privileges. Like a
/// repository hook, it must not be loaded, offered to the model, or executed
/// until the user has approved that exact definition. Trust is a SHA-256 of
/// the project path and everything that decides what the tool does, so an
/// edited definition (a `git pull` that rewrites the command) loses trust
/// and asks again. User-level (global) tools were written by the user and are
/// always trusted.
public final class CustomToolTrustStore: @unchecked Sendable {
    /// Replaceable so tests can use an isolated file or memory-only store.
    public static var shared = CustomToolTrustStore()

    private let lock = NSLock()
    private var trusted: Set<String>
    private let fileURL: URL?

    /// `fileURL == nil` keeps the store in memory only.
    public init(fileURL: URL? = AppStorageRoot.file("trusted_project_tools.json")) {
        self.fileURL = fileURL
        if let fileURL, let data = try? Data(contentsOf: fileURL),
           let list = try? JSONDecoder().decode([String].self, from: data)
        {
            trusted = Set(list)
        } else {
            trusted = []
        }
    }

    /// Content fingerprint of a project-scoped tool.
    public static func fingerprint(of tool: CustomToolDefinition) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let execution = (try? encoder.encode(tool.execution)).flatMap {
            String(data: $0, encoding: .utf8)
        } ?? ""
        var projectPath = ""
        if case .projectLocal(let path) = tool.scope {
            projectPath = URL(fileURLWithPath: path).standardizedFileURL.path
        }
        let material = [
            projectPath, tool.sourcePath ?? "", tool.name.lowercased(),
            tool.effectiveCategory.rawValue, execution,
        ].joined(separator: "\u{1F}")
        return SHA256.hash(data: Data(material.utf8))
            .map { String(format: "%02x", $0) }.joined()
    }

    /// Global, bundled and plugin tools are trusted by construction.
    public func isTrusted(_ tool: CustomToolDefinition) -> Bool {
        guard tool.scope.isProjectScope else { return true }
        let key = Self.fingerprint(of: tool)
        lock.lock()
        defer { lock.unlock() }
        return trusted.contains(key)
    }

    public func trust(_ tool: CustomToolDefinition) {
        guard tool.scope.isProjectScope else { return }
        lock.lock()
        trusted.insert(Self.fingerprint(of: tool))
        let snapshot = trusted
        lock.unlock()
        persist(snapshot)
    }

    public func revoke(_ tool: CustomToolDefinition) {
        lock.lock()
        trusted.remove(Self.fingerprint(of: tool))
        let snapshot = trusted
        lock.unlock()
        persist(snapshot)
    }

    private func persist(_ snapshot: Set<String>) {
        guard let fileURL, let data = try? JSONEncoder().encode(snapshot.sorted()) else { return }
        try? data.write(to: fileURL, options: .atomic)
    }
}
