import Foundation

/// Profile-wide memory. This is deliberately separate from the project memory
/// store: a user's preferences should follow them between projects, while
/// project decisions should not.
public final class ProfileMemoryStore {
    public static let shared = ProfileMemoryStore()

    private let lock = NSLock()
    private let fileManager = FileManager.default
    private let injectedBase: URL?

    public init(base: URL? = nil) { self.injectedBase = base }

    public var directory: URL {
        if let injectedBase { return injectedBase }
        if AppStorageRoot.isRunningTests {
            return AppStorageRoot.machineRoot.appendingPathComponent("profile-memory", isDirectory: true)
        }
        return UserProfileStore.userScopeSubdirectory("memory")
    }

    public var fileURL: URL { directory.appendingPathComponent("MEMORY.md") }
    private var legacyIndexURL: URL { directory.appendingPathComponent(".embeddings.json") }
    private var usesVault: Bool { injectedBase == nil && ProfileRepository.shared.isAvailable }
    private let memoryKey = "memory:file:profile/MEMORY.md"
    private let indexKey = "memory:file:profile/.embeddings.json"

    public func load() -> String {
        lock.lock(); defer { lock.unlock() }
        return loadUnlocked()
    }

    @discardableResult
    public func append(_ text: String, timestamp: Date = Date()) throws -> String {
        let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else { return "" }
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withColonSeparatorInTime, .withColonSeparatorInTimeZone]
        let entry = "\n- [\(formatter.string(from: timestamp))] \(value.replacingOccurrences(of: "\n", with: " "))\n"
        lock.lock(); defer { lock.unlock() }
        if !usesVault {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        }
        let old = loadUnlocked()
        try atomicWrite(old.isEmpty ? entry.trimmingCharacters(in: .newlines) + "\n" : old + entry)
        return value
    }

    public func chunks(maxCharacters: Int = 900) -> [String] {
        let text = load()
        guard !text.isEmpty else { return [] }
        return text.components(separatedBy: "\n\n").flatMap { paragraph in
            guard paragraph.count > maxCharacters else { return [paragraph] }
            return stride(from: 0, to: paragraph.count, by: maxCharacters).map {
                let start = paragraph.index(paragraph.startIndex, offsetBy: $0)
                let end = paragraph.index(start, offsetBy: min(maxCharacters, paragraph.distance(from: start, to: paragraph.endIndex)))
                return String(paragraph[start..<end])
            }
        }.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    }

    public func lexicalSearch(_ query: String, limit: Int = 5) -> [String] {
        let terms = Set(query.lowercased().split { $0.isWhitespace || $0.isPunctuation }.map(String.init))
        return chunks().map { chunk in
            let lower = chunk.lowercased()
            let score = terms.reduce(0) { $0 + (lower.contains($1) ? 1 : 0) }
            return (score, chunk)
        }.filter { $0.0 > 0 }.sorted { $0.0 > $1.0 }.prefix(limit).map(\.1)
    }

    public func write(_ text: String) throws {
        lock.lock(); defer { lock.unlock() }
        if !usesVault {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        }
        try atomicWrite(text)
    }

    public func clearLegacyIndex() {
        if usesVault { try? ProfileRepository.shared.deleteRecord(key: indexKey) }
        else { try? fileManager.removeItem(at: legacyIndexURL) }
    }

    private func loadUnlocked() -> String {
        if usesVault {
            if let data = (try? ProfileRepository.shared.rawRecord(key: memoryKey)) ?? nil,
               let text = String(data: data, encoding: .utf8) {
                return text
            }
            if let data = try? Data(contentsOf: fileURL),
               let text = String(data: data, encoding: .utf8) {
                try? ProfileRepository.shared.saveRawRecord(data, key: memoryKey)
                return text
            }
            return ""
        }
        return (try? String(contentsOf: fileURL, encoding: .utf8)) ?? ""
    }

    private func atomicWrite(_ text: String) throws {
        if usesVault {
            try ProfileRepository.shared.saveRawRecord(Data(text.utf8), key: memoryKey)
            return
        }
        let temporary = fileURL.appendingPathExtension("tmp-\(UUID().uuidString)")
        try text.write(to: temporary, atomically: true, encoding: .utf8)
        if fileManager.fileExists(atPath: fileURL.path) { try fileManager.removeItem(at: fileURL) }
        try fileManager.moveItem(at: temporary, to: fileURL)
    }
}
