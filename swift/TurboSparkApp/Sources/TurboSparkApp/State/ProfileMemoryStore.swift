import Foundation
import TurboSpark
import CryptoKit

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
    public var indexURL: URL { directory.appendingPathComponent(".embeddings.json") }
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

    public func clearIndex() {
        if usesVault { try? ProfileRepository.shared.deleteRecord(key: indexKey) }
        else { try? fileManager.removeItem(at: indexURL) }
    }

    public var hasIndex: Bool {
        if usesVault { return ((try? ProfileRepository.shared.rawRecord(key: indexKey)) ?? nil) != nil }
        return fileManager.fileExists(atPath: indexURL.path)
    }

    public var indexedModel: String? {
        let stored: Data? = usesVault
            ? ((try? ProfileRepository.shared.rawRecord(key: indexKey)) ?? nil)
            : try? Data(contentsOf: indexURL)
        guard let data = stored,
              let index = try? JSONDecoder().decode(ProfileMemoryEmbeddingIndex.self, from: data)
        else { return nil }
        return index.model
    }

    public func rebuildIndex(modelPath: String) async throws {
        let documents = chunks()
        guard !documents.isEmpty, !modelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            clearIndex()
            return
        }
        let vectors = try await TurboSparkEmbedding.encode(texts: documents, modelPath: modelPath)
        guard vectors.count == documents.count else { return }
        let hash = sourceHash(for: load())
        let index = ProfileMemoryEmbeddingIndex(
            model: modelPath,
            sourceHash: hash,
            rows: zip(documents, vectors).map { .init(text: $0.0, vector: $0.1) })
        let data = try JSONEncoder().encode(index)
        if usesVault { try ProfileRepository.shared.saveRawRecord(data, key: indexKey) }
        else { try data.write(to: indexURL, options: .atomic) }
    }

    public func semanticSearch(_ query: String, modelPath: String, limit: Int = 5) async -> [String] {
        let stored: Data? = usesVault
            ? ((try? ProfileRepository.shared.rawRecord(key: indexKey)) ?? nil)
            : try? Data(contentsOf: indexURL)
        guard let data = stored,
              let index = try? JSONDecoder().decode(ProfileMemoryEmbeddingIndex.self, from: data),
              index.model == modelPath,
              index.sourceHash == sourceHash(for: load()) else { return lexicalSearch(query, limit: limit) }
        return (try? await index.topK(query: query, limit: limit, modelPath: modelPath)) ?? lexicalSearch(query, limit: limit)
    }

    private func sourceHash(for text: String) -> String {
        let digest = SHA256.hash(data: text.data(using: .utf8) ?? Data())
        return digest.map { String(format: "%02x", $0) }.joined()
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

/// Disposable semantic index for profile memory. The sidecar stores vectors,
/// never user-facing Markdown, and can always be rebuilt from MEMORY.md.
public struct ProfileMemoryEmbeddingIndex: Codable, Equatable {
    public struct Row: Codable, Equatable {
        public var text: String
        public var vector: [Float]
    }
    public var model: String
    public var sourceHash: String
    public var rows: [Row]

    public func topK(query: String, limit: Int = 5, modelPath: String) async throws -> [String] {
        guard !rows.isEmpty else { return [] }
        let queryVector = try await TurboSparkEmbedding.encode(text: query, modelPath: modelPath)
        return rows.map { row in
            (TurboSparkEmbedding.cosineSimilarity(queryVector, row.vector), row.text)
        }.sorted { $0.0 > $1.0 }.prefix(limit).map(\.1)
    }
}
