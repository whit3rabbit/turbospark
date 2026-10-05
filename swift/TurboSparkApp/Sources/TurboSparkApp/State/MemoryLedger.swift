import CryptoKit
import Foundation
import TurboSpark

public enum MemoryClaimStatus: String, Codable, CaseIterable, Sendable {
    case pending, active, legacy, superseded, retracted, dismissed
}

public struct MemoryEvidence: Codable, Equatable, Sendable, Identifiable {
    public var id: UUID
    public var chatID: UUID
    public var messageID: UUID
    public var quote: String
    public var sourceHash: String

    public init(chatID: UUID, messageID: UUID, quote: String, source: String) {
        id = UUID()
        self.chatID = chatID
        self.messageID = messageID
        self.quote = quote
        sourceHash = MemoryLedgerStore.hash(source)
    }
}

public struct MemoryClaim: Codable, Equatable, Sendable, Identifiable {
    public var id: UUID
    public var scope: String
    public var kind: String
    public var text: String
    public var status: MemoryClaimStatus
    public var confidence: String
    public var createdAt: Date
    public var updatedAt: Date
    public var supersedesClaimID: UUID?
    public var evidence: [MemoryEvidence]
    public var projectTopicName: String?
    public var legacySourceText: String?
    public var requestedForgetID: UUID?

    public init(
        id: UUID = UUID(), scope: String, kind: String, text: String,
        status: MemoryClaimStatus = .pending, confidence: String = "source-matched",
        evidence: [MemoryEvidence] = [], projectTopicName: String? = nil
    ) {
        self.id = id
        self.scope = scope
        self.kind = kind
        self.text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        self.status = status
        self.confidence = confidence
        createdAt = Date()
        updatedAt = createdAt
        self.evidence = evidence
        self.projectTopicName = projectTopicName
        legacySourceText = nil
        requestedForgetID = nil
    }
}

public struct MemoryRunReceipt: Codable, Equatable, Sendable, Identifiable {
    public var id = UUID()
    public var kind: String
    public var date = Date()
    public var outcome: String
    public var sourceIDs: [UUID]
}

public struct MemoryReflection: Codable, Equatable, Sendable, Identifiable {
    public var id = UUID()
    public var date = Date()
    public var prose: String
    public var proposedGuidance: String
    public var sourceClaimIDs: [UUID]
    public var approved = false
}

public struct MemoryReviewBatch: Codable, Equatable, Sendable, Identifiable {
    public var id = UUID()
    public var createdAt = Date()
    public var origin: String
    public var sourceChatID: UUID?
    public var sourceMessageIDs: [UUID]
    public var claimIDs: [UUID]
}

public struct MemoryClaimEmbedding: Codable, Equatable, Sendable {
    public var claimID: UUID
    public var model: String
    public var sourceHash: String
    public var vector: [Float]
}

public struct MemoryLedgerState: Codable, Equatable, Sendable {
    public var version = 1
    public var claims: [MemoryClaim] = []
    public var processedSources: Set<String> = []
    public var suppressedSources: Set<String> = []
    public var receipts: [MemoryRunReceipt] = []
    public var reflections: [MemoryReflection] = []
    public var reviewBatches: [MemoryReviewBatch]? = []
    public var activeGuidance = ""
    public var lastHourlyRun: Date?
    public var lastNightlyDay: String?
    public var legacyImported = false
    public var revision = UUID()
}

/// All writes to remembered claims pass through this store. The ledger and
/// its Markdown projections commit together in the encrypted profile vault.
public final class MemoryLedgerStore: @unchecked Sendable {
    public static let shared = MemoryLedgerStore()

    private let lock = NSRecursiveLock()
    private let fallbackURL: URL?
    private var fallbackState = MemoryLedgerState()

    public init(base: URL? = nil) {
        fallbackURL = base.map { $0.appendingPathComponent("ledger.json") }
    }

    public static func hash(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    private static func embeddingModelKey(_ model: String) -> String {
        guard FileManager.default.fileExists(atPath: model) else { return "alias:\(model)" }
        let directory = URL(fileURLWithPath: model).resolvingSymlinksInPath()
        let stamps = ["config.json", "model.safetensors", "tokenizer.json"].map { name in
            let url = directory.appendingPathComponent(name)
            let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
            let size = attributes?[.size] as? NSNumber
            let modified = attributes?[.modificationDate] as? Date
            return "\(name):\(size?.int64Value ?? 0):\(modified?.timeIntervalSince1970 ?? 0)"
        }
        return "path:\(Self.hash(([directory.path] + stamps).joined(separator: "|")))"
    }

    public static func sourceKey(chatID: UUID, messageID: UUID, source: String) -> String {
        "\(chatID.uuidString)/\(messageID.uuidString)/\(hash(source))"
    }

    public static func sourceKey(chatID: UUID, messageID: UUID, sourceHash: String) -> String {
        "\(chatID.uuidString)/\(messageID.uuidString)/\(sourceHash)"
    }

    public func snapshot() -> MemoryLedgerState {
        lock.lock(); defer { lock.unlock() }
        if fallbackURL == nil, ProfileRepository.shared.isAvailable,
           let data = try? ProfileRepository.shared.loadMemoryLedger(),
           let state = try? JSONDecoder().decode(MemoryLedgerState.self, from: data) {
            return state
        }
        if let fallbackURL, let data = try? Data(contentsOf: fallbackURL),
           let state = try? JSONDecoder().decode(MemoryLedgerState.self, from: data) {
            return state
        }
        return fallbackState
    }

    public func update(_ change: (inout MemoryLedgerState) throws -> Void) throws {
        lock.lock(); defer { lock.unlock() }
        let stateData: Data?
        if fallbackURL == nil, ProfileRepository.shared.isAvailable {
            stateData = try ProfileRepository.shared.loadMemoryLedger()
        } else if let fallbackURL {
            stateData = FileManager.default.fileExists(atPath: fallbackURL.path)
                ? try Data(contentsOf: fallbackURL) : nil
        } else {
            stateData = nil
        }
        var state = try stateData.map { try JSONDecoder().decode(MemoryLedgerState.self, from: $0) }
            ?? fallbackState
        try change(&state)
        state.revision = UUID()
        let projections = Self.projections(state)
        if fallbackURL == nil, ProfileRepository.shared.isAvailable {
            try ProfileRepository.shared.saveMemoryLedger(
                try JSONEncoder().encode(state), projections: projections)
        } else if let fallbackURL {
            try FileManager.default.createDirectory(
                at: fallbackURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONEncoder().encode(state).write(to: fallbackURL, options: .atomic)
        } else if AppStorageRoot.isRunningTests {
            fallbackState = state
        } else {
            throw NSError(domain: "TurboSparkMemory", code: 8, userInfo: [
                NSLocalizedDescriptionKey: "Unlock the profile before changing memory."])
        }
    }

    public func propose(_ claim: MemoryClaim) throws {
        guard !claim.text.isEmpty else { return }
        try update { state in
            state.claims.append(claim)
            if claim.status == .pending {
                state.reviewBatches = (state.reviewBatches ?? []) + [MemoryReviewBatch(
                    origin: "model-tool", sourceChatID: nil,
                    sourceMessageIDs: [], claimIDs: [claim.id])]
            }
        }
    }

    public func saveUserClaim(_ text: String, scope: String, kind: String = "preference") throws {
        var claim = MemoryClaim(scope: scope, kind: kind, text: text,
                                status: .active, confidence: "user-authored")
        claim.updatedAt = Date()
        try propose(claim)
    }

    public func approve(_ id: UUID, text: String? = nil, supersedes: UUID? = nil) throws {
        try update { state in
            guard let index = state.claims.firstIndex(where: {
                $0.id == id && ($0.status == .pending || $0.status == .legacy)
            }) else { return }
            if let targetID = state.claims[index].requestedForgetID {
                Self.retract(targetID, in: &state)
                state.claims[index].status = .dismissed
                return
            }
            if let text {
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else { return }
                state.claims[index].text = trimmed
            }
            state.claims[index].status = .active
            state.claims[index].confidence = "user-confirmed"
            state.claims[index].updatedAt = Date()
            let selectedSupersession = supersedes ?? state.claims.first(where: {
                $0.id != id && $0.status == .active &&
                    $0.scope == state.claims[index].scope &&
                    $0.projectTopicName != nil &&
                    $0.projectTopicName == state.claims[index].projectTopicName
            })?.id
            if let supersedes = selectedSupersession,
               let old = state.claims.firstIndex(where: { $0.id == supersedes && $0.scope == state.claims[index].scope && $0.status == .active }) {
                state.claims[old].status = .superseded
                state.claims[index].supersedesClaimID = supersedes
            }
        }
    }

    public func dismiss(_ id: UUID) throws {
        try update { state in
            guard let index = state.claims.firstIndex(where: { $0.id == id && $0.status == .pending }) else { return }
            state.claims[index].status = .dismissed
            state.claims[index].updatedAt = Date()
        }
    }

    public func edit(_ id: UUID, text: String) throws {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        try update { state in
            guard let index = state.claims.firstIndex(where: { $0.id == id && $0.status == .active }) else { return }
            state.claims[index].text = trimmed
            state.claims[index].confidence = "user-confirmed"
            state.claims[index].updatedAt = Date()
        }
    }

    public func approveReflection(_ id: UUID) throws {
        try update { state in
            guard let index = state.reflections.firstIndex(where: { $0.id == id }) else { return }
            state.reflections[index].approved = true
            state.activeGuidance = state.reflections[index].proposedGuidance
        }
    }

    public func forget(_ id: UUID) throws {
        try update { state in
            Self.retract(id, in: &state)
        }
    }

    private static func retract(_ id: UUID, in state: inout MemoryLedgerState) {
        guard state.claims.contains(where: { $0.id == id }) else { return }
        var targets: Set<UUID> = [id]
        var changed = true
        while changed {
            let count = targets.count
            for claim in state.claims where claim.supersedesClaimID.map(targets.contains) == true {
                targets.insert(claim.id)
            }
            changed = targets.count != count
        }
        let sourceKeys = Set(state.claims.filter { targets.contains($0.id) }
            .flatMap(\.evidence).map {
                sourceKey(chatID: $0.chatID, messageID: $0.messageID, sourceHash: $0.sourceHash)
            })
        state.suppressedSources.formUnion(sourceKeys)
        for index in state.claims.indices {
            let claim = state.claims[index]
            let linked = claim.evidence.contains {
                sourceKeys.contains(sourceKey(
                    chatID: $0.chatID, messageID: $0.messageID, sourceHash: $0.sourceHash))
            }
            guard targets.contains(claim.id) || (claim.status == .pending && linked) else { continue }
            state.claims[index].evidence = []
            state.claims[index].text = ""
            state.claims[index].legacySourceText = nil
            state.claims[index].status = .retracted
            state.claims[index].updatedAt = Date()
            targets.insert(claim.id)
        }
        let removedApprovedGuidance = state.reflections.contains {
            $0.approved && !$0.sourceClaimIDs.filter(targets.contains).isEmpty
        }
        state.reflections.removeAll { !$0.sourceClaimIDs.filter(targets.contains).isEmpty }
        if removedApprovedGuidance { state.activeGuidance = "" }
        state.receipts.removeAll { !$0.sourceIDs.filter(targets.contains).isEmpty }
        state.reviewBatches?.removeAll { !$0.claimIDs.filter(targets.contains).isEmpty }
    }

    public func search(_ query: String, scope: String?, limit: Int = 5) -> [MemoryClaim] {
        let terms = Set(query.lowercased().split { $0.isWhitespace || $0.isPunctuation }.map(String.init))
        guard !terms.isEmpty else { return [] }
        let ftsIDs = (try? ProfileRepository.shared.memoryClaimSearchIDs(
            query: query, limit: 20)) ?? []
        let ftsRank = Dictionary(uniqueKeysWithValues: ftsIDs.enumerated().map { ($0.element, $0.offset) })
        var ranked: [(claim: MemoryClaim, score: Int)] = []
        for claim in snapshot().claims {
            guard claim.status == .active else { continue }
            guard claim.scope == "profile" || (scope != nil && claim.scope == scope) else { continue }
            let lower = claim.text.lowercased()
            let lexical = terms.reduce(0) { sum, term in sum + (lower.contains(term) ? 1 : 0) }
            let score = lexical + max(0, 100 - (ftsRank[claim.id] ?? 100))
            if score > 0 { ranked.append((claim, score)) }
        }
        ranked.sort { left, right in
            if left.score != right.score { return left.score > right.score }
            return left.claim.id.uuidString < right.claim.id.uuidString
        }
        return Array(ranked.prefix(max(0, limit)).map(\.claim))
    }

    public func rebuildEmbeddings(modelPath: String) async throws {
        let model = modelPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !model.isEmpty, ProfileRepository.shared.isAvailable else { return }
        let modelKey = Self.embeddingModelKey(model)
        let claims = snapshot().claims.filter { $0.status == .active && !$0.text.isEmpty }
        var rows: [MemoryClaimEmbedding] = []
        for batchStart in stride(from: 0, to: claims.count, by: 16) {
            let batch = Array(claims[batchStart..<min(batchStart + 16, claims.count)])
            let vectors = try await TurboSparkEmbedding.encode(
                texts: batch.map(\.text), modelPath: model)
            guard vectors.count == batch.count else { return }
            for (claim, vector) in zip(batch, vectors) {
                rows.append(MemoryClaimEmbedding(
                    claimID: claim.id, model: modelKey,
                    sourceHash: Self.hash(claim.text), vector: vector))
            }
        }
        guard modelKey == Self.embeddingModelKey(model) else { return }
        try ProfileRepository.shared.saveMemoryEmbeddings(rows)
    }

    public func hasEmbeddingIndex(modelPath: String) -> Bool {
        guard ProfileRepository.shared.isAvailable else { return false }
        let active = snapshot().claims.filter { $0.status == .active && !$0.text.isEmpty }
        guard !active.isEmpty else { return false }
        let rows = (try? ProfileRepository.shared.loadMemoryEmbeddings(
            model: Self.embeddingModelKey(modelPath))) ?? []
        let indexed = Dictionary(uniqueKeysWithValues: rows.map { ($0.claimID, $0.sourceHash) })
        return active.allSatisfy { indexed[$0.id] == Self.hash($0.text) }
    }

    public func clearEmbeddings() throws {
        try ProfileRepository.shared.clearMemoryEmbeddings()
    }

    public func semanticSearch(
        _ query: String, modelPath: String, scope: String?, limit: Int = 5
    ) async -> [MemoryClaim] {
        let lexical = search(query, scope: scope, limit: 20)
        let model = modelPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let modelKey = Self.embeddingModelKey(model)
        guard !model.isEmpty, ProfileRepository.shared.isAvailable,
              hasEmbeddingIndex(modelPath: model),
              let rows = try? ProfileRepository.shared.loadMemoryEmbeddings(model: modelKey),
              !rows.isEmpty,
              let queryVector = try? await TurboSparkEmbedding.encode(text: query, modelPath: model)
        else { return Array(lexical.prefix(limit)) }
        guard modelKey == Self.embeddingModelKey(model) else { return Array(lexical.prefix(limit)) }
        let state = snapshot()
        let active = Dictionary(uniqueKeysWithValues: state.claims.filter {
            $0.status == .active && ($0.scope == "profile" || (scope != nil && $0.scope == scope))
        }.map { ($0.id, $0) })
        let semantic = rows.compactMap { row -> (MemoryClaim, Float)? in
            guard let claim = active[row.claimID], row.sourceHash == Self.hash(claim.text),
                  row.vector.count == queryVector.count else { return nil }
            return (claim, TurboSparkEmbedding.cosineSimilarity(queryVector, row.vector))
        }.sorted { $0.1 > $1.1 }.prefix(20).map(\.0)
        var scores: [UUID: Double] = [:]
        for (rank, claim) in lexical.enumerated() {
            scores[claim.id, default: 0] += 1.0 / Double(60 + rank + 1)
        }
        for (rank, claim) in semantic.enumerated() {
            scores[claim.id, default: 0] += 1.0 / Double(60 + rank + 1)
        }
        return scores.keys.compactMap { active[$0] }.sorted { left, right in
            let leftScore = scores[left.id] ?? 0
            let rightScore = scores[right.id] ?? 0
            return leftScore == rightScore
                ? left.id.uuidString < right.id.uuidString
                : leftScore > rightScore
        }.prefix(max(0, limit)).map { $0 }
    }

    public func explain(_ id: UUID) -> MemoryClaim? {
        snapshot().claims.first { $0.id == id }
    }

    /// Existing Markdown has no trustworthy source IDs. Import each line as
    /// a legacy record and retain it for review instead of inventing a quote.
    public func importLegacyIfNeeded() throws {
        guard ProfileRepository.shared.isAvailable, !snapshot().legacyImported else { return }
        let profileText = ProfileMemoryStore.shared.load()
        let topics = try ProfileRepository.shared.rawRecords(prefix: "memory:file:projects/")
        try update { state in
            Self.importLegacy(profileText: profileText, topics: topics, into: &state)
        }
        ProfileMemoryStore.shared.clearLegacyIndex()
    }

    public static func importLegacy(
        profileText: String, topics: [(String, Data)], into state: inout MemoryLedgerState
    ) {
        guard !state.legacyImported else { return }
        for line in profileText.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            var claim = MemoryClaim(scope: "profile", kind: "legacy", text: trimmed,
                                    status: .legacy, confidence: "source-unverified")
            claim.legacySourceText = line
            state.claims.append(claim)
        }
        for (key, data) in topics where key.hasSuffix(".md") && !key.hasSuffix("/MEMORY.md") {
            let segments = key.split(separator: "/")
            guard segments.count == 4, segments[2] == "memory",
                  let text = String(data: data, encoding: .utf8) else { continue }
            let normalized = text.replacingOccurrences(of: "\r\n", with: "\n")
            let body: String
            if normalized.hasPrefix("---\n"),
               let close = normalized.range(of: "\n---\n", range:
                    normalized.index(normalized.startIndex, offsetBy: 4)..<normalized.endIndex) {
                body = String(normalized[close.upperBound...])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            } else {
                body = text
            }
            var claim = MemoryClaim(scope: "project:\(segments[1])", kind: "legacy",
                                    text: body, status: .legacy,
                                    confidence: "source-unverified",
                                    projectTopicName: String(segments.last!.dropLast(3)))
            claim.legacySourceText = text
            state.claims.append(claim)
        }
        state.legacyImported = true
    }

    public func invalidateEvidence(chatID: UUID, messages: [AppChatMessage]) throws {
        let current = Dictionary(uniqueKeysWithValues: messages.map { ($0.id, $0.content) })
        let affected = snapshot().claims.contains { claim in
            claim.evidence.contains { evidence in
                evidence.chatID == chatID && current[evidence.messageID].map(Self.hash) != evidence.sourceHash
            }
        }
        guard affected else { return }
        try update { state in
            var invalidatedIDs: Set<UUID> = []
            for index in state.claims.indices {
                let oldCount = state.claims[index].evidence.count
                state.claims[index].evidence.removeAll { evidence in
                    guard evidence.chatID == chatID else { return false }
                    guard let content = current[evidence.messageID] else { return true }
                    return Self.hash(content) != evidence.sourceHash
                }
                if oldCount > 0 && state.claims[index].evidence.isEmpty &&
                    state.claims[index].confidence != "user-authored" &&
                    state.claims[index].status == .active {
                    state.claims[index].status = .pending
                    invalidatedIDs.insert(state.claims[index].id)
                }
            }
            if !invalidatedIDs.isEmpty {
                let removedApprovedGuidance = state.reflections.contains {
                    $0.approved && !$0.sourceClaimIDs.filter(invalidatedIDs.contains).isEmpty
                }
                state.reflections.removeAll {
                    !$0.sourceClaimIDs.allSatisfy { !invalidatedIDs.contains($0) }
                }
                if removedApprovedGuidance { state.activeGuidance = "" }
            }
        }
    }

    public static func curatedProfileText(_ state: MemoryLedgerState) -> String {
        var lines: [String] = []
        var bytes = 0
        let profile = state.claims.filter { $0.scope == "profile" && $0.status == .active }
            .sorted { $0.updatedAt > $1.updatedAt }
        for claim in profile.prefix(35) {
            let compact = claim.text.replacingOccurrences(of: "\n", with: " ")
            let line = "- [\(claim.id.uuidString)] \(String(compact.prefix(300)))"
            guard bytes + line.utf8.count + 1 <= 6_000 else { break }
            lines.append(line)
            bytes += line.utf8.count + 1
        }
        return lines.joined(separator: "\n")
    }

    private static func projections(_ state: MemoryLedgerState) -> [String: Data] {
        let active = state.claims.filter { $0.status == .active }
        let profile = active.filter { $0.scope == "profile" }
        let summary = curatedProfileText(state)
        var result = ["memory:file:profile/MEMORY.md": Data((summary + "\n").utf8)]
        let day = DateFormatter()
        day.dateFormat = "yyyy-MM-dd"
        day.locale = Locale(identifier: "en_US_POSIX")
        for (key, claims) in Dictionary(grouping: profile, by: { day.string(from: $0.createdAt) }) {
            let body = claims.map { "- [\($0.id.uuidString)] \($0.text)" }.joined(separator: "\n")
            result["memory:file:profile/daily/\(key).md"] = Data((body + "\n").utf8)
        }
        for (kind, claims) in Dictionary(grouping: profile, by: { $0.kind }) {
            let safeKind = String(kind.lowercased().map { $0.isLetter || $0.isNumber ? $0 : "-" })
            let body = claims.map { claim in
                let sources = claim.evidence.map { "\($0.chatID.uuidString)/\($0.messageID.uuidString)" }.joined(separator: ", ")
                return "- [\(claim.id.uuidString)] \(claim.text)\(sources.isEmpty ? "" : " (sources: \(sources))")"
            }.joined(separator: "\n")
            result["memory:file:profile/bank/\(safeKind).md"] = Data((body + "\n").utf8)
        }
        let legacy = state.claims.filter {
            $0.scope == "profile" && $0.status != .retracted && $0.legacySourceText != nil
        }
        if !legacy.isEmpty {
            let body = legacy.compactMap(\.legacySourceText).joined(separator: "\n")
            result["memory:file:profile/legacy/IMPORTED_MEMORY.md"] = Data((body + "\n").utf8)
        }
        for reflection in state.reflections {
            let key = day.string(from: reflection.date)
            let content = "prompt_hoisted: false\nsource_claim_ids: \(reflection.sourceClaimIDs.map(\.uuidString).joined(separator: ", "))\n\n\(reflection.prose)\n"
            result["memory:file:profile/dreams/\(key).md"] = Data(content.utf8)
        }
        if !state.activeGuidance.isEmpty {
            result["memory:file:profile/ALIGNMENT_SYNTHESIS.md"] = Data(state.activeGuidance.utf8)
        }
        return result
    }
}
