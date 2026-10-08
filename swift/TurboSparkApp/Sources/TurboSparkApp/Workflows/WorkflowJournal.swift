import Foundation

enum WorkflowJournalError: Error, Equatable, Sendable, LocalizedError {
    case schemaUnavailable(String)
    case runAlreadyExists(UUID)
    case runNotFound(UUID)
    case malformedHistory(String)
    case sequenceExhausted(UUID)
    case invalidEventIdentity
    case resolutionMustBeAtomic
    case actorTranscriptRequired(String)
    case actorTranscriptLaneMismatch(expected: String, actual: String)
    case invalidTimestamp
    case questionAlreadyExists(UUID)
    case questionNotPending(UUID)
    case unsupportedCanonicalizationVersion(Int64)
    case unsupportedTranscriptVersion(actor: String, version: Int)
    case unreadableTranscript(actor: String)
    case transcriptHashMismatch(actor: String)
    case actorTranscriptEventMissing(actor: String)
    case actorTranscriptHistoryMismatch(actor: String)

    var errorDescription: String? {
        switch self {
        case .schemaUnavailable(let message):
            return "Workflow history is unavailable: \(message)"
        case .runAlreadyExists(let id):
            return "Workflow run '\(id.uuidString)' already exists."
        case .runNotFound(let id):
            return "Workflow run '\(id.uuidString)' was not found."
        case .malformedHistory(let message):
            return "Workflow history is malformed: \(message)"
        case .sequenceExhausted(let id):
            return "Workflow run '\(id.uuidString)' exhausted its event sequence."
        case .invalidEventIdentity:
            return "This workflow journal event requires a request identity."
        case .resolutionMustBeAtomic:
            return "Resolved requests must be appended with their actor transcript in one transaction."
        case .actorTranscriptRequired(let actor):
            return "Resolving actor '\(actor)' work requires its transcript snapshot."
        case .actorTranscriptLaneMismatch(let expected, let actual):
            return "Actor lane '\(expected)' cannot commit a transcript for '\(actual)'."
        case .invalidTimestamp:
            return "Workflow journal timestamp must be finite."
        case .questionAlreadyExists(let id):
            return "Workflow question '\(id.uuidString)' already exists."
        case .questionNotPending(let id):
            return "Workflow question '\(id.uuidString)' is not pending."
        case .unsupportedCanonicalizationVersion(let version):
            return "Workflow history uses unsupported canonicalization version \(version)."
        case .unsupportedTranscriptVersion(let actor, let version):
            return "Actor '\(actor)' has unsupported transcript version \(version)."
        case .unreadableTranscript(let actor):
            return "Actor '\(actor)' has a transcript that cannot be decoded."
        case .transcriptHashMismatch(let actor):
            return "Actor '\(actor)' transcript hash does not match its snapshot."
        case .actorTranscriptEventMissing(let actor):
            return "Actor '\(actor)' has a resolved request without its replay transcript snapshot."
        case .actorTranscriptHistoryMismatch(let actor):
            return "Actor '\(actor)' transcript table does not match its resolved event history."
        }
    }
}

enum WorkflowJournalEventKind: String, Codable, Equatable, Sendable {
    case runStarted
    case phaseEntered
    case requestStarted
    case requestResolved
    case finding
    case questionOpened
    case questionResolved
    case stateChanged
}

enum WorkflowJournalOutcome: Codable, Equatable, Sendable {
    case value(WorkflowCanonicalValue)
    case failure(WorkflowError)
}

struct WorkflowJournalFinding: Codable, Equatable, Sendable {
    var phase: String?
    var title: String
    var body: String
}

enum WorkflowJournalQuestionState: String, Codable, Equatable, Sendable {
    case pending
    case answered
    case unanswered
}

struct WorkflowJournalQuestion: Codable, Equatable, Sendable {
    var id: UUID
    var prompt: String
    var state: WorkflowJournalQuestionState
    var answer: String?
    var createdAt: Date
    var resolvedAt: Date?
}

enum WorkflowJournalEventPayload: Codable, Equatable, Sendable {
    case runStarted
    case phaseEntered(String)
    case requestStarted
    case requestResolved(WorkflowJournalOutcome)
    case requestResolvedWithActorTranscript(
        outcome: WorkflowJournalOutcome,
        actorName: String,
        transcript: WorkflowActorTranscriptSnapshot,
        transcriptHash: String)
    case finding(WorkflowJournalFinding)
    case questionOpened(WorkflowJournalQuestion)
    case questionResolved(WorkflowJournalQuestion)
    case stateChanged(WorkflowRunState)

    var kind: WorkflowJournalEventKind {
        switch self {
        case .runStarted: return .runStarted
        case .phaseEntered: return .phaseEntered
        case .requestStarted: return .requestStarted
        case .requestResolved, .requestResolvedWithActorTranscript: return .requestResolved
        case .finding: return .finding
        case .questionOpened: return .questionOpened
        case .questionResolved: return .questionResolved
        case .stateChanged: return .stateChanged
        }
    }
}

struct WorkflowJournalEvent: Codable, Equatable, Sendable {
    var runID: UUID
    var sequence: Int64
    var identity: WorkflowRequestIdentity?
    var payload: WorkflowJournalEventPayload
    var createdAt: Date
}

struct WorkflowActorTranscriptSnapshot: Codable, Equatable, Sendable {
    static let currentVersion = 1

    var version: Int
    var payload: WorkflowCanonicalValue

    init(payload: WorkflowCanonicalValue) {
        version = Self.currentVersion
        self.payload = payload
    }

    init(version: Int, payload: WorkflowCanonicalValue) {
        self.version = version
        self.payload = payload
    }

    func sha256() throws -> String {
        try WorkflowCanonicalSerialization.sha256Hex(.object([
            "payload": payload,
            "version": .integer(Int64(version))
        ]))
    }

    func encodedJSON() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return String(decoding: try encoder.encode(self), as: UTF8.self)
    }
}

struct WorkflowJournalActorTranscript: Codable, Equatable, Sendable {
    var actorName: String
    var sha256: String
    var encodedSnapshot: String
    var snapshot: WorkflowActorTranscriptSnapshot?

    var version: Int? {
        snapshot?.version ?? Self.version(in: encodedSnapshot)
    }

    private static func version(in json: String) -> Int? {
        guard let data = json.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let number = object["version"] as? NSNumber
        else { return nil }
        return number.intValue
    }
}

struct WorkflowJournalHistory: Codable, Equatable, Sendable {
    var runID: UUID
    var state: String
    var canonicalizationVersion: Int64
    var source: String
    var sourceHash: String
    var facadeVersion: Int
    var args: [String: String]
    var events: [WorkflowJournalEvent]
    var questions: [WorkflowJournalQuestion]
    var actorTranscripts: [WorkflowJournalActorTranscript]
}

struct WorkflowJournalResumeSnapshot: Codable, Equatable, Sendable {
    var history: WorkflowJournalHistory
}

protocol WorkflowJournalStore: Sendable {
    func createRun(
        _ descriptor: WorkflowRunDescriptor,
        state: WorkflowRunState,
        at date: Date
    ) async throws

    func append(
        runID: UUID,
        identity: WorkflowRequestIdentity?,
        payload: WorkflowJournalEventPayload,
        at date: Date
    ) async throws -> WorkflowJournalEvent

    func resolveRequest(
        runID: UUID,
        identity: WorkflowRequestIdentity,
        outcome: WorkflowJournalOutcome,
        actorName: String?,
        transcript: WorkflowActorTranscriptSnapshot?,
        at date: Date
    ) async throws -> WorkflowJournalEvent

    func recordedOutcome(
        runID: UUID,
        identity: WorkflowRequestIdentity
    ) async throws -> WorkflowJournalEvent?

    func openQuestion(
        runID: UUID,
        questionID: UUID,
        prompt: String,
        at date: Date
    ) async throws -> WorkflowJournalEvent

    func resolveQuestion(
        runID: UUID,
        questionID: UUID,
        answer: String?,
        at date: Date
    ) async throws -> WorkflowJournalEvent

    func history(runID: UUID) async throws -> WorkflowJournalHistory
}

/// Owns the typed journal contract while the store serializes database writes.
struct WorkflowJournal: Sendable {
    private let store: any WorkflowJournalStore
    private let now: @Sendable () -> Date

    init(store: any WorkflowJournalStore, now: (@Sendable () -> Date)? = nil) {
        self.store = store
        self.now = now ?? { Date() }
    }

    func createRun(_ descriptor: WorkflowRunDescriptor, state: WorkflowRunState = .running) async throws {
        try await store.createRun(descriptor, state: state, at: now())
    }

    func append(
        runID: UUID,
        identity: WorkflowRequestIdentity? = nil,
        payload: WorkflowJournalEventPayload
    ) async throws -> WorkflowJournalEvent {
        try await store.append(runID: runID, identity: identity, payload: payload, at: now())
    }

    func beginRequest(runID: UUID, identity: WorkflowRequestIdentity) async throws -> WorkflowJournalEvent {
        try await store.append(
            runID: runID,
            identity: identity,
            payload: .requestStarted,
            at: now())
    }

    func resolveRequest(
        runID: UUID,
        identity: WorkflowRequestIdentity,
        outcome: WorkflowJournalOutcome,
        actorName: String? = nil,
        transcript: WorkflowActorTranscriptSnapshot? = nil
    ) async throws -> WorkflowJournalEvent {
        try await store.resolveRequest(
            runID: runID,
            identity: identity,
            outcome: outcome,
            actorName: actorName,
            transcript: transcript,
            at: now())
    }

    func recordedOutcome(
        runID: UUID,
        identity: WorkflowRequestIdentity
    ) async throws -> WorkflowJournalEvent? {
        try await store.recordedOutcome(runID: runID, identity: identity)
    }

    func recordFinding(runID: UUID, finding: WorkflowJournalFinding) async throws -> WorkflowJournalEvent {
        try await append(runID: runID, payload: .finding(finding))
    }

    func openQuestion(runID: UUID, questionID: UUID, prompt: String) async throws -> WorkflowJournalEvent {
        try await store.openQuestion(runID: runID, questionID: questionID, prompt: prompt, at: now())
    }

    func resolveQuestion(
        runID: UUID,
        questionID: UUID,
        answer: String?
    ) async throws -> WorkflowJournalEvent {
        try await store.resolveQuestion(runID: runID, questionID: questionID, answer: answer, at: now())
    }

    func history(runID: UUID) async throws -> WorkflowJournalHistory {
        try await store.history(runID: runID)
    }

    func resumeSnapshot(runID: UUID) async throws -> WorkflowJournalResumeSnapshot {
        let history = try await store.history(runID: runID)
        guard history.canonicalizationVersion == Int64(WorkflowCanonicalSerialization.currentVersion) else {
            throw WorkflowJournalError.unsupportedCanonicalizationVersion(history.canonicalizationVersion)
        }

        var latestEventTranscripts: [String: (snapshot: WorkflowActorTranscriptSnapshot, hash: String)] = [:]
        for event in history.events {
            switch event.payload {
            case .requestResolved:
                guard let identity = event.identity else {
                    throw WorkflowJournalError.invalidEventIdentity
                }
                if identity.site.lane != "main" {
                    throw WorkflowJournalError.actorTranscriptEventMissing(actor: identity.site.lane)
                }
            case .requestResolvedWithActorTranscript(_, let actorName, let snapshot, let digest):
                guard let identity = event.identity, identity.site.lane == actorName else {
                    throw WorkflowJournalError.actorTranscriptHistoryMismatch(actor: actorName)
                }
                guard snapshot.version == WorkflowActorTranscriptSnapshot.currentVersion else {
                    throw WorkflowJournalError.unsupportedTranscriptVersion(
                        actor: actorName,
                        version: snapshot.version)
                }
                guard try snapshot.sha256() == digest else {
                    throw WorkflowJournalError.transcriptHashMismatch(actor: actorName)
                }
                latestEventTranscripts[actorName] = (snapshot, digest)
            default:
                continue
            }
        }

        // A corrupt or hand-edited profile can hold two rows for one actor;
        // uniqueKeysWithValues would trap, so fail closed instead.
        var duplicateActor: String?
        let transcriptsByActor = Dictionary(
            history.actorTranscripts.map { ($0.actorName, $0) },
            uniquingKeysWith: { first, _ in
                duplicateActor = first.actorName
                return first
            })
        if let duplicateActor {
            throw WorkflowJournalError.malformedHistory(
                "duplicate actor transcript rows for '\(duplicateActor)'")
        }
        for transcript in history.actorTranscripts {
            guard let snapshot = transcript.snapshot else {
                throw WorkflowJournalError.unreadableTranscript(actor: transcript.actorName)
            }
            guard snapshot.version == WorkflowActorTranscriptSnapshot.currentVersion else {
                throw WorkflowJournalError.unsupportedTranscriptVersion(
                    actor: transcript.actorName,
                    version: snapshot.version)
            }
            guard try snapshot.sha256() == transcript.sha256 else {
                throw WorkflowJournalError.transcriptHashMismatch(actor: transcript.actorName)
            }
            guard let latest = latestEventTranscripts[transcript.actorName],
                  latest.snapshot == snapshot,
                  latest.hash == transcript.sha256
            else {
                throw WorkflowJournalError.actorTranscriptHistoryMismatch(actor: transcript.actorName)
            }
        }
        for actorName in latestEventTranscripts.keys where transcriptsByActor[actorName] == nil {
            throw WorkflowJournalError.actorTranscriptHistoryMismatch(actor: actorName)
        }
        return WorkflowJournalResumeSnapshot(history: history)
    }
}
