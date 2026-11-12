import Foundation
import SQLCipher

extension ProfileDatabase: WorkflowSavedDefinitionStore {
    func save(_ definition: WorkflowSavedDefinition) async throws {
        try requireWorkflowSchemaAvailable()

        let timestamp = definition.updatedAt.timeIntervalSince1970
        guard timestamp.isFinite else {
            throw WorkflowSavedDefinitionStorageError.invalidUpdatedAt
        }
        let declarationData = try JSONEncoder().encode(definition.declarations)
        let declarationJSON = String(decoding: declarationData, as: UTF8.self)

        try transaction {
            try withStatement(
                """
                INSERT INTO saved_workflows(
                    workflow_id, name, scope, script_source, args_declaration_json,
                    description, updated_at
                )
                VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)
                ON CONFLICT(workflow_id) DO UPDATE SET
                    name = excluded.name,
                    scope = excluded.scope,
                    script_source = excluded.script_source,
                    args_declaration_json = excluded.args_declaration_json,
                    description = excluded.description,
                    updated_at = excluded.updated_at
                """) { statement in
                try bind(definition.id.uuidString, at: 1, to: statement)
                try bind(definition.name, at: 2, to: statement)
                try bind(definition.scope.rawValue, at: 3, to: statement)
                try bind(definition.source, at: 4, to: statement)
                try bind(declarationJSON, at: 5, to: statement)
                try bind(definition.description, at: 6, to: statement)
                try bind(String(timestamp), at: 7, to: statement)
                try stepDone(statement)
            }
        }
    }

    func definition(id: UUID) async throws -> WorkflowSavedDefinition? {
        try requireWorkflowSchemaAvailable()
        return try withStatement(
            """
            SELECT workflow_id, name, scope, script_source, args_declaration_json,
                   description, updated_at
            FROM saved_workflows
            WHERE workflow_id = ?1
            """) { (statement: OpaquePointer) -> WorkflowSavedDefinition? in
            try bind(id.uuidString, at: 1, to: statement)
            let result = sqlite3_step(statement)
            if result == SQLITE_DONE { return nil }
            guard result == SQLITE_ROW else { throw DatabaseError.step(errorMessage) }
            return try decodeSavedDefinition(statement)
        }
    }

    func listDefinitions() async throws -> [WorkflowSavedDefinition] {
        try requireWorkflowSchemaAvailable()
        var definitions: [WorkflowSavedDefinition] = []
        try withStatement(
            """
            SELECT workflow_id, name, scope, script_source, args_declaration_json,
                   description, updated_at
            FROM saved_workflows
            ORDER BY workflow_id COLLATE BINARY ASC
            """) { statement in
            while true {
                let result = sqlite3_step(statement)
                if result == SQLITE_DONE { break }
                guard result == SQLITE_ROW else { throw DatabaseError.step(errorMessage) }
                definitions.append(try decodeSavedDefinition(statement))
            }
        }
        return definitions
    }

    private func requireWorkflowSchemaAvailable() throws {
        let status = workflowSchemaStatus
        guard status == .available(version: 1) else {
            throw WorkflowSavedDefinitionStorageError.schemaUnavailable(status)
        }
    }

    private func decodeSavedDefinition(_ statement: OpaquePointer) throws -> WorkflowSavedDefinition {
        let storedID = columnText(statement, index: 0) ?? "<null>"
        guard let id = UUID(uuidString: storedID) else {
            throw WorkflowSavedDefinitionStorageError.malformedRow(
                id: storedID,
                reason: "workflow_id is not a UUID")
        }
        guard let name = columnText(statement, index: 1),
              let scopeText = columnText(statement, index: 2),
              let scope = WorkflowDefinitionScope(rawValue: scopeText),
              let source = columnText(statement, index: 3),
              let declarationJSON = columnText(statement, index: 4),
              let description = columnText(statement, index: 5),
              let timestampText = columnText(statement, index: 6),
              let timestamp = Double(timestampText),
              timestamp.isFinite
        else {
            throw WorkflowSavedDefinitionStorageError.malformedRow(
                id: storedID,
                reason: "required field is missing or invalid")
        }

        let declarations: [WorkflowArgumentDeclaration]
        do {
            declarations = try JSONDecoder().decode(
                [WorkflowArgumentDeclaration].self,
                from: Data(declarationJSON.utf8))
        } catch {
            throw WorkflowSavedDefinitionStorageError.malformedRow(
                id: storedID,
                reason: "argument declarations could not be decoded: \(error.localizedDescription)")
        }

        return WorkflowSavedDefinition(
            id: id,
            name: name,
            scope: scope,
            source: source,
            declarations: declarations,
            description: description,
            updatedAt: Date(timeIntervalSince1970: timestamp))
    }
}

extension ProfileDatabase: WorkflowJournalStore {
    func createRun(
        _ descriptor: WorkflowRunDescriptor,
        state: WorkflowRunState,
        at date: Date
    ) async throws {
        try requireWorkflowJournalSchemaAvailable()
        let timestamp = try journalTimestamp(date)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let argsJSON = String(decoding: try encoder.encode(descriptor.args.allValues), as: UTF8.self)

        try withWorkflowTransaction {
            guard try !workflowRunExists(descriptor.id) else {
                throw WorkflowJournalError.runAlreadyExists(descriptor.id)
            }
            try withStatement(
                """
                INSERT INTO workflow_runs(
                    run_id, name, state, script_source, script_hash, facade_version,
                    args_json, parent_run_id, launch_source, canonicalization_version,
                    usage_json, created_at, updated_at
                )
                VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, NULL, NULL, ?8, NULL, ?9, ?9)
                """) { statement in
                try bind(descriptor.id.uuidString, at: 1, to: statement)
                try bind(descriptor.name, at: 2, to: statement)
                try bind(state.rawValue, at: 3, to: statement)
                try bind(descriptor.source, at: 4, to: statement)
                try bind(descriptor.sourceHash, at: 5, to: statement)
                try bind(Int64(descriptor.facadeVersion), at: 6, to: statement)
                try bind(argsJSON, at: 7, to: statement)
                try bind(Int64(WorkflowCanonicalSerialization.currentVersion), at: 8, to: statement)
                try bind(timestamp, at: 9, to: statement)
                try stepDone(statement)
            }
            _ = try insertWorkflowEvent(
                runID: descriptor.id,
                identity: nil,
                payload: .runStarted,
                at: date)
        }
    }

    func append(
        runID: UUID,
        identity: WorkflowRequestIdentity?,
        payload: WorkflowJournalEventPayload,
        at date: Date
    ) async throws -> WorkflowJournalEvent {
        try requireWorkflowJournalSchemaAvailable()
        _ = try journalTimestamp(date)
        switch payload {
        case .runStarted, .requestResolved, .requestResolvedWithActorTranscript:
            throw WorkflowJournalError.resolutionMustBeAtomic
        case .questionOpened, .questionResolved:
            throw WorkflowJournalError.malformedHistory("Question events must update the question table atomically.")
        case .requestStarted:
            guard identity != nil else { throw WorkflowJournalError.invalidEventIdentity }
        case .phaseEntered:
            if let identity {
                guard identity.site.lane == "main",
                      identity.site.siteIndex == WorkflowJournalSystemSites.phase else {
                    throw WorkflowJournalError.invalidEventIdentity
                }
            }
        default:
            guard identity == nil else { throw WorkflowJournalError.invalidEventIdentity }
        }

        return try withWorkflowTransaction {
            try requireWorkflowRun(runID)
            if case .stateChanged(let state) = payload {
                let timestamp = try journalTimestamp(date)
                try withStatement("UPDATE workflow_runs SET state = ?2, updated_at = ?3 WHERE run_id = ?1") {
                    statement in
                    try bind(runID.uuidString, at: 1, to: statement)
                    try bind(state.rawValue, at: 2, to: statement)
                    try bind(timestamp, at: 3, to: statement)
                    try stepDone(statement)
                }
            }
            let event = try insertWorkflowEvent(runID: runID, identity: identity, payload: payload, at: date)
            try touchWorkflowRun(runID, at: date)
            return event
        }
    }

    func resolveRequest(
        runID: UUID,
        identity: WorkflowRequestIdentity,
        outcome: WorkflowJournalOutcome,
        actorName: String?,
        transcript: WorkflowActorTranscriptSnapshot?,
        at date: Date
    ) async throws -> WorkflowJournalEvent {
        try requireWorkflowJournalSchemaAvailable()
        _ = try journalTimestamp(date)
        guard (actorName == nil) == (transcript == nil) else {
            throw WorkflowJournalError.malformedHistory("Actor transcript updates require both an actor name and a snapshot.")
        }
        if identity.site.lane == "main" {
            guard actorName == nil else {
                throw WorkflowJournalError.malformedHistory("The entry lane cannot commit an actor transcript.")
            }
        } else {
            guard let actorName, transcript != nil else {
                throw WorkflowJournalError.actorTranscriptRequired(identity.site.lane)
            }
            guard actorName == identity.site.lane else {
                throw WorkflowJournalError.actorTranscriptLaneMismatch(
                    expected: identity.site.lane,
                    actual: actorName)
            }
        }
        if let actorName, actorName.isEmpty {
            throw WorkflowJournalError.malformedHistory("Actor name cannot be empty.")
        }
        if let transcript, transcript.version != WorkflowActorTranscriptSnapshot.currentVersion {
            throw WorkflowJournalError.unsupportedTranscriptVersion(actor: actorName ?? "", version: transcript.version)
        }

        return try withWorkflowTransaction {
            try requireWorkflowRun(runID)
            if let existing = try findResolvedWorkflowEvent(runID: runID, identity: identity) {
                return existing
            }

            let payload: WorkflowJournalEventPayload
            if let actorName, let transcript {
                payload = .requestResolvedWithActorTranscript(
                    outcome: outcome,
                    actorName: actorName,
                    transcript: transcript,
                    transcriptHash: try transcript.sha256())
            } else {
                payload = .requestResolved(outcome)
            }
            let event = try insertWorkflowEvent(
                runID: runID,
                identity: identity,
                payload: payload,
                at: date)
            if let actorName, let transcript {
                try saveActorTranscript(runID: runID, actorName: actorName, snapshot: transcript)
            }
            try touchWorkflowRun(runID, at: date)
            return event
        }
    }

    func recordedOutcome(
        runID: UUID,
        identity: WorkflowRequestIdentity
    ) async throws -> WorkflowJournalEvent? {
        try requireWorkflowJournalSchemaAvailable()
        try requireWorkflowRun(runID)
        return try findResolvedWorkflowEvent(runID: runID, identity: identity)
    }

    func openQuestion(
        runID: UUID,
        questionID: UUID,
        prompt: String,
        at date: Date
    ) async throws -> WorkflowJournalEvent {
        try requireWorkflowJournalSchemaAvailable()
        let timestamp = try journalTimestamp(date)
        guard !prompt.isEmpty else {
            throw WorkflowJournalError.malformedHistory("Question prompt cannot be empty.")
        }

        return try withWorkflowTransaction {
            try requireWorkflowRun(runID)
            guard try !workflowQuestionExists(runID: runID, questionID: questionID) else {
                throw WorkflowJournalError.questionAlreadyExists(questionID)
            }
            let question = WorkflowJournalQuestion(
                id: questionID,
                prompt: prompt,
                state: .pending,
                answer: nil,
                createdAt: date,
                resolvedAt: nil)
            try withStatement(
                """
                INSERT INTO workflow_questions(
                    run_id, question_id, prompt, answered, answer, created_at, resolved_at
                ) VALUES(?1, ?2, ?3, 0, NULL, ?4, NULL)
                """) { statement in
                try bind(runID.uuidString, at: 1, to: statement)
                try bind(questionID.uuidString, at: 2, to: statement)
                try bind(prompt, at: 3, to: statement)
                try bind(timestamp, at: 4, to: statement)
                try stepDone(statement)
            }
            let event = try insertWorkflowEvent(
                runID: runID,
                identity: nil,
                payload: .questionOpened(question),
                at: date)
            try touchWorkflowRun(runID, at: date)
            return event
        }
    }

    func resolveQuestion(
        runID: UUID,
        questionID: UUID,
        answer: String?,
        at date: Date
    ) async throws -> WorkflowJournalEvent {
        try requireWorkflowJournalSchemaAvailable()
        let timestamp = try journalTimestamp(date)

        return try withWorkflowTransaction {
            try requireWorkflowRun(runID)
            guard let question = try loadWorkflowQuestion(runID: runID, questionID: questionID),
                  question.state == .pending
            else { throw WorkflowJournalError.questionNotPending(questionID) }
            let resolved = WorkflowJournalQuestion(
                id: question.id,
                prompt: question.prompt,
                state: answer == nil ? .unanswered : .answered,
                answer: answer,
                createdAt: question.createdAt,
                resolvedAt: date)
            try withStatement(
                """
                UPDATE workflow_questions
                SET answered = ?3, answer = ?4, resolved_at = ?5
                WHERE run_id = ?1 AND question_id = ?2
                """) { statement in
                try bind(runID.uuidString, at: 1, to: statement)
                try bind(questionID.uuidString, at: 2, to: statement)
                try bind(Int64(answer == nil ? 0 : 1), at: 3, to: statement)
                try bind(answer, at: 4, to: statement)
                try bind(timestamp, at: 5, to: statement)
                try stepDone(statement)
            }
            let event = try insertWorkflowEvent(
                runID: runID,
                identity: nil,
                payload: .questionResolved(resolved),
                at: date)
            try touchWorkflowRun(runID, at: date)
            return event
        }
    }

    func history(runID: UUID) async throws -> WorkflowJournalHistory {
        try requireWorkflowJournalSchemaAvailable()
        return try withWorkflowReadTransaction {
            let run = try loadWorkflowRunHeader(runID)
            return WorkflowJournalHistory(
                runID: runID,
                state: run.state,
                canonicalizationVersion: run.canonicalizationVersion,
                source: run.source,
                sourceHash: run.sourceHash,
                facadeVersion: run.facadeVersion,
                args: run.args,
                events: try loadWorkflowEvents(runID),
                questions: try loadWorkflowQuestions(runID),
                actorTranscripts: try loadActorTranscripts(runID))
        }
    }

    private func requireWorkflowJournalSchemaAvailable() throws {
        let status = workflowSchemaStatus
        guard status == .available(version: 1) else {
            throw WorkflowJournalError.schemaUnavailable(String(describing: status))
        }
    }

    private func journalTimestamp(_ date: Date) throws -> String {
        let value = date.timeIntervalSince1970
        guard value.isFinite else { throw WorkflowJournalError.invalidTimestamp }
        return String(value)
    }

    private func withWorkflowTransaction<T>(_ body: () throws -> T) throws -> T {
        lock.lock()
        defer { lock.unlock() }
        try execute("BEGIN IMMEDIATE;")
        do {
            let value = try body()
            try execute("COMMIT;")
            return value
        } catch {
            try? execute("ROLLBACK;")
            throw error
        }
    }

    private func withWorkflowReadTransaction<T>(_ body: () throws -> T) throws -> T {
        lock.lock()
        defer { lock.unlock() }
        try execute("BEGIN;")
        do {
            let value = try body()
            try execute("COMMIT;")
            return value
        } catch {
            try? execute("ROLLBACK;")
            throw error
        }
    }

    private func workflowRunExists(_ runID: UUID) throws -> Bool {
        try withStatement("SELECT 1 FROM workflow_runs WHERE run_id = ?1") { statement in
            try bind(runID.uuidString, at: 1, to: statement)
            let result = sqlite3_step(statement)
            if result == SQLITE_DONE { return false }
            guard result == SQLITE_ROW else { throw DatabaseError.step(errorMessage) }
            return true
        }
    }

    private func requireWorkflowRun(_ runID: UUID) throws {
        guard try workflowRunExists(runID) else { throw WorkflowJournalError.runNotFound(runID) }
    }

    private func workflowQuestionExists(runID: UUID, questionID: UUID) throws -> Bool {
        try withStatement(
            "SELECT 1 FROM workflow_questions WHERE run_id = ?1 AND question_id = ?2") { statement in
            try bind(runID.uuidString, at: 1, to: statement)
            try bind(questionID.uuidString, at: 2, to: statement)
            let result = sqlite3_step(statement)
            if result == SQLITE_DONE { return false }
            guard result == SQLITE_ROW else { throw DatabaseError.step(errorMessage) }
            return true
        }
    }

    private func nextWorkflowSequence(runID: UUID) throws -> Int64 {
        try withStatement("SELECT MAX(seq) FROM workflow_events WHERE run_id = ?1") { statement in
            try bind(runID.uuidString, at: 1, to: statement)
            guard sqlite3_step(statement) == SQLITE_ROW else { throw DatabaseError.step(errorMessage) }
            guard sqlite3_column_type(statement, 0) != SQLITE_NULL else { return 1 }
            let last = sqlite3_column_int64(statement, 0)
            guard last >= 0, last < Int64.max else {
                throw WorkflowJournalError.sequenceExhausted(runID)
            }
            return last + 1
        }
    }

    private func insertWorkflowEvent(
        runID: UUID,
        identity: WorkflowRequestIdentity?,
        payload: WorkflowJournalEventPayload,
        at date: Date
    ) throws -> WorkflowJournalEvent {
        if let identity, !identity.site.isValidJournalKey {
            throw WorkflowJournalError.invalidEventIdentity
        }
        let sequence = try nextWorkflowSequence(runID: runID)
        let timestamp = try journalTimestamp(date)
        let payloadData = try Self.workflowJournalEncoder().encode(payload)
        let payloadJSON = String(decoding: payloadData, as: UTF8.self)
        try withStatement(
            """
            INSERT INTO workflow_events(
                run_id, seq, kind, lane, site_index, site_ordinal, input_hash,
                payload_json, created_at
            ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            """) { statement in
            try bind(runID.uuidString, at: 1, to: statement)
            try bind(sequence, at: 2, to: statement)
            try bind(payload.kind.rawValue, at: 3, to: statement)
            try bind(identity?.site.lane, at: 4, to: statement)
            try bind(identity.map { Int64($0.site.siteIndex) }, at: 5, to: statement)
            try bind(identity.map { Int64($0.site.ordinal) }, at: 6, to: statement)
            try bind(identity?.inputHash, at: 7, to: statement)
            try bind(payloadJSON, at: 8, to: statement)
            try bind(timestamp, at: 9, to: statement)
            try stepDone(statement)
        }
        return WorkflowJournalEvent(
            runID: runID,
            sequence: sequence,
            identity: identity,
            payload: payload,
            createdAt: date)
    }

    private func findResolvedWorkflowEvent(
        runID: UUID,
        identity: WorkflowRequestIdentity
    ) throws -> WorkflowJournalEvent? {
        try withStatement(
            """
            SELECT run_id, seq, kind, lane, site_index, site_ordinal, input_hash,
                   payload_json, created_at
            FROM workflow_events
            WHERE run_id = ?1 AND kind = ?2 AND lane = ?3 AND site_index = ?4
              AND site_ordinal = ?5 AND input_hash = ?6
            ORDER BY seq ASC LIMIT 1
            """) { (statement: OpaquePointer) -> WorkflowJournalEvent? in
            try bind(runID.uuidString, at: 1, to: statement)
            try bind(WorkflowJournalEventKind.requestResolved.rawValue, at: 2, to: statement)
            try bind(identity.site.lane, at: 3, to: statement)
            try bind(Int64(identity.site.siteIndex), at: 4, to: statement)
            try bind(Int64(identity.site.ordinal), at: 5, to: statement)
            try bind(identity.inputHash, at: 6, to: statement)
            let result = sqlite3_step(statement)
            if result == SQLITE_DONE { return nil }
            guard result == SQLITE_ROW else { throw DatabaseError.step(errorMessage) }
            return try decodeWorkflowEvent(statement)
        }
    }

    private func saveActorTranscript(
        runID: UUID,
        actorName: String,
        snapshot: WorkflowActorTranscriptSnapshot
    ) throws {
        let snapshotJSON = try snapshot.encodedJSON()
        let digest = try snapshot.sha256()
        try withStatement(
            """
            INSERT INTO workflow_actors(run_id, actor_name, usage_json, transcript_json, transcript_sha256)
            VALUES(?1, ?2, NULL, ?3, ?4)
            ON CONFLICT(run_id, actor_name) DO UPDATE SET
                transcript_json = excluded.transcript_json,
                transcript_sha256 = excluded.transcript_sha256
            """) { statement in
            try bind(runID.uuidString, at: 1, to: statement)
            try bind(actorName, at: 2, to: statement)
            try bind(snapshotJSON, at: 3, to: statement)
            try bind(digest, at: 4, to: statement)
            try stepDone(statement)
        }
    }

    private func touchWorkflowRun(_ runID: UUID, at date: Date) throws {
        let timestamp = try journalTimestamp(date)
        try withStatement("UPDATE workflow_runs SET updated_at = ?2 WHERE run_id = ?1") { statement in
            try bind(runID.uuidString, at: 1, to: statement)
            try bind(timestamp, at: 2, to: statement)
            try stepDone(statement)
        }
    }

    private struct WorkflowRunHeader {
        var state: String
        var canonicalizationVersion: Int64
        var source: String
        var sourceHash: String
        var facadeVersion: Int
        var args: [String: String]
    }

    private func loadWorkflowRunHeader(_ runID: UUID) throws -> WorkflowRunHeader {
        try withStatement(
            """
            SELECT state, canonicalization_version, script_source, script_hash,
                   facade_version, args_json
            FROM workflow_runs WHERE run_id = ?1
            """) { statement in
            try bind(runID.uuidString, at: 1, to: statement)
            let result = sqlite3_step(statement)
            if result == SQLITE_DONE { throw WorkflowJournalError.runNotFound(runID) }
            guard result == SQLITE_ROW,
                  let state = columnText(statement, index: 0),
                  sqlite3_column_type(statement, 1) != SQLITE_NULL,
                  let source = columnText(statement, index: 2),
                  let sourceHash = columnText(statement, index: 3),
                  sqlite3_column_type(statement, 4) != SQLITE_NULL,
                  let argsJSON = columnText(statement, index: 5)
            else {
                throw WorkflowJournalError.malformedHistory("Run header is missing required fields.")
            }
            let args: [String: String]
            do {
                args = try JSONDecoder().decode([String: String].self, from: Data(argsJSON.utf8))
            } catch {
                throw WorkflowJournalError.malformedHistory("Run arguments could not be decoded.")
            }
            let facadeVersion = sqlite3_column_int64(statement, 4)
            guard facadeVersion >= 0, facadeVersion <= Int64(Int.max) else {
                throw WorkflowJournalError.malformedHistory("Facade version is outside the supported integer range.")
            }
            return WorkflowRunHeader(
                state: state,
                canonicalizationVersion: sqlite3_column_int64(statement, 1),
                source: source,
                sourceHash: sourceHash,
                facadeVersion: Int(facadeVersion),
                args: args)
        }
    }

    private func loadWorkflowEvents(_ runID: UUID) throws -> [WorkflowJournalEvent] {
        var events: [WorkflowJournalEvent] = []
        try withStatement(
            """
            SELECT run_id, seq, kind, lane, site_index, site_ordinal, input_hash,
                   payload_json, created_at
            FROM workflow_events WHERE run_id = ?1 ORDER BY seq ASC
            """) { statement in
            try bind(runID.uuidString, at: 1, to: statement)
            while true {
                let result = sqlite3_step(statement)
                if result == SQLITE_DONE { break }
                guard result == SQLITE_ROW else { throw DatabaseError.step(errorMessage) }
                events.append(try decodeWorkflowEvent(statement))
            }
        }
        return events
    }

    private func decodeWorkflowEvent(_ statement: OpaquePointer) throws -> WorkflowJournalEvent {
        let storedRunID = columnText(statement, index: 0) ?? "<null>"
        guard let runID = UUID(uuidString: storedRunID),
              sqlite3_column_type(statement, 1) != SQLITE_NULL,
              let kindText = columnText(statement, index: 2),
              let kind = WorkflowJournalEventKind(rawValue: kindText),
              let payloadJSON = columnText(statement, index: 7),
              let timestampText = columnText(statement, index: 8),
              let timestamp = Double(timestampText),
              timestamp.isFinite
        else {
            throw WorkflowJournalError.malformedHistory("Event row has a missing or invalid field.")
        }
        let sequence = sqlite3_column_int64(statement, 1)
        guard sequence > 0 else {
            throw WorkflowJournalError.malformedHistory("Event sequence must be positive.")
        }
        let payload: WorkflowJournalEventPayload
        do {
            payload = try JSONDecoder().decode(
                WorkflowJournalEventPayload.self,
                from: Data(payloadJSON.utf8))
        } catch {
            throw WorkflowJournalError.malformedHistory("Event payload could not be decoded.")
        }
        guard payload.kind == kind else {
            throw WorkflowJournalError.malformedHistory("Event kind does not match its payload.")
        }

        let lane = columnText(statement, index: 3)
        let siteIndex = optionalInt64(statement, index: 4)
        let ordinal = optionalInt64(statement, index: 5)
        let inputHash = columnText(statement, index: 6)
        let identity: WorkflowRequestIdentity?
        if lane == nil, siteIndex == nil, ordinal == nil, inputHash == nil {
            identity = nil
        } else {
            guard let lane, !lane.isEmpty,
                  let siteIndex, let decodedSiteIndex = Int(exactly: siteIndex),
                  let ordinal, ordinal >= 0, ordinal <= Int64(Int.max),
                  let inputHash
            else {
                throw WorkflowJournalError.malformedHistory("Event request identity is incomplete.")
            }
            let site = WorkflowSiteKey(lane: lane, siteIndex: decodedSiteIndex, ordinal: Int(ordinal))
            guard site.isValidJournalKey else { throw WorkflowJournalError.invalidEventIdentity }
            identity = WorkflowRequestIdentity(
                site: site,
                inputHash: inputHash)
        }
        switch payload {
        case .requestStarted, .requestResolved, .requestResolvedWithActorTranscript:
            guard identity != nil else { throw WorkflowJournalError.invalidEventIdentity }
            if case .requestResolvedWithActorTranscript(_, let actorName, _, _) = payload {
                guard let identity, identity.site.lane == actorName, actorName != "main" else {
                    throw WorkflowJournalError.malformedHistory(
                        "Resolved actor transcript does not match its request lane.")
                }
            }
        case .phaseEntered:
            if let identity {
                guard identity.site.lane == "main",
                      identity.site.siteIndex == WorkflowJournalSystemSites.phase else {
                    throw WorkflowJournalError.invalidEventIdentity
                }
            }
        default:
            guard identity == nil else { throw WorkflowJournalError.invalidEventIdentity }
        }
        return WorkflowJournalEvent(
            runID: runID,
            sequence: sequence,
            identity: identity,
            payload: payload,
            createdAt: Date(timeIntervalSince1970: timestamp))
    }

    private func loadWorkflowQuestions(_ runID: UUID) throws -> [WorkflowJournalQuestion] {
        var questions: [WorkflowJournalQuestion] = []
        try withStatement(
            """
            SELECT question_id, prompt, answered, answer, created_at, resolved_at
            FROM workflow_questions WHERE run_id = ?1
            ORDER BY CAST(created_at AS REAL) ASC, question_id COLLATE BINARY ASC
            """) { statement in
            try bind(runID.uuidString, at: 1, to: statement)
            while true {
                let result = sqlite3_step(statement)
                if result == SQLITE_DONE { break }
                guard result == SQLITE_ROW else { throw DatabaseError.step(errorMessage) }
                questions.append(try decodeWorkflowQuestion(statement))
            }
        }
        return questions
    }

    private func loadWorkflowQuestion(runID: UUID, questionID: UUID) throws -> WorkflowJournalQuestion? {
        try withStatement(
            """
            SELECT question_id, prompt, answered, answer, created_at, resolved_at
            FROM workflow_questions WHERE run_id = ?1 AND question_id = ?2
            """) { (statement: OpaquePointer) -> WorkflowJournalQuestion? in
            try bind(runID.uuidString, at: 1, to: statement)
            try bind(questionID.uuidString, at: 2, to: statement)
            let result = sqlite3_step(statement)
            if result == SQLITE_DONE { return nil }
            guard result == SQLITE_ROW else { throw DatabaseError.step(errorMessage) }
            return try decodeWorkflowQuestion(statement)
        }
    }

    private func decodeWorkflowQuestion(_ statement: OpaquePointer) throws -> WorkflowJournalQuestion {
        let storedID = columnText(statement, index: 0) ?? "<null>"
        guard let id = UUID(uuidString: storedID),
              let prompt = columnText(statement, index: 1),
              sqlite3_column_type(statement, 2) != SQLITE_NULL,
              let createdAtText = columnText(statement, index: 4),
              let createdAtValue = Double(createdAtText),
              createdAtValue.isFinite
        else {
            throw WorkflowJournalError.malformedHistory("Question row has a missing or invalid field.")
        }
        let answer = columnText(statement, index: 3)
        let resolvedAtText = columnText(statement, index: 5)
        let resolvedAt: Date?
        if let resolvedAtText {
            guard let value = Double(resolvedAtText), value.isFinite else {
                throw WorkflowJournalError.malformedHistory("Question resolution time is invalid.")
            }
            resolvedAt = Date(timeIntervalSince1970: value)
        } else {
            resolvedAt = nil
        }
        let wasAnswered = sqlite3_column_int64(statement, 2) != 0
        let state: WorkflowJournalQuestionState = resolvedAt == nil
            ? .pending
            : (wasAnswered ? .answered : .unanswered)
        return WorkflowJournalQuestion(
            id: id,
            prompt: prompt,
            state: state,
            answer: answer,
            createdAt: Date(timeIntervalSince1970: createdAtValue),
            resolvedAt: resolvedAt)
    }

    private func loadActorTranscripts(_ runID: UUID) throws -> [WorkflowJournalActorTranscript] {
        var transcripts: [WorkflowJournalActorTranscript] = []
        try withStatement(
            """
            SELECT actor_name, transcript_json, transcript_sha256
            FROM workflow_actors WHERE run_id = ?1
            ORDER BY actor_name COLLATE BINARY ASC
            """) { statement in
            try bind(runID.uuidString, at: 1, to: statement)
            while true {
                let result = sqlite3_step(statement)
                if result == SQLITE_DONE { break }
                guard result == SQLITE_ROW else { throw DatabaseError.step(errorMessage) }
                let actorName = columnText(statement, index: 0) ?? "<null>"
                let encodedSnapshot = columnText(statement, index: 1)
                let digest = columnText(statement, index: 2)
                if encodedSnapshot == nil, digest == nil { continue }
                guard let encodedSnapshot, let digest else {
                    throw WorkflowJournalError.malformedHistory(
                        "Actor '\(actorName)' has an incomplete transcript row.")
                }
                let snapshot = try? JSONDecoder().decode(
                    WorkflowActorTranscriptSnapshot.self,
                    from: Data(encodedSnapshot.utf8))
                transcripts.append(WorkflowJournalActorTranscript(
                    actorName: actorName,
                    sha256: digest,
                    encodedSnapshot: encodedSnapshot,
                    snapshot: snapshot))
            }
        }
        return transcripts
    }

    private static func workflowJournalEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }
}

