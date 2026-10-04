import Foundation
import SQLCipher
import XCTest

@testable import TurboSparkApp

final class WorkflowJournalTests: XCTestCase {
    private var root: URL!
    private let key = Data(repeating: 0x4A, count: 32)
    private let runID = UUID(uuidString: "00000000-0000-0000-0000-000000000071")!
    private let fixedTime = Date(timeIntervalSince1970: 1_800_000_123.5)

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("WorkflowJournalTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    func testJournalOrdersEventsAndReplaysClassifiedOutcomeAcrossReopen() async throws {
        let url = root.appendingPathComponent("profile.sqlite3")
        let database = try makeDatabase(at: url)
        let journal = makeJournal(store: database)
        let descriptor = makeDescriptor()
        try await journal.createRun(descriptor)

        let site = WorkflowSiteKey(lane: "writer", siteIndex: 0, ordinal: 0)
        let identity = try WorkflowRequestIdentity.make(
            site: site,
            input: .object(["prompt": .string("draft")]),
            priorActorTranscriptHash: "prior-transcript-hash")
        _ = try await journal.append(runID: runID, payload: .phaseEntered("draft"))
        _ = try await journal.recordFinding(
            runID: runID,
            finding: WorkflowJournalFinding(phase: "draft", title: "Finding", body: "Evidence"))
        _ = try await journal.beginRequest(runID: runID, identity: identity)

        let failure = WorkflowJournalOutcome.failure(WorkflowError(
            kind: .permissionDenied,
            message: "Command approval was declined.",
            site: site))
        let transcript = WorkflowActorTranscriptSnapshot(payload: .array([
            .object(["role": .string("assistant"), "content": .string("I need approval.")])
        ]))
        let resolved = try await journal.resolveRequest(
            runID: runID,
            identity: identity,
            outcome: failure,
            actorName: "writer",
            transcript: transcript)
        XCTAssertEqual(resolved.sequence, 5)

        _ = try await journal.openQuestion(
            runID: runID,
            questionID: UUID(uuidString: "00000000-0000-0000-0000-000000000072")!,
            prompt: "Which branch should continue?")
        _ = try await journal.resolveQuestion(
            runID: runID,
            questionID: UUID(uuidString: "00000000-0000-0000-0000-000000000072")!,
            answer: "Continue with the first branch.")
        database.close()

        let reopened = try makeDatabase(at: url)
        defer { reopened.close() }
        let reopenedJournal = makeJournal(store: reopened)
        let history = try await reopenedJournal.history(runID: runID)
        let recorded = try await reopenedJournal.recordedOutcome(runID: runID, identity: identity)

        XCTAssertEqual(history.events.map(\.sequence), Array(1...7).map(Int64.init))
        XCTAssertEqual(history.events.first?.payload, .runStarted)
        XCTAssertEqual(
            history.events[2].payload,
            .finding(WorkflowJournalFinding(phase: "draft", title: "Finding", body: "Evidence")))
        XCTAssertEqual(
            recorded?.payload,
            .requestResolvedWithActorTranscript(
                outcome: failure,
                actorName: "writer",
                transcript: transcript,
                transcriptHash: try transcript.sha256()))
        XCTAssertEqual(history.questions.count, 1)
        XCTAssertEqual(history.questions.first?.state, .answered)
        XCTAssertEqual(history.questions.first?.answer, "Continue with the first branch.")
        XCTAssertEqual(history.actorTranscripts.first?.snapshot, transcript)
        XCTAssertEqual(history.actorTranscripts.first?.sha256, try transcript.sha256())

        let duplicate = try await reopenedJournal.resolveRequest(
            runID: runID,
            identity: identity,
            outcome: .value(.string("must not replace the recorded failure")),
            actorName: "writer",
            transcript: WorkflowActorTranscriptSnapshot(payload: .array([])))
        XCTAssertEqual(duplicate, resolved)
        let finalHistory = try await reopenedJournal.history(runID: runID)
        XCTAssertEqual(finalHistory.events.count, 7)
    }

    func testPersistedEnginePhaseCompletesAndSurvivesReopen() async throws {
        let url = root.appendingPathComponent("phase.sqlite3")
        let database = try makeDatabase(at: url)
        let journal = makeJournal(store: database)
        let checked = try XCTUnwrap(WorkflowScriptChecker.check(
            source: "async function workflow() { phase(\"draft\"); await report(\"done\"); }",
            name: "Persisted phase", args: [:]).checked)
        try await journal.createRun(checked.descriptor)
        let engine = WorkflowRunEngine(
            program: checked, journal: journal, capabilities: WorkflowJournalPhaseCapabilities())

        await engine.start()

        let progress = await engine.progress()
        XCTAssertEqual(progress.state, .completed)
        let history = try await journal.history(runID: checked.descriptor.id)
        let phase = try XCTUnwrap(history.events.first { $0.payload == .phaseEntered("draft") })
        XCTAssertEqual(phase.identity?.site.lane, "main")
        database.close()

        let reopened = try makeDatabase(at: url)
        defer { reopened.close() }
        let snapshot = try await makeJournal(store: reopened).resumeSnapshot(runID: checked.descriptor.id)
        XCTAssertEqual(snapshot.history.state, WorkflowRunState.completed.rawValue)
        XCTAssertEqual(snapshot.history.events.first { $0.payload == .phaseEntered("draft") }, phase)
    }

    func testPersistedEngineCancellationKeepsItsResolutionBeforeTerminalState() async throws {
        let database = try makeDatabase(at: root.appendingPathComponent("cancelled-engine.sqlite3"))
        defer { database.close() }
        let journal = makeJournal(store: database)
        let checked = try XCTUnwrap(WorkflowScriptChecker.check(
            source: "async function workflow() { await report(\"done\"); }",
            name: "Persisted cancellation", args: [:]).checked)
        try await journal.createRun(checked.descriptor)
        let engine = WorkflowRunEngine(
            program: checked, journal: journal, capabilities: WorkflowJournalCancelledCapabilities())

        await engine.start()

        let history = try await journal.resumeSnapshot(runID: checked.descriptor.id).history
        let cancellation = try XCTUnwrap(history.events.first { event in
            guard case .requestResolved(.failure(let error)) = event.payload else { return false }
            return error.kind == .cancelled
        })
        let terminal = try XCTUnwrap(history.events.first { $0.payload == .stateChanged(.cancelled) })
        XCTAssertEqual(cancellation.identity?.site.siteIndex, WorkflowJournalSystemSites.cancellation)
        XCTAssertLessThan(cancellation.sequence, terminal.sequence)
    }

    func testUnreservedNegativeSitesRemainInvalid() async throws {
        let database = try makeDatabase(at: root.appendingPathComponent("invalid-site.sqlite3"))
        defer { database.close() }
        let journal = makeJournal(store: database)
        try await journal.createRun(makeDescriptor())

        for site in [
            WorkflowSiteKey(lane: "main", siteIndex: -1, ordinal: 0),
            WorkflowSiteKey(lane: "writer", siteIndex: WorkflowJournalSystemSites.cancellation, ordinal: 0),
        ] {
            do {
                _ = try await journal.beginRequest(
                    runID: runID, identity: WorkflowRequestIdentity(site: site, inputHash: "invalid"))
                XCTFail("Source and actor lanes must not acquire reserved system sites")
            } catch let error as WorkflowJournalError {
                XCTAssertEqual(error, .invalidEventIdentity)
            }
        }
        let history = try await journal.history(runID: runID)
        XCTAssertEqual(history.events.count, 1)
    }

    func testResolvedActorEventsPreservePerAskSnapshotsForReplay() async throws {
        let database = try makeDatabase(at: root.appendingPathComponent("actor-replay.sqlite3"))
        defer { database.close() }
        let journal = makeJournal(store: database)
        try await journal.createRun(makeDescriptor())

        let firstTranscript = WorkflowActorTranscriptSnapshot(payload: .array([
            .object(["role": .string("assistant"), "content": .string("First turn")])
        ]))
        let firstIdentity = try WorkflowRequestIdentity.make(
            site: WorkflowSiteKey(lane: "writer", siteIndex: 0, ordinal: 0),
            input: .string("first ask"),
            priorActorTranscriptHash: "initial")
        _ = try await journal.beginRequest(runID: runID, identity: firstIdentity)
        _ = try await journal.resolveRequest(
            runID: runID,
            identity: firstIdentity,
            outcome: .value(.string("first answer")),
            actorName: "writer",
            transcript: firstTranscript)

        let secondTranscript = WorkflowActorTranscriptSnapshot(payload: .array([
            .object(["role": .string("assistant"), "content": .string("Second turn")])
        ]))
        let secondIdentity = try WorkflowRequestIdentity.make(
            site: WorkflowSiteKey(lane: "writer", siteIndex: 0, ordinal: 1),
            input: .string("second ask"),
            priorActorTranscriptHash: try firstTranscript.sha256())
        _ = try await journal.beginRequest(runID: runID, identity: secondIdentity)
        _ = try await journal.resolveRequest(
            runID: runID,
            identity: secondIdentity,
            outcome: .value(.string("second answer")),
            actorName: "writer",
            transcript: secondTranscript)

        let firstReplay = try await journal.recordedOutcome(runID: runID, identity: firstIdentity)
        let history = try await journal.history(runID: runID)
        guard case .requestResolvedWithActorTranscript(
            let outcome,
            let actorName,
            let replayedTranscript,
            let replayedHash)? = firstReplay?.payload else {
            return XCTFail("Each actor resolution must retain its own replay snapshot")
        }
        XCTAssertEqual(outcome, .value(.string("first answer")))
        XCTAssertEqual(actorName, "writer")
        XCTAssertEqual(replayedTranscript, firstTranscript)
        XCTAssertEqual(replayedHash, try firstTranscript.sha256())
        XCTAssertEqual(history.actorTranscripts.first?.snapshot, secondTranscript)
        _ = try await journal.resumeSnapshot(runID: runID)
    }

    func testResumeRejectsLegacyActorEventsAndCorruptReplayState() async throws {
        let url = root.appendingPathComponent("invalid-replay-state.sqlite3")
        let database = try makeDatabase(at: url)
        let journal = makeJournal(store: database)
        let legacyRun = UUID(uuidString: "00000000-0000-0000-0000-000000000075")!
        let hashMismatchRun = UUID(uuidString: "00000000-0000-0000-0000-000000000076")!
        let projectionMismatchRun = UUID(uuidString: "00000000-0000-0000-0000-000000000077")!
        let missingProjectionRun = UUID(uuidString: "00000000-0000-0000-0000-000000000078")!
        let actorTranscript = WorkflowActorTranscriptSnapshot(payload: .array([
            .object(["role": .string("assistant"), "content": .string("Recorded transcript")])
        ]))
        let outcome = WorkflowJournalOutcome.value(.string("answer"))

        for id in [legacyRun, hashMismatchRun, projectionMismatchRun, missingProjectionRun] {
            try await journal.createRun(makeDescriptor(id: id))
            let identity = try WorkflowRequestIdentity.make(
                site: WorkflowSiteKey(lane: "writer", siteIndex: 0, ordinal: 0),
                input: .string(id.uuidString),
                priorActorTranscriptHash: "initial")
            _ = try await journal.resolveRequest(
                runID: id,
                identity: identity,
                outcome: outcome,
                actorName: "writer",
                transcript: actorTranscript)
        }
        database.close()

        try updateDatabase(
            at: url,
            sql: "UPDATE workflow_events SET payload_json = ?1 WHERE run_id = ?2 AND seq = 2",
            values: [try encode(WorkflowJournalEventPayload.requestResolved(outcome)), legacyRun.uuidString])
        try updateDatabase(
            at: url,
            sql: "UPDATE workflow_events SET payload_json = ?1 WHERE run_id = ?2 AND seq = 2",
            values: [
                try encode(WorkflowJournalEventPayload.requestResolvedWithActorTranscript(
                    outcome: outcome,
                    actorName: "writer",
                    transcript: actorTranscript,
                    transcriptHash: "incorrect-event-hash")),
                hashMismatchRun.uuidString
            ])

        let mismatchedProjection = WorkflowActorTranscriptSnapshot(payload: .array([
            .object(["role": .string("assistant"), "content": .string("Different table projection")])
        ]))
        try updateDatabase(
            at: url,
            sql: "UPDATE workflow_actors SET transcript_json = ?1, transcript_sha256 = ?2 WHERE run_id = ?3 AND actor_name = ?4",
            values: [
                try encode(mismatchedProjection),
                try mismatchedProjection.sha256(),
                projectionMismatchRun.uuidString,
                "writer"
            ])
        try executeDatabase(
            at: url,
            sql: "DELETE FROM workflow_actors WHERE run_id = '\(missingProjectionRun.uuidString)' AND actor_name = 'writer'")

        let reopened = try makeDatabase(at: url)
        defer { reopened.close() }
        let reopenedJournal = makeJournal(store: reopened)

        let legacyHistory = try await reopenedJournal.history(runID: legacyRun)
        XCTAssertEqual(legacyHistory.events.count, 2)
        XCTAssertEqual(legacyHistory.actorTranscripts.first?.snapshot, actorTranscript)
        do {
            _ = try await reopenedJournal.resumeSnapshot(runID: legacyRun)
            XCTFail("Legacy actor outcomes without event snapshots must refuse resume")
        } catch let error as WorkflowJournalError {
            XCTAssertEqual(error, .actorTranscriptEventMissing(actor: "writer"))
        }

        let hashHistory = try await reopenedJournal.history(runID: hashMismatchRun)
        XCTAssertEqual(hashHistory.events.count, 2)
        do {
            _ = try await reopenedJournal.resumeSnapshot(runID: hashMismatchRun)
            XCTFail("An altered event snapshot hash must refuse resume")
        } catch let error as WorkflowJournalError {
            XCTAssertEqual(error, .transcriptHashMismatch(actor: "writer"))
        }

        let projectionHistory = try await reopenedJournal.history(runID: projectionMismatchRun)
        XCTAssertEqual(projectionHistory.actorTranscripts.first?.snapshot, mismatchedProjection)
        do {
            _ = try await reopenedJournal.resumeSnapshot(runID: projectionMismatchRun)
            XCTFail("An actor-table projection that differs from its latest event must refuse resume")
        } catch let error as WorkflowJournalError {
            XCTAssertEqual(error, .actorTranscriptHistoryMismatch(actor: "writer"))
        }

        let missingProjectionHistory = try await reopenedJournal.history(runID: missingProjectionRun)
        XCTAssertEqual(missingProjectionHistory.events.count, 2)
        XCTAssertTrue(missingProjectionHistory.actorTranscripts.isEmpty)
        do {
            _ = try await reopenedJournal.resumeSnapshot(runID: missingProjectionRun)
            XCTFail("An actor-table row missing from the latest event projection must refuse resume")
        } catch let error as WorkflowJournalError {
            XCTAssertEqual(error, .actorTranscriptHistoryMismatch(actor: "writer"))
        }
    }

    func testUnknownVersionsRefuseResumeWhileHistoryStaysReadable() async throws {
        let url = root.appendingPathComponent("unknown-versions.sqlite3")
        let database = try makeDatabase(at: url)
        let journal = makeJournal(store: database)
        try await journal.createRun(makeDescriptor())
        let identity = try WorkflowRequestIdentity.make(
            site: WorkflowSiteKey(lane: "writer", siteIndex: 0, ordinal: 0),
            input: .string("ask"),
            priorActorTranscriptHash: "empty-transcript")
        let transcript = WorkflowActorTranscriptSnapshot(payload: .array([]))
        _ = try await journal.resolveRequest(
            runID: runID,
            identity: identity,
            outcome: .value(.string("answer")),
            actorName: "writer",
            transcript: transcript)

        let transcriptOnlyRun = UUID(uuidString: "00000000-0000-0000-0000-000000000073")!
        try await journal.createRun(makeDescriptor(id: transcriptOnlyRun))
        _ = try await journal.resolveRequest(
            runID: transcriptOnlyRun,
            identity: identity,
            outcome: .value(.string("answer")),
            actorName: "writer",
            transcript: transcript)

        let eventTranscriptRun = UUID(uuidString: "00000000-0000-0000-0000-000000000074")!
        try await journal.createRun(makeDescriptor(id: eventTranscriptRun))
        let eventIdentity = try WorkflowRequestIdentity.make(
            site: WorkflowSiteKey(lane: "writer", siteIndex: 0, ordinal: 0),
            input: .string("event ask"),
            priorActorTranscriptHash: "empty-transcript")
        _ = try await journal.resolveRequest(
            runID: eventTranscriptRun,
            identity: eventIdentity,
            outcome: .value(.string("answer")),
            actorName: "writer",
            transcript: transcript)
        database.close()

        let futureTranscript = WorkflowActorTranscriptSnapshot(version: 778, payload: .array([]))
        let futureEventPayload = WorkflowJournalEventPayload.requestResolvedWithActorTranscript(
            outcome: .value(.string("answer")),
            actorName: "writer",
            transcript: futureTranscript,
            transcriptHash: try futureTranscript.sha256())
        try updateDatabase(at: url, sql: "UPDATE workflow_runs SET canonicalization_version = ?1 WHERE run_id = ?2", values: [
            "777", runID.uuidString
        ])
        try updateDatabase(
            at: url,
            sql: "UPDATE workflow_actors SET transcript_json = ?1, transcript_sha256 = ?2 WHERE run_id = ?3 AND actor_name = ?4",
            values: [
                try encode(futureTranscript),
                "future-transcript-hash",
                transcriptOnlyRun.uuidString,
                "writer"
            ])
        try updateDatabase(
            at: url,
            sql: "UPDATE workflow_events SET payload_json = ?1 WHERE run_id = ?2 AND seq = 2",
            values: [try encode(futureEventPayload), eventTranscriptRun.uuidString])

        let reopened = try makeDatabase(at: url)
        defer { reopened.close() }
        let reopenedJournal = makeJournal(store: reopened)
        let canonicalHistory = try await reopenedJournal.history(runID: runID)
        XCTAssertEqual(canonicalHistory.canonicalizationVersion, 777)
        XCTAssertEqual(canonicalHistory.events.count, 2)
        do {
            _ = try await reopenedJournal.resumeSnapshot(runID: runID)
            XCTFail("Unknown canonicalization versions must fail closed for resume")
        } catch let error as WorkflowJournalError {
            XCTAssertEqual(error, .unsupportedCanonicalizationVersion(777))
            XCTAssertTrue(error.localizedDescription.contains("unsupported canonicalization version 777"))
        }

        let transcriptHistory = try await reopenedJournal.history(runID: transcriptOnlyRun)
        XCTAssertEqual(transcriptHistory.events.count, 2)
        XCTAssertEqual(transcriptHistory.actorTranscripts.first?.version, 778)
        XCTAssertEqual(transcriptHistory.actorTranscripts.first?.encodedSnapshot, try encode(futureTranscript))
        do {
            _ = try await reopenedJournal.resumeSnapshot(runID: transcriptOnlyRun)
            XCTFail("Unknown transcript versions must fail closed for resume")
        } catch let error as WorkflowJournalError {
            XCTAssertEqual(error, .unsupportedTranscriptVersion(actor: "writer", version: 778))
            XCTAssertTrue(error.localizedDescription.contains("unsupported transcript version 778"))
        }

        let eventTranscriptHistory = try await reopenedJournal.history(runID: eventTranscriptRun)
        XCTAssertEqual(eventTranscriptHistory.events.count, 2)
        guard case .requestResolvedWithActorTranscript(_, "writer", futureTranscript, _) =
            eventTranscriptHistory.events.last?.payload else {
            return XCTFail("Readable event history should retain the future transcript payload")
        }
        do {
            _ = try await reopenedJournal.resumeSnapshot(runID: eventTranscriptRun)
            XCTFail("Unknown transcript versions in historical events must fail closed for resume")
        } catch let error as WorkflowJournalError {
            XCTAssertEqual(error, .unsupportedTranscriptVersion(actor: "writer", version: 778))
            XCTAssertTrue(error.localizedDescription.contains("unsupported transcript version 778"))
        }
    }

    func testAskOutcomeAndActorTranscriptRollbackTogether() async throws {
        let url = root.appendingPathComponent("atomic-ask.sqlite3")
        let database = try makeDatabase(at: url)
        let journal = makeJournal(store: database)
        try await journal.createRun(makeDescriptor())
        let identity = try WorkflowRequestIdentity.make(
            site: WorkflowSiteKey(lane: "writer", siteIndex: 0, ordinal: 0),
            input: .string("ask"),
            priorActorTranscriptHash: "empty-transcript")
        database.close()

        try executeDatabase(
            at: url,
            sql: """
            CREATE TRIGGER fail_workflow_actor_snapshot
            BEFORE INSERT ON workflow_actors
            BEGIN
                SELECT RAISE(ABORT, 'injected transcript write failure');
            END;
            """)

        let reopened = try makeDatabase(at: url)
        defer { reopened.close() }
        let reopenedJournal = makeJournal(store: reopened)
        do {
            _ = try await reopenedJournal.resolveRequest(
                runID: runID,
                identity: identity,
                outcome: .value(.string("answer")),
                actorName: "writer",
                transcript: WorkflowActorTranscriptSnapshot(payload: .array([])))
            XCTFail("The injected transcript write failure should abort the ask transaction")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("injected transcript write failure"))
        }

        let history = try await reopenedJournal.history(runID: runID)
        XCTAssertEqual(history.events.map(\.payload), [.runStarted])
        XCTAssertTrue(history.actorTranscripts.isEmpty)
        let recorded = try await reopenedJournal.recordedOutcome(runID: runID, identity: identity)
        XCTAssertNil(recorded)
    }

    func testActorLaneResolutionRequiresItsMatchingTranscriptSnapshot() async throws {
        let database = try makeDatabase(at: root.appendingPathComponent("actor-lane.sqlite3"))
        defer { database.close() }
        let journal = makeJournal(store: database)
        try await journal.createRun(makeDescriptor())
        let identity = try WorkflowRequestIdentity.make(
            site: WorkflowSiteKey(lane: "writer", siteIndex: 0, ordinal: 0),
            input: .string("ask"),
            priorActorTranscriptHash: "empty-transcript")

        do {
            _ = try await journal.resolveRequest(
                runID: runID,
                identity: identity,
                outcome: .value(.string("answer")))
            XCTFail("An actor ask cannot resolve without its transcript snapshot")
        } catch let error as WorkflowJournalError {
            XCTAssertEqual(error, .actorTranscriptRequired("writer"))
        }
        let history = try await journal.history(runID: runID)
        XCTAssertEqual(history.events.map(\.payload), [.runStarted])
    }

    private func makeDatabase(at url: URL) throws -> ProfileDatabase {
        try ProfileDatabase(url: url, key: key)
    }

    private func makeJournal(store: any WorkflowJournalStore) -> WorkflowJournal {
        let date = fixedTime
        return WorkflowJournal(store: store, now: { date })
    }

    private func makeDescriptor(id: UUID? = nil) -> WorkflowRunDescriptor {
        WorkflowRunDescriptor(
            id: id ?? runID,
            name: "Journal test",
            source: "async function workflow() { await report(\"done\"); }",
            sourceHash: "source-hash",
            facadeVersion: WorkflowFacade.version,
            args: WorkflowFrozenArguments(["topic": "history"]),
            manifest: WorkflowLaunchManifest(actors: [
                WorkflowActorSpec(name: "writer", rolePrompt: "Draft a response"),
            ]))
    }

    private func encode<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }

    private func executeDatabase(at url: URL, sql: String) throws {
        try withDatabase(at: url) { connection in
            var message: UnsafeMutablePointer<CChar>?
            guard sqlite3_exec(connection, sql, nil, nil, &message) == SQLITE_OK else {
                let detail = message.map { String(cString: $0) } ?? String(cString: sqlite3_errmsg(connection))
                sqlite3_free(message)
                throw ProfileDatabase.DatabaseError.statement(detail)
            }
        }
    }

    private func updateDatabase(at url: URL, sql: String, values: [String]) throws {
        try withDatabase(at: url) { connection in
            var statement: OpaquePointer?
            guard sqlite3_prepare_v2(connection, sql, -1, &statement, nil) == SQLITE_OK,
                  let statement
            else { throw ProfileDatabase.DatabaseError.statement(String(cString: sqlite3_errmsg(connection))) }
            defer { sqlite3_finalize(statement) }
            for (offset, value) in values.enumerated() {
                guard sqlite3_bind_text(statement, Int32(offset + 1), value, -1, Self.transient) == SQLITE_OK else {
                    throw ProfileDatabase.DatabaseError.bind(String(cString: sqlite3_errmsg(connection)))
                }
            }
            guard sqlite3_step(statement) == SQLITE_DONE else {
                throw ProfileDatabase.DatabaseError.step(String(cString: sqlite3_errmsg(connection)))
            }
        }
    }

    private func withDatabase<T>(at url: URL, _ body: (OpaquePointer) throws -> T) throws -> T {
        var connection: OpaquePointer?
        guard sqlite3_open_v2(
            url.path,
            &connection,
            SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX,
            nil) == SQLITE_OK,
            let connection
        else { throw ProfileDatabase.DatabaseError.open("Could not open test database") }
        defer { sqlite3_close(connection) }

        let keyResult = key.withUnsafeBytes { bytes in
            sqlite3_key(connection, bytes.baseAddress, Int32(key.count))
        }
        guard keyResult == SQLITE_OK else {
            throw ProfileDatabase.DatabaseError.open("Could not unlock test database")
        }
        return try body(connection)
    }

    private static let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
}

private struct WorkflowJournalPhaseCapabilities: WorkflowCapabilities {
    func perform(
        _ operation: WorkflowInterpreterOperation,
        at site: WorkflowSiteKey?,
        context: WorkflowAttemptContext
    ) async throws -> WorkflowCapabilityResult {
        WorkflowCapabilityResult(value: .null)
    }
}

private struct WorkflowJournalCancelledCapabilities: WorkflowCapabilities {
    func perform(
        _ operation: WorkflowInterpreterOperation,
        at site: WorkflowSiteKey?,
        context: WorkflowAttemptContext
    ) async throws -> WorkflowCapabilityResult {
        throw WorkflowError(kind: .cancelled, message: "Cancelled fixture", site: site)
    }
}
