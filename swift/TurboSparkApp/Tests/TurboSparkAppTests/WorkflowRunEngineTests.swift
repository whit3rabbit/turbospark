import Foundation
import XCTest

@testable import TurboSparkApp

final class WorkflowRunEngineTests: XCTestCase {
  func testRecordedRequestsShortCircuitAndPhasesAreNotEnteredTwice() async throws {
    let checked = try makeCheckedProgram(
      """
      async function workflow() {
        phase("draft");
        await report("cached");
      }
      """)
    let store = WorkflowRunEngineMemoryStore()
    let journal = WorkflowJournal(store: store)
    try await journal.createRun(checked.descriptor)
    let capabilities = WorkflowRunEngineTestCapabilities()
    let prefixBuilder = WorkflowRunEngine(
      program: checked,
      journal: journal,
      capabilities: capabilities)

    _ = try await prefixBuilder.perform(
      .phase(name: "draft"),
      at: nil,
      context: WorkflowAttemptContext(
        cancellation: WorkflowCancellationToken(),
        deadline: nil))
    _ = try await prefixBuilder.perform(
      .report(value: .string("cached")),
      at: WorkflowSiteKey(lane: "main", siteIndex: 0, ordinal: 0),
      context: WorkflowAttemptContext(
        cancellation: WorkflowCancellationToken(),
        deadline: nil))
    let callsBeforeReplay = await capabilities.calls()
    let engine = WorkflowRunEngine(
      program: checked,
      journal: journal,
      capabilities: capabilities)

    await engine.start()

    let callsAfterReplay = await capabilities.calls()
    let history = try await journal.history(runID: checked.descriptor.id)
    let phaseEvents = history.events.filter { $0.payload == .phaseEntered("draft") }
    let progress = await engine.progress()
    XCTAssertEqual(callsBeforeReplay.count, 1)
    XCTAssertEqual(callsAfterReplay, callsBeforeReplay)
    XCTAssertEqual(phaseEvents.count, 1)
    XCTAssertEqual(progress.currentPhase, "draft")
    XCTAssertEqual(progress.state, .completed)
  }

  func testActorLaneKeepsFIFOOrderAcrossConcurrentRequests() async throws {
    let checked = try makeCheckedProgram("async function workflow() { await report(\"done\"); }")
    let store = WorkflowRunEngineMemoryStore()
    let journal = WorkflowJournal(store: store)
    try await journal.createRun(checked.descriptor)
    let capabilities = WorkflowRunEngineTestCapabilities(mode: .holdFirst)
    let engine = WorkflowRunEngine(
      program: checked,
      journal: journal,
      capabilities: capabilities)
    let first = Task {
      try await engine.perform(
        .ask(actor: "writer", prompt: .string("first"), shape: .null),
        at: WorkflowSiteKey(lane: "writer", siteIndex: 0, ordinal: 0),
        context: WorkflowAttemptContext(
          cancellation: WorkflowCancellationToken(),
          deadline: nil))
    }

    await capabilities.waitForCallCount(1)
    let second = Task {
      try await engine.perform(
        .ask(actor: "writer", prompt: .string("second"), shape: .null),
        at: WorkflowSiteKey(lane: "writer", siteIndex: 0, ordinal: 1),
        context: WorkflowAttemptContext(
          cancellation: WorkflowCancellationToken(),
          deadline: nil))
    }
    while await engine.progress().lanes.first(where: { $0.name == "writer" })?.queuedCount != 1 {
      await Task.yield()
    }

    let callsBeforeRelease = await capabilities.calls()
    XCTAssertEqual(callsBeforeRelease.count, 1)
    await capabilities.releaseFirst()
    _ = try await first.value
    _ = try await second.value

    let calls = await capabilities.calls()
    let history = try await journal.history(runID: checked.descriptor.id)
    XCTAssertEqual(
      calls,
      [
        .ask(actor: "writer", prompt: .string("first"), shape: .null),
        .ask(actor: "writer", prompt: .string("second"), shape: .null),
      ])
    XCTAssertEqual(
      history.events.filter {
        if case .requestResolvedWithActorTranscript = $0.payload { return true }
        return false
      }.count, 2)
  }

  func testCancellationIsJournaledBeforeStateAndInterruptedRequestCanResume() async throws {
    let checked = try makeCheckedProgram(
      "async function workflow() { phase(\"draft\"); await report(\"resume\"); }")
    let store = WorkflowRunEngineMemoryStore()
    let journal = WorkflowJournal(store: store)
    try await journal.createRun(checked.descriptor)
    let capabilities = WorkflowRunEngineTestCapabilities(mode: .cancelFirstThenSucceed)
    let engine = WorkflowRunEngine(
      program: checked,
      journal: journal,
      capabilities: capabilities)
    let firstStart = Task { await engine.start() }
    await capabilities.waitForCallCount(1)
    await engine.cancel()
    await firstStart.value

    let cancelledProgress = await engine.progress()
    let cancelledHistory = try await journal.history(runID: checked.descriptor.id)
    let cancellationIndex = try XCTUnwrap(
      cancelledHistory.events.firstIndex { event in
        guard case .requestResolved(.failure(let error)) = event.payload else { return false }
        return error.kind == .cancelled
      })
    let cancelledStateIndex = try XCTUnwrap(
      cancelledHistory.events.firstIndex {
        $0.payload == .stateChanged(.cancelled)
      })
    XCTAssertEqual(cancelledProgress.state, .cancelled)
    XCTAssertLessThan(cancellationIndex, cancelledStateIndex)

    await engine.start()

    let calls = await capabilities.calls()
    let resumedProgress = await engine.progress()
    let resumedHistory = try await journal.history(runID: checked.descriptor.id)
    XCTAssertEqual(calls.count, 2, "Only the interrupted request should be redispatched on resume")
    XCTAssertEqual(resumedProgress.state, .completed)
    XCTAssertEqual(
      resumedHistory.events.filter { $0.payload == .phaseEntered("draft") }.count,
      1)
  }

  func testCancelledCapabilityOutcomeWithoutTokenCanResume() async throws {
    let checked = try makeCheckedProgram("async function workflow() { await report(\"retry\"); }")
    let store = WorkflowRunEngineMemoryStore()
    let journal = WorkflowJournal(store: store)
    try await journal.createRun(checked.descriptor)
    let capabilities = WorkflowRunEngineTestCapabilities(mode: .cancelKindThenSucceed)
    let engine = WorkflowRunEngine(
      program: checked,
      journal: journal,
      capabilities: capabilities)

    await engine.start()
    let firstProgress = await engine.progress()
    XCTAssertEqual(firstProgress.state, .cancelled)

    await engine.start()

    let calls = await capabilities.calls()
    let progress = await engine.progress()
    XCTAssertEqual(
      calls.count, 2, "A cancelled capability outcome should be redispatched on resume")
    XCTAssertEqual(progress.state, .completed)
  }

  func testEveryErrorKindIsRecordedBeforeTheFailedRunState() async throws {
    for kind in WorkflowErrorKind.allCases {
      let checked = try makeCheckedProgram("async function workflow() { await report(\"fails\"); }")
      let store = WorkflowRunEngineMemoryStore()
      let journal = WorkflowJournal(store: store)
      try await journal.createRun(checked.descriptor)
      let capabilities = WorkflowRunEngineTestCapabilities(
        mode: .fail(WorkflowError(kind: kind, message: "classified", site: nil)))
      let engine = WorkflowRunEngine(
        program: checked,
        journal: journal,
        capabilities: capabilities)

      await engine.start()

      let history = try await journal.history(runID: checked.descriptor.id)
      let failureIndex = try XCTUnwrap(
        history.events.firstIndex { event in
          guard case .requestResolved(.failure(let error)) = event.payload else { return false }
          return error.kind == kind
        }, "Missing recorded outcome for \(kind)")
      let terminalState = kind == .cancelled ? WorkflowRunState.cancelled : .failed
      let stateIndex = try XCTUnwrap(
        history.events.firstIndex {
          $0.payload == .stateChanged(terminalState)
        }, "Missing terminal state for \(kind)")
      XCTAssertLessThan(failureIndex, stateIndex)
    }
  }

  func testWorldReadOutcomeIsReplayedWithoutRereadingWorkspace() async throws {
    let checked = try makeCheckedProgram("async function workflow() { await report(\"done\"); }")
    let store = WorkflowRunEngineMemoryStore()
    let journal = WorkflowJournal(store: store)
    try await journal.createRun(checked.descriptor)
    let executor = WorkflowWorldTestExecutor()
    let world = WorkflowWorld(
      workspaceRoot: FileManager.default.temporaryDirectory,
      executor: executor)
    let engine = WorkflowRunEngine(program: checked, journal: journal, capabilities: world)
    let site = WorkflowSiteKey(lane: "main", siteIndex: 41, ordinal: 0)
    let operation = WorkflowInterpreterOperation.worldRead(operation: .object([
      "kind": .string("git"),
      "arguments": .array([.string("status")]),
    ]))
    let context = WorkflowAttemptContext(
      cancellation: WorkflowCancellationToken(),
      deadline: nil)

    let first = try await engine.perform(operation, at: site, context: context)
    let callsAfterFirst = await executor.requests()
    let replayed = try await engine.perform(operation, at: site, context: context)
    let callsAfterReplay = await executor.requests()
    let history = try await journal.history(runID: checked.descriptor.id)
    let progress = await engine.progress()
    let resolvedReads = history.events.filter { event in
      guard event.identity?.site == site,
            case .requestResolved(let outcome) = event.payload
      else { return false }
      if case .value = outcome { return true }
      return false
    }

    XCTAssertEqual(first, replayed)
    XCTAssertEqual(callsAfterFirst.count, 1)
    XCTAssertEqual(callsAfterReplay.count, 1)
    XCTAssertEqual(resolvedReads.count, 1)
    XCTAssertEqual(progress.usage.replayedRequests, 1)
  }

  func testWorldSandboxRefusalIsJournaledAndReplayed() async throws {
    let checked = try makeCheckedProgram("async function workflow() { await report(\"done\"); }")
    let store = WorkflowRunEngineMemoryStore()
    let journal = WorkflowJournal(store: store)
    try await journal.createRun(checked.descriptor)
    let executor = WorkflowWorldTestExecutor()
    let world = WorkflowWorld(
      workspaceRoot: FileManager.default.temporaryDirectory,
      executor: executor)
    let engine = WorkflowRunEngine(program: checked, journal: journal, capabilities: world)
    let site = WorkflowSiteKey(lane: "main", siteIndex: 42, ordinal: 0)
    let operation = WorkflowInterpreterOperation.worldRead(operation: .object([
      "kind": .string("read"),
      "arguments": .array([.string("../outside.txt"), .integer(32)]),
    ]))
    let context = WorkflowAttemptContext(
      cancellation: WorkflowCancellationToken(),
      deadline: nil)

    for _ in 0..<2 {
      do {
        _ = try await engine.perform(operation, at: site, context: context)
        XCTFail("a workspace traversal must be refused")
      } catch {
        // The engine exposes its internal journaled failure wrapper to direct callers.
      }
    }

    let history = try await journal.history(runID: checked.descriptor.id)
    let progress = await engine.progress()
    let refusals = history.events.filter { event in
      guard event.identity?.site == site,
            case .requestResolved(let outcome) = event.payload,
            case .failure(let error) = outcome
      else { return false }
      return error.kind == .sandboxRefusal
    }
    let calls = await executor.requests()

    XCTAssertEqual(refusals.count, 1)
    XCTAssertEqual(calls.count, 0)
    XCTAssertEqual(progress.usage.replayedRequests, 1)
  }

  private func makeCheckedProgram(_ source: String) throws -> WorkflowCheckedProgram {
    let result = WorkflowScriptChecker.check(source: source, name: "Engine test", args: [:])
    return try XCTUnwrap(result.checked, result.diagnostics.map(\.message).joined(separator: "\n"))
  }
}

private actor WorkflowRunEngineTestCapabilities: WorkflowCapabilities {
  enum Mode: Sendable {
    case immediate
    case fail(WorkflowError)
    case holdFirst
    case cancelFirstThenSucceed
    case cancelKindThenSucceed
  }

  private struct CallWaiter {
    var count: Int
    var continuation: CheckedContinuation<Void, Never>
  }

  private var mode: Mode
  private var recordedCalls: [WorkflowInterpreterOperation] = []
  private var callWaiters: [CallWaiter] = []
  private var heldContinuation: CheckedContinuation<WorkflowCapabilityResult, Error>?

  init(mode: Mode = .immediate) {
    self.mode = mode
  }

  func perform(
    _ operation: WorkflowInterpreterOperation,
    at site: WorkflowSiteKey?,
    context: WorkflowAttemptContext
  ) async throws -> WorkflowCapabilityResult {
    let callIndex = recordedCalls.count
    recordedCalls.append(operation)
    let ready = callWaiters.filter { recordedCalls.count >= $0.count }
    callWaiters.removeAll { recordedCalls.count >= $0.count }
    for waiter in ready { waiter.continuation.resume() }

    switch mode {
    case .immediate:
      return result(for: operation, callIndex: callIndex)
    case .fail(let error):
      throw error
    case .holdFirst where callIndex == 0:
      return try await withCheckedThrowingContinuation { continuation in
        heldContinuation = continuation
      }
    case .cancelFirstThenSucceed where callIndex == 0:
      _ = await context.cancellation.waitUntilCancelled()
      throw CancellationError()
    case .cancelKindThenSucceed where callIndex == 0:
      throw WorkflowError(kind: .cancelled, message: "cancelled by capability", site: site)
    case .holdFirst, .cancelFirstThenSucceed:
      return result(for: operation, callIndex: callIndex)
    case .cancelKindThenSucceed:
      return result(for: operation, callIndex: callIndex)
    }
  }

  func calls() -> [WorkflowInterpreterOperation] {
    recordedCalls
  }

  func waitForCallCount(_ count: Int) async {
    guard recordedCalls.count < count else { return }
    await withCheckedContinuation { continuation in
      callWaiters.append(CallWaiter(count: count, continuation: continuation))
    }
  }

  func releaseFirst() {
    heldContinuation?.resume(
      returning: result(
        for: recordedCalls[0],
        callIndex: 0))
    heldContinuation = nil
  }

  private func result(
    for operation: WorkflowInterpreterOperation,
    callIndex: Int
  ) -> WorkflowCapabilityResult {
    let transcript: WorkflowActorTranscriptSnapshot?
    if case .ask = operation {
      transcript = WorkflowActorTranscriptSnapshot(
        payload: .array([
          .object([
            "turn": .integer(Int64(callIndex)),
            "operation": .string(String(describing: operation)),
          ])
        ]))
    } else {
      transcript = nil
    }
    return WorkflowCapabilityResult(
      value: .string("result-\(callIndex)"), actorTranscript: transcript)
  }
}

private actor WorkflowRunEngineMemoryStore: WorkflowJournalStore {
  private var descriptors: [UUID: WorkflowRunDescriptor] = [:]
  private var runStates: [UUID: WorkflowRunState] = [:]
  private var events: [UUID: [WorkflowJournalEvent]] = [:]
  private var outcomes: [UUID: [WorkflowRequestIdentity: WorkflowJournalEvent]] = [:]
  private var transcripts: [UUID: [String: WorkflowJournalActorTranscript]] = [:]

  func createRun(_ descriptor: WorkflowRunDescriptor, state: WorkflowRunState, at date: Date)
    async throws
  {
    guard descriptors[descriptor.id] == nil else {
      throw WorkflowJournalError.runAlreadyExists(descriptor.id)
    }
    descriptors[descriptor.id] = descriptor
    runStates[descriptor.id] = state
    events[descriptor.id] = [
      WorkflowJournalEvent(
        runID: descriptor.id,
        sequence: 1,
        identity: nil,
        payload: .runStarted,
        createdAt: date)
    ]
  }

  func append(
    runID: UUID,
    identity: WorkflowRequestIdentity?,
    payload: WorkflowJournalEventPayload,
    at date: Date
  ) async throws -> WorkflowJournalEvent {
    _ = try requireRun(runID)
    return appendEvent(runID: runID, identity: identity, payload: payload, at: date)
  }

  func resolveRequest(
    runID: UUID,
    identity: WorkflowRequestIdentity,
    outcome: WorkflowJournalOutcome,
    actorName: String?,
    transcript: WorkflowActorTranscriptSnapshot?,
    at date: Date
  ) async throws -> WorkflowJournalEvent {
    _ = try requireRun(runID)
    if let existing = outcomes[runID]?[identity] { return existing }
    if identity.site.lane != "main" {
      guard actorName == identity.site.lane, let transcript else {
        throw WorkflowJournalError.actorTranscriptRequired(identity.site.lane)
      }
      let encoded = try transcript.encodedJSON()
      transcripts[runID, default: [:]][identity.site.lane] = WorkflowJournalActorTranscript(
        actorName: identity.site.lane,
        sha256: try transcript.sha256(),
        encodedSnapshot: encoded,
        snapshot: transcript)
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
    let event = appendEvent(runID: runID, identity: identity, payload: payload, at: date)
    outcomes[runID, default: [:]][identity] = event
    return event
  }

  func recordedOutcome(runID: UUID, identity: WorkflowRequestIdentity) async throws
    -> WorkflowJournalEvent?
  {
    _ = try requireRun(runID)
    return outcomes[runID]?[identity]
  }

  func openQuestion(runID: UUID, questionID: UUID, prompt: String, at date: Date) async throws
    -> WorkflowJournalEvent
  {
    _ = try requireRun(runID)
    let question = WorkflowJournalQuestion(
      id: questionID,
      prompt: prompt,
      state: .pending,
      answer: nil,
      createdAt: date,
      resolvedAt: nil)
    return appendEvent(runID: runID, identity: nil, payload: .questionOpened(question), at: date)
  }

  func resolveQuestion(runID: UUID, questionID: UUID, answer: String?, at date: Date) async throws
    -> WorkflowJournalEvent
  {
    _ = try requireRun(runID)
    let question = WorkflowJournalQuestion(
      id: questionID,
      prompt: "",
      state: answer == nil ? .unanswered : .answered,
      answer: answer,
      createdAt: date,
      resolvedAt: date)
    return appendEvent(runID: runID, identity: nil, payload: .questionResolved(question), at: date)
  }

  func history(runID: UUID) async throws -> WorkflowJournalHistory {
    let descriptor = try requireRun(runID)
    return WorkflowJournalHistory(
      runID: runID,
      state: runStates[runID, default: .running].rawValue,
      canonicalizationVersion: Int64(WorkflowCanonicalSerialization.currentVersion),
      source: descriptor.source,
      sourceHash: descriptor.sourceHash,
      facadeVersion: descriptor.facadeVersion,
      args: descriptor.args.allValues,
      events: events[runID, default: []],
      questions: [],
      actorTranscripts: transcripts[runID, default: [:]].values.sorted {
        $0.actorName < $1.actorName
      })
  }

  private func requireRun(_ runID: UUID) throws -> WorkflowRunDescriptor {
    guard let descriptor = descriptors[runID] else {
      throw WorkflowJournalError.runNotFound(runID)
    }
    return descriptor
  }

  private func appendEvent(
    runID: UUID,
    identity: WorkflowRequestIdentity?,
    payload: WorkflowJournalEventPayload,
    at date: Date
  ) -> WorkflowJournalEvent {
    let event = WorkflowJournalEvent(
      runID: runID,
      sequence: Int64(events[runID, default: []].count + 1),
      identity: identity,
      payload: payload,
      createdAt: date)
    events[runID, default: []].append(event)
    if case .stateChanged(let state) = payload { runStates[runID] = state }
    return event
  }
}
