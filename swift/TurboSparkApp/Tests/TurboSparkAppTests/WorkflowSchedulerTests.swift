import Foundation
import XCTest

@testable import TurboSparkApp

final class WorkflowSchedulerTests: XCTestCase {
  func testCancellationAfterDispatchPersistsOutcomeButDoesNotStartDependentWork() async throws {
    let (descriptor, journal) = try await makeJournal()
    let graph = try graph(nodes: [
      node("first", actor: "writer", siteIndex: 10),
      node("second", actor: "reviewer", siteIndex: 11, dependencies: ["first"]),
    ])
    let dispatcher = WorkflowSchedulerTestDispatcher()
    let gate = WorkflowSchedulerCancellationGate()
    let task = Task {
      try await WorkflowScheduler().runGraph(
        graph,
        joinedAt: graph.joinSite,
        runID: descriptor.id,
        journal: journal,
        maximumConcurrency: 1
      ) { node, dependencies in
        let outcome = try await dispatcher.dispatch(node, dependencies: dependencies)
        if node.id == "first" { await gate.wait() }
        return outcome
      }
    }
    while !(await gate.started) { await Task.yield() }
    task.cancel()
    await gate.release()

    do {
      _ = try await task.value
      XCTFail("Cancellation should prevent the next batch from starting.")
    } catch is CancellationError {}
    let calls = await dispatcher.calls()
    let history = try await journal.history(runID: descriptor.id)
    XCTAssertEqual(calls, ["first"])
    XCTAssertEqual(history.events.filter { $0.payload == .requestStarted }.count, 1)
    XCTAssertEqual(history.events.filter { isResolution($0.payload) }.count, 1)
  }

  func testCancelledGraphDoesNotJournalOrDispatchNewAttempts() async throws {
    let (descriptor, journal) = try await makeJournal()
    let graph = try graph(nodes: [node("first", actor: "writer", siteIndex: 10)])
    let dispatcher = WorkflowSchedulerTestDispatcher()
    let task = Task {
      withUnsafeCurrentTask { $0?.cancel() }
      return try await WorkflowScheduler().runGraph(
        graph,
        joinedAt: graph.joinSite,
        runID: descriptor.id,
        journal: journal,
        maximumConcurrency: 1
      ) { node, dependencies in
        try await dispatcher.dispatch(node, dependencies: dependencies)
      }
    }

    do {
      _ = try await task.value
      XCTFail("Cancelled graph should stop before starting work.")
    } catch is CancellationError {}
    let calls = await dispatcher.calls()
    let history = try await journal.history(runID: descriptor.id)
    XCTAssertTrue(calls.isEmpty)
    XCTAssertFalse(history.events.contains { $0.payload == .requestStarted })
  }

  func testInvalidGraphsAreRejectedBeforeAnyDispatch() async throws {
    let scheduler = WorkflowScheduler(maximumRetriesPerRun: 4)
    let a = node("a", actor: "writer", siteIndex: 0)
    let b = node("b", actor: "writer", siteIndex: 1)

    let duplicateIDs = try graph(nodes: [a, node("a", actor: "writer", siteIndex: 2)])
    let unknownActor = try graph(nodes: [node("a", actor: "unknown", siteIndex: 0)])
    let unknownDependency = try graph(nodes: [
      node("a", actor: "writer", siteIndex: 0, dependencies: ["missing"])
    ])
    let cycle = try graph(nodes: [
      node("a", actor: "writer", siteIndex: 0, dependencies: ["b"]),
      node("b", actor: "writer", siteIndex: 1, dependencies: ["a"]),
    ])
    let invalidRetryBound = try graph(nodes: [
      node("a", actor: "writer", siteIndex: 0, maxRetries: 4)
    ])
    let graphOverLimit = try graph(
      nodes: (0..<101).map {
        node("node-\($0)", actor: "writer", siteIndex: $0)
      })
    let valid = try graph(nodes: [a, b])
    let badIdentity = WorkflowGraph(
      identity: WorkflowRequestIdentity(site: valid.identity.site, inputHash: "not-the-graph-hash"),
      sourceSite: valid.sourceSite,
      joinSite: valid.joinSite,
      declaredActors: valid.declaredActors,
      nodes: valid.nodes)

    for candidate in [
      duplicateIDs, unknownActor, unknownDependency, cycle, invalidRetryBound, graphOverLimit,
      badIdentity,
    ] {
      try await assertRejected(candidate, joinedAt: candidate.joinSite, scheduler: scheduler)
    }
    try await assertRejected(
      valid,
      joinedAt: WorkflowStaticSite(lane: "main", siteIndex: valid.joinSite.siteIndex + 1),
      scheduler: scheduler)
  }

  func testReadyNodesDispatchInDeclarationOrderAndReceiveOnlyDeclaredDependencies() async throws {
    let (descriptor, journal) = try await makeJournal()
    let graph = try graph(nodes: [
      node("first", actor: "writer", siteIndex: 10),
      node("second", actor: "writer", siteIndex: 11, dependencies: ["first"]),
      node("independent", actor: "writer", siteIndex: 12),
    ])
    let dispatcher = WorkflowSchedulerTestDispatcher()
    let scheduler = WorkflowScheduler(maximumRetriesPerRun: 4)

    let outcome = try await scheduler.runGraph(
      graph,
      joinedAt: graph.joinSite,
      runID: descriptor.id,
      journal: journal,
      maximumConcurrency: 1
    ) { node, dependencies in
      try await dispatcher.dispatch(node, dependencies: dependencies)
    }

    let calls = await dispatcher.calls()
    let dependencyCalls = await dispatcher.dependencies(for: "second")
    XCTAssertEqual(calls, ["first", "second", "independent"])
    XCTAssertEqual(
      dependencyCalls, [[WorkflowOutcomeRecord(nodeID: "first", value: .string("first"))]])
    XCTAssertEqual(outcome.nodes["first"]?.state, .succeeded(.string("first")))
    XCTAssertEqual(outcome.nodes["second"]?.state, .succeeded(.string("second")))
    XCTAssertEqual(outcome.nodes["independent"]?.state, .succeeded(.string("independent")))
  }

  func testFailedDependenciesAreBlockedAndReasonsAreRecordedOnce() async throws {
    let (descriptor, journal) = try await makeJournal()
    let graph = try graph(nodes: [
      node("root", actor: "writer", siteIndex: 20),
      node("child", actor: "reviewer", siteIndex: 21, dependencies: ["root"]),
      node("leaf", actor: "writer", siteIndex: 22, dependencies: ["child"]),
    ])
    let dispatcher = WorkflowSchedulerTestDispatcher(failingNodeIDs: ["root"])
    let scheduler = WorkflowScheduler(maximumRetriesPerRun: 4)

    let firstOutcome = try await scheduler.runGraph(
      graph,
      joinedAt: graph.joinSite,
      runID: descriptor.id,
      journal: journal,
      maximumConcurrency: 2
    ) { node, dependencies in
      try await dispatcher.dispatch(node, dependencies: dependencies)
    }
    let firstHistory = try await journal.history(runID: descriptor.id)

    let secondOutcome = try await scheduler.runGraph(
      graph,
      joinedAt: graph.joinSite,
      runID: descriptor.id,
      journal: journal,
      maximumConcurrency: 2
    ) { node, dependencies in
      try await dispatcher.dispatch(node, dependencies: dependencies)
    }
    let secondHistory = try await journal.history(runID: descriptor.id)

    let calls = await dispatcher.calls()
    XCTAssertEqual(calls, ["root"])
    XCTAssertEqual(firstOutcome.nodes["child"]?.state, .blocked("Dependency 'root' failed."))
    XCTAssertEqual(firstOutcome.nodes["leaf"]?.state, .blocked("Dependency 'child' was blocked."))
    XCTAssertEqual(secondOutcome.nodes, firstOutcome.nodes)
    XCTAssertEqual(blockedReasonEvents(firstHistory).count, 2)
    XCTAssertEqual(blockedReasonEvents(secondHistory).count, 2)
  }

  func testPerNodeRetryBoundJournalsTheInitialAttemptAndEachRetry() async throws {
    let (descriptor, journal) = try await makeJournal()
    let graph = try graph(nodes: [node("flaky", actor: "writer", siteIndex: 30, maxRetries: 1)])
    let dispatcher = WorkflowSchedulerTestDispatcher(failingNodeIDs: ["flaky"])
    let scheduler = WorkflowScheduler(maximumRetriesPerRun: 10)

    let outcome = try await scheduler.runGraph(
      graph,
      joinedAt: graph.joinSite,
      runID: descriptor.id,
      journal: journal,
      maximumConcurrency: 1
    ) { node, dependencies in
      try await dispatcher.dispatch(node, dependencies: dependencies)
    }

    let calls = await dispatcher.calls()
    let history = try await journal.history(runID: descriptor.id)
    let attemptEvents = history.events.filter {
      $0.identity?.site.lane == "main"
        && $0.identity?.site.siteIndex == 30
        && ($0.identity?.site.ordinal ?? Int.max) < Int.max - 1
        && ($0.payload == .requestStarted || isResolution($0.payload))
    }
    XCTAssertEqual(calls, ["flaky", "flaky"])
    XCTAssertEqual(outcome.nodes["flaky"]?.attempts, 2)
    XCTAssertEqual(
      Set(attemptEvents.compactMap { $0.identity?.site.ordinal }),
      Set([0, 1]))
    XCTAssertEqual(attemptEvents.filter { $0.payload == .requestStarted }.count, 2)
    XCTAssertEqual(attemptEvents.filter { isResolution($0.payload) }.count, 2)
  }

  func testRunWideRetryCeilingBoundsRetriesAcrossNodes() async throws {
    let (descriptor, journal) = try await makeJournal()
    let graph = try graph(nodes: [
      node("first", actor: "writer", siteIndex: 40, maxRetries: 2),
      node("second", actor: "reviewer", siteIndex: 41, maxRetries: 2),
    ])
    let dispatcher = WorkflowSchedulerTestDispatcher(failingNodeIDs: ["first", "second"])
    let scheduler = WorkflowScheduler(maximumRetriesPerRun: 1)

    let outcome = try await scheduler.runGraph(
      graph,
      joinedAt: graph.joinSite,
      runID: descriptor.id,
      journal: journal,
      maximumConcurrency: 1
    ) { node, dependencies in
      try await dispatcher.dispatch(node, dependencies: dependencies)
    }

    let calls = await dispatcher.calls()
    let history = try await journal.history(runID: descriptor.id)
    let attemptStarts = history.events.filter {
      $0.identity?.site.lane == "main"
        && [40, 41].contains($0.identity?.site.siteIndex)
        && ($0.identity?.site.ordinal ?? Int.max) < Int.max - 1
        && $0.payload == .requestStarted
    }
    XCTAssertEqual(calls, ["first", "first", "second"])
    XCTAssertEqual(outcome.nodes["first"]?.attempts, 2)
    XCTAssertEqual(outcome.nodes["second"]?.attempts, 1)
    XCTAssertEqual(outcome.nodes["first"]?.state.errorKind, .resourceLimit)
    XCTAssertEqual(outcome.nodes["second"]?.state.errorKind, .resourceLimit)
    XCTAssertEqual(attemptStarts.count, 3)
  }

  private func assertRejected(
    _ graph: WorkflowGraph,
    joinedAt: WorkflowStaticSite,
    scheduler: WorkflowScheduler
  ) async throws {
    let (descriptor, journal) = try await makeJournal()
    let dispatcher = WorkflowSchedulerTestDispatcher()
    do {
      _ = try await scheduler.runGraph(
        graph,
        joinedAt: joinedAt,
        runID: descriptor.id,
        journal: journal,
        maximumConcurrency: 2
      ) { node, dependencies in
        try await dispatcher.dispatch(node, dependencies: dependencies)
      }
      XCTFail("Invalid graph should be rejected before dispatch.")
    } catch let error as WorkflowError {
      XCTAssertEqual(error.kind, .validation)
    }
    let calls = await dispatcher.calls()
    let history = try await journal.history(runID: descriptor.id)
    XCTAssertTrue(calls.isEmpty)
    XCTAssertFalse(history.events.contains { $0.payload == .requestStarted })
  }

  private func makeJournal() async throws -> (WorkflowRunDescriptor, WorkflowJournal) {
    let descriptor = WorkflowRunDescriptor(
      id: UUID(),
      name: "scheduler-test",
      source: "async function workflow() {}",
      sourceHash: "scheduler-test-hash",
      facadeVersion: WorkflowFacade.version,
      args: WorkflowFrozenArguments([:]),
      manifest: WorkflowLaunchManifest())
    let store = WorkflowSchedulerMemoryStore()
    let journal = WorkflowJournal(store: store)
    try await journal.createRun(descriptor)
    return (descriptor, journal)
  }

  private func graph(
    nodes: [WorkflowNode],
    actors: Set<String> = ["writer", "reviewer"],
    sourceIndex: Int = 90,
    joinIndex: Int = 91
  ) throws -> WorkflowGraph {
    try WorkflowGraph.make(
      sourceSite: WorkflowStaticSite(lane: "main", siteIndex: sourceIndex),
      joinSite: WorkflowStaticSite(lane: "main", siteIndex: joinIndex),
      declaredActors: actors,
      nodes: nodes)
  }

  private func node(
    _ id: String,
    actor: String,
    siteIndex: Int,
    dependencies: [String] = [],
    maxRetries: Int = 0
  ) -> WorkflowNode {
    WorkflowNode(
      id: id,
      actor: actor,
      prompt: .string("prompt-\(id)"),
      shape: WorkflowResultShape(fields: []),
      dependsOn: dependencies,
      maxRetries: maxRetries,
      siteIndex: siteIndex)
  }

  private func blockedReasonEvents(_ history: WorkflowJournalHistory) -> [WorkflowJournalEvent] {
    history.events.filter { event in
      guard case .requestResolved(.failure(let error)) = event.payload else { return false }
      return error.message.hasPrefix("Blocked node '")
    }
  }

  private func isResolution(_ payload: WorkflowJournalEventPayload) -> Bool {
    switch payload {
    case .requestResolved, .requestResolvedWithActorTranscript: return true
    default: return false
    }
  }
}

extension WorkflowNodeOutcome.State {
  fileprivate var errorKind: WorkflowErrorKind? {
    guard case .failed(let error) = self else { return nil }
    return error.kind
  }
}

private actor WorkflowSchedulerCancellationGate {
  private var continuation: CheckedContinuation<Void, Never>?
  private(set) var started = false

  func wait() async {
    await withCheckedContinuation { continuation in
      self.continuation = continuation
      started = true
    }
  }

  func release() {
    continuation?.resume()
    continuation = nil
  }
}

private actor WorkflowSchedulerTestDispatcher {
  private let failingNodeIDs: Set<String>
  private var dispatched: [String] = []
  private var recordedDependencies: [String: [[WorkflowOutcomeRecord]]] = [:]

  init(failingNodeIDs: Set<String> = []) {
    self.failingNodeIDs = failingNodeIDs
  }

  func dispatch(
    _ node: WorkflowNode,
    dependencies: [WorkflowOutcomeRecord]
  ) throws -> WorkflowOutcomeRecord {
    dispatched.append(node.id)
    recordedDependencies[node.id, default: []].append(dependencies)
    if failingNodeIDs.contains(node.id) {
      throw WorkflowError(
        kind: .modelFailure,
        message: "Synthetic failure for \(node.id).",
        site: WorkflowSiteKey(
          lane: node.actor, siteIndex: node.siteIndex, ordinal: dispatched.count - 1))
    }
    return WorkflowOutcomeRecord(nodeID: node.id, value: .string(node.id))
  }

  func calls() -> [String] {
    dispatched
  }

  func dependencies(for nodeID: String) -> [[WorkflowOutcomeRecord]] {
    recordedDependencies[nodeID, default: []]
  }
}

private actor WorkflowSchedulerMemoryStore: WorkflowJournalStore {
  private var descriptors: [UUID: WorkflowRunDescriptor] = [:]
  private var states: [UUID: WorkflowRunState] = [:]
  private var events: [UUID: [WorkflowJournalEvent]] = [:]
  private var outcomes: [UUID: [WorkflowRequestIdentity: WorkflowJournalEvent]] = [:]

  func createRun(_ descriptor: WorkflowRunDescriptor, state: WorkflowRunState, at date: Date)
    async throws
  {
    guard descriptors[descriptor.id] == nil else {
      throw WorkflowJournalError.runAlreadyExists(descriptor.id)
    }
    descriptors[descriptor.id] = descriptor
    states[descriptor.id] = state
    events[descriptor.id] = [
      event(runID: descriptor.id, sequence: 1, identity: nil, payload: .runStarted, at: date)
    ]
  }

  func append(
    runID: UUID,
    identity: WorkflowRequestIdentity?,
    payload: WorkflowJournalEventPayload,
    at date: Date
  ) async throws -> WorkflowJournalEvent {
    _ = try requireDescriptor(runID)
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
    _ = try requireDescriptor(runID)
    if let existing = outcomes[runID]?[identity] { return existing }
    guard identity.site.lane == "main", actorName == nil, transcript == nil else {
      throw WorkflowJournalError.actorTranscriptLaneMismatch(
        expected: "main",
        actual: identity.site.lane)
    }
    let resolved = appendEvent(
      runID: runID,
      identity: identity,
      payload: .requestResolved(outcome),
      at: date)
    outcomes[runID, default: [:]][identity] = resolved
    return resolved
  }

  func recordedOutcome(runID: UUID, identity: WorkflowRequestIdentity) async throws
    -> WorkflowJournalEvent?
  {
    _ = try requireDescriptor(runID)
    return outcomes[runID]?[identity]
  }

  func openQuestion(runID: UUID, questionID: UUID, prompt: String, at date: Date)
    async throws -> WorkflowJournalEvent
  {
    let question = WorkflowJournalQuestion(
      id: questionID,
      prompt: prompt,
      state: .pending,
      answer: nil,
      createdAt: date,
      resolvedAt: nil)
    return try await append(
      runID: runID, identity: nil, payload: .questionOpened(question), at: date)
  }

  func resolveQuestion(runID: UUID, questionID: UUID, answer: String?, at date: Date)
    async throws -> WorkflowJournalEvent
  {
    let question = WorkflowJournalQuestion(
      id: questionID,
      prompt: "",
      state: answer == nil ? .unanswered : .answered,
      answer: answer,
      createdAt: date,
      resolvedAt: date)
    return try await append(
      runID: runID, identity: nil, payload: .questionResolved(question), at: date)
  }

  func history(runID: UUID) async throws -> WorkflowJournalHistory {
    let descriptor = try requireDescriptor(runID)
    return WorkflowJournalHistory(
      runID: runID,
      state: states[runID, default: .running].rawValue,
      canonicalizationVersion: Int64(WorkflowCanonicalSerialization.currentVersion),
      source: descriptor.source,
      sourceHash: descriptor.sourceHash,
      facadeVersion: descriptor.facadeVersion,
      args: descriptor.args.allValues,
      events: events[runID, default: []],
      questions: [],
      actorTranscripts: [])
  }

  private func requireDescriptor(_ runID: UUID) throws -> WorkflowRunDescriptor {
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
    let sequence = Int64(events[runID, default: []].count + 1)
    let appended = event(
      runID: runID, sequence: sequence, identity: identity, payload: payload, at: date)
    events[runID, default: []].append(appended)
    if case .stateChanged(let state) = payload {
      states[runID] = state
    }
    return appended
  }

  private func event(
    runID: UUID,
    sequence: Int64,
    identity: WorkflowRequestIdentity?,
    payload: WorkflowJournalEventPayload,
    at date: Date
  ) -> WorkflowJournalEvent {
    WorkflowJournalEvent(
      runID: runID,
      sequence: sequence,
      identity: identity,
      payload: payload,
      createdAt: date)
  }
}
