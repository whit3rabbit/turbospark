import Foundation

struct WorkflowEngineLimits: Sendable {
  var interpreter: WorkflowInterpreterLimits

  init(interpreter: WorkflowInterpreterLimits = WorkflowInterpreterLimits()) {
    self.interpreter = interpreter
  }
}

struct WorkflowCapabilityResult: Sendable {
  var value: WorkflowCanonicalValue
  var actorTranscript: WorkflowActorTranscriptSnapshot?

  init(value: WorkflowCanonicalValue, actorTranscript: WorkflowActorTranscriptSnapshot? = nil) {
    self.value = value
    self.actorTranscript = actorTranscript
  }
}

protocol WorkflowCapabilities: Sendable {
  func perform(
    _ operation: WorkflowInterpreterOperation,
    at site: WorkflowSiteKey?,
    context: WorkflowAttemptContext
  ) async throws -> WorkflowCapabilityResult
}

struct WorkflowLaneProgress: Equatable, Sendable {
  var name: String
  var runningCount: Int
  var finishedCount: Int
  var queuedCount: Int
}

struct WorkflowUsageSnapshot: Equatable, Sendable {
  var totalRequests: Int
  var replayedRequests: Int
  var liveRequests: Int
}

struct WorkflowRunProgress: Equatable, Sendable {
  var state: WorkflowRunState
  var currentPhase: String?
  var lanes: [WorkflowLaneProgress]
  var usage: WorkflowUsageSnapshot
}

private struct WorkflowRunEngineFailure: Error {
  var workflowError: WorkflowError
}

private actor WorkflowRunRequestLane {
  private struct Waiter {
    var continuation: CheckedContinuation<Bool, Never>
  }

  private var occupied = false
  private var waiters: [Waiter] = []

  var queuedCount: Int { waiters.count }

  func acquire() async -> Bool {
    guard occupied else {
      occupied = true
      return true
    }
    return await withCheckedContinuation { continuation in
      waiters.append(Waiter(continuation: continuation))
    }
  }

  func release() {
    guard occupied else { return }
    guard !waiters.isEmpty else {
      occupied = false
      return
    }
    waiters.removeFirst().continuation.resume(returning: true)
  }

  func cancelQueued() {
    let pending = waiters
    waiters.removeAll()
    for waiter in pending {
      waiter.continuation.resume(returning: false)
    }
  }
}

/// Serializes each individual append while allowing execution in different lanes to overlap.
private actor WorkflowJournalAppendSerialiser {
  private var tail: Task<Void, Never>?

  func run<Value: Sendable>(
    _ operation: @escaping @Sendable () async throws -> Value
  ) async throws -> Value {
    let predecessor = tail
    let current = Task<Value, Error> {
      await predecessor?.value
      return try await operation()
    }
    tail = Task {
      _ = await current.result
    }
    return try await current.value
  }
}

/// Executes a checked workflow, consulting its journal before each capability call.
actor WorkflowRunEngine: WorkflowCommandSink {
  private static let initialActorTranscriptHash = "initial"
  private static let cancellationSiteIndex = WorkflowJournalSystemSites.cancellation
  private static let runFailureSiteIndex = WorkflowJournalSystemSites.runFailure
  private static let phaseSiteIndex = WorkflowJournalSystemSites.phase

  private let program: WorkflowCheckedProgram
  private let journal: WorkflowJournal
  private let capabilities: any WorkflowCapabilities
  private let limits: WorkflowEngineLimits
  private let appender = WorkflowJournalAppendSerialiser()

  private var interpreter: WorkflowInterpreter?
  private var lanes: [String: WorkflowRunRequestLane] = [:]
  private var laneProgress: [String: WorkflowLaneProgress] = [:]
  private var runState: WorkflowRunState = .running
  private var currentPhase: String?
  private var totalRequests = 0
  private var replayedRequests = 0
  private var liveRequests = 0
  private var phaseOrdinal = 0
  private var activeStart = false
  private var cancelBeforeStart = false
  private var recordedRunFailure: WorkflowError?

  init(
    program: WorkflowCheckedProgram,
    journal: WorkflowJournal,
    capabilities: any WorkflowCapabilities,
    limits: WorkflowEngineLimits = WorkflowEngineLimits()
  ) {
    self.program = program
    self.journal = journal
    self.capabilities = capabilities
    self.limits = limits
  }

  /// Runs a checked program or resumes it from its last journaled boundary.
  func start() async {
    guard !activeStart else { return }
    activeStart = true
    recordedRunFailure = nil
    defer {
      activeStart = false
      recordedRunFailure = nil
    }

    // A refused resume (newer canonicalization or transcript version, a
    // mismatched descriptor, a corrupt journal) is a refusal, not a run
    // failure. Nothing has executed, so nothing is journaled: the recorded
    // run keeps its state and can be resumed by a build that understands it.
    let snapshot: WorkflowJournalResumeSnapshot
    do {
      snapshot = try await journal.resumeSnapshot(runID: program.descriptor.id)
      try validateResume(snapshot.history)
    } catch {
      runState = .failed
      return
    }

    do {
      let persistedState =
        snapshot.history.events.reversed().compactMap { event -> WorkflowRunState? in
          guard case .stateChanged(let state) = event.payload else { return nil }
          return state
        }.first ?? WorkflowRunState(rawValue: snapshot.history.state) ?? .running
      switch persistedState {
      case .completed, .failed, .declined, .superseded:
        runState = persistedState
        return
      case .checking, .awaitingApproval:
        runState = persistedState
        return
      case .cancelled, .interrupted:
        try await appendState(.running)
        runState = .running
      case .running:
        runState = .running
      }

      if interpreter == nil {
        interpreter = WorkflowInterpreter(limits: limits.interpreter, engine: self)
      }
      interpreter?.clearPendingCancel()
      if cancelBeforeStart {
        cancelBeforeStart = false
        interpreter?.requestCancel()
      }
      phaseOrdinal = 0
      try await interpreter?.execute(program: program)
      try await appendState(.completed)
      runState = .completed
    } catch {
      let failure = recordedRunFailure ?? normalized(error, site: nil)
      if recordedRunFailure == nil {
        do {
          try await recordRunFailure(failure)
        } catch {
          // A failed journal write cannot be repaired by another append.
        }
      }
      let terminalState: WorkflowRunState = failure.kind == .cancelled ? .cancelled : .failed
      do {
        try await appendState(terminalState)
      } catch {
        // Keep the observed in-memory state even when persistence is unavailable.
      }
      runState = terminalState
    }
  }

  /// Requests cooperative cancellation. The active await owns the cancellation boundary.
  func cancel() async {
    if let interpreter {
      // Only arm the interpreter while start() is running. A Stop pressed
      // after the run ended would otherwise leave the pre-start flag set and
      // silently cancel the next resume.
      if activeStart {
        interpreter.requestCancel()
      }
    } else {
      cancelBeforeStart = true
    }
    let requestLanes = Array(lanes.values)
    for lane in requestLanes {
      await lane.cancelQueued()
    }
  }

  func progress() async -> WorkflowRunProgress {
    var snapshots: [WorkflowLaneProgress] = []
    for name in laneProgress.keys.sorted() {
      guard let lane = lanes[name], var snapshot = laneProgress[name] else { continue }
      snapshot.queuedCount = await lane.queuedCount
      snapshots.append(snapshot)
    }
    return WorkflowRunProgress(
      state: runState,
      currentPhase: currentPhase,
      lanes: snapshots,
      usage: WorkflowUsageSnapshot(
        totalRequests: totalRequests,
        replayedRequests: replayedRequests,
        liveRequests: liveRequests))
  }

  func perform(
    _ operation: WorkflowInterpreterOperation,
    at site: WorkflowSiteKey?,
    context: WorkflowAttemptContext
  ) async throws -> WorkflowCanonicalValue {
    let laneName: String
    let isPhase: Bool
    if case .ask(let actor, _, _) = operation {
      laneName = actor
      isPhase = false
      guard site?.lane == actor else {
        throw WorkflowError(
          kind: .validation,
          message: "Actor request site does not match its actor lane.",
          site: site)
      }
    } else if case .phase = operation {
      laneName = "main"
      isPhase = true
      if let site, site.lane != "main" {
        throw WorkflowError(
          kind: .validation,
          message: "Entry phase must use the main lane.",
          site: site)
      }
    } else {
      laneName = "main"
      isPhase = false
      guard site?.lane == "main" else {
        throw WorkflowError(
          kind: .validation,
          message: "Entry request must use the main lane.",
          site: site)
      }
    }
    guard isPhase || site != nil else {
      throw WorkflowError(
        kind: .validation,
        message: "Every workflow request requires a checked site key.",
        site: nil)
    }

    let lane = requestLane(named: laneName)
    guard await lane.acquire() else {
      throw CancellationError()
    }
    var progress =
      laneProgress[laneName]
      ?? WorkflowLaneProgress(
        name: laneName,
        runningCount: 0,
        finishedCount: 0,
        queuedCount: 0)
    progress.runningCount += 1
    laneProgress[laneName] = progress

    do {
      let value = try await executeJournaled(
        operation,
        site: site,
        laneName: laneName,
        context: context)
      finishLane(laneName)
      await lane.release()
      return value
    } catch {
      finishLane(laneName)
      await lane.release()
      throw error
    }
  }

  private func executeJournaled(
    _ operation: WorkflowInterpreterOperation,
    site: WorkflowSiteKey?,
    laneName: String,
    context: WorkflowAttemptContext
  ) async throws -> WorkflowCanonicalValue {
    let history = try await journal.history(runID: program.descriptor.id)
    if case .phase(let name) = operation {
      let phaseSite = WorkflowSiteKey(
        lane: "main",
        siteIndex: Self.phaseSiteIndex,
        ordinal: phaseOrdinal)
      phaseOrdinal += 1
      let phaseIdentity = try WorkflowRequestIdentity.make(
        site: phaseSite,
        input: canonicalInput(for: operation))
      if let prior = history.events.first(where: {
        $0.identity?.site == phaseSite && $0.payload.kind == .phaseEntered
      }) {
        guard prior.identity == phaseIdentity, prior.payload == .phaseEntered(name) else {
          throw WorkflowError(
            kind: .validation,
            message: "Recorded phase does not match this checked workflow site.",
            site: phaseSite)
        }
      } else {
        _ = try await appendJournal(identity: phaseIdentity, payload: .phaseEntered(name))
      }
      currentPhase = name
      return .null
    }
    guard let site else {
      throw WorkflowError(
        kind: .validation,
        message: "Every workflow request requires a checked site key.",
        site: nil)
    }
    let input = canonicalInput(for: operation)
    let priorActorHash: String?
    if case .ask(let actor, _, _) = operation {
      priorActorHash = priorTranscriptHash(actor: actor, site: site, history: history)
    } else {
      priorActorHash = nil
    }
    let identity: WorkflowRequestIdentity
    do {
      identity = try makeIdentity(site: site, input: input, priorActorHash: priorActorHash)
    } catch {
      throw normalized(error, site: site)
    }

    if let prior = history.events.first(where: {
      $0.identity?.site == site && resolvedOutcome(in: $0.payload) != nil
    }), prior.identity != identity {
      throw WorkflowError(
        kind: .validation,
        message: "Recorded request inputs do not match this checked workflow site.",
        site: site)
    }
    if let priorStart = history.events.last(where: {
      $0.identity?.site == site && $0.payload == .requestStarted
    }), priorStart.identity != identity {
      throw WorkflowError(
        kind: .validation,
        message: "An unfinished request identity does not match this checked workflow site.",
        site: site)
    }

    if let event = try await journal.recordedOutcome(
      runID: program.descriptor.id, identity: identity),
      let outcome = resolvedOutcome(in: event.payload)
    {
      replayedRequests += 1
      switch outcome {
      case .value(let value):
        return value
      case .failure(let error):
        recordedRunFailure = error
        throw WorkflowRunEngineFailure(workflowError: error)
      }
    }

    let unresolvedStart = history.events.last(where: {
      $0.identity == identity && $0.payload == .requestStarted
    })
    if let unresolvedStart,
      !history.events.contains(where: {
        $0.identity == identity && resolvedOutcome(in: $0.payload) != nil
      }),
      !hasCancellationAfter(unresolvedStart.sequence, history: history)
    {
      throw WorkflowError(
        kind: .validation,
        message: "A request started without a recorded outcome; refusing to repeat uncertain work.",
        site: site)
    }

    liveRequests += 1
    _ = try await appendBeginRequest(identity)
    if context.cancellation.isCancelled {
      let error = normalized(CancellationError(), site: site)
      try await recordCooperativeCancellation(error)
      recordedRunFailure = error
      throw WorkflowRunEngineFailure(workflowError: error)
    }

    do {
      let result = try await capabilities.perform(operation, at: site, context: context)
      if case .ask(let actor, _, _) = operation {
        guard let transcript = result.actorTranscript else {
          let error = WorkflowError(
            kind: .modelFailure,
            message: "Actor capability completed without a replay transcript.",
            site: site)
          _ = try await appendResolution(
            identity,
            outcome: .failure(error),
            actor: actor,
            transcript: priorTranscript(for: actor, history: history))
          recordedRunFailure = error
          throw WorkflowRunEngineFailure(workflowError: error)
        }
        _ = try await appendResolution(
          identity,
          outcome: .value(result.value),
          actor: actor,
          transcript: transcript)
      } else {
        guard result.actorTranscript == nil else {
          let error = WorkflowError(
            kind: .validation,
            message: "Only actor requests may update an actor transcript.",
            site: site)
          _ = try await appendResolution(
            identity, outcome: .failure(error), actor: nil, transcript: nil)
          recordedRunFailure = error
          throw WorkflowRunEngineFailure(workflowError: error)
        }
        _ = try await appendResolution(
          identity,
          outcome: .value(result.value),
          actor: nil,
          transcript: nil)
      }
      return result.value
    } catch let failure as WorkflowRunEngineFailure {
      throw failure
    } catch {
      let failure = normalized(error, site: site)
      if failure.kind == .cancelled {
        try await recordCooperativeCancellation(failure)
      } else {
        let actor: String?
        let transcript: WorkflowActorTranscriptSnapshot?
        if case .ask(let name, _, _) = operation {
          actor = name
          transcript = priorTranscript(for: name, history: history)
        } else {
          actor = nil
          transcript = nil
        }
        _ = try await appendResolution(
          identity,
          outcome: .failure(failure),
          actor: actor,
          transcript: transcript)
      }
      recordedRunFailure = failure
      throw WorkflowRunEngineFailure(workflowError: failure)
    }
  }

  private func validateResume(_ history: WorkflowJournalHistory) throws {
    let descriptor = program.descriptor
    guard history.runID == descriptor.id,
      history.events.first?.payload == .runStarted
    else {
      throw WorkflowError(
        kind: .validation,
        message: "Workflow journal is missing its matching run-start event.",
        site: nil)
    }
    guard history.source == descriptor.source,
      history.sourceHash == descriptor.sourceHash,
      history.facadeVersion == descriptor.facadeVersion,
      history.args == descriptor.args.allValues
    else {
      throw WorkflowError(
        kind: .validation,
        message: "Checked workflow descriptor does not match the recorded run.",
        site: nil)
    }
  }

  private func makeIdentity(
    site: WorkflowSiteKey,
    input: WorkflowCanonicalValue,
    priorActorHash: String?
  ) throws -> WorkflowRequestIdentity {
    if let priorActorHash {
      return try WorkflowRequestIdentity.make(
        site: site,
        input: input,
        priorActorTranscriptHash: priorActorHash)
    }
    return try WorkflowRequestIdentity.make(site: site, input: input)
  }

  private func priorTranscriptHash(
    actor: String,
    site: WorkflowSiteKey,
    history: WorkflowJournalHistory
  ) -> String {
    let matchingEvent = history.events.first(where: {
      $0.identity?.site == site && resolvedOutcome(in: $0.payload) != nil
    })
    let matchingStart = history.events.last(where: {
      $0.identity?.site == site && $0.payload == .requestStarted
    })
    let boundary = matchingEvent?.sequence ?? matchingStart?.sequence
    let candidates = history.events.filter { event in
      guard event.sequence < (boundary ?? Int64.max), event.identity?.site.lane == actor else {
        return false
      }
      return actorTranscriptHash(in: event.payload) != nil
    }
    if let prior = candidates.last, let hash = actorTranscriptHash(in: prior.payload) {
      return hash
    }
    if boundary == nil,
      let latest = history.actorTranscripts.first(where: { $0.actorName == actor })?.sha256
    {
      return latest
    }
    return Self.initialActorTranscriptHash
  }

  private func priorTranscript(for actor: String, history: WorkflowJournalHistory)
    -> WorkflowActorTranscriptSnapshot
  {
    if let latest = history.actorTranscripts.first(where: { $0.actorName == actor })?.snapshot {
      return latest
    }
    return WorkflowActorTranscriptSnapshot(payload: .array([]))
  }

  private func hasCancellationAfter(_ sequence: Int64, history: WorkflowJournalHistory) -> Bool {
    history.events.contains { event in
      guard event.sequence > sequence,
        event.identity?.site.lane == "main",
        event.identity?.site.siteIndex == Self.cancellationSiteIndex
      else { return false }
      guard case .requestResolved(.failure(let error)) = event.payload else { return false }
      return error.kind == .cancelled
    }
  }

  private func recordCooperativeCancellation(_ error: WorkflowError) async throws {
    let history = try await journal.history(runID: program.descriptor.id)
    let ordinal = history.events.filter { event in
      event.identity?.site.lane == "main"
        && event.identity?.site.siteIndex == Self.cancellationSiteIndex
        && event.payload == .requestStarted
    }.count
    let site = WorkflowSiteKey(
      lane: "main",
      siteIndex: Self.cancellationSiteIndex,
      ordinal: ordinal)
    let identity = try WorkflowRequestIdentity.make(
      site: site,
      input: .object(["kind": .string("cooperative-cancellation")]))
    _ = try await appendBeginRequest(identity)
    _ = try await appendResolution(
      identity,
      outcome: .failure(error),
      actor: nil,
      transcript: nil)
  }

  private func recordRunFailure(_ error: WorkflowError) async throws {
    let history = try await journal.history(runID: program.descriptor.id)
    let ordinal = history.events.filter { event in
      event.identity?.site.lane == "main"
        && event.identity?.site.siteIndex == Self.runFailureSiteIndex
        && event.payload == .requestStarted
    }.count
    let site = WorkflowSiteKey(
      lane: "main",
      siteIndex: Self.runFailureSiteIndex,
      ordinal: ordinal)
    let identity = try WorkflowRequestIdentity.make(
      site: site,
      input: .object([
        "kind": .string("engine-failure"),
        "errorKind": .string(error.kind.rawValue),
        "message": .string(error.message),
      ]))
    _ = try await appendBeginRequest(identity)
    _ = try await appendResolution(
      identity,
      outcome: .failure(error),
      actor: nil,
      transcript: nil)
    recordedRunFailure = error
  }

  private func appendState(_ state: WorkflowRunState) async throws {
    _ = try await appendJournal(identity: nil, payload: .stateChanged(state))
  }

  private func appendBeginRequest(_ identity: WorkflowRequestIdentity) async throws
    -> WorkflowJournalEvent
  {
    let journal = self.journal
    let runID = program.descriptor.id
    return try await appender.run {
      try await journal.beginRequest(runID: runID, identity: identity)
    }
  }

  private func appendResolution(
    _ identity: WorkflowRequestIdentity,
    outcome: WorkflowJournalOutcome,
    actor: String?,
    transcript: WorkflowActorTranscriptSnapshot?
  ) async throws -> WorkflowJournalEvent {
    let journal = self.journal
    let runID = program.descriptor.id
    return try await appender.run {
      try await journal.resolveRequest(
        runID: runID,
        identity: identity,
        outcome: outcome,
        actorName: actor,
        transcript: transcript)
    }
  }

  private func appendJournal(
    identity: WorkflowRequestIdentity?,
    payload: WorkflowJournalEventPayload
  ) async throws -> WorkflowJournalEvent {
    let journal = self.journal
    let runID = program.descriptor.id
    return try await appender.run {
      try await journal.append(runID: runID, identity: identity, payload: payload)
    }
  }

  private func requestLane(named name: String) -> WorkflowRunRequestLane {
    if let lane = lanes[name] { return lane }
    let lane = WorkflowRunRequestLane()
    lanes[name] = lane
    laneProgress[name] = WorkflowLaneProgress(
      name: name,
      runningCount: 0,
      finishedCount: 0,
      queuedCount: 0)
    return lane
  }

  private func finishLane(_ name: String) {
    guard var progress = laneProgress[name] else { return }
    progress.runningCount = max(0, progress.runningCount - 1)
    progress.finishedCount += 1
    laneProgress[name] = progress
    totalRequests += 1
  }

  private func normalized(_ error: Error, site: WorkflowSiteKey?) -> WorkflowError {
    if let failure = error as? WorkflowRunEngineFailure { return failure.workflowError }
    if let error = error as? WorkflowError {
      return WorkflowError(kind: error.kind, message: error.message, site: error.site ?? site)
    }
    if error is CancellationError {
      return WorkflowError(
        kind: .cancelled,
        message: "Workflow run was cancelled.",
        site: site)
    }
    if let error = error as? WorkflowJournalError {
      switch error {
      case .unsupportedCanonicalizationVersion, .unsupportedTranscriptVersion,
        .unreadableTranscript, .transcriptHashMismatch, .actorTranscriptHistoryMismatch,
        .actorTranscriptEventMissing, .malformedHistory:
        return WorkflowError(kind: .validation, message: error.localizedDescription, site: site)
      default:
        return WorkflowError(kind: .modelFailure, message: error.localizedDescription, site: site)
      }
    }
    if let error = error as? WorkflowCanonicalSerializationError {
      return WorkflowError(kind: .validation, message: String(describing: error), site: site)
    }
    if let error = error as? WorkflowSiteSequenceError {
      return WorkflowError(kind: .resourceLimit, message: String(describing: error), site: site)
    }
    return WorkflowError(kind: .modelFailure, message: error.localizedDescription, site: site)
  }

  private func resolvedOutcome(in payload: WorkflowJournalEventPayload) -> WorkflowJournalOutcome? {
    switch payload {
    case .requestResolved(let outcome),
      .requestResolvedWithActorTranscript(let outcome, _, _, _):
      return outcome
    default:
      return nil
    }
  }

  private func actorTranscriptHash(in payload: WorkflowJournalEventPayload) -> String? {
    guard case .requestResolvedWithActorTranscript(_, _, _, let hash) = payload else { return nil }
    return hash
  }

  private func canonicalInput(for operation: WorkflowInterpreterOperation) -> WorkflowCanonicalValue
  {
    switch operation {
    case .ask(let actor, let prompt, let shape):
      return .object([
        "kind": .string("ask"), "actor": .string(actor), "prompt": prompt, "shape": shape,
      ])
    case .join(let graph):
      return .object(["kind": .string("join"), "graph": graph])
    case .criticLoop(let policy):
      return .object(["kind": .string("criticLoop"), "policy": policy])
    case .worldRead(let operation):
      return .object(["kind": .string("worldRead"), "operation": operation])
    case .run(let commandKey, let values):
      return .object([
        "kind": .string("run"),
        "commandKey": .string(commandKey),
        "values": .object(values),
      ])
    case .report(let value):
      return .object(["kind": .string("report"), "value": value])
    case .artifact(let value):
      return .object(["kind": .string("artifact"), "value": value])
    case .phase(let name):
      return .object(["kind": .string("phase"), "name": .string(name)])
    }
  }
}
