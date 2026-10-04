import Foundation

struct WorkflowNode: Sendable, Equatable {
  var id: String
  var actor: String
  var prompt: WorkflowCanonicalValue
  var shape: WorkflowResultShape
  var dependsOn: [String]
  var maxRetries: Int
  var siteIndex: Int
}

struct WorkflowOutcomeRecord: Sendable, Equatable {
  var nodeID: String
  var value: WorkflowCanonicalValue
}

struct WorkflowNodeOutcome: Sendable, Equatable {
  enum State: Sendable, Equatable {
    case succeeded(WorkflowCanonicalValue)
    case failed(WorkflowError)
    case blocked(String)
  }

  var state: State
  var attempts: Int
}

struct WorkflowGraphOutcome: Sendable, Equatable {
  var identity: WorkflowRequestIdentity
  var nodes: [String: WorkflowNodeOutcome]
}

struct WorkflowGraph: Sendable, Equatable {
  var identity: WorkflowRequestIdentity
  var sourceSite: WorkflowStaticSite
  var joinSite: WorkflowStaticSite
  var declaredActors: Set<String>
  var nodes: [WorkflowNode]

  static func make(
    sourceSite: WorkflowStaticSite,
    joinSite: WorkflowStaticSite,
    declaredActors: Set<String>,
    nodes: [WorkflowNode]
  ) throws -> WorkflowGraph {
    let input = graphInput(
      sourceSite: sourceSite,
      joinSite: joinSite,
      declaredActors: declaredActors,
      nodes: nodes)
    let identity = try WorkflowRequestIdentity.make(
      site: WorkflowSiteKey(site: sourceSite, ordinal: 0),
      input: input)
    return WorkflowGraph(
      identity: identity,
      sourceSite: sourceSite,
      joinSite: joinSite,
      declaredActors: declaredActors,
      nodes: nodes)
  }

  fileprivate var canonicalInput: WorkflowCanonicalValue {
    Self.graphInput(
      sourceSite: sourceSite,
      joinSite: joinSite,
      declaredActors: declaredActors,
      nodes: nodes)
  }

  private static func graphInput(
    sourceSite: WorkflowStaticSite,
    joinSite: WorkflowStaticSite,
    declaredActors: Set<String>,
    nodes: [WorkflowNode]
  ) -> WorkflowCanonicalValue {
    .object([
      "actors": .array(declaredActors.sorted().map(WorkflowCanonicalValue.string)),
      "join": siteValue(joinSite),
      "nodes": .array(nodes.map(nodeValue)),
      "source": siteValue(sourceSite),
    ])
  }

  private static func siteValue(_ site: WorkflowStaticSite) -> WorkflowCanonicalValue {
    .object(["lane": .string(site.lane), "siteIndex": .integer(Int64(site.siteIndex))])
  }

  private static func nodeValue(_ node: WorkflowNode) -> WorkflowCanonicalValue {
    .object([
      "actor": .string(node.actor),
      "dependsOn": .array(node.dependsOn.map(WorkflowCanonicalValue.string)),
      "id": .string(node.id),
      "maxRetries": .integer(Int64(node.maxRetries)),
      "prompt": node.prompt,
      "shape": shapeValue(node.shape),
      "siteIndex": .integer(Int64(node.siteIndex)),
    ])
  }

  private static func shapeValue(_ shape: WorkflowResultShape) -> WorkflowCanonicalValue {
    .array(
      shape.fields.map { field in
        .object([
          "name": .string(field.name),
          "required": .boolean(field.required),
          "value": shapeValue(field.value),
        ])
      })
  }

  private static func shapeValue(_ shape: WorkflowShapeValue) -> WorkflowCanonicalValue {
    switch shape {
    case .string(let enumValues):
      return .object([
        "enum": enumValues.map { .array($0.map(WorkflowCanonicalValue.string)) } ?? .null,
        "type": .string("string"),
      ])
    case .integer:
      return .string("integer")
    case .number:
      return .string("number")
    case .boolean:
      return .string("boolean")
    case .null:
      return .string("null")
    case .array(let item):
      return .object(["item": shapeValue(item), "type": .string("array")])
    case .object(let fields):
      return .object([
        "fields": .array(
          fields.map { field in
            .object([
              "name": .string(field.name),
              "required": .boolean(field.required),
              "value": shapeValue(field.value),
            ])
          }),
        "type": .string("object"),
      ])
    }
  }
}

struct WorkflowScheduler: Sendable {
  let maximumRetriesPerRun: Int

  init(maximumRetriesPerRun: Int = 100) {
    self.maximumRetriesPerRun = maximumRetriesPerRun
  }

  func runGraph(
    _ graph: WorkflowGraph,
    joinedAt: WorkflowStaticSite,
    runID: UUID,
    journal: WorkflowJournal,
    maximumConcurrency: Int,
    dispatch:
      @escaping @Sendable (WorkflowNode, [WorkflowOutcomeRecord]) async throws ->
      WorkflowOutcomeRecord
  ) async throws -> WorkflowGraphOutcome {
    try validate(graph, joinedAt: joinedAt, maximumConcurrency: maximumConcurrency)
    try Task.checkCancellation()

    let history = try await journal.history(runID: runID)
    var outcomes: [String: WorkflowNodeOutcome] = [:]
    var successfulRecords: [String: WorkflowOutcomeRecord] = [:]
    var retriesUsed = 0

    while outcomes.count < graph.nodes.count {
      try Task.checkCancellation()
      var addedBlockedNode = true
      while addedBlockedNode {
        addedBlockedNode = false
        let failedIDs = Set(
          outcomes.compactMap { (id, outcome) -> String? in
            switch outcome.state {
            case .succeeded: return nil
            case .failed, .blocked: return id
            }
          })
        let unresolved = graph.nodes.filter { outcomes[$0.id] == nil }
        for node in blockedNodes(of: unresolved, failed: failedIDs) {
          guard let dependencyID = node.dependsOn.first(where: { failedIDs.contains($0) }),
            let dependency = outcomes[dependencyID]
          else { continue }
          let reason: String
          switch dependency.state {
          case .failed:
            reason = "Dependency '\(dependencyID)' failed."
          case .blocked:
            reason = "Dependency '\(dependencyID)' was blocked."
          case .succeeded:
            continue
          }
          try await recordBlockedNode(
            node,
            reason: reason,
            graph: graph,
            runID: runID,
            journal: journal,
            originalHistory: history)
          outcomes[node.id] = WorkflowNodeOutcome(state: .blocked(reason), attempts: 0)
          addedBlockedNode = true
        }
      }
      if outcomes.count == graph.nodes.count { break }

      let successIDs = Set(successfulRecords.keys)
      let failedIDs = Set(
        outcomes.compactMap { (id, outcome) -> String? in
          switch outcome.state {
          case .succeeded: return nil
          case .failed, .blocked: return id
          }
        })
      let ready = readyNodes(
        of: graph.nodes.filter { outcomes[$0.id] == nil },
        finished: successIDs,
        failed: failedIDs)
      guard !ready.isEmpty else {
        throw validationError(
          "Workflow graph could not make scheduling progress.", site: graph.joinSite)
      }

      let batch = concurrencyBatch(ready, maximumConcurrency: maximumConcurrency)
      let dependencyRecords: [String: [WorkflowOutcomeRecord]] = Dictionary(
        uniqueKeysWithValues: batch.map { node in
          (node.id, node.dependsOn.compactMap { successfulRecords[$0] })
        })
      let batchOutcomes = try await executeBatch(
        batch,
        dependencyRecords: dependencyRecords,
        graph: graph,
        runID: runID,
        journal: journal,
        originalHistory: history,
        retriesUsed: &retriesUsed,
        dispatch: dispatch)
      for node in batch {
        guard let nodeOutcome = batchOutcomes[node.id] else {
          throw validationError(
            "Workflow node '\(node.id)' completed without an outcome.", site: graph.joinSite)
        }
        outcomes[node.id] = nodeOutcome
        if case .succeeded(let value) = nodeOutcome.state {
          successfulRecords[node.id] = WorkflowOutcomeRecord(nodeID: node.id, value: value)
        }
      }
    }

    return WorkflowGraphOutcome(identity: graph.identity, nodes: outcomes)
  }

  func validateDAG(_ nodes: [WorkflowNode]) throws {
    guard nodes.count <= 100 else {
      throw validationError(
        "Workflow graph exceeds the 100-node limit.", site: Optional<WorkflowStaticSite>.none)
    }

    var nodeIDs = Set<String>()
    var siteIndices = Set<Int>()
    for node in nodes {
      guard !node.id.isEmpty, nodeIDs.insert(node.id).inserted else {
        throw validationError(
          "Workflow graph contains an empty or duplicate node ID.", site: site(node, ordinal: 0))
      }
      guard !node.actor.isEmpty, node.actor != "main" else {
        throw validationError(
          "Workflow node '\(node.id)' has an invalid actor.", site: site(node, ordinal: 0))
      }
      guard (0...3).contains(node.maxRetries) else {
        throw validationError(
          "Workflow node '\(node.id)' retry bound must be between 0 and 3.",
          site: site(node, ordinal: 0))
      }
      guard node.siteIndex >= 0, siteIndices.insert(node.siteIndex).inserted else {
        throw validationError(
          "Workflow graph contains an invalid or duplicate node site.", site: site(node, ordinal: 0)
        )
      }
      guard Set(node.dependsOn).count == node.dependsOn.count else {
        throw validationError(
          "Workflow node '\(node.id)' contains a duplicate dependency.",
          site: site(node, ordinal: 0))
      }
    }

    for node in nodes {
      for dependency in node.dependsOn {
        guard nodeIDs.contains(dependency), dependency != node.id else {
          throw validationError(
            "Workflow node '\(node.id)' references an unknown or self dependency '\(dependency)'.",
            site: site(node, ordinal: 0))
        }
      }
    }

    var visited = Set<String>()
    while visited.count < nodes.count {
      guard
        let next = nodes.first(where: { node in
          !visited.contains(node.id) && node.dependsOn.allSatisfy(visited.contains)
        })
      else {
        throw validationError(
          "Workflow graph contains a dependency cycle.", site: Optional<WorkflowStaticSite>.none)
      }
      visited.insert(next.id)
    }
  }

  func readyNodes(
    of nodes: [WorkflowNode],
    finished: Set<String>,
    failed: Set<String>
  ) -> [WorkflowNode] {
    nodes.filter { node in
      !node.dependsOn.contains(where: failed.contains)
        && node.dependsOn.allSatisfy(finished.contains)
    }
  }

  func blockedNodes(of nodes: [WorkflowNode], failed: Set<String>) -> [WorkflowNode] {
    nodes.filter { node in node.dependsOn.contains(where: failed.contains) }
  }

  private func validate(
    _ graph: WorkflowGraph,
    joinedAt: WorkflowStaticSite,
    maximumConcurrency: Int
  ) throws {
    guard maximumRetriesPerRun >= 0 else {
      throw validationError("Run retry ceiling cannot be negative.", site: graph.joinSite)
    }
    guard maximumConcurrency > 0 else {
      throw validationError("Maximum workflow concurrency must be positive.", site: graph.joinSite)
    }
    guard graph.sourceSite.lane == "main", graph.joinSite.lane == "main",
      joinedAt == graph.joinSite, graph.sourceSite != graph.joinSite
    else {
      throw validationError(
        "Workflow graph must be joined at its declared main-lane owner.", site: joinedAt)
    }

    try validateDAG(graph.nodes)
    guard !graph.declaredActors.contains("main"),
      graph.nodes.allSatisfy({ graph.declaredActors.contains($0.actor) })
    else {
      throw validationError("Workflow graph references an undeclared actor.", site: graph.joinSite)
    }
    let graphSites = Set([graph.sourceSite.siteIndex, graph.joinSite.siteIndex])
    guard graphSites.isDisjoint(with: Set(graph.nodes.map(\.siteIndex))) else {
      throw validationError("Workflow graph sites overlap a node site.", site: graph.joinSite)
    }

    let expectedIdentity = try WorkflowRequestIdentity.make(
      site: WorkflowSiteKey(site: graph.sourceSite, ordinal: 0),
      input: graph.canonicalInput)
    guard graph.identity == expectedIdentity else {
      throw validationError(
        "Workflow graph identity does not match its canonical contents.", site: graph.sourceSite)
    }
  }

  private func concurrencyBatch(_ ready: [WorkflowNode], maximumConcurrency: Int) -> [WorkflowNode]
  {
    var actors = Set<String>()
    var batch: [WorkflowNode] = []
    for node in ready where actors.insert(node.actor).inserted {
      batch.append(node)
      if batch.count == maximumConcurrency { break }
    }
    return batch
  }

  private func executeBatch(
    _ nodes: [WorkflowNode],
    dependencyRecords: [String: [WorkflowOutcomeRecord]],
    graph: WorkflowGraph,
    runID: UUID,
    journal: WorkflowJournal,
    originalHistory: WorkflowJournalHistory,
    retriesUsed: inout Int,
    dispatch:
      @escaping @Sendable (WorkflowNode, [WorkflowOutcomeRecord]) async throws ->
      WorkflowOutcomeRecord
  ) async throws -> [String: WorkflowNodeOutcome] {
    var attempts: [String: Int] = Dictionary(uniqueKeysWithValues: nodes.map { ($0.id, 0) })
    var nextOrdinal: [String: Int] = Dictionary(uniqueKeysWithValues: nodes.map { ($0.id, 0) })
    var finished: [String: WorkflowNodeOutcome] = [:]
    var active = nodes

    while !active.isEmpty {
      try Task.checkCancellation()
      var currentOutcomes: [String: WorkflowJournalOutcome] = [:]
      var newAttempts:
        [(
          node: WorkflowNode, identity: WorkflowRequestIdentity,
          dependencies: [WorkflowOutcomeRecord]
        )] = []

      for node in active {
        let ordinal = nextOrdinal[node.id, default: 0]
        let dependencies = dependencyRecords[node.id, default: []]
        let identity = try requestIdentity(
          graph: graph,
          node: node,
          dependencies: dependencies,
          ordinal: ordinal)
        if let event = try await journal.recordedOutcome(runID: runID, identity: identity) {
          guard let resolved = resolvedOutcome(in: event.payload) else {
            throw validationError(
              "Workflow attempt has a malformed journal outcome.", site: identity.site)
          }
          attempts[node.id, default: 0] += 1
          currentOutcomes[node.id] = resolved
          continue
        }

        if originalHistory.events.contains(where: {
          $0.identity == identity && $0.payload == .requestStarted
        }) {
          throw validationError(
            "A workflow node attempt started without a recorded outcome; refusing to repeat uncertain work.",
            site: identity.site)
        }
        _ = try await journal.beginRequest(runID: runID, identity: identity)
        attempts[node.id, default: 0] += 1
        newAttempts.append((node, identity, dependencies))
      }

      let dispatchedOutcomes = await dispatchAttempts(newAttempts, dispatch: dispatch)
      for attempt in newAttempts {
        guard let outcome = dispatchedOutcomes[attempt.node.id] else {
          throw validationError(
            "Workflow dispatch returned no attempt outcome.", site: attempt.identity.site)
        }
        _ = try await journal.resolveRequest(
          runID: runID, identity: attempt.identity, outcome: outcome)
        currentOutcomes[attempt.node.id] = outcome
      }

      var retry = Set<String>()
      for node in active {
        guard let outcome = currentOutcomes[node.id] else {
          throw validationError(
            "Workflow node attempt has no recorded result.", site: site(node, ordinal: 0))
        }
        switch outcome {
        case .value(let value):
          guard let record = outcomeRecord(value), record.nodeID == node.id else {
            finished[node.id] = WorkflowNodeOutcome(
              state: .failed(
                WorkflowError(
                  kind: .validation,
                  message: "Workflow node '\(node.id)' has a mismatched journaled outcome.",
                  site: site(node, ordinal: nextOrdinal[node.id, default: 0]))),
              attempts: attempts[node.id, default: 0])
            continue
          }
          finished[node.id] = WorkflowNodeOutcome(
            state: .succeeded(record.value),
            attempts: attempts[node.id, default: 0])
        case .failure(let error):
          let ordinal = nextOrdinal[node.id, default: 0]
          guard error.kind == .modelFailure, ordinal < node.maxRetries else {
            finished[node.id] = WorkflowNodeOutcome(
              state: .failed(error),
              attempts: attempts[node.id, default: 0])
            continue
          }
          guard retriesUsed < maximumRetriesPerRun else {
            let limitError = WorkflowError(
              kind: .resourceLimit,
              message: "Workflow run exhausted its retry ceiling.",
              site: site(node, ordinal: ordinal))
            finished[node.id] = WorkflowNodeOutcome(
              state: .failed(limitError),
              attempts: attempts[node.id, default: 0])
            continue
          }
          retriesUsed += 1
          nextOrdinal[node.id] = ordinal + 1
          retry.insert(node.id)
        }
      }
      active = nodes.filter { retry.contains($0.id) }
    }

    return finished
  }

  private func dispatchAttempts(
    _ attempts: [(
      node: WorkflowNode, identity: WorkflowRequestIdentity, dependencies: [WorkflowOutcomeRecord]
    )],
    dispatch:
      @escaping @Sendable (WorkflowNode, [WorkflowOutcomeRecord]) async throws ->
      WorkflowOutcomeRecord
  ) async -> [String: WorkflowJournalOutcome] {
    await withTaskGroup(of: (String, WorkflowJournalOutcome).self) { group in
      for attempt in attempts {
        group.addTask {
          do {
            try Task.checkCancellation()
            let record = try await dispatch(attempt.node, attempt.dependencies)
            guard record.nodeID == attempt.node.id else {
              return (
                attempt.node.id,
                .failure(
                  WorkflowError(
                    kind: .validation,
                    message: "Workflow dispatcher returned an outcome for another node.",
                    site: attempt.identity.site))
              )
            }
            return (attempt.node.id, .value(encodedOutcome(record)))
          } catch let error as WorkflowError {
            return (attempt.node.id, .failure(error))
          } catch is CancellationError {
            return (
              attempt.node.id,
              .failure(
                WorkflowError(
                  kind: .cancelled,
                  message: "Workflow node dispatch was cancelled.",
                  site: attempt.identity.site))
            )
          } catch {
            return (
              attempt.node.id,
              .failure(
                WorkflowError(
                  kind: .modelFailure,
                  message: String(describing: error),
                  site: attempt.identity.site))
            )
          }
        }
      }

      var outcomes: [String: WorkflowJournalOutcome] = [:]
      for await (nodeID, outcome) in group {
        outcomes[nodeID] = outcome
      }
      return outcomes
    }
  }

  private func requestIdentity(
    graph: WorkflowGraph,
    node: WorkflowNode,
    dependencies: [WorkflowOutcomeRecord],
    ordinal: Int
  ) throws -> WorkflowRequestIdentity {
    let dependencyValues = dependencies.map { record in
      WorkflowCanonicalValue.object([
        "nodeID": .string(record.nodeID),
        "value": record.value,
      ])
    }
    let input = WorkflowCanonicalValue.object([
      "dependencies": .array(dependencyValues),
      "graphIdentity": .string(graph.identity.inputHash),
      "node": .string(node.id),
      "prompt": node.prompt,
    ])
    return try WorkflowRequestIdentity.make(
      site: WorkflowSiteKey(lane: "main", siteIndex: node.siteIndex, ordinal: ordinal),
      input: input)
  }

  private func recordBlockedNode(
    _ node: WorkflowNode,
    reason: String,
    graph: WorkflowGraph,
    runID: UUID,
    journal: WorkflowJournal,
    originalHistory: WorkflowJournalHistory
  ) async throws {
    let journalReason = "Blocked node '\(node.id)': \(reason)"
    let input = WorkflowCanonicalValue.object([
      "graphIdentity": .string(graph.identity.inputHash),
      "node": .string(node.id),
      "reason": .string(journalReason),
    ])
    let identity = try WorkflowRequestIdentity.make(
      site: WorkflowSiteKey(lane: "main", siteIndex: node.siteIndex, ordinal: Int.max),
      input: input)
    if let event = try await journal.recordedOutcome(runID: runID, identity: identity) {
      guard case .failure(let previous) = resolvedOutcome(in: event.payload),
        previous.message == journalReason
      else {
        throw validationError(
          "A blocked-node journal record does not match the current dependency failure.",
          site: identity.site)
      }
      return
    }
    if originalHistory.events.contains(where: {
      $0.identity == identity && $0.payload == .requestStarted
    }) {
      throw validationError(
        "A blocked-node record started without a resolution.", site: identity.site)
    }
    _ = try await journal.beginRequest(runID: runID, identity: identity)
    _ = try await journal.resolveRequest(
      runID: runID,
      identity: identity,
      outcome: .failure(
        WorkflowError(kind: .validation, message: journalReason, site: site(node, ordinal: Int.max))
      ))
  }

  private func resolvedOutcome(in payload: WorkflowJournalEventPayload) -> WorkflowJournalOutcome? {
    switch payload {
    case .requestResolved(let outcome): return outcome
    case .requestResolvedWithActorTranscript(let outcome, _, _, _): return outcome
    default: return nil
    }
  }

  private func validationError(_ message: String, site: WorkflowStaticSite?) -> WorkflowError {
    WorkflowError(
      kind: .validation,
      message: message,
      site: site.map { WorkflowSiteKey(site: $0, ordinal: 0) })
  }

  private func validationError(_ message: String, site: WorkflowSiteKey?) -> WorkflowError {
    WorkflowError(kind: .validation, message: message, site: site)
  }

  private func site(_ node: WorkflowNode, ordinal: Int) -> WorkflowSiteKey {
    WorkflowSiteKey(lane: "main", siteIndex: node.siteIndex, ordinal: ordinal)
  }
}

private func encodedOutcome(_ outcome: WorkflowOutcomeRecord) -> WorkflowCanonicalValue {
  .object([
    "nodeID": .string(outcome.nodeID),
    "value": outcome.value,
  ])
}

private func outcomeRecord(_ value: WorkflowCanonicalValue) -> WorkflowOutcomeRecord? {
  guard case .object(let object) = value,
    case .string(let nodeID)? = object["nodeID"],
    let value = object["value"]
  else { return nil }
  return WorkflowOutcomeRecord(nodeID: nodeID, value: value)
}
