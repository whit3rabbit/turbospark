import Foundation
import TurboSpark

struct WorkflowAskLimits: Sendable, Equatable {
  static let maximumRepairAsks = 2

  let maxRepairAsks: Int

  init(maxRepairAsks: Int = Self.maximumRepairAsks) {
    self.maxRepairAsks = min(max(0, maxRepairAsks), Self.maximumRepairAsks)
  }
}

struct WorkflowAskOutcome: Sendable {
  var value: WorkflowCanonicalValue
  var rawText: String
  var repairAttempts: Int
  var transcriptDelta: [ChatMessage]
}

struct WorkflowActorTranscript: Codable, Sendable, Equatable {
  static let currentVersion = 1

  var version: Int
  var messages: [ChatMessage]
}

protocol SubagentDispatcherPort: Sendable {
  func run(
    actor: WorkflowActorSpec,
    prompt: String,
    priorHistory: [ChatMessage],
    depth: Int
  ) async -> SubagentRunResult
}

final class WorkflowActorBridge: Sendable {
  private let subagents: any SubagentDispatcherPort
  private let limits: WorkflowAskLimits

  init(subagents: any SubagentDispatcherPort, limits: WorkflowAskLimits = WorkflowAskLimits()) {
    self.subagents = subagents
    self.limits = limits
  }

  func ask(
    actor: WorkflowActorSpec,
    prompt: String,
    shape: WorkflowResultShape,
    priorTranscript: WorkflowActorTranscript?,
    context: WorkflowAttemptContext,
    depth: Int
  ) async throws -> WorkflowAskOutcome {
    if let priorTranscript, priorTranscript.version != WorkflowActorTranscript.currentVersion {
      throw failure(.validation, "Actor '\(actor.name)' has an unsupported transcript version.")
    }

    let originalHistory = priorTranscript?.messages ?? []
    var history = originalHistory
    let schema = WorkflowResultShapeCodec.canonicalValue(for: shape)
    guard let schemaText = WorkflowResultShapeCodec.jsonText(schema) else {
      throw failure(.validation, "Actor '\(actor.name)' has an unserializable result shape.")
    }
    var currentPrompt = initialPrompt(prompt, schema: schemaText)

    for repairCount in 0...limits.maxRepairAsks {
      if let reason = context.cancellation.cancellationReason {
        throw cancellationFailure(actor: actor.name, reason: reason)
      }

      let result = try await dispatch(
        actor: actor,
        prompt: currentPrompt,
        priorHistory: history,
        depth: depth,
        cancellation: context.cancellation)

      if let reason = context.cancellation.cancellationReason {
        throw cancellationFailure(actor: actor.name, reason: reason)
      }
      if result.status == "cancelled" {
        throw failure(.cancelled, "Actor '\(actor.name)' dispatch was cancelled.")
      }
      guard result.status == "completed" else {
        throw failure(.modelFailure, "Actor '\(actor.name)' dispatch ended with status '\(result.status)'.")
      }
      guard let updatedHistory = result.transcript,
            updatedHistory.count >= history.count,
            Array(updatedHistory.prefix(history.count)) == history
      else {
        throw failure(.modelFailure, "Actor '\(actor.name)' dispatch did not return its transcript continuation.")
      }

      history = updatedHistory
      let answer = WorkflowResultShapeCodec.parseAnswer(result.finalResponse)
      let issues: [String]
      if let answer {
        issues = WorkflowResultShapeCodec.validate(answer, against: shape)
      } else {
        issues = ["The response must be a JSON value with no surrounding prose or Markdown fences."]
      }
      if issues.isEmpty, let answer {
        if let reason = context.cancellation.cancellationReason {
          throw cancellationFailure(actor: actor.name, reason: reason)
        }
        let delta = Array(history.dropFirst(originalHistory.count))
        return WorkflowAskOutcome(
          value: answer,
          rawText: result.finalResponse,
          repairAttempts: repairCount,
          transcriptDelta: delta)
      }

      guard repairCount < limits.maxRepairAsks else {
        if let reason = context.cancellation.cancellationReason {
          throw cancellationFailure(actor: actor.name, reason: reason)
        }
        throw validationFailure(actor: actor.name, issues: issues, attempts: repairCount)
      }
      currentPrompt = repairPrompt(originalPrompt: prompt, issues: issues, schema: schemaText)
    }

    throw failure(.validation, "Actor '\(actor.name)' did not satisfy the declared result shape.")
  }

  private func dispatch(
    actor: WorkflowActorSpec,
    prompt: String,
    priorHistory: [ChatMessage],
    depth: Int,
    cancellation: WorkflowCancellationToken
  ) async throws -> SubagentRunResult {
    let child = Task {
      await subagents.run(actor: actor, prompt: prompt, priorHistory: priorHistory, depth: depth)
    }
    let observer = cancellation.observeCancellation { _ in child.cancel() }
    let result = await child.value
    if let observer { cancellation.removeCancellationObserver(observer) }
    if let reason = cancellation.cancellationReason {
      throw cancellationFailure(actor: actor.name, reason: reason)
    }
    return result
  }

  private func initialPrompt(_ prompt: String, schema: String) -> String {
    """
    \(prompt)

    Return only one JSON object matching this declared result shape. Do not include Markdown fences or surrounding prose.
    \(schema)
    """
  }

  private func repairPrompt(originalPrompt: String, issues: [String], schema: String) -> String {
    let complaints = issues.map { "- \($0)" }.joined(separator: "\n")
    return """
      \(originalPrompt)

      Your previous answer did not satisfy the declared result shape. Correct it using the same task and return only JSON, without Markdown fences or surrounding prose.
      Validation issues:
      \(complaints)
      Required shape:
      \(schema)
      """
  }

  private func validationFailure(actor: String, issues: [String], attempts: Int) -> WorkflowError {
    let details = issues.joined(separator: " ")
    return failure(
      .validation,
      "Actor '\(actor)' did not satisfy its declared result shape after \(attempts) repair attempts. \(details)")
  }

  private func cancellationFailure(
    actor: String,
    reason: WorkflowCancellationReason
  ) -> WorkflowError {
    switch reason {
    case .requested:
      return failure(.cancelled, "Actor '\(actor)' dispatch was cancelled.")
    case .deadlineExceeded:
      return failure(.cancelled, "Actor '\(actor)' dispatch exceeded its deadline.")
    }
  }

  private func failure(_ kind: WorkflowErrorKind, _ message: String) -> WorkflowError {
    WorkflowError(kind: kind, message: message, site: nil)
  }
}

struct WorkflowActorCapabilityAdapter: Sendable {
  static let maximumActorTranscriptBytes = 256 * 1024

  private let bridge: WorkflowActorBridge
  private let actors: [WorkflowActorSpec]
  private let actorDepth: Int

  init(bridge: WorkflowActorBridge, actors: [WorkflowActorSpec], actorDepth: Int) {
    self.bridge = bridge
    self.actors = actors
    self.actorDepth = actorDepth
  }

  func perform(
    _ operation: WorkflowInterpreterOperation,
    context: WorkflowAttemptContext,
    priorActorTranscript: WorkflowActorTranscriptSnapshot?
  ) async throws -> WorkflowCapabilityResult {
    guard case .ask(let alias, let promptValue, let shapeValue) = operation else {
      throw WorkflowError(
        kind: .validation,
        message: "The actor capability adapter only accepts ask operations.",
        site: nil)
    }
    let matches = actors.filter { $0.name == alias }
    guard matches.count == 1, let actor = matches.first else {
      throw WorkflowError(
        kind: .validation,
        message: "Workflow actor alias '\(alias)' is missing or ambiguous in the checked manifest.",
        site: nil)
    }
    let prompt: String
    if case .string(let string) = promptValue {
      prompt = string
    } else if let json = WorkflowResultShapeCodec.jsonText(promptValue) {
      prompt = json
    } else {
      throw WorkflowError(
        kind: .validation,
        message: "Workflow actor '\(alias)' prompt cannot be serialized.",
        site: nil)
    }

    let shape: WorkflowResultShape
    do {
      shape = try WorkflowResultShapeCodec.decode(shapeValue)
    } catch let error as WorkflowResultShapeCodec.Failure {
      throw WorkflowError(kind: .validation, message: error.message, site: nil)
    }

    if let priorActorTranscript {
      let encoded: Data
      do {
        encoded = try WorkflowActorTranscriptCodec.encodedSnapshotData(priorActorTranscript)
      } catch {
        throw WorkflowError(
          kind: .validation,
          message: "Workflow actor '\(alias)' transcript cannot be serialized.",
          site: nil)
      }
      guard encoded.count <= Self.maximumActorTranscriptBytes else {
        throw WorkflowError(
          kind: .resourceLimit,
          message: "Workflow actor '\(alias)' transcript exceeds 256 KiB.",
          site: nil)
      }
    }
    let prior: WorkflowActorTranscript?
    do {
      prior = try priorActorTranscript.map(WorkflowActorTranscriptCodec.decode)
    } catch let error as WorkflowActorTranscriptCodec.Failure {
      throw WorkflowError(
        kind: .validation,
        message: "Workflow actor '\(alias)' transcript is invalid: \(error.message)",
        site: nil)
    } catch {
      throw WorkflowError(
        kind: .validation,
        message: "Workflow actor '\(alias)' transcript is invalid.",
        site: nil)
    }
    let outcome = try await bridge.ask(
      actor: actor,
      prompt: prompt,
      shape: shape,
      priorTranscript: prior,
      context: context,
      depth: actorDepth)
    if let reason = context.cancellation.cancellationReason {
      throw WorkflowError(
        kind: .cancelled,
        message: cancellationMessage(actor: alias, reason: reason),
        site: nil)
    }
    let transcript = WorkflowActorTranscript(
      version: WorkflowActorTranscript.currentVersion,
      messages: (prior?.messages ?? []) + outcome.transcriptDelta)
    let snapshot: WorkflowActorTranscriptSnapshot
    do {
      snapshot = try WorkflowActorTranscriptCodec.snapshot(from: transcript)
    } catch let error as WorkflowActorTranscriptCodec.Failure {
      throw WorkflowError(kind: .validation, message: error.message, site: nil)
    } catch {
      throw WorkflowError(
        kind: .validation,
        message: "Workflow actor '\(alias)' transcript cannot be serialized.",
        site: nil)
    }
    let encodedSnapshot: Data
    do {
      encodedSnapshot = try WorkflowActorTranscriptCodec.encodedSnapshotData(snapshot)
    } catch {
      throw WorkflowError(
        kind: .validation,
        message: "Workflow actor '\(alias)' transcript cannot be serialized.",
        site: nil)
    }
    guard encodedSnapshot.count <= Self.maximumActorTranscriptBytes else {
      throw WorkflowError(
        kind: .resourceLimit,
        message: "Workflow actor '\(alias)' transcript exceeds 256 KiB.",
        site: nil)
    }
    if let reason = context.cancellation.cancellationReason {
      throw WorkflowError(
        kind: .cancelled,
        message: cancellationMessage(actor: alias, reason: reason),
        site: nil)
    }
    return WorkflowCapabilityResult(value: outcome.value, actorTranscript: snapshot)
  }

  private func cancellationMessage(actor: String, reason: WorkflowCancellationReason) -> String {
    switch reason {
    case .requested:
      "Workflow actor '\(actor)' dispatch was cancelled."
    case .deadlineExceeded:
      "Workflow actor '\(actor)' dispatch exceeded its deadline."
    }
  }
}

struct WorkflowAppSubagentDispatcher: SubagentDispatcherPort {
  typealias Dispatch = @Sendable (
    AppAgentDefinition,
    String,
    [ChatMessage],
    Int
  ) async -> SubagentRunResult

  private let dispatch: Dispatch

  init(dispatch: @escaping Dispatch) {
    self.dispatch = dispatch
  }

  func run(
    actor: WorkflowActorSpec,
    prompt: String,
    priorHistory: [ChatMessage],
    depth: Int
  ) async -> SubagentRunResult {
    await dispatch(Self.transientAgent(for: actor), prompt, priorHistory, depth)
  }

  static func transientAgent(for actor: WorkflowActorSpec) -> AppAgentDefinition {
    AppAgentDefinition(
      name: actor.name,
      displayName: actor.name,
      agentDescription: "Workflow-local actor declared for this run.",
      systemPrompt: actor.rolePrompt,
      tools: nil,
      disallowedTools: nil,
      model: nil,
      maxTurns: 5,
      omitsProjectInstructions: false,
      sourceAgent: .turboSpark,
      scope: .builtIn,
      filePath: nil,
      isEnabled: true)
  }
}
