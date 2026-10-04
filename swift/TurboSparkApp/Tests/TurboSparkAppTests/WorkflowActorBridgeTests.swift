import Foundation
import XCTest
import TurboSpark

@testable import TurboSparkApp

final class WorkflowActorBridgeTests: XCTestCase {
  func testTypedRepairKeepsActorHistoryAndReturnsOnlyNewTranscriptDelta() async throws {
    let original = [
      ChatMessage(role: .user, content: "Earlier workflow ask"),
      ChatMessage(role: .assistant, content: "Earlier workflow answer"),
    ]
    let dispatcher = WorkflowActorBridgeTestDispatcher(responses: [
      "{\"verdict\":\"maybe\",\"sources\":[7]}",
      "{\"verdict\":\"accept\",\"sources\":[\"spec.md\"]}",
    ])
    let bridge = WorkflowActorBridge(subagents: dispatcher, limits: WorkflowAskLimits(maxRepairAsks: 1))

    let outcome = try await bridge.ask(
      actor: WorkflowActorSpec(name: "reviewer", rolePrompt: "Review claims"),
      prompt: "Check the proposal",
      shape: resultShape(),
      priorTranscript: WorkflowActorTranscript(version: 1, messages: original),
      context: attemptContext(),
      depth: 7)

    XCTAssertEqual(outcome.value, .object([
      "verdict": .string("accept"),
      "sources": .array([.string("spec.md")]),
    ]))
    XCTAssertEqual(outcome.repairAttempts, 1)
    XCTAssertEqual(outcome.rawText, "{\"verdict\":\"accept\",\"sources\":[\"spec.md\"]}")
    XCTAssertEqual(outcome.transcriptDelta.count, 4)

    let calls = await dispatcher.calls()
    XCTAssertEqual(calls.map(\.actor), [
      WorkflowActorSpec(name: "reviewer", rolePrompt: "Review claims"),
      WorkflowActorSpec(name: "reviewer", rolePrompt: "Review claims"),
    ])
    XCTAssertEqual(calls[0].priorHistory, original)
    XCTAssertEqual(calls[1].priorHistory.count, original.count + 2)
    XCTAssertTrue(calls[1].prompt.contains("verdict"))
    XCTAssertTrue(calls[1].prompt.contains("one of"))
    XCTAssertEqual(calls.map(\.depth), [7, 7])
  }

  func testFirstAskPassesExplicitEmptyHistoryAndActorDepthUnchanged() async throws {
    let dispatcher = WorkflowActorBridgeTestDispatcher(responses: ["{\"answer\":\"ready\"}"])
    let bridge = WorkflowActorBridge(subagents: dispatcher, limits: WorkflowAskLimits())

    _ = try await bridge.ask(
      actor: WorkflowActorSpec(name: "writer", rolePrompt: "Draft"),
      prompt: "Write",
      shape: WorkflowResultShape(fields: [
        WorkflowShapeField(name: "answer", required: true, value: .string(enumValues: nil)),
      ]),
      priorTranscript: nil,
      context: attemptContext(),
      depth: 13)

    let calls = await dispatcher.calls()
    XCTAssertEqual(calls.count, 1)
    XCTAssertEqual(calls[0].priorHistory, [])
    XCTAssertEqual(calls[0].depth, 13)
  }

  func testRepairExhaustionNamesActorAndUnmetFields() async throws {
    let dispatcher = WorkflowActorBridgeTestDispatcher(responses: [
      "{\"verdict\":\"accept\"}",
      "{\"verdict\":\"wrong\",\"sources\":[7]}",
    ])
    let bridge = WorkflowActorBridge(subagents: dispatcher, limits: WorkflowAskLimits(maxRepairAsks: 1))

    do {
      _ = try await bridge.ask(
        actor: WorkflowActorSpec(name: "reviewer", rolePrompt: "Review"),
        prompt: "Review",
        shape: resultShape(),
        priorTranscript: nil,
        context: attemptContext(),
        depth: 1)
      XCTFail("Invalid results should exhaust as a validation error")
    } catch let error as WorkflowError {
      XCTAssertEqual(error.kind, .validation)
      XCTAssertTrue(error.message.contains("reviewer"))
      XCTAssertTrue(error.message.contains("verdict"))
      XCTAssertTrue(error.message.contains("sources"))
    }

    let calls = await dispatcher.calls()
    XCTAssertEqual(calls.count, 2)
  }

  func testAdapterResolvesFullSpecAndRoleOnlyForNamedActor() async throws {
    let dispatcher = WorkflowActorBridgeTestDispatcher(responses: ["{\"answer\":\"reviewed\"}"])
    let bridge = WorkflowActorBridge(subagents: dispatcher, limits: WorkflowAskLimits())
    let actors = [
      WorkflowActorSpec(name: "writer", rolePrompt: "Draft only"),
      WorkflowActorSpec(name: "reviewer", rolePrompt: "Review only"),
    ]
    let adapter = WorkflowActorCapabilityAdapter(bridge: bridge, actors: actors, actorDepth: 4)

    let result = try await adapter.perform(
      .ask(actor: "reviewer", prompt: .string("Review this"), shape: .object(["answer": .string("string")])),
      context: attemptContext(),
      priorActorTranscript: nil)

    XCTAssertEqual(result.value, .object(["answer": .string("reviewed")]))
    let calls = await dispatcher.calls()
    XCTAssertEqual(calls.map(\.actor), [actors[1]])
    XCTAssertEqual(calls[0].depth, 4)
  }

  func testAdapterDecodesNestedShapeAndCarriesPriorTranscriptIntoSnapshot() async throws {
    let priorMessages = [
      ChatMessage(role: .user, content: "Earlier ask"),
      ChatMessage(role: .assistant, content: "Earlier answer"),
    ]
    let priorSnapshot = try WorkflowActorTranscriptCodec.snapshot(
      from: WorkflowActorTranscript(version: 1, messages: priorMessages))
    let rawAnswer = "{\"profile\":{\"name\":\"Ada\"},\"scores\":[0.91]}"
    let dispatcher = WorkflowActorBridgeTestDispatcher(responses: [rawAnswer])
    let adapter = WorkflowActorCapabilityAdapter(
      bridge: WorkflowActorBridge(subagents: dispatcher, limits: WorkflowAskLimits()),
      actors: [WorkflowActorSpec(name: "writer", rolePrompt: "Draft")],
      actorDepth: 5)
    let shape: WorkflowCanonicalValue = .object([
      "profile": .object([
        "type": .string("object"),
        "fields": .array([
          .object(["name": .string("name"), "required": .boolean(true), "value": .string("string")]),
          .object(["name": .string("nickname"), "required": .boolean(false), "value": .string("string")]),
        ]),
      ]),
      "scores": .object(["type": .string("array"), "item": .string("number")]),
    ])

    let result = try await adapter.perform(
      .ask(actor: "writer", prompt: .string("Draft a profile"), shape: shape),
      context: attemptContext(),
      priorActorTranscript: priorSnapshot)

    XCTAssertEqual(result.value, .object([
      "profile": .object(["name": .string("Ada")]),
      "scores": .array([.number(0.91)]),
    ]))
    let calls = await dispatcher.calls()
    XCTAssertEqual(calls[0].priorHistory, priorMessages)
    XCTAssertEqual(calls[0].depth, 5)
    let finalSnapshot = try XCTUnwrap(result.actorTranscript)
    let finalTranscript = try WorkflowActorTranscriptCodec.decode(finalSnapshot)
    XCTAssertEqual(finalTranscript.messages, priorMessages + [
      ChatMessage(role: .user, content: calls[0].prompt),
      ChatMessage(role: .assistant, content: rawAnswer),
    ])
  }

  func testAdapterFailsClosedForUnknownActorAlias() async throws {
    let dispatcher = WorkflowActorBridgeTestDispatcher(responses: ["{}"])
    let adapter = WorkflowActorCapabilityAdapter(
      bridge: WorkflowActorBridge(subagents: dispatcher, limits: WorkflowAskLimits()),
      actors: [WorkflowActorSpec(name: "writer", rolePrompt: "Draft")],
      actorDepth: 0)

    do {
      _ = try await adapter.perform(
        .ask(actor: "missing", prompt: .string("Run"), shape: .object([:])),
        context: attemptContext(),
        priorActorTranscript: nil)
      XCTFail("Unknown actor aliases must not dispatch")
    } catch let error as WorkflowError {
      XCTAssertEqual(error.kind, .validation)
      XCTAssertTrue(error.message.contains("missing"))
    }

    let calls = await dispatcher.calls()
    XCTAssertEqual(calls.count, 0)
  }

  func testCancellationStopsActiveDispatchWithoutReturningTranscriptDelta() async throws {
    let dispatcher = WorkflowActorBridgeBlockingDispatcher()
    let checked = WorkflowScriptChecker.check(
      source: """
        async function workflow() {
          agent("writer", "Draft");
          const answer = await ask("writer", "Write", { answer: "string" });
          await report(answer);
        }
        """,
      name: "Cancellation",
      args: [:])
    XCTAssertTrue(checked.isValid, checked.diagnostics.map(\.message).joined(separator: "\n"))
    let program = try XCTUnwrap(checked.checked)
    let interpreter = WorkflowInterpreter(limits: WorkflowInterpreterLimits(), engine: dispatcher)
    let task = Task { try await interpreter.execute(program: program) }

    await dispatcher.waitUntilStarted()
    interpreter.requestCancel()
    do {
      try await task.value
      XCTFail("A cancelled child must not return a transcript delta")
    } catch let error as WorkflowError {
      XCTAssertEqual(error.kind, .cancelled)
    }
    let cancellationObserved = await dispatcher.cancellationObserved()
    XCTAssertTrue(cancellationObserved)
  }

  func testDeadlineStopsActiveDispatchWithoutReturningTranscriptDelta() async throws {
    let dispatcher = WorkflowActorBridgeBlockingDispatcher()
    let checked = WorkflowScriptChecker.check(
      source: """
        async function workflow() {
          agent("writer", "Draft");
          const answer = await ask("writer", "Write", { answer: "string" });
          await report(answer);
        }
        """,
      name: "Deadline",
      args: [:])
    XCTAssertTrue(checked.isValid, checked.diagnostics.map(\.message).joined(separator: "\n"))
    let program = try XCTUnwrap(checked.checked)
    let interpreter = WorkflowInterpreter(limits: WorkflowInterpreterLimits(), engine: dispatcher)
    let deadline = ContinuousClock().now.advanced(by: .milliseconds(200))
    let task = Task { try await interpreter.execute(program: program, deadline: deadline) }

    await dispatcher.waitUntilStarted()
    do {
      try await task.value
      XCTFail("A deadline-cancelled child must not return a transcript delta")
    } catch let error as WorkflowError {
      XCTAssertEqual(error.kind, .resourceLimit)
      XCTAssertEqual(error.message, "Workflow attempt deadline exceeded.")
    }
    let cancellationObserved = await dispatcher.cancellationObserved()
    XCTAssertTrue(cancellationObserved)
  }

  func testAdapterRejectsOversizedPriorTranscriptBeforeDispatch() async throws {
    let dispatcher = WorkflowActorBridgeTestDispatcher(responses: [])
    let adapter = WorkflowActorCapabilityAdapter(
      bridge: WorkflowActorBridge(subagents: dispatcher, limits: WorkflowAskLimits()),
      actors: [WorkflowActorSpec(name: "writer", rolePrompt: "Draft")],
      actorDepth: 0)
    let prior = try WorkflowActorTranscriptCodec.snapshot(
      from: WorkflowActorTranscript(
        version: WorkflowActorTranscript.currentVersion,
        messages: [ChatMessage(
          role: .user,
          content: String(repeating: "x", count: WorkflowActorCapabilityAdapter.maximumActorTranscriptBytes))]))

    do {
      _ = try await adapter.perform(
        .ask(actor: "writer", prompt: .string("Write"), shape: .object(["answer": .string("string")])),
        context: attemptContext(),
        priorActorTranscript: prior)
      XCTFail("Oversized prior history must fail before dispatch")
    } catch let error as WorkflowError {
      XCTAssertEqual(error.kind, .resourceLimit)
    }
    let calls = await dispatcher.calls()
    XCTAssertEqual(calls.count, 0)
  }

  func testAdapterRejectsOversizedReturnedTranscriptWithoutAnswer() async throws {
    let answer = String(repeating: "x", count: WorkflowActorCapabilityAdapter.maximumActorTranscriptBytes)
    let dispatcher = WorkflowActorBridgeTestDispatcher(responses: ["{\"answer\":\"\(answer)\"}"])
    let adapter = WorkflowActorCapabilityAdapter(
      bridge: WorkflowActorBridge(subagents: dispatcher, limits: WorkflowAskLimits()),
      actors: [WorkflowActorSpec(name: "writer", rolePrompt: "Draft")],
      actorDepth: 0)

    do {
      _ = try await adapter.perform(
        .ask(actor: "writer", prompt: .string("Write"), shape: .object(["answer": .string("string")])),
        context: attemptContext(),
        priorActorTranscript: nil)
      XCTFail("Oversized new history must not return a validated answer")
    } catch let error as WorkflowError {
      XCTAssertEqual(error.kind, .resourceLimit)
    }
    let calls = await dispatcher.calls()
    XCTAssertEqual(calls.count, 1)
  }

  private func resultShape() -> WorkflowResultShape {
    WorkflowResultShape(fields: [
      WorkflowShapeField(
        name: "verdict",
        required: true,
        value: .string(enumValues: ["accept", "revise"])),
      WorkflowShapeField(
        name: "sources",
        required: true,
        value: .array(item: .string(enumValues: nil))),
    ])
  }

  private func attemptContext() -> WorkflowAttemptContext {
    WorkflowAttemptContext(cancellation: WorkflowCancellationToken(), deadline: nil)
  }
}

final class WorkflowAppSubagentDispatcherTests: XCTestCase {
  func testRolePromptCreatesTransientAgentWithoutModelOrToolOverrides() async {
    let probe = WorkflowAppSubagentDispatcherProbe()
    let dispatcher = WorkflowAppSubagentDispatcher { agent, prompt, history, depth in
      await probe.record(agent: agent, prompt: prompt, history: history, depth: depth)
      return SubagentRunResult(
        agentName: agent.name,
        finalResponse: "done",
        totalTurns: 1,
        totalToolCalls: 0,
        durationSeconds: 0,
        transcript: history)
    }
    let actor = WorkflowActorSpec(name: "workflow-reviewer", rolePrompt: "Inspect only the declared issue")

    _ = await dispatcher.run(
      actor: actor,
      prompt: "Inspect this change",
      priorHistory: [],
      depth: 6)

    let captured = await probe.value()
    XCTAssertEqual(captured?.agent.name, "workflow-reviewer")
    XCTAssertEqual(captured?.agent.systemPrompt, actor.rolePrompt)
    XCTAssertNil(captured?.agent.model)
    XCTAssertNil(captured?.agent.tools)
    XCTAssertNil(captured?.agent.disallowedTools)
    XCTAssertEqual(captured?.prompt, "Inspect this change")
    XCTAssertEqual(captured?.history, [])
    XCTAssertEqual(captured?.depth, 6)
  }
}

private actor WorkflowAppSubagentDispatcherProbe {
  struct Capture: Sendable {
    let agent: AppAgentDefinition
    let prompt: String
    let history: [ChatMessage]
    let depth: Int
  }

  private var captured: Capture?

  func record(agent: AppAgentDefinition, prompt: String, history: [ChatMessage], depth: Int) {
    captured = Capture(agent: agent, prompt: prompt, history: history, depth: depth)
  }

  func value() -> Capture? { captured }
}

private actor WorkflowActorBridgeTestDispatcher: SubagentDispatcherPort {
  struct Call: Sendable, Equatable {
    let actor: WorkflowActorSpec
    let prompt: String
    let priorHistory: [ChatMessage]
    let depth: Int
  }

  private var responses: [String]
  private var recordedCalls: [Call] = []

  init(responses: [String]) {
    self.responses = responses
  }

  func run(actor: WorkflowActorSpec, prompt: String, priorHistory: [ChatMessage], depth: Int) async -> SubagentRunResult {
    recordedCalls.append(Call(actor: actor, prompt: prompt, priorHistory: priorHistory, depth: depth))
    let response = responses.removeFirst()
    let transcript = priorHistory + [
      ChatMessage(role: .user, content: prompt),
      ChatMessage(role: .assistant, content: response),
    ]
    return SubagentRunResult(
      agentName: actor.name,
      finalResponse: response,
      totalTurns: 1,
      totalToolCalls: 0,
      durationSeconds: 0,
      transcript: transcript)
  }

  func calls() -> [Call] { recordedCalls }
}

private actor WorkflowActorBridgeBlockingDispatcher: SubagentDispatcherPort, WorkflowCommandSink {
  private var startWaiter: CheckedContinuation<Void, Never>?
  private var dispatchContinuation: CheckedContinuation<SubagentRunResult, Never>?
  private var didStart = false
  private var didCancel = false

  func perform(
    _ operation: WorkflowInterpreterOperation,
    at _: WorkflowSiteKey?,
    context: WorkflowAttemptContext
  ) async throws -> WorkflowCanonicalValue {
    let adapter = WorkflowActorCapabilityAdapter(
      bridge: WorkflowActorBridge(subagents: self, limits: WorkflowAskLimits()),
      actors: [WorkflowActorSpec(name: "writer", rolePrompt: "Draft")],
      actorDepth: 0)
    let result = try await adapter.perform(operation, context: context, priorActorTranscript: nil)
    return result.value
  }

  func run(actor: WorkflowActorSpec, prompt: String, priorHistory: [ChatMessage], depth: Int) async -> SubagentRunResult {
    await withTaskCancellationHandler {
      await withCheckedContinuation { continuation in
        dispatchContinuation = continuation
        didStart = true
        startWaiter?.resume()
        startWaiter = nil
      }
    } onCancel: {
      Task { await self.finishCancelled(priorHistory: priorHistory) }
    }
  }

  func waitUntilStarted() async {
    if didStart { return }
    await withCheckedContinuation { startWaiter = $0 }
  }

  func cancellationObserved() -> Bool { didCancel }

  private func finishCancelled(priorHistory: [ChatMessage]) {
    didCancel = true
    dispatchContinuation?.resume(returning: SubagentRunResult(
      agentName: "writer",
      status: "cancelled",
      finalResponse: "",
      totalTurns: 0,
      totalToolCalls: 0,
      durationSeconds: 0,
      transcript: priorHistory))
    dispatchContinuation = nil
  }
}
