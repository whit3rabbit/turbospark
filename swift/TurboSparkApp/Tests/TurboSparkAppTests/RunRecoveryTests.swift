import XCTest

@testable import TurboSparkApp

/// When a failed tool card offers Retry, Skip and Tell the model
/// (`RunRecoveryActions`). The buttons only send or draft an ordinary user
/// reply, so the rules here are what keep them from racing a live agent.
final class RunRecoveryTests: XCTestCase {
    private func failedCall(_ name: String = "bash") -> AppToolCall {
        AppToolCall(name: name, arguments: ["command": "make"], status: .failed)
    }

    private func result(for call: AppToolCall, isError: Bool = true) -> AppToolResult {
        AppToolResult(callID: call.id, output: "exit 2", isError: isError)
    }

    private func turn(_ calls: [AppToolCall]) -> AppChatMessage {
        AppChatMessage(role: .assistant, content: "", toolCalls: calls, toolResults: calls.map { result(for: $0) })
    }

    func testOfferedWhenAFailedCallEndsAnIdleTurn() {
        let call = failedCall()
        let messages = [
            AppChatMessage(role: .user, content: "build it"),
            turn([call]),
            // The model giving up in prose does not hide the offer.
            AppChatMessage(role: .assistant, content: "The build failed."),
        ]
        XCTAssertTrue(RunRecoveryActions.isOffered(callID: call.id, in: messages, isIdle: true))
    }

    func testNotOfferedWhileTheTurnIsRunning() {
        let call = failedCall()
        XCTAssertFalse(RunRecoveryActions.isOffered(callID: call.id, in: [turn([call])], isIdle: false))
    }

    func testNotOfferedOnceTheUserRepliedOrTheAgentActedAgain() {
        let call = failedCall()
        let next = AppToolCall(name: "bash", status: .completed)
        XCTAssertFalse(RunRecoveryActions.isOffered(callID: call.id, in: [
            turn([call]), AppChatMessage(role: .user, content: "try again"),
        ], isIdle: true))
        XCTAssertFalse(RunRecoveryActions.isOffered(callID: call.id, in: [
            turn([call]),
            AppChatMessage(role: .assistant, content: "", toolCalls: [next],
                toolResults: [AppToolResult(callID: next.id, output: "ok")]),
        ], isIdle: true))
    }

    /// A batch runs its members together, so any failed member of the last
    /// batch is part of the last thing that happened.
    func testOfferedForAFailedMemberOfTheLastBatch() {
        let failed = failedCall()
        let ok = AppToolCall(name: "read_file", status: .completed)
        let message = AppChatMessage(role: .assistant, content: "", toolCalls: [failed, ok],
            toolResults: [result(for: failed), AppToolResult(callID: ok.id, output: "text")])
        XCTAssertTrue(RunRecoveryActions.isOffered(callID: failed.id, in: [message], isIdle: true))
        XCTAssertFalse(RunRecoveryActions.isOffered(callID: ok.id, in: [message], isIdle: true))
    }

    /// A denial is the user's or the policy's answer, not a failure to
    /// recover from: retrying it would ask the same question again.
    func testNotOfferedForADeniedCall() {
        var call = failedCall()
        call.status = .denied
        XCTAssertFalse(RunRecoveryActions.isOffered(callID: call.id, in: [turn([call])], isIdle: true))
    }

    func testNotOfferedWithoutAnErrorResultOrForAnUnknownCall() {
        let call = failedCall()
        XCTAssertFalse(RunRecoveryActions.isOffered(callID: call.id, in: [
            AppChatMessage(role: .assistant, content: "", toolCalls: [call])
        ], isIdle: true))
        XCTAssertFalse(RunRecoveryActions.isOffered(callID: call.id, in: [
            AppChatMessage(role: .assistant, content: "", toolCalls: [call],
                toolResults: [result(for: call, isError: false)])
        ], isIdle: true))
        XCTAssertFalse(RunRecoveryActions.isOffered(callID: UUID(), in: [turn([call])], isIdle: true))
    }

    /// The app quitting mid-call leaves `.failed` plus an error result
    /// (`reconcilingOrphanedToolCalls`), so a relaunched chat offers recovery.
    func testOfferedForACallInterruptedByAQuit() {
        let running = AppChatMessage(role: .assistant, content: "",
            toolCalls: [AppToolCall(name: "bash", status: .running)])
        guard let repaired = AppModel.reconcilingOrphanedToolCalls(
            [AppChat(title: "t", messages: [running])]).first?.messages.first,
            let call = repaired.toolCalls.first
        else { return XCTFail("repair dropped the message") }
        XCTAssertTrue(RunRecoveryActions.isOffered(callID: call.id, in: [repaired], isIdle: true))
    }
}
