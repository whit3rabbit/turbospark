import XCTest

@testable import TurboSpark
@testable import TurboSparkApp

final class AppChatMicrocompactIntegrationTests: XCTestCase {
    private let placeholder = "[Tool output omitted from model context.]"

    @MainActor
    func testChangedProjectionCountsThenPublishesEventBeforeWindowFit() async throws {
        let model = AppModel()
        model.stopCronScheduler()
        model.microcompactEnabled = true
        model.compactionKeepRecentTurns = 1
        model.microcompactMinimumSavingsTokens = 1

        let chatID = UUID()
        var chat = AppChat(title: "Integration")
        chat.id = chatID
        chat.messages = [AppChatMessage(role: .user, content: "Keep this transcript row.")]
        model.chats = [chat]
        model.selectedChatID = chatID
        let transcriptBeforeRequest = model.chats[0].messages

        let staleOutput = String(repeating: "stale-tool-output-", count: 40)
        let history = AppChatHistoryProjection(
            messages: [
                .tool("<tool_response>\n\(staleOutput)\n</tool_response>"),
                .user("current question"),
            ],
            sourceRowIndexByMessage: [0, 1],
            instructionPinBlockIndex: nil,
            sourceTranscriptRowCount: 2)
        var order: [String] = []
        let requestMessages = try await model.fitRequestHistoryAfterMicrocompact(
            chatID: chatID,
            history: history,
            countTokens: { messages in
                order.append("count")
                return messages.reduce(0) { $0 + $1.content.utf8.count }
            },
            fitWindow: { messages in
                order.append("fit")
                let event = try XCTUnwrap(model.compactionBoundaryEvents[chatID])
                XCTAssertEqual(event.chatID, chatID)
                XCTAssertEqual(event.rowsCleared, 1)
                XCTAssertGreaterThan(event.estimatedTokensSaved, 0)
                XCTAssertTrue(messages[0].content.contains(self.placeholder))
                XCTAssertFalse(messages.contains {
                    $0.content.contains("Some older tool output was shortened for this request.")
                })
                return messages
            })

        XCTAssertEqual(order, ["count", "count", "fit"])
        XCTAssertEqual(requestMessages.count, history.messages.count)
        XCTAssertEqual(history.messages[0].content, "<tool_response>\n\(staleOutput)\n</tool_response>")
        XCTAssertEqual(model.chats[0].messages, transcriptBeforeRequest)
        XCTAssertEqual(model.selectedCompactionBoundaryEvent?.chatID, chatID)
    }

    @MainActor
    func testDisabledMicrocompactFitsTheOriginalProjectionAndClearsItsPriorEvent() async throws {
        let model = AppModel()
        model.stopCronScheduler()
        model.compactionKeepRecentTurns = 1
        model.microcompactMinimumSavingsTokens = 1

        let chatID = UUID()
        let history = compactableHistory()
        _ = try await model.fitRequestHistoryAfterMicrocompact(
            chatID: chatID,
            history: history,
            countTokens: { messages in
                messages.reduce(0) { $0 + $1.content.utf8.count }
            },
            fitWindow: { $0 })
        XCTAssertNotNil(model.compactionBoundaryEvents[chatID])

        model.microcompactEnabled = false
        var tokenCountCalls = 0
        let requestMessages = try await model.fitRequestHistoryAfterMicrocompact(
            chatID: chatID,
            history: history,
            countTokens: { _ in
                tokenCountCalls += 1
                return 0
            },
            fitWindow: { $0 })

        XCTAssertEqual(requestMessages, history.messages)
        XCTAssertEqual(tokenCountCalls, 0)
        XCTAssertNil(model.compactionBoundaryEvents[chatID])
    }

    @MainActor
    func testNextNoOpRequestClearsOnlyThatChatsTransientEvent() async throws {
        let model = AppModel()
        model.stopCronScheduler()
        model.compactionKeepRecentTurns = 1
        model.microcompactMinimumSavingsTokens = 1

        let firstChatID = UUID()
        let secondChatID = UUID()
        _ = try await model.fitRequestHistoryAfterMicrocompact(
            chatID: firstChatID,
            history: compactableHistory(),
            countTokens: { messages in
                messages.reduce(0) { $0 + $1.content.utf8.count }
            },
            fitWindow: { $0 })
        XCTAssertNotNil(model.compactionBoundaryEvents[firstChatID])

        model.selectedChatID = secondChatID
        XCTAssertNil(model.selectedCompactionBoundaryEvent)

        let noOpHistory = AppChatHistoryProjection(
            messages: [.user("No stale tool output."), .assistant("Current answer")],
            sourceRowIndexByMessage: [0, 1],
            instructionPinBlockIndex: nil,
            sourceTranscriptRowCount: 2)
        var tokenCountCalls = 0
        _ = try await model.fitRequestHistoryAfterMicrocompact(
            chatID: firstChatID,
            history: noOpHistory,
            countTokens: { _ in
                tokenCountCalls += 1
                return 1
            },
            fitWindow: { $0 })

        XCTAssertNil(model.compactionBoundaryEvents[firstChatID])
        XCTAssertTrue(model.compactionBoundaryEvents.isEmpty)
        XCTAssertEqual(tokenCountCalls, 0)
    }

    func testDiagnosticsCarryAndRenderTheRequestBoundaryCounts() throws {
        let event = CompactionBoundaryEvent(
            chatID: UUID(), rowsCleared: 2, estimatedTokensSaved: 125)
        let resultData = Data(
            #"{"promptTokens":300,"newTokens":12,"prefillSeconds":0.5,"decodeSeconds":1.0,"stopReason":"endOfTurn","content":"done"}"#.utf8)
        let result = try JSONDecoder().decode(GenerationResult.self, from: resultData)

        let diagnostics = AppDiagnostics(
            result: result, compactionBoundaryEvent: event)
        XCTAssertEqual(diagnostics.compactionBoundaryEvent, event)

        let renderedRows = RunnerDiagnosticsSection.compactionBoundaryRows(for: event)
        XCTAssertEqual(renderedRows.map(\.label), [
            String(localized: "Rows shortened", bundle: .module),
            String(localized: "Estimated tokens saved", bundle: .module),
        ])
        XCTAssertEqual(renderedRows.map(\.value), ["2", "125"])
    }

    private func compactableHistory() -> AppChatHistoryProjection {
        let staleOutput = String(repeating: "stale-tool-output-", count: 40)
        return AppChatHistoryProjection(
            messages: [
                .tool("<tool_response>\n\(staleOutput)\n</tool_response>"),
                .user("current question"),
            ],
            sourceRowIndexByMessage: [0, 1],
            instructionPinBlockIndex: nil,
            sourceTranscriptRowCount: 2)
    }
}
