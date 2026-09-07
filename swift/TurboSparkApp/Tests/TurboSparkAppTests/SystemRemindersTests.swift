import XCTest

@testable import TurboSpark
@testable import TurboSparkApp

/// `SystemReminders` -- the assembly-time `<system-reminder>` injection for
/// todo and plan-mode state. The matrix under test: WHICH state fires, WHEN
/// the todo reminder counts as stale, and the EPHEMERAL contract (the
/// reminder reaches the model-bound history, never the stored transcript).
@MainActor
final class SystemRemindersTests: XCTestCase {
    // MARK: - the pure builder

    func testNoTodosAndNoPlanModeProducesNothing() {
        XCTAssertNil(SystemReminders.reminder(
            todos: [], messages: [], planModeActive: false))
    }

    func testCompletedTodosProduceNothing() {
        let done = [TodoItem(content: "ship it", status: "completed")]
        XCTAssertNil(SystemReminders.reminder(
            todos: done, messages: [], planModeActive: false))
    }

    private func assistant(_ content: String, calls: [AppToolCall] = []) -> AppChatMessage {
        AppChatMessage(role: .assistant, content: content, toolCalls: calls)
    }

    private func todoWriteCall() -> AppToolCall {
        AppToolCall(approvalID: "a1", name: "todowrite", arguments: [:], rawInvocation: "todowrite()")
    }

    func testFreshTodosDoNotFireYet() {
        // The write happened THIS step: zero assistant turns since.
        let messages = [assistant("working", calls: [todoWriteCall()])]
        let todos = [TodoItem(content: "step one")]
        XCTAssertNil(SystemReminders.reminder(
            todos: todos, messages: messages, planModeActive: false))
    }

    func testStaleTodosFireWithTheListRendered() {
        var messages = [assistant("writing", calls: [todoWriteCall()])]
        messages.append(assistant("still working"))
        messages.append(assistant("done with this step"))
        let todos = [
            TodoItem(content: "step one", status: "completed"),
            TodoItem(content: "step two", status: "in_progress", activeForm: "Stepping two"),
        ]
        let reminder = SystemReminders.reminder(
            todos: todos, messages: messages, planModeActive: false)
        let text = try! XCTUnwrap(reminder)
        XCTAssertTrue(text.contains("<system-reminder>"))
        XCTAssertTrue(text.contains("2 assistant turns"))
        XCTAssertTrue(text.contains("step one"))
        XCTAssertTrue(text.contains("step two"))
        XCTAssertTrue(text.contains("todowrite"))
        // Completed items are listed for context but marked done.
        XCTAssertTrue(text.contains("[x]"))
        XCTAssertTrue(text.contains("[ ]"))
    }

    func testANeverWrittenListCountsEveryAssistantTurn() {
        let messages = [assistant("a"), assistant("b")]
        let todos = [TodoItem(content: "orphaned item")]
        let reminder = SystemReminders.reminder(
            todos: todos, messages: messages, planModeActive: false)
        XCTAssertNotNil(reminder, "Todos with no TodoWrite in sight are at least as stale.")
    }

    func testPlanModeAloneFiresWithTheExitInstruction() {
        let reminder = SystemReminders.reminder(
            todos: [], messages: [], planModeActive: true)
        let text = try! XCTUnwrap(reminder)
        XCTAssertTrue(text.contains("Plan mode is ACTIVE"))
        XCTAssertTrue(text.contains("exit_plan_mode"))
    }

    func testBothSectionsWrapIndependently() {
        let messages = [assistant("writing", calls: [todoWriteCall()]),
                        assistant("a"), assistant("b")]
        let todos = [TodoItem(content: "step two")]
        let reminder = SystemReminders.reminder(
            todos: todos, messages: messages, planModeActive: true)
        let text = try! XCTUnwrap(reminder)
        XCTAssertEqual(text.components(separatedBy: "<system-reminder>").count - 1, 2)
    }

    func testAssistantTurnsSinceLastTodoWriteCountsCorrectly() {
        let none = [assistant("a"), assistant("b")]
        XCTAssertEqual(SystemReminders.assistantTurnsSinceLastTodoWrite(in: none), 2)
        let fresh = [assistant("a"), assistant("b"), assistant("c", calls: [todoWriteCall()])]
        XCTAssertEqual(SystemReminders.assistantTurnsSinceLastTodoWrite(in: fresh), 0)
        // Other tool calls do not reset the clock, and name matching is
        // case-insensitive against what the dialect emitted.
        let otherTool = [assistant("a"), assistant("b", calls: [todoWriteCall()]),
                         assistant("c", calls: [AppToolCall(approvalID: "a2", name: "read_file", arguments: [:], rawInvocation: "x")])]
        XCTAssertEqual(SystemReminders.assistantTurnsSinceLastTodoWrite(in: otherTool), 1)
    }

    // MARK: - the assembly hook

    func testTheReminderReachesThePromptButNeverTheTranscript() {
        let model = AppModel()
        var chat = AppChat(title: "Reminders")
        chat.messages = [
            AppChatMessage(role: .user, content: "please do the thing"),
            assistant("working"),
            assistant("still working"),
            AppChatMessage(role: .user, content: "and then the other thing"),
        ]
        chat.todos = [TodoItem(content: "step two")]
        model.chats = [chat]
        model.selectedChatID = chat.id

        let history = model.buildAppendOnlyHistory(chatIndex: 0, project: nil)
        let lastUser = history.last(where: { $0.role == .user })
        let content = try! XCTUnwrap(lastUser?.content)
        XCTAssertTrue(
            content.contains("<system-reminder>"),
            "The reminder rides the current turn's model-bound copy.")
        XCTAssertTrue(content.hasPrefix("and then the other thing"))

        // The stored row is untouched by assembly.
        let row = model.chats[0]
        XCTAssertEqual(row.messages.last?.content, "and then the other thing")
        XCTAssertFalse(
            row.messages.contains(where: { $0.content.contains("<system-reminder>") }),
            "A reminder must never be persisted into the transcript.")
    }
}
