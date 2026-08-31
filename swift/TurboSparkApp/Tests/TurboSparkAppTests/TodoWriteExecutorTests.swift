import XCTest
@testable import TurboSparkApp

final class TodoWriteExecutorTests: XCTestCase {
    func testParseTodosFromStandardJSONArray() throws {
        let json = """
        [
            {"id": "t1", "content": "Design architecture", "status": "completed", "activeForm": "Designing architecture"},
            {"id": "t2", "content": "Implement executor", "status": "in_progress", "activeForm": "Implementing executor"},
            {"id": "t3", "content": "Write unit tests", "status": "pending", "activeForm": "Writing unit tests"}
        ]
        """
        let args = ["todos": json]
        let items = try TodoWriteExecutor.parseTodos(from: args)

        XCTAssertEqual(items.count, 3)
        XCTAssertEqual(items[0].id, "t1")
        XCTAssertEqual(items[0].content, "Design architecture")
        XCTAssertTrue(items[0].isCompleted)
        XCTAssertFalse(items[0].isInProgress)

        XCTAssertEqual(items[1].id, "t2")
        XCTAssertEqual(items[1].content, "Implement executor")
        XCTAssertTrue(items[1].isInProgress)
        XCTAssertEqual(items[1].activeForm, "Implementing executor")

        XCTAssertEqual(items[2].id, "t3")
        XCTAssertEqual(items[2].content, "Write unit tests")
        XCTAssertTrue(items[2].isPending)
    }

    func testParseTodosFromObjectWithTodosKey() throws {
        let json = """
        {
            "todos": [
                {"content": "Step 1", "status": "done"},
                {"content": "Step 2", "status": "running", "activeForm": "Processing step 2"}
            ]
        }
        """
        let args = ["todos": json]
        let items = try TodoWriteExecutor.parseTodos(from: args)

        XCTAssertEqual(items.count, 2)
        XCTAssertEqual(items[0].content, "Step 1")
        XCTAssertTrue(items[0].isCompleted)

        XCTAssertEqual(items[1].content, "Step 2")
        XCTAssertTrue(items[1].isInProgress)
        XCTAssertEqual(items[1].activeForm, "Processing step 2")
    }

    func testParseTodosFromStringArray() throws {
        let json = """
        ["Setup project", "Build frontend", "Run validation"]
        """
        let args = ["todos": json]
        let items = try TodoWriteExecutor.parseTodos(from: args)

        XCTAssertEqual(items.count, 3)
        XCTAssertEqual(items[0].content, "Setup project")
        XCTAssertTrue(items[0].isPending)
        XCTAssertEqual(items[0].activeForm, "Setup project")
    }

    func testParseTodosFromSingleItemArguments() throws {
        let args = [
            "content": "Refactor parser",
            "status": "in_progress",
            "activeForm": "Refactoring parser"
        ]
        let items = try TodoWriteExecutor.parseTodos(from: args)

        XCTAssertEqual(items.count, 1)
        XCTAssertEqual(items[0].content, "Refactor parser")
        XCTAssertTrue(items[0].isInProgress)
        XCTAssertEqual(items[0].activeForm, "Refactoring parser")
    }

    func testExecuteGeneratesFormattedOutputAndTriggersCallback() throws {
        let expectation = expectation(description: "onTodosUpdated callback invoked")
        let targetChatID = UUID()

        TodoWriteExecutor.onTodosUpdated = { chatID, todos in
            XCTAssertEqual(chatID, targetChatID)
            XCTAssertEqual(todos.count, 2)
            expectation.fulfill()
        }

        let json = """
        [
            {"content": "Read spec", "status": "completed", "activeForm": "Reading spec"},
            {"content": "Write code", "status": "in_progress", "activeForm": "Writing code"}
        ]
        """
        let args = ["todos": json]
        let result = try TodoWriteExecutor.execute(arguments: args, chatID: targetChatID)

        waitForExpectations(timeout: 1.0)

        XCTAssertEqual(result.todos.count, 2)
        XCTAssertTrue(result.output.contains("1 completed, 1 in progress, 0 pending"))
        XCTAssertTrue(result.output.contains("[x] Read spec"))
        XCTAssertTrue(result.output.contains("[>] Write code (Active: Writing code)"))

        // Clean up callback
        TodoWriteExecutor.onTodosUpdated = nil
    }

    func testTodoItemStatusHelpers() {
        let completedItem = TodoItem(content: "A", status: "completed")
        let inProgressItem = TodoItem(content: "B", status: "in_progress")
        let pendingItem = TodoItem(content: "C", status: "pending")
        let cancelledItem = TodoItem(content: "D", status: "cancelled")

        XCTAssertTrue(completedItem.isCompleted)
        XCTAssertFalse(completedItem.isInProgress)

        XCTAssertTrue(inProgressItem.isInProgress)
        XCTAssertFalse(inProgressItem.isCompleted)

        XCTAssertTrue(pendingItem.isPending)
        XCTAssertFalse(pendingItem.isInProgress)

        XCTAssertTrue(cancelledItem.isCancelled)
    }

    func testAppChatTolerantDecodingWithTodos() throws {
        let chat = AppChat(
            title: "Task Chat",
            todos: [
                TodoItem(id: "1", content: "Item 1", status: "completed", activeForm: "Doing 1"),
                TodoItem(id: "2", content: "Item 2", status: "in_progress", activeForm: "Doing 2")
            ]
        )

        let encoded = try JSONEncoder().encode(chat)
        let decoded = try JSONDecoder().decode(AppChat.self, from: encoded)

        XCTAssertEqual(decoded.todos.count, 2)
        XCTAssertEqual(decoded.todos[0].content, "Item 1")
        XCTAssertTrue(decoded.todos[0].isCompleted)
        XCTAssertEqual(decoded.todos[1].content, "Item 2")
        XCTAssertTrue(decoded.todos[1].isInProgress)

        // Legacy JSON without "todos" key should decode cleanly with empty todos
        let legacyJSON = """
        {
            "id": "\(UUID().uuidString)",
            "title": "Legacy Chat",
            "draft": "hello",
            "messages": [],
            "createdAt": 0,
            "updatedAt": 0
        }
        """
        let decodedLegacy = try JSONDecoder().decode(AppChat.self, from: legacyJSON.data(using: .utf8)!)
        XCTAssertEqual(decodedLegacy.todos, [])
    }

    func testToolCallDiffFormatterTodoSummary() {
        let json = """
        [
            {"content": "T1", "status": "completed"},
            {"content": "T2", "status": "completed"},
            {"content": "T3", "status": "in_progress", "activeForm": "Working on T3"},
            {"content": "T4", "status": "pending"}
        ]
        """
        let summary = ToolCallDiffFormatter.summarize(callName: "todowrite", arguments: ["todos": json])
        XCTAssertEqual(summary.action, "Tasks")
        XCTAssertEqual(summary.target, "2/4 done (1 in progress)")
    }
}
