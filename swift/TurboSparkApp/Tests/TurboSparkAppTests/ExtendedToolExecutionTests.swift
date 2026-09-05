import XCTest
@testable import TurboSparkApp

final class ExtendedToolExecutionTests: XCTestCase {
    var tempDir: URL!
    var project: AppProject!

    override func setUp() async throws {
        try await super.setUp()
        tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        project = AppProject(name: "test-proj", rootDirectoryPath: tempDir.path)
    }

    override func tearDown() async throws {
        if let tempDir {
            try? FileManager.default.removeItem(at: tempDir)
        }
        try await super.tearDown()
    }

    // MARK: - Planning & Interactive Tools

    func testAskUserQuestionExecution() async throws {
        let json = """
        [{"question": "Which architecture?", "header": "Arch", "options": [{"label": "MVC", "description": "Classic"}, {"label": "MVVM", "description": "Reactive"}], "multiSelect": false}]
        """
        let call = AppToolCall(name: "AskUserQuestion", arguments: ["questions": json], category: .automation)
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError, "AskUserQuestion must execute without error: \(result.output)")
        XCTAssertTrue(result.output.contains("Arch"))
        XCTAssertTrue(result.output.contains("MVC"))
    }

    func testPlanModeLifecycle() async {
        let chatID = UUID()
        let enterCall = AppToolCall(name: "EnterPlanMode", arguments: [:], category: .automation)
        let enterResult = await AppToolRegistry.execute(call: enterCall, in: project, chatID: chatID)
        XCTAssertFalse(enterResult.isError)
        XCTAssertTrue(PlanModeExecutor.isPlanModeActive(for: chatID))

        let exitCall = AppToolCall(name: "ExitPlanMode", arguments: ["plan": "# Refactoring Plan\n1. Step A\n2. Step B"], category: .automation)
        let exitResult = await AppToolRegistry.execute(call: exitCall, in: project, chatID: chatID)
        XCTAssertFalse(exitResult.isError)
        XCTAssertFalse(PlanModeExecutor.isPlanModeActive(for: chatID))
        XCTAssertTrue(exitResult.output.contains("Refactoring Plan"))
    }

    func testReportFindingsExecution() async {
        let json = """
        [{"file": "src/main.swift", "line": 42, "summary": "Unchecked optional", "failure_scenario": "Null token passed", "category": "Crash"}]
        """
        let call = AppToolCall(name: "ReportFindings", arguments: ["findings": json, "level": "high"], category: .automation)
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError)
        XCTAssertTrue(result.output.contains("src/main.swift:42"))
        XCTAssertTrue(result.output.contains("Unchecked optional"))
    }

    func testProposeSkillsAndGoals() async throws {
        let callSkill = AppToolCall(
            name: "ProposeSkills",
            arguments: [
                "name": "swift-helper",
                "skillMd": "---\nname: swift-helper\n---\nHelp with swift"
            ],
            category: .fileWrite
        )
        let resSkill = await AppToolRegistry.execute(call: callSkill, in: project)
        XCTAssertFalse(resSkill.isError)
        let skillFile = tempDir.appendingPathComponent(".turbospark/skills/swift-helper/SKILL.md")
        XCTAssertTrue(FileManager.default.fileExists(atPath: skillFile.path))

        let callGoal = AppToolCall(name: "ProposeGoal", arguments: ["condition": "Build passes with 0 errors"], category: .automation)
        let resGoal = await AppToolRegistry.execute(call: callGoal, in: project)
        XCTAssertFalse(resGoal.isError)
        XCTAssertTrue(resGoal.output.contains("Build passes with 0 errors"))
    }

    // MARK: - File & Utility Tools

    func testSnipExecution() async throws {
        let file = tempDir.appendingPathComponent("code.swift")
        let lines = (1...20).map { "let line\($0) = \($0)" }.joined(separator: "\n")
        try lines.write(to: file, atomically: true, encoding: .utf8)

        let call = AppToolCall(name: "Snip", arguments: ["path": "code.swift", "start_line": "5", "end_line": "8"], category: .fileRead)
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError)
        XCTAssertTrue(result.output.contains("5 | let line5 = 5"))
        XCTAssertTrue(result.output.contains("8 | let line8 = 8"))
        XCTAssertFalse(result.output.contains("4 | let line4 = 4"))
    }

    func testNotebookEditExecution() async throws {
        let nbFile = tempDir.appendingPathComponent("analysis.ipynb")
        let initialJson = """
        {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "id": "c1",
                    "metadata": {},
                    "outputs": [],
                    "source": ["import math\\n", "print(math.pi)"]
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5
        }
        """
        try initialJson.write(to: nbFile, atomically: true, encoding: .utf8)

        let call = AppToolCall(
            name: "NotebookEdit",
            arguments: [
                "notebook_path": "analysis.ipynb",
                "cell_id": "c1",
                "new_source": "import numpy as np\nprint(np.zeros(5))",
                "edit_mode": "replace"
            ],
            category: .fileWrite
        )
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError, "NotebookEdit failed: \(result.output)")

        let modifiedData = try Data(contentsOf: nbFile)
        let json = try JSONSerialization.jsonObject(with: modifiedData) as! [String: Any]
        let cells = json["cells"] as! [[String: Any]]
        XCTAssertEqual(cells.count, 1)
        let source = (cells[0]["source"] as! [String]).joined()
        XCTAssertTrue(source.contains("import numpy as np"))
    }

    func testSendUserFileExecution() async throws {
        let file = tempDir.appendingPathComponent("report.pdf")
        try "dummy pdf data".write(to: file, atomically: true, encoding: .utf8)

        let call = AppToolCall(name: "SendUserFile", arguments: ["path": "report.pdf", "message": "Final report"], category: .fileRead)
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError)
        XCTAssertTrue(result.output.contains("report.pdf"))
        XCTAssertTrue(result.output.contains("Final report"))
    }

    // MARK: - Task Management Tools

    func testTaskManagerFullLifecycle() async {
        let chatID = UUID()

        // 1. Create
        let createCall = AppToolCall(name: "TaskCreate", arguments: ["subject": "Optimize memory", "description": "Check slot cache"], category: .automation)
        let createRes = await AppToolRegistry.execute(call: createCall, in: project, chatID: chatID)
        XCTAssertFalse(createRes.isError)

        // 2. List
        let listCall = AppToolCall(name: "TaskList", arguments: [:], category: .automation)
        let listRes = await AppToolRegistry.execute(call: listCall, in: project, chatID: chatID)
        XCTAssertFalse(listRes.isError)
        XCTAssertTrue(listRes.output.contains("Optimize memory"))

        // 3. Get first task
        let tasks = TaskManager.shared.listTasks(chatID: chatID)
        guard let firstTask = tasks.first else {
            XCTFail("No task created")
            return
        }

        let getCall = AppToolCall(name: "TaskGet", arguments: ["taskId": firstTask.id], category: .automation)
        let getRes = await AppToolRegistry.execute(call: getCall, in: project, chatID: chatID)
        XCTAssertFalse(getRes.isError)
        XCTAssertTrue(getRes.output.contains("Check slot cache"))

        // 4. Update
        let updateCall = AppToolCall(name: "TaskUpdate", arguments: ["taskId": firstTask.id, "status": "in_progress", "output": "Running profiler"], category: .automation)
        let updateRes = await AppToolRegistry.execute(call: updateCall, in: project, chatID: chatID)
        XCTAssertFalse(updateRes.isError)
        XCTAssertTrue(updateRes.output.contains("in_progress"))

        // 5. Output
        let outCall = AppToolCall(name: "TaskOutput", arguments: ["taskId": firstTask.id], category: .automation)
        let outRes = await AppToolRegistry.execute(call: outCall, in: project, chatID: chatID)
        XCTAssertFalse(outRes.isError)
        XCTAssertTrue(outRes.output.contains("Running profiler"))

        // 6. Stop
        let stopCall = AppToolCall(name: "TaskStop", arguments: ["taskId": firstTask.id], category: .automation)
        let stopRes = await AppToolRegistry.execute(call: stopCall, in: project, chatID: chatID)
        XCTAssertFalse(stopRes.isError)
        XCTAssertTrue(stopRes.output.contains("stopped"))
    }

    // MARK: - Automation Tools

    func testSleepAndConfigAndCtxInspect() async {
        let sleepCall = AppToolCall(name: "Sleep", arguments: ["seconds": "0.1"], category: .automation)
        let sleepRes = await AppToolRegistry.execute(call: sleepCall, in: nil)
        XCTAssertFalse(sleepRes.isError)
        XCTAssertTrue(sleepRes.output.contains("Slept for 0.1"))

        let configCall = AppToolCall(name: "Config", arguments: ["action": "get", "key": "permissions.mode"], category: .automation)
        let configRes = await AppToolRegistry.execute(call: configCall, in: project)
        XCTAssertFalse(configRes.isError)
        XCTAssertTrue(configRes.output.contains("permissions.mode"))

        let ctxCall = AppToolCall(name: "CtxInspect", arguments: [:], category: .automation)
        let ctxRes = await AppToolRegistry.execute(call: ctxCall, in: project)
        XCTAssertFalse(ctxRes.isError)
        XCTAssertTrue(ctxRes.output.contains("Registered Tools"))
    }
}
