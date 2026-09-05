import XCTest
@testable import TurboSparkApp

final class PlanningAndInteractiveToolTests: XCTestCase {
    private func makeProject() throws -> (AppProject, URL) {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let project = AppProject(name: "test-planning", rootDirectoryPath: dir.path)
        return (project, dir)
    }

    // MARK: - ask_user_question / askuserquestion / question

    func testAskUserQuestionSingleSelect() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        var receivedQuestions: [UserQuestionItem]?
        AskUserQuestionExecutor.onQuestionsAsked = { _, items in
            receivedQuestions = items
        }
        defer { AskUserQuestionExecutor.onQuestionsAsked = nil }

        for alias in ["ask_user_question", "askuserquestion", "ask_question", "question"] {
            let call = AppToolCall(
                name: alias,
                arguments: [
                    "question": "Which architecture do you prefer?",
                    "options": "[\"Option 1\", \"Option 2\", \"Option 3\"]",
                    "multi_select": "false"
                ],
                category: .automation
            )
            let result = await AppToolRegistry.execute(call: call, in: project)
            XCTAssertFalse(result.isError, "Alias \(alias) should succeed: \(result.output)")
            XCTAssertEqual(receivedQuestions?.first?.question, "Which architecture do you prefer?")
            XCTAssertEqual(receivedQuestions?.first?.options.count, 3)
            XCTAssertEqual(receivedQuestions?.first?.multiSelect, false)
            XCTAssertTrue(result.output.contains("Option 1"))
        }
    }

    func testAskUserQuestionMultiSelectAndMissingQuestion() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        var isMulti: Bool?
        AskUserQuestionExecutor.onQuestionsAsked = { _, items in
            isMulti = items.first?.multiSelect
        }
        defer { AskUserQuestionExecutor.onQuestionsAsked = nil }

        let call = AppToolCall(
            name: "ask_user_question",
            arguments: [
                "question": "Select features",
                "options": "[\"A\", \"B\", \"C\"]",
                "multi_select": "true"
            ],
            category: .automation
        )
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError)
        XCTAssertEqual(isMulti, true)
        XCTAssertTrue(result.output.contains("Select features"))

        let missingCall = AppToolCall(
            name: "ask_user_question",
            arguments: [:],
            category: .automation
        )
        let missingResult = await AppToolRegistry.execute(call: missingCall, in: project)
        XCTAssertTrue(missingResult.isError, "Missing question parameter must report error")
    }

    // MARK: - enter_plan_mode / exit_plan_mode / plan_mode

    func testPlanModeTransitionsAndLifecycle() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        var planState: (active: Bool, plan: String?)?
        PlanModeExecutor.onPlanModeChanged = { _, active, plan in
            planState = (active, plan)
        }
        defer { PlanModeExecutor.onPlanModeChanged = nil }

        // Enter plan mode
        for alias in ["enter_plan_mode", "enterplanmode", "plan_mode", "plan"] {
            let enterCall = AppToolCall(
                name: alias,
                arguments: ["plan": "Step 1: Inspect\nStep 2: Implement"],
                category: .automation
            )
            let enterResult = await AppToolRegistry.execute(call: enterCall, in: project)
            XCTAssertFalse(enterResult.isError, "Alias \(alias) should enter plan mode")
            XCTAssertEqual(planState?.active, true)
        }

        // Exit plan mode
        for alias in ["exit_plan_mode", "exitplanmode"] {
            let exitCall = AppToolCall(
                name: alias,
                arguments: ["plan": "Step 1: Done\nStep 2: Done"],
                category: .automation
            )
            let exitResult = await AppToolRegistry.execute(call: exitCall, in: project)
            XCTAssertFalse(exitResult.isError, "Alias \(alias) should exit plan mode")
            XCTAssertEqual(planState?.active, false)
            XCTAssertEqual(planState?.plan, "Step 1: Done\nStep 2: Done")
        }
    }

    // MARK: - report_findings / findings

    func testReportFindingsStructuredFormatting() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        for alias in ["report_findings", "reportfindings", "findings"] {
            let call = AppToolCall(
                name: alias,
                arguments: [
                    "summary": "Completed root cause analysis",
                    "findings": "[{\"file\": \"main.swift\", \"summary\": \"Bug Root Cause\", \"failure_scenario\": \"Off by one error\"}]",
                    "level": "high"
                ],
                category: .automation
            )
            let result = await AppToolRegistry.execute(call: call, in: project)
            XCTAssertFalse(result.isError, "Alias \(alias) should report findings: \(result.output)")
            XCTAssertTrue(result.output.contains("Bug Root Cause"))
            XCTAssertTrue(result.output.contains("Off by one error"))
        }
    }

    // MARK: - propose_skills / propose_goal / send_feedback

    func testProposeSkillsAndGoalsAndFeedback() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let skillsCall = AppToolCall(
            name: "propose_skills",
            arguments: [
                "name": "rust-expert",
                "description": "Guides safe concurrency in Rust",
                "rules": "Never use unsafe without soundness doc"
            ],
            category: .automation
        )
        let skillsResult = await AppToolRegistry.execute(call: skillsCall, in: project)
        XCTAssertFalse(skillsResult.isError, "propose_skills should succeed: \(skillsResult.output)")
        XCTAssertTrue(skillsResult.output.contains("proposed skill"))
        let skillFile = dir.appendingPathComponent(".turbospark/skills/rust-expert/SKILL.md")
        XCTAssertTrue(FileManager.default.fileExists(atPath: skillFile.path), "Skill file should exist on disk")

        let goalCall = AppToolCall(
            name: "propose_goal",
            arguments: [
                "goal": "Refactor all network clients to use async/await",
                "success_criteria": "All unit tests pass and no thread locks"
            ],
            category: .automation
        )
        let goalResult = await AppToolRegistry.execute(call: goalCall, in: project)
        XCTAssertFalse(goalResult.isError, "propose_goal should succeed: \(goalResult.output)")
        XCTAssertTrue(goalResult.output.contains("Refactor all network clients"))

        let feedbackCall = AppToolCall(
            name: "send_feedback",
            arguments: [
                "feedback": "The proposal looks great, proceed with phase 1."
            ],
            category: .automation
        )
        let feedbackResult = await AppToolRegistry.execute(call: feedbackCall, in: project)
        XCTAssertFalse(feedbackResult.isError, "send_feedback should succeed: \(feedbackResult.output)")
        XCTAssertTrue(feedbackResult.output.contains("The proposal looks great"))
    }

    // MARK: - todo_write / todowrite

    func testTodoWriteItemManagement() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        for alias in ["todo_write", "todowrite"] {
            let call = AppToolCall(
                name: alias,
                arguments: [
                    "action": "create",
                    "title": "Write unit tests",
                    "status": "pending"
                ],
                category: .automation
            )
            let result = await AppToolRegistry.execute(call: call, in: project)
            XCTAssertFalse(result.isError, "Alias \(alias) should manage todo: \(result.output)")
        }
    }
}
