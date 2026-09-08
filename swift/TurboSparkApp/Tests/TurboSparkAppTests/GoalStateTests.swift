import XCTest

@testable import TurboSparkApp

/// Goal state at the persistence and assembly boundaries: the archive
/// round-trip (including the pre-goal decode), the ghost payload, and the
/// assembly-time `<goal>` reminder. No AppModel, no session.
final class GoalStateTests: XCTestCase {
    // MARK: - AppChat archive decode

    func testGoalRoundTripsThroughTheArchive() throws {
        var chat = AppChat()
        chat.goal = ChatGoalState(condition: "all tests pass")
        let data = try JSONEncoder().encode(AppChatArchive(selectedChatID: chat.id, chats: [chat]))
        let decoded = try JSONDecoder().decode(AppChatArchive.self, from: data)
        XCTAssertEqual(decoded.chats.first?.goal?.condition, "all tests pass")
    }

    func testArchiveWrittenBeforeGoalsExistedDecodesWithoutOne() throws {
        let chat = AppChat()
        let data = try JSONEncoder().encode(AppChatArchive(selectedChatID: chat.id, chats: [chat]))
        // The lenient form must not even flinch on a WRONG-TYPED goal: the
        // field is lost, never the chat (the samplingOverride precedent).
        var damaged = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        var chatJSON = (damaged["chats"] as! [[String: Any]])[0]
        chatJSON["goal"] = 42
        damaged["chats"] = [chatJSON]
        let repaired = try JSONSerialization.data(withJSONObject: damaged)
        let decoded = try JSONDecoder().decode(AppChatArchive.self, from: repaired)
        XCTAssertNil(decoded.chats.first?.goal)
        XCTAssertEqual(decoded.chats.count, 1, "one hand-edited key took the whole row down")
    }

    // MARK: - Ghost payload

    func testGhostPayloadCarriesAndReportsAGoal() throws {
        var payload = GhostChatPayload()
        XCTAssertNil(payload.goal)
        XCTAssertFalse(payload.hasContent, "an empty payload has no content")
        payload.goal = ChatGoalState(condition: "migrate the config")
        XCTAssertTrue(payload.hasContent, "a goal alone is something a user could lose")
        // The vault seals with a per-launch key, so its decode never meets
        // an older schema in practice -- but the field is optional anyway,
        // which this round-trip pins.
        let data = try JSONEncoder().encode(payload)
        let decoded = try JSONDecoder().decode(GhostChatPayload.self, from: data)
        XCTAssertEqual(decoded.goal?.condition, "migrate the config")
    }

    // MARK: - Assembly-time reminder

    func testReminderCarriesConditionAndLatestReason() {
        var goal = ChatGoalState(condition: "all tests pass")
        goal.iterations = 2
        goal.lastReason = "one test still fails"
        let section = SystemReminders.goalSection(goal)
        XCTAssertTrue(section.contains("ACTIVE GOAL"))
        XCTAssertTrue(section.contains("all tests pass"))
        XCTAssertTrue(section.contains("one test still fails"))
        // Without a reason yet (first round), the section still states the
        // goal rather than going quiet.
        let fresh = SystemReminders.goalSection(ChatGoalState(condition: "ship it"))
        XCTAssertTrue(fresh.contains("ship it"))
        XCTAssertTrue(fresh.contains("PAUSED") == false)
        // The paused variant tells the model to answer the user instead of
        // spinning on.
        var paused = ChatGoalState(condition: "ship it")
        paused.isPaused = true
        XCTAssertTrue(SystemReminders.goalSection(paused).contains("PAUSED"))
    }

    func testReminderIncludesTheGoalSectionAlongsideTheOthers() {
        var goal = ChatGoalState(condition: "all tests pass")
        goal.lastReason = "lint fails"
        let reminder = SystemReminders.reminder(
            todos: [], messages: [], planModeActive: false, goal: goal)
        XCTAssertNotNil(reminder)
        XCTAssertTrue(reminder!.contains("<system-reminder>"))
        XCTAssertTrue(reminder!.contains("all tests pass"))
        // No goal, no section, and the pre-existing nil contract holds.
        XCTAssertNil(
            SystemReminders.reminder(todos: [], messages: [], planModeActive: false, goal: nil))
    }

    // MARK: - Elapsed formatting

    func testElapsedFormatting() {
        let now = Date()
        XCTAssertEqual(AppModel.formatGoalElapsed(since: now, now: now), "0m")
        XCTAssertEqual(
            AppModel.formatGoalElapsed(
                since: now.addingTimeInterval(-5 * 60), now: now), "5m")
        XCTAssertEqual(
            AppModel.formatGoalElapsed(
                since: now.addingTimeInterval(-2 * 3600), now: now), "2h")
        XCTAssertEqual(
            AppModel.formatGoalElapsed(
                since: now.addingTimeInterval(-2 * 3600 - 15 * 60), now: now), "2h 15m")
    }
}
