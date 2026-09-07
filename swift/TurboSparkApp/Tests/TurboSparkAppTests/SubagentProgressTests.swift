import XCTest

@testable import TurboSparkApp

/// The subagent progress seam and the `agent` tool's result contract
/// (swift/docs/SWIFT_TOOLS.md section 12): the event bracket, the run state a
/// card renders from, and the trailer + no-output sentence the parent
/// reads. All reachable without a model session -- the no-session run is
/// the shortest path through `run` that still emits the full event bracket.
final class SubagentProgressTests: XCTestCase {
    /// Mutable capture across the `@Sendable` progress sink, boxed because
    /// the sink is ordered by the runner (it awaits each event before the
    /// next) and the read happens only after `run` returns.
    private final class EventBox: @unchecked Sendable {
        var events: [SubagentProgressEvent] = []
    }

    private func collectEvents(
        prompt: String = "do a thing"
    ) async -> [SubagentProgressEvent] {
        let box = EventBox()
        let agent = AgentManager.shared.builtInAgents[0]
        _ = await SubagentRunner.run(
            agent: agent, taskPrompt: prompt, session: nil, project: nil,
            progress: { event in
                box.events.append(event)
            })
        return box.events
    }

    // MARK: - the event bracket

    func testEveryRunIsBracketedByStartedAndFinishedEvenWhenItStartsNowhere() async {
        let events = await collectEvents()
        guard case .started = events.first else {
            return XCTFail("The first event must be `.started`, got \(String(describing: events.first))")
        }
        guard case .finished(let status) = events.last else {
            return XCTFail("The last event must be `.finished`, got \(String(describing: events.last))")
        }
        // No session: `runBody` returns before the loop, and the wrapper
        // still reports the same status the result carries.
        XCTAssertEqual(status, "failed")
    }

    func testTheStartedEventCarriesTheRunIdentityAndBoundedPromptHead() async {
        let events = await collectEvents(prompt: String(repeating: "x", count: 5_000))
        guard case .started(let agentName, let displayName, let description, let head, let chatID) = events.first else {
            return XCTFail("No `.started` event")
        }
        let builtIn = AgentManager.shared.builtInAgents[0]
        XCTAssertEqual(agentName, builtIn.name)
        XCTAssertEqual(displayName, builtIn.displayName)
        XCTAssertEqual(description, "")
        XCTAssertLessThanOrEqual(head.count, 303, "A card header must not carry a 5,000-char prompt.")
        XCTAssertTrue(head.hasSuffix("..."))
        XCTAssertNil(chatID)
    }

    // MARK: - SubagentRunState, the thing a card observes

    @MainActor
    func testRunStateAccumulatesContentCountsTurnsAndMarksToolRowsInPlace() {
        let state = SubagentRunState(id: "k", mode: .foreground, chatID: nil)
        state.apply(.started(
            agentName: "explore", displayName: "Explore", taskDescription: "find usages",
            promptHead: "find usages of Foo", chatID: nil))
        state.apply(.turnStarted(number: 1))
        state.apply(.content("hel"))
        state.apply(.content("lo"))
        state.apply(.toolStarted(name: "search_code", summary: "\"Foo\""))
        state.apply(.toolFinished(name: "search_code", summary: "\"Foo\"", isError: false))
        state.apply(.toolStarted(name: "read_file", summary: "src/a.rs"))
        state.apply(.toolFinished(name: "read_file", summary: "src/a.rs", isError: true))
        state.apply(.turnStarted(number: 2))
        state.apply(.finished(status: "completed"))

        XCTAssertEqual(state.agentName, "explore")
        XCTAssertEqual(state.streamedText, "hello")
        XCTAssertEqual(state.turns, 2)
        XCTAssertEqual(state.status, "completed")
        // One row per call, with its OUTCOME -- not one row for started and
        // another for finished.
        XCTAssertEqual(state.toolRows.count, 2)
        XCTAssertEqual(state.toolRows[0].name, "search_code")
        XCTAssertFalse(state.toolRows[0].isError)
        XCTAssertFalse(state.toolRows[0].isRunning)
        XCTAssertEqual(state.toolRows[1].name, "read_file")
        XCTAssertTrue(state.toolRows[1].isError)
        XCTAssertFalse(state.toolRows[1].isRunning)
    }

    @MainActor
    func testToolRowTracksRunningStateUntilFinished() {
        let state = SubagentRunState(id: "k2", mode: .foreground, chatID: nil)
        state.apply(.toolStarted(name: "grep_search", summary: "\"foo\""))
        XCTAssertEqual(state.toolRows.count, 1)
        XCTAssertTrue(state.toolRows[0].isRunning, "Newly started tool must have isRunning == true")
        XCTAssertFalse(state.toolRows[0].isError)

        state.apply(.toolFinished(name: "grep_search", summary: "\"foo\"", isError: false))
        XCTAssertEqual(state.toolRows.count, 1)
        XCTAssertFalse(state.toolRows[0].isRunning, "Finished tool must have isRunning == false")

        // Interrupted/finished run clears isRunning on any running tool
        state.apply(.toolStarted(name: "bash", summary: "ls"))
        XCTAssertTrue(state.toolRows.last?.isRunning ?? false)
        state.apply(.finished(status: "cancelled"))
        XCTAssertFalse(state.toolRows.last?.isRunning ?? true, "Cancelled run must clear isRunning")
    }

    // MARK: - the result trailer

    func testTheTrailerCarriesIdentityAndStatsAndEmptyOutputIsSaidSo() {
        let result = SubagentRunResult(
            agentName: "explore", status: "completed", finalResponse: "  found 3 uses  ",
            totalTurns: 3, totalToolCalls: 4, durationSeconds: 1.25, runID: "run-1")
        let output = SubagentRunner.toolOutput(for: result, agentName: "explore")

        XCTAssertTrue(output.hasPrefix("found 3 uses"), "The answer comes first, untrimmed content aside.")
        XCTAssertTrue(output.contains("<subagent_meta>"))
        XCTAssertTrue(output.contains("id: run-1"))
        XCTAssertTrue(output.contains("status: completed"))
        XCTAssertTrue(output.contains("turns: 3"))
        XCTAssertTrue(output.contains("tool_calls: 4"))
        XCTAssertTrue(output.contains("duration_s: 1.2"))
        XCTAssertTrue(output.contains("</subagent_meta>"))
        XCTAssertFalse(output.contains("\n\n  found"), "Whitespace-padded output is trimmed before the trailer.")

        let empty = SubagentRunner.toolOutput(
            for: SubagentRunResult(
                agentName: "explore", status: "completed", finalResponse: "   ",
                totalTurns: 1, totalToolCalls: 0, durationSeconds: 0.1, runID: "run-2"),
            agentName: "explore")
        XCTAssertTrue(
            empty.hasPrefix("(Subagent completed but returned no output.)"),
            "Silence must be named, or the parent reads it as an empty tool result.")
    }
}
