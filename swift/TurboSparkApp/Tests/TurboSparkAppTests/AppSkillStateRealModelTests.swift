import XCTest

@testable import TurboSparkApp

/// Does a REAL local model emit valid patches against the SHIPPED coding-agent
/// schema? `docs/SKILL_STATE.md` measured a two-key warehouse schema; this one
/// has five fields including a map, and that page names schema comprehension as
/// the failure class most likely to reappear as a schema grows. So the growth
/// gets its own check rather than an assumption.
///
/// The point of putting it here rather than in `scripts/` is that it uses the
/// shipped code as its own oracle: the same `AppSkillStateSchema` text the app
/// sends, and the same `AppSkillStatePatch` validator and merge the app runs.
/// A Python copy of the schema would be a second spelling that rots.
///
/// Skipped unless a server is running. Start one first:
///
///     ./target/release/turbospark-server --model ~/models/gemma4.gturbo \
///         --port 8123 --max-context 4096
///     TURBOSPARK_SKILL_STATE_SERVER=http://127.0.0.1:8123 \
///         swift test --filter AppSkillStateRealModelTests
final class AppSkillStateRealModelTests: XCTestCase {

    /// A realistic short agent run: a task, then tool results arriving one at a
    /// time. Each entry is what the runtime would put in front of the model as
    /// the newest observation.
    private static let observations = [
        "<tool_response>\nrun_command\nls src/\nauth.rs  main.rs  parser.rs\n</tool_response>",
        "<tool_response>\nread_file\nsrc/auth.rs\npub fn verify(token: &str) -> bool { token.len() > 8 }\n</tool_response>",
        "<tool_response>\nrun_command\ncargo test auth\ntest auth::rejects_short_token ... FAILED\n</tool_response>",
        "<tool_error>\nrun_command\ncargo test --all\nerror: could not compile `parser` (lib test)\n</tool_error>",
        "<tool_response>\nread_file\nsrc/parser.rs\nfn parse(s: &str) -> Option<u32> { s.parse().ok() }\n</tool_response>",
        "NOTE: the CI machine was rebooted; unrelated to this task.",
    ]

    private var endpoint: URL {
        get throws {
            guard let raw = ProcessInfo.processInfo.environment["TURBOSPARK_SKILL_STATE_SERVER"],
                let url = URL(string: raw + "/v1/chat/completions")
            else {
                throw XCTSkip(
                    "set TURBOSPARK_SKILL_STATE_SERVER to a running turbospark-server origin")
            }
            return url
        }
    }

    private func complete(messages: [[String: String]]) async throws -> String {
        var request = URLRequest(url: try endpoint)
        request.httpMethod = "POST"
        request.timeoutInterval = 600
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": "local",
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 512,
            "stream": false,
        ])
        let (data, response) = try await URLSession.shared.data(for: request)
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard code == 200 else {
            throw XCTSkip("server returned \(code): \(String(decoding: data, as: UTF8.self))")
        }
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let choices = json?["choices"] as? [[String: Any]]
        let message = choices?.first?["message"] as? [String: Any]
        return message?["content"] as? String ?? ""
    }

    /// Drives the real protocol over a real model and asserts every proposed
    /// patch validates, that the state actually accumulates, and that the
    /// irrelevant observation does not wipe it.
    func testARealModelEmitsValidPatchesAgainstTheShippedSchema() async throws {
        let url = try endpoint  // throws XCTSkip when unset
        _ = url

        let model = await AppModel()
        let spec = await model.skillStateProtocol()
        var state = AppSkillState()
        var validPatches = 0
        var proposedPatches = 0
        var prompts: [Int] = []

        for (index, observation) in Self.observations.enumerated() {
            var latest = "CURRENT STATE:\n\(state.rendered)\n\nLATEST RESULT:\n\(observation)"
            latest += "\n\nContinue. Emit a state patch for anything you learned, "
                + "then either call one tool or give your final answer."
            let messages = [
                ["role": "system", "content": spec],
                ["role": "user", "content": "Find and fix the failing auth test in this Rust project."],
                ["role": "user", "content": latest],
            ]
            prompts.append(latest.count)

            let reply = try await complete(messages: messages)
            guard let patch = AppSkillStatePatch.extract(from: reply) else {
                // No patch is legal for a step that learned nothing, though on
                // these observations it would be surprising.
                continue
            }
            proposedPatches += 1
            let errors = AppSkillStatePatch.validate(patch)
            XCTAssertTrue(
                errors.isEmpty,
                "step \(index) proposed an invalid patch: \(errors.joined(separator: "; "))\n\(reply)")
            if errors.isEmpty {
                validPatches += 1
                state.apply(patch: patch)
            }
        }

        XCTAssertGreaterThan(
            proposedPatches, 0,
            "the model never proposed a patch, so this asserted nothing")
        XCTAssertEqual(
            validPatches, proposedPatches,
            "every proposed patch must validate against the shipped schema")
        XCTAssertFalse(
            state.isEmpty,
            "state never accumulated, so the run carried nothing forward")

        // The bound is the whole point: the last observation's prompt must not
        // be dramatically larger than the first. Compared as characters, which
        // is a proxy for tokens but a fair one across a single run.
        let growth = Double(prompts.last ?? 0) / Double(max(prompts.first ?? 1, 1))
        XCTAssertLessThan(
            growth, 6.0,
            "the bounded prompt grew \(String(format: "%.1f", growth))x, which is not bounded")

        print("skill-state real check: \(validPatches)/\(proposedPatches) patches valid")
        print("final state:\n\(state.rendered)")
    }
}
