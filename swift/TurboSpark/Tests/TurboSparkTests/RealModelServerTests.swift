import Foundation
import XCTest

@testable import TurboSpark

/// End-to-end server integration tests against a real install.
final class RealModelServerTests: RealModelTestCase {

    /// End to end for the in-process server: start it from a REAL open
    /// session, make an actual HTTP request against the port it bound, and
    /// confirm the reply carries real generated text. No other gate proves
    /// this whole path together -- `crates/ffi/tests/c_surface.rs` covers
    /// the same shape against a SCRIPTED session with no model, and
    /// `testServerCABISymbolsLinkAndValidateNullArgs` only proves the header
    /// links, never that a real forward pass answers through it.
    func testInProcessServerServesARealGeneration() async throws {
        let session = try await TurboSparkSession(modelPath: try modelPath())
        let server = try await session.startServer()
        defer { server.stop() }

        let info = try server.info()
        XCTAssertNotEqual(info.port, 0, "port 0 must resolve to the actually bound port")
        XCTAssertFalse(info.authEnabled)
        XCTAssertEqual(
            info.host, "127.0.0.1",
            "the engine binds loopback and info must REPORT it, not leave a caller to assume it")

        // The URL comes from `info`, not from a literal beside it. Spelling
        // the host by hand here would keep this test green against any bind
        // the reported host no longer matched, which is exactly the gap on
        // the app side that this field closes.
        let base = try XCTUnwrap(info.baseURL)
        var request = URLRequest(
            url: base.appendingPathComponent("v1/chat/completions"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": "m",
            "max_tokens": 8,
            "temperature": 0.0,
            "messages": [["role": "user", "content": "Say hello in one short sentence."]],
        ])

        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 200)

        let body = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choices = try XCTUnwrap(body["choices"] as? [[String: Any]])
        let message = try XCTUnwrap(choices.first?["message"] as? [String: Any])
        let content = try XCTUnwrap(message["content"] as? String)
        XCTAssertFalse(content.isEmpty)
    }

    /// **`info()` AND `stop()` RACING MUST NOT TOUCH A FREED HANDLE.**
    ///
    /// `ts_server_stop` frees its C handle, so any window between reading the
    /// `stopped` flag and dereferencing the pointer is a use-after-free, and
    /// `TurboSparkServer` exists to have no such window. The bug this covers
    /// read the flag under the lock, UNLOCKED, and then made the C call --
    /// which looks careful and is exactly the race.
    ///
    /// The assertion is that this terminates without crashing and that every
    /// call after the stop throws rather than answering. A crash here is the
    /// failure; there is no softer signal a use-after-free gives.
    func testServerInfoRacingStopNeverTouchesAFreedHandle() async throws {
        let session = try await TurboSparkSession(modelPath: try modelPath())
        let server = try await session.startServer()

        // Prove it answers before the race, or a passing race proves nothing.
        XCTAssertNotEqual(try server.info().port, 0)

        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<64 {
                group.addTask {
                    // Either outcome is legal: a read that beat the stop, or
                    // the already-stopped error. Reading a freed pointer is
                    // not, and would crash rather than land here.
                    _ = try? server.info()
                }
            }
            group.addTask { server.stop() }
            group.addTask { server.stop() }
        }

        XCTAssertThrowsError(try server.info()) { error in
            let message = String(describing: error)
            XCTAssertTrue(
                message.contains("already been stopped"),
                "a stopped server must refuse by name, got: \(message)")
        }
    }
}
