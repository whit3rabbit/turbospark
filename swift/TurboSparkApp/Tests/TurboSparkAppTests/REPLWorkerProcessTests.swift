import Foundation
import XCTest
@testable import TurboSparkApp

/// Task 3.1: launch and serialize worker requests. Integration checks run
/// the real packaged executable in serve mode: bounded requests return one
/// result each, bindings persist across process boundary calls, and
/// concurrent calls for one session never overlap (3.5).
final class REPLWorkerProcessTests: XCTestCase {
    // MARK: Helpers

    private func makeArtifactDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("repl-process-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }

    private func makeExecutableURL() throws -> URL {
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try XCTUnwrap(
            [
                ".build/out/Products/Debug/TurboSparkApp",
                ".build/out/Products/Release/TurboSparkApp",
                ".build/arm64-apple-macosx/debug/TurboSparkApp",
                ".build/arm64-apple-macosx/release/TurboSparkApp",
                ".build/debug/TurboSparkApp",
                ".build/release/TurboSparkApp"
            ]
            .map { packageRoot.appendingPathComponent($0, isDirectory: false) }
            .first { FileManager.default.isExecutableFile(atPath: $0.path) },
            "The TurboSparkApp executable should be available in the SwiftPM build products")
    }

    private func makeWorker(
        artifacts: URL,
        maximumRequestBytes: Int = 1_024 * 1_024
    ) throws -> REPLWorkerProcess {
        let worker = try REPLWorkerProcess(
            executableURL: makeExecutableURL(),
            configuration: REPLSessionConfiguration(artifactDirectory: artifacts),
            limits: REPLLimits(maximumRequestBytes: maximumRequestBytes))
        addTeardownBlock { worker.terminateAndWait() }
        return worker
    }

    // MARK: Bounded requests return one result each

    func testEvaluationsRoundTripThroughTheWorkerProcessAndPersistBindings() async throws {
        let worker = try makeWorker(artifacts: try makeArtifactDirectory())

        let declaration = await worker.evaluate(code: "var x = 21; x")
        XCTAssertEqual(declaration.status, .completed, declaration.errorText ?? "")

        let use = await worker.evaluate(code: "x * 2")
        XCTAssertEqual(use.status, .completed, use.errorText ?? "")
        XCTAssertEqual(use.completionText, "42",
                       "bindings must persist across process-boundary calls")

        let logged = await worker.evaluate(code: #"console.log("across the wire"); "done";"#)
        XCTAssertEqual(logged.status, .completed, logged.errorText ?? "")
        XCTAssertEqual(logged.consoleText, "across the wire")
        XCTAssertEqual(logged.completionText, "done")
    }

    func testMultilineAndQuotedScriptsSurviveTheWire() async throws {
        let worker = try makeWorker(artifacts: try makeArtifactDirectory())

        let result = await worker.evaluate(code: """
        const text = "line one\\nline two \\"quoted\\"";
        text.length
        """)

        XCTAssertEqual(result.status, .completed, result.errorText ?? "")
        XCTAssertNotNil(Int(result.completionText ?? ""),
                        "the newline and quote content must survive JSON transport")
    }

    // MARK: One evaluation in flight, concurrent calls never overlap (3.5)

    func testConcurrentEvaluationsNeverOverlapAndEachReturnsOneResult() async throws {
        let worker = try makeWorker(artifacts: try makeArtifactDirectory())
        let callCount = 6

        let results = await withTaskGroup(of: REPLCallResult.self) { group in
            for index in 0..<callCount {
                group.addTask {
                    await worker.evaluate(code: """
                    if (globalThis.busy) { globalThis.overlapped = true; }
                    globalThis.busy = true;
                    await Promise.resolve(1);
                    globalThis.busy = false;
                    "ok:\(index)"
                    """)
                }
            }
            var collected: [REPLCallResult] = []
            for await result in group {
                collected.append(result)
            }
            return collected
        }

        XCTAssertEqual(results.count, callCount, "each concurrent call returns exactly one result")
        for result in results {
            XCTAssertEqual(result.status, .completed, result.errorText ?? "")
            XCTAssertTrue((result.completionText ?? "").hasPrefix("ok:"))
        }
        let completions = Set(results.compactMap(\.completionText))
        XCTAssertEqual(
            completions.count, callCount,
            "every caller must see its own result, got: \(completions.sorted())")

        let probe = await worker.evaluate(code: """
        (globalThis.overlapped === true ? "OVERLAPPED" : "clean")
            + "|count:" + (globalThis.count = (globalThis.count || 0) + 1)
        """)
        XCTAssertEqual(probe.status, .completed, probe.errorText ?? "")
        XCTAssertEqual(
            probe.completionText, "clean|count:1",
            "no evaluation may observe another in flight, got: "
                + (probe.completionText ?? ""))
    }

    // MARK: Bounded requests reject oversized scripts locally

    func testOversizedScriptIsRejectedLocallyAndTheWorkerStaysUsable() async throws {
        let worker = try makeWorker(
            artifacts: try makeArtifactDirectory(),
            maximumRequestBytes: 64)

        let oversized = await worker.evaluate(
            code: String(repeating: "1", count: 200))
        XCTAssertEqual(oversized.status, .failed)
        XCTAssertTrue(
            (oversized.errorText ?? "").contains("request limit"),
            "the rejection must explain the bound, got: " + (oversized.errorText ?? ""))

        let small = await worker.evaluate(code: "40 + 2")
        XCTAssertEqual(small.status, .completed, small.errorText ?? "")
        XCTAssertEqual(small.completionText, "42",
                       "the worker must stay usable after a locally rejected request")
    }

    // MARK: Termination

    func testTerminateAndWaitStopsTheWorkerAndLaterCallsFailFast() async throws {
        let worker = try makeWorker(artifacts: try makeArtifactDirectory())

        let first = await worker.evaluate(code: "1 + 1")
        XCTAssertEqual(first.status, .completed, first.errorText ?? "")

        worker.terminateAndWait(grace: 10)
        XCTAssertFalse(worker.isRunning, "the worker process must exit on termination")

        let after = await worker.evaluate(code: "1 + 1")
        XCTAssertEqual(after.status, .failed)
        XCTAssertTrue(
            (after.errorText ?? "").contains("not running"),
            "calls after termination must fail fast, got: " + (after.errorText ?? ""))
    }
}

