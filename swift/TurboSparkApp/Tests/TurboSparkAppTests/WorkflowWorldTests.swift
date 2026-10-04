import Foundation
import XCTest
@testable import TurboSparkApp

final class WorkflowWorldTests: XCTestCase {
    func testByteCapsKeepCompleteUTF8PrefixForReadAndGrep() async throws {
        let root = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("needle\n\u{1F600}tail".utf8).write(to: root.appendingPathComponent("unicode.txt"))
        let world = WorkflowWorld(
            workspaceRoot: root,
            executor: WorkflowWorldTestExecutor(),
            limits: WorkflowWorldLimits(maximumScanBytes: 9))

        let read = try await world.read(
            .read(path: "unicode.txt", maxBytes: 9),
            identity: identity(siteIndex: 0))
        XCTAssertEqual(read.outputText, "needle\n")
        XCTAssertTrue(read.truncated)

        let grep = try await world.read(
            .grep(pattern: "needle", pathHint: "unicode.txt"),
            identity: identity(siteIndex: 1))
        XCTAssertEqual(grep.outputText, "unicode.txt:1:needle")
        XCTAssertTrue(grep.truncated)
    }

    func testInvalidUTF8StillRefusesCappedFileRead() async throws {
        let root = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        try Data([0xFF, 0x61, 0x62, 0x63, 0x64, 0x65]).write(
            to: root.appendingPathComponent("invalid.txt"))
        let world = WorkflowWorld(workspaceRoot: root, executor: WorkflowWorldTestExecutor())

        do {
            _ = try await world.read(
                .read(path: "invalid.txt", maxBytes: 4),
                identity: identity(siteIndex: 0))
            XCTFail("An invalid leading byte must remain a validation failure")
        } catch let error as WorkflowError {
            XCTAssertEqual(error.kind, .validation)
        }
    }

    func testInvalidMultibytePrefixesAreNotTrimmedAsPartialScalars() async throws {
        let root = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let world = WorkflowWorld(workspaceRoot: root, executor: WorkflowWorldTestExecutor())
        let invalidPrefixes: [[UInt8]] = [[0xE0, 0x80], [0xED, 0xA0], [0xF0, 0x80], [0xF4, 0x90]]
        for (index, prefix) in invalidPrefixes.enumerated() {
            let path = "invalid-\(index).txt"
            try Data([0x61] + prefix + [0x80, 0x80, 0x62]).write(to: root.appendingPathComponent(path))
            do {
                _ = try await world.read(
                    .read(path: path, maxBytes: 3),
                    identity: identity(siteIndex: index))
                XCTFail("An overlong, surrogate, or out-of-range prefix must remain invalid")
            } catch let error as WorkflowError {
                XCTAssertEqual(error.kind, .validation)
            }
        }
    }

    func testGlobFileGrepAndGitReadsUseFixedBoundedObservations() async throws {
        let root = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("Sources"),
            withIntermediateDirectories: true)
        try Data("first\nneedle value\nlast\n".utf8).write(
            to: root.appendingPathComponent("Sources/main.swift"))

        let executor = WorkflowWorldTestExecutor()
        let world = WorkflowWorld(workspaceRoot: root, executor: executor)

        let glob = try await world.read(
            .glob(pattern: "Sources/**"),
            identity: identity(siteIndex: 0))
        XCTAssertEqual(glob.outputText, "Sources/main.swift")
        XCTAssertEqual(glob.argv, ["glob", "Sources/**"])

        let file = try await world.read(
            .read(path: "Sources/main.swift", maxBytes: 6),
            identity: identity(siteIndex: 1))
        XCTAssertEqual(file.outputText, "first\n")
        XCTAssertTrue(file.truncated)
        XCTAssertEqual(file.argv, ["read", "Sources/main.swift", "6"])

        let grep = try await world.read(
            .grep(pattern: "needle", pathHint: "Sources"),
            identity: identity(siteIndex: 2))
        XCTAssertEqual(grep.outputText, "Sources/main.swift:2:needle value")
        XCTAssertEqual(grep.argv, ["grep", "needle", "Sources"])

        let git = try await world.read(
            .git(op: .status),
            identity: identity(siteIndex: 3))
        XCTAssertEqual(git.outputText, "git status")
        XCTAssertEqual(git.argv.first, "git")
        let requests = await executor.requests()
        XCTAssertEqual(requests.count, 1)
        XCTAssertEqual(
            requests[0].arguments,
            [
                "--no-pager", "-c", "core.fsmonitor=false",
                "status", "--short", "--untracked-files=all", "--no-renames",
            ])
        XCTAssertEqual(requests[0].workingDirectoryURL, PathContainment.canonical(root))
    }

    func testAbsoluteAndParentTraversalPathsAreSandboxRefusals() async throws {
        let root = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let world = WorkflowWorld(workspaceRoot: root, executor: WorkflowWorldTestExecutor())

        await assertSandboxRefusal(
            world,
            operation: .read(path: "/etc/passwd", maxBytes: 128),
            identity: identity(siteIndex: 0))
        await assertSandboxRefusal(
            world,
            operation: .read(path: "../outside.txt", maxBytes: 128),
            identity: identity(siteIndex: 1))
        await assertSandboxRefusal(
            world,
            operation: .glob(pattern: "../**"),
            identity: identity(siteIndex: 2))
    }

    func testSymlinkEscapesAreRefusedForReadGrepAndGlob() async throws {
        let root = try makeWorkspace()
        let outside = try makeWorkspace()
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: outside)
        }

        try Data("secret".utf8).write(to: outside.appendingPathComponent("secret.txt"))
        let link = root.appendingPathComponent("secret.txt")
        try FileManager.default.createSymbolicLink(
            atPath: link.path,
            withDestinationPath: outside.appendingPathComponent("secret.txt").path)
        let world = WorkflowWorld(workspaceRoot: root, executor: WorkflowWorldTestExecutor())

        await assertSandboxRefusal(
            world,
            operation: .read(path: "secret.txt", maxBytes: 128),
            identity: identity(siteIndex: 0))
        await assertSandboxRefusal(
            world,
            operation: .grep(pattern: "secret", pathHint: "secret.txt"),
            identity: identity(siteIndex: 1))
        await assertSandboxRefusal(
            world,
            operation: .glob(pattern: "secret.txt"),
            identity: identity(siteIndex: 2))
    }

    func testChangedFilesKeepsCompleteRecordsAndDropsOnlyPartialTail() async throws {
        let root = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let completeExecutor = WorkflowWorldTestExecutor(
            outputText: " M first.swift\0?? second.swift\0",
            truncated: true)
        let completeWorld = WorkflowWorld(workspaceRoot: root, executor: completeExecutor)

        let complete = try await completeWorld.read(
            .git(op: .changedFiles),
            identity: identity(siteIndex: 0))
        XCTAssertEqual(complete.outputText, "first.swift\nsecond.swift")
        XCTAssertTrue(complete.truncated)
        let completeRequests = await completeExecutor.requests()
        XCTAssertEqual(completeRequests.count, 1)
        XCTAssertEqual(
            completeRequests[0].arguments,
            [
                "--no-pager", "-c", "core.fsmonitor=false",
                "status", "--porcelain=v1", "-z", "--untracked-files=all", "--no-renames",
            ])

        let partialExecutor = WorkflowWorldTestExecutor(
            outputText: " M first.swift\0?? partial.swift",
            truncated: true)
        let partialWorld = WorkflowWorld(workspaceRoot: root, executor: partialExecutor)
        let partial = try await partialWorld.read(
            .git(op: .changedFiles),
            identity: identity(siteIndex: 1))
        XCTAssertEqual(partial.outputText, "first.swift")
        XCTAssertTrue(partial.truncated)
    }

    func testUnsupportedCanonicalReadFailsClosed() async throws {
        let root = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let world = WorkflowWorld(workspaceRoot: root, executor: WorkflowWorldTestExecutor())
        let operation = WorkflowInterpreterOperation.worldRead(operation: .object([
            "kind": .string("shell"),
            "arguments": .array([]),
        ]))

        do {
            _ = try await world.perform(
                operation,
                at: WorkflowSiteKey(lane: "main", siteIndex: 0, ordinal: 0),
                context: WorkflowAttemptContext(
                    cancellation: WorkflowCancellationToken(),
                    deadline: nil))
            XCTFail("unsupported world reads must fail closed")
        } catch let error as WorkflowError {
            XCTAssertEqual(error.kind, .sandboxRefusal)
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    private func makeWorkspace() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("workflow-world-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true)
        return root
    }

    private func identity(siteIndex: Int) -> WorkflowRequestIdentity {
        WorkflowRequestIdentity(
            site: WorkflowSiteKey(lane: "main", siteIndex: siteIndex, ordinal: 0),
            inputHash: "test")
    }

    private func assertSandboxRefusal(
        _ world: WorkflowWorld,
        operation: WorkflowWorldRead,
        identity: WorkflowRequestIdentity,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async {
        do {
            _ = try await world.read(operation, identity: identity)
            XCTFail("expected a named sandbox refusal", file: file, line: line)
        } catch let error as WorkflowError {
            XCTAssertEqual(error.kind, .sandboxRefusal, file: file, line: line)
            XCTAssertEqual(error.site, identity.site, file: file, line: line)
        } catch {
            XCTFail("unexpected error: \(error)", file: file, line: line)
        }
    }
}

actor WorkflowWorldTestExecutor: WorkflowWorldExecutorPort {
    private(set) var recordedRequests: [WorkflowWorldProcessInvocation] = []
    private let outputText: String
    private let truncated: Bool

    init(outputText: String = "git status", truncated: Bool = false) {
        self.outputText = outputText
        self.truncated = truncated
    }

    func execute(
        _ invocation: WorkflowWorldProcessInvocation
    ) async throws -> WorkflowWorldObservation {
        recordedRequests.append(invocation)
        return WorkflowWorldObservation(
            argv: ["git"] + invocation.arguments,
            exitStatus: 0,
            outputText: outputText,
            truncated: truncated)
    }

    func requests() -> [WorkflowWorldProcessInvocation] {
        recordedRequests
    }
}
