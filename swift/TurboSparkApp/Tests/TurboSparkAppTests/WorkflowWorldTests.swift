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

    func testStdoutOnlyInvocationExcludesStderrWarnings() async throws {
        func invocation(stdoutOnly: Bool) -> WorkflowWorldProcessInvocation {
            WorkflowWorldProcessInvocation(
                executableURL: URL(fileURLWithPath: "/bin/sh"),
                arguments: ["-c", "printf 'a.txt'; echo 'warning: denied' >&2"],
                workingDirectoryURL: FileManager.default.temporaryDirectory,
                environment: [:],
                timeoutSeconds: 10,
                outputCapBytes: 4096,
                stdoutOnly: stdoutOnly)
        }
        let executor = WorkflowWorldSystemExecutor()
        let clean = try await executor.execute(invocation(stdoutOnly: true))
        XCTAssertEqual(clean.outputText, "a.txt")
        let merged = try await executor.execute(invocation(stdoutOnly: false))
        XCTAssertTrue(merged.outputText.contains("warning: denied"))
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
        // Three repository-path probes precede filter enumeration and the read.
        XCTAssertEqual(requests.count, 5)
        XCTAssertEqual(requests[0].arguments, ["rev-parse", "--show-toplevel"])
        XCTAssertEqual(requests[0].environment["GIT_CEILING_DIRECTORIES"],
            root.deletingLastPathComponent().path)
        XCTAssertNil(requests[0].environment["GIT_DIR"])
        XCTAssertEqual(requests[3].arguments, ["config", "--get-regexp", "^filter\\."])
        XCTAssertEqual(
            requests[4].arguments,
            [
                "--no-pager", "-c", "core.fsmonitor=false",
                "status", "--short", "--untracked-files=all", "--no-renames", "--", ".",
            ])
        XCTAssertEqual(requests[4].workingDirectoryURL, PathContainment.canonical(root))
        XCTAssertEqual(requests[4].environment["GIT_DIR"], root.path)
        XCTAssertEqual(requests[4].environment["GIT_WORK_TREE"], root.path)
        XCTAssertEqual(requests[4].environment["GIT_COMMON_DIR"], root.path)
    }

    func testFilterDriversAreNeutralizedForGitReads() {
        let config = """
        filter.x.clean ./payload.sh
        filter.x.smudge cat
        filter.a.b.process ./p
        core.other nope
        """
        XCTAssertEqual(
            WorkflowWorld.filterNeutralizingOverrides(configOutput: config),
            [
                "-c", "filter.x.clean=", "-c", "filter.x.smudge=", "-c", "filter.x.process=",
                "-c", "filter.a.b.clean=", "-c", "filter.a.b.smudge=", "-c", "filter.a.b.process=",
            ])
        XCTAssertEqual(WorkflowWorld.filterNeutralizingOverrides(configOutput: ""), [])
    }

    func testStatusReadCarriesFilterOverridesAndLogDisablesSignatures() async throws {
        let root = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let executor = WorkflowWorldTestExecutor(outputText: "filter.x.clean ./payload.sh")
        let world = WorkflowWorld(workspaceRoot: root, executor: executor)
        _ = try await world.read(.git(op: .status), identity: identity(siteIndex: 0))
        _ = try await world.read(.git(op: .log), identity: identity(siteIndex: 1))
        let requests = await executor.requests()
        XCTAssertEqual(requests.count, 10)
        XCTAssertEqual(Array(requests[4].arguments.prefix(6)), [
            "-c", "filter.x.clean=", "-c", "filter.x.smudge=", "-c", "filter.x.process=",
        ])
        XCTAssertTrue(requests[9].arguments.contains("log.showSignature=false"))
        XCTAssertEqual(Array(requests[9].arguments.suffix(2)), ["--", "."])
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
        XCTAssertEqual(completeRequests.count, 5)
        XCTAssertEqual(
            completeRequests[4].arguments,
            [
                "--no-pager", "-c", "core.fsmonitor=false",
                "status", "--porcelain=v1", "-z", "--untracked-files=all", "--no-renames",
                "--", ".",
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

    func testGitDiffRefusesARepositoryDiscoveredAboveWorkspace() async throws {
        let repository = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: repository) }
        try runGit(["init", "--quiet"], at: repository)
        let sibling = repository.appendingPathComponent("sibling-secret.txt")
        try Data().write(to: sibling)
        try runGit(["add", "--", "sibling-secret.txt"], at: repository)
        try Data("ANCESTOR_SIBLING_SECRET".utf8).write(to: sibling)

        let workspace = repository.appendingPathComponent("workspace", isDirectory: true)
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        let world = WorkflowWorld(workspaceRoot: workspace)
        do {
            _ = try await world.read(.git(op: .diff), identity: identity(siteIndex: 0))
            XCTFail("Git must not discover a repository above the declared workspace")
        } catch let error as WorkflowError {
            XCTAssertEqual(error.kind, .sandboxRefusal)
        }
    }

    func testGitDiffRefusesRepositoryLocalWorkTreeOutsideWorkspace() async throws {
        let workspace = try makeWorkspace()
        let outside = try makeWorkspace()
        defer {
            try? FileManager.default.removeItem(at: workspace)
            try? FileManager.default.removeItem(at: outside)
        }
        try runGit(["init", "--quiet"], at: workspace)
        try runGit(["config", "core.worktree", outside.path], at: workspace)

        let credential = outside.appendingPathComponent("credential.txt")
        try Data().write(to: credential)
        try runGit(
            ["--git-dir", workspace.appendingPathComponent(".git").path,
             "--work-tree", outside.path, "add", "--", "credential.txt"],
            at: workspace)
        try Data("CANARY_SECRET_OUTSIDE_WORKSPACE".utf8).write(to: credential)

        let world = WorkflowWorld(workspaceRoot: workspace)
        do {
            _ = try await world.read(.git(op: .diff), identity: identity(siteIndex: 0))
            XCTFail("Git must not read a repository-local work tree outside the workspace")
        } catch let error as WorkflowError {
            XCTAssertEqual(error.kind, .sandboxRefusal)
        }
    }

    func testGitDiffRefusesSymlinkedRepositoryWorkTreeOutsideWorkspace() async throws {
        let workspace = try makeWorkspace()
        let outside = try makeWorkspace()
        defer {
            try? FileManager.default.removeItem(at: workspace)
            try? FileManager.default.removeItem(at: outside)
        }
        try runGit(["init", "--quiet"], at: workspace)
        let linkedWorkTree = workspace.appendingPathComponent("linked-worktree")
        try FileManager.default.createSymbolicLink(
            atPath: linkedWorkTree.path,
            withDestinationPath: outside.path)
        try runGit(["config", "core.worktree", linkedWorkTree.path], at: workspace)

        let world = WorkflowWorld(workspaceRoot: workspace)
        do {
            _ = try await world.read(.git(op: .diff), identity: identity(siteIndex: 0))
            XCTFail("A work tree symlink that resolves outside the workspace must be refused")
        } catch let error as WorkflowError {
            XCTAssertEqual(error.kind, .sandboxRefusal)
        }
    }

    func testGitDiffRefusesGitDirectoryOutsideWorkspace() async throws {
        let workspace = try makeWorkspace()
        let externalRepository = try makeWorkspace()
        defer {
            try? FileManager.default.removeItem(at: workspace)
            try? FileManager.default.removeItem(at: externalRepository)
        }
        try runGit(["init", "--quiet"], at: externalRepository)
        try runGit(["config", "core.worktree", workspace.path], at: externalRepository)
        let credential = workspace.appendingPathComponent("credential.txt")
        try Data().write(to: credential)
        try runGit(
            ["--git-dir", externalRepository.appendingPathComponent(".git").path,
             "--work-tree", workspace.path, "add", "--", "credential.txt"],
            at: workspace)
        try Data("CANARY_SECRET_IN_EXTERNAL_GIT_DIR".utf8).write(to: credential)

        try Data("gitdir: \(externalRepository.appendingPathComponent(".git").path)\n".utf8)
            .write(to: workspace.appendingPathComponent(".git"))

        let world = WorkflowWorld(workspaceRoot: workspace)
        do {
            _ = try await world.read(.git(op: .diff), identity: identity(siteIndex: 0))
            XCTFail("Git metadata outside the declared workspace must be refused")
        } catch let error as WorkflowError {
            XCTAssertEqual(error.kind, .sandboxRefusal)
        }
    }

    func testGitDiffWorksForContainedRepositoryAndUsesWorkspacePathspec() async throws {
        let workspace = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: workspace) }
        try runGit(["init", "--quiet"], at: workspace)
        try runGit(["config", "user.name", "Workflow Tests"], at: workspace)
        try runGit(["config", "user.email", "workflow-tests@example.invalid"], at: workspace)
        let file = workspace.appendingPathComponent("tracked.txt")
        try Data("baseline\n".utf8).write(to: file)
        try runGit(["add", "--", "tracked.txt"], at: workspace)
        try runGit(["commit", "--quiet", "-m", "baseline"], at: workspace)
        try Data("contained diff content\n".utf8).write(to: file)

        let world = WorkflowWorld(workspaceRoot: workspace)
        let diff = try await world.read(.git(op: .diff), identity: identity(siteIndex: 0))
        XCTAssertTrue(diff.outputText.contains("contained diff content"))
        XCTAssertEqual(Array(diff.argv.suffix(2)), ["--", "."])
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

    private func runGit(_ arguments: [String], at directory: URL) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = ["-C", directory.path] + arguments
        process.environment = [
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": "/var/empty",
            "PATH": "/usr/bin:/bin",
        ]
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw WorkflowWorldTestError.gitCommandFailed(Int(process.terminationStatus))
        }
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

private enum WorkflowWorldTestError: Error {
    case gitCommandFailed(Int)
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
        let outputText: String
        let outputTruncated: Bool
        switch invocation.arguments {
        case ["rev-parse", "--show-toplevel"],
             ["rev-parse", "--absolute-git-dir"],
             ["rev-parse", "--git-common-dir"]:
            outputText = invocation.workingDirectoryURL.path + "\n"
            outputTruncated = false
        default:
            outputText = self.outputText
            outputTruncated = truncated
        }
        return WorkflowWorldObservation(
            argv: ["git"] + invocation.arguments,
            exitStatus: 0,
            outputText: outputText,
            truncated: outputTruncated)
    }

    func requests() -> [WorkflowWorldProcessInvocation] {
        recordedRequests
    }
}
