import JavaScriptCore
import XCTest
@testable import TurboSparkApp

/// Task 2.2: the rooted file facade behind explicit capability grants.
/// Broker tests cover the per-operation current-grant read (5.5, 5.7),
/// canonicalization with symlink resolution and containment under granted
/// roots (5.4), and fail-closed denial when nothing is granted (5.3).
/// Facade tests cover the frozen `repl.fs` surface that forwards every
/// operation to the request channel instead of touching the filesystem.
final class REPLFileAccessTests: XCTestCase {
    private var neighborCounter = 0

    // MARK: Helpers

    private func makeRootDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("repl-file-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }

    /// A file directly outside every granted root, used for traversal and
    /// symlink-pivot probes.
    private func makeOutsideFile(content: String) throws -> URL {
        neighborCounter += 1
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("repl-file-outside-\(neighborCounter)-\(UUID().uuidString).txt")
        try content.write(to: url, atomically: true, encoding: .utf8)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private func request(
        _ operation: REPLFileAccessRequest.Operation,
        _ path: String,
        payload: Data? = nil
    ) -> REPLFileAccessRequest {
        REPLFileAccessRequest(id: UUID(), operation: operation, path: path, payload: payload)
    }

    private func makeChannel(box: GrantBox, chatID: UUID? = UUID()) -> REPLInProcessFileChannel {
        REPLInProcessFileChannel(broker: REPLFileAccessBroker(grants: box.source), chatID: chatID)
    }

    private func deniedMessage(_ result: REPLFileAccessResult) -> String? {
        guard case let .denied(message) = result else { return nil }
        return message
    }

    private func dataText(_ result: REPLFileAccessResult) -> String? {
        guard case let .data(data) = result else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private func entryNames(_ result: REPLFileAccessResult) -> [String]? {
        guard case let .entries(names) = result else { return nil }
        return names
    }

    private func isWritten(_ result: REPLFileAccessResult) -> Bool {
        guard case .written = result else { return false }
        return true
    }

    // MARK: Broker: fail-closed denial (5.3, 5.5)

    func testEveryOperationIsDeniedWhenNoRootsAreGranted() async {
        let box = GrantBox()
        let channel = makeChannel(box: box)

        for request in [
            request(.readFile, "notes.txt"),
            request(.writeFile, "notes.txt", payload: Data("x".utf8)),
            request(.list, ".")
        ] {
            let result = await channel.send(request)
            let message = deniedMessage(result)
            XCTAssertNotNil(
                message,
                "expected a denial for \(request.operation) with no grants, got: \(result)")
            XCTAssertTrue(
                message?.lowercased().contains("denied") == true,
                "the denial must read as a permission error, got: \(message ?? "")")
        }
    }

    func testNilChatIdentityIsDeniedEvenWhenRootsExist() async throws {
        let root = try makeRootDirectory()
        let box = GrantBox()
        box.setRoots([root.path])

        let result = await makeChannel(box: box, chatID: nil)
            .send(request(.readFile, "anything.txt"))

        let message = deniedMessage(result)
        XCTAssertNotNil(
            message,
            "a nil chat identity cannot resolve a project grant, got: \(result)")
        XCTAssertTrue(message?.lowercased().contains("denied") == true, message ?? "")
    }

    func testIOFailureDenialsDoNotDiscloseTheGrantedRootLocation() async throws {
        let root = try makeRootDirectory()
        let channel = makeChannel(box: GrantBox(roots: [root.path]))

        let missing = await channel.send(request(.readFile, "missing.txt"))

        let message = deniedMessage(missing)
        XCTAssertNotNil(message, "expected an IO-failure denial, got: \(missing)")
        XCTAssertTrue(
            message?.contains("missing.txt") == true,
            "the denial should still name the requested file, got: \(message ?? "")")
        let canonicalRoot = root.standardizedFileURL.resolvingSymlinksInPath().path
        XCTAssertFalse(
            message?.contains(root.path) == true,
            "a denial must not disclose the granted root as configured, got: \(message ?? "")")
        XCTAssertFalse(
            message?.contains(canonicalRoot) == true,
            "a denial must not disclose the granted root's canonical location either, "
                + "got: \(message ?? "")")
    }

    // MARK: Broker: working access inside granted roots (5.3)

    func testReadWriteAndListWorkInsideAGrantedRoot() async throws {
        let root = try makeRootDirectory()
        let channel = makeChannel(box: GrantBox(roots: [root.path]))

        let written = await channel.send(
            request(.writeFile, "first.txt", payload: Data("hello rooted world".utf8)))
        XCTAssertTrue(isWritten(written), "expected the in-root write to succeed, got: \(written)")

        let read = await channel.send(request(.readFile, "first.txt"))
        XCTAssertEqual(dataText(read), "hello rooted world")

        let absoluteRead = await channel.send(
            request(.readFile, root.appendingPathComponent("first.txt").path))
        XCTAssertEqual(
            dataText(absoluteRead),
            "hello rooted world",
            "an absolute path under the granted root must resolve to the same file")

        let listed = await channel.send(request(.list, "."))
        XCTAssertTrue(
            entryNames(listed)?.contains("first.txt") == true,
            "expected the listing to contain the written file, got: \(listed)")
    }

    // MARK: Broker: traversal and symlink containment (5.4)

    func testTraversalOutsideTheGrantedRootIsRejected() async throws {
        let root = try makeRootDirectory()
        let outside = try makeOutsideFile(content: "outside secret")
        let channel = makeChannel(box: GrantBox(roots: [root.path]))

        let relative = await channel.send(request(.readFile, "../\(outside.lastPathComponent)"))
        XCTAssertNotNil(deniedMessage(relative), "expected ../ traversal to be denied, got: \(relative)")

        let deep = await channel.send(request(.readFile, "sub/../../\(outside.lastPathComponent)"))
        XCTAssertNotNil(deniedMessage(deep), "expected nested traversal to be denied, got: \(deep)")

        let absolute = await channel.send(request(.readFile, outside.path))
        XCTAssertNotNil(
            deniedMessage(absolute),
            "an absolute path outside the root must not bypass containment, got: \(absolute)")

        let escapeName = "escape-probe-\(UUID().uuidString).txt"
        let escapeWrite = await channel.send(
            request(.writeFile, "../\(escapeName)", payload: Data("no".utf8)))
        XCTAssertNotNil(deniedMessage(escapeWrite), "expected an escaping write to be denied")
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: root.deletingLastPathComponent()
                    .appendingPathComponent(escapeName).path),
            "no escaping write may land on disk")
    }

    func testSymlinkPivotInsideTheRootPointingOutsideIsRejected() async throws {
        let root = try makeRootDirectory()
        let outside = try makeOutsideFile(content: "outside secret")
        let outsideDirectory = outside.deletingLastPathComponent()
        let channel = makeChannel(box: GrantBox(roots: [root.path]))

        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("pivot.txt"),
            withDestinationURL: outside)
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("pivot-dir"),
            withDestinationURL: outsideDirectory)

        let readThroughPivot = await channel.send(request(.readFile, "pivot.txt"))
        let readMessage = deniedMessage(readThroughPivot)
        XCTAssertNotNil(
            readMessage,
            "reading through an outward symlink must be denied, got: \(readThroughPivot)")
        XCTAssertFalse(
            readMessage?.contains(root.path) == true,
            "the denial must not disclose the granted root, got: \(readMessage ?? "")")

        let writeThroughPivot = await channel.send(
            request(.writeFile, "pivot.txt", payload: Data("clobber".utf8)))
        XCTAssertNotNil(
            deniedMessage(writeThroughPivot),
            "writing through an outward symlink must be denied, got: \(writeThroughPivot)")
        XCTAssertEqual(
            try String(contentsOf: outside, encoding: .utf8),
            "outside secret",
            "the denied write must not modify the outside target")

        let listThroughPivot = await channel.send(request(.list, "pivot-dir"))
        XCTAssertNotNil(
            deniedMessage(listThroughPivot),
            "listing through an outward directory symlink must be denied, got: \(listThroughPivot)")
    }

    func testSiblingPathSharingTheRootNamePrefixIsDeniedForEveryOperation() async throws {
        // A sibling that merely shares the granted root's name prefix must
        // not pass containment: ".../proj-backup/a.txt" and ".../proj.txt"
        // both start with the string ".../proj" but diverge at the
        // component boundary, so only a prefix compare that demands the
        // "/" separator rejects them.
        let containment = FileManager.default.temporaryDirectory
            .appendingPathComponent("repl-containment-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: containment, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: containment) }

        let root = containment.appendingPathComponent("proj", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let backupDirectory = containment.appendingPathComponent("proj-backup", isDirectory: true)
        try FileManager.default.createDirectory(at: backupDirectory, withIntermediateDirectories: true)
        try "sibling".write(
            to: containment.appendingPathComponent("proj.txt"),
            atomically: true, encoding: .utf8)
        try "backup".write(
            to: backupDirectory.appendingPathComponent("a.txt"),
            atomically: true, encoding: .utf8)

        let channel = makeChannel(box: GrantBox(roots: [root.path]))
        let absoluteRequests = [
            request(.readFile, backupDirectory.appendingPathComponent("a.txt").path),
            request(.writeFile, backupDirectory.appendingPathComponent("new.txt").path,
                    payload: Data("x".utf8)),
            request(.list, backupDirectory.path),
            request(.readFile, containment.appendingPathComponent("proj.txt").path),
            request(.writeFile, containment.appendingPathComponent("proj.txt").path,
                    payload: Data("x".utf8)),
            request(.list, containment.appendingPathComponent("proj.txt").path)
        ]
        for request in absoluteRequests {
            let result = await channel.send(request)
            XCTAssertNotNil(
                deniedMessage(result),
                "a sibling sharing only the root's name prefix must be denied for "
                    + "\(request.operation) at \(request.path), got: \(result)")
        }
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: backupDirectory.appendingPathComponent("new.txt").path),
            "the denied sibling write must not land on disk")

        let relativeDirectory = await channel.send(request(.readFile, "../proj-backup/a.txt"))
        XCTAssertNotNil(
            deniedMessage(relativeDirectory),
            "the relative spelling must be denied at the same boundary, got: \(relativeDirectory)")
        let relativeFile = await channel.send(request(.readFile, "../proj.txt"))
        XCTAssertNotNil(
            deniedMessage(relativeFile),
            "the relative file-sibling spelling must be denied, got: \(relativeFile)")

        let inside = await channel.send(request(.list, "."))
        XCTAssertTrue(
            entryNames(inside)?.isEmpty == true,
            "the granted root itself must still work in the same fixture, got: \(inside)")
    }

    func testInRootSymlinkResolvingInsideTheRootIsAllowed() async throws {
        let root = try makeRootDirectory()
        try "shared payload".write(
            to: root.appendingPathComponent("real.txt"), atomically: true, encoding: .utf8)
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("alias.txt"),
            withDestinationURL: root.appendingPathComponent("real.txt"))
        let channel = makeChannel(box: GrantBox(roots: [root.path]))

        let read = await channel.send(request(.readFile, "alias.txt"))

        XCTAssertEqual(
            dataText(read),
            "shared payload",
            "a symlink whose resolution stays inside the root is legitimate under the "
                + "resolveSecurePath discipline, got: \(read)")
    }

    // MARK: Broker: current grants are re-read per operation (5.5, 5.7)

    func testSymlinkCreatedMidSessionAfterEarlierAccessIsRejected() async throws {
        let root = try makeRootDirectory()
        try "first read".write(
            to: root.appendingPathComponent("data.txt"), atomically: true, encoding: .utf8)
        let outside = try makeOutsideFile(content: "outside secret")
        let channel = makeChannel(box: GrantBox(roots: [root.path]))

        let before = await channel.send(request(.readFile, "data.txt"))
        XCTAssertEqual(dataText(before), "first read")

        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("pivot.txt"),
            withDestinationURL: outside)

        let after = await channel.send(request(.readFile, "pivot.txt"))
        XCTAssertNotNil(
            deniedMessage(after),
            "canonicalization must run again per operation, so a symlink planted after "
                + "an earlier successful access is still denied, got: \(after)")
    }

    func testGrantRevocationTakesEffectOnTheNextOperation() async throws {
        let root = try makeRootDirectory()
        try "kept".write(
            to: root.appendingPathComponent("data.txt"), atomically: true, encoding: .utf8)
        let box = GrantBox(roots: [root.path])
        let channel = makeChannel(box: box)

        let before = await channel.send(request(.readFile, "data.txt"))
        XCTAssertEqual(dataText(before), "kept")

        box.setRoots([])

        let after = await channel.send(request(.readFile, "data.txt"))
        let message = deniedMessage(after)
        XCTAssertNotNil(
            message,
            "a revoked grant must deny the very next operation, got: \(after)")
        XCTAssertTrue(message?.lowercased().contains("denied") == true, message ?? "")
    }

    func testGrantNarrowingDeniesTheRemovedRootOnTheNextOperation() async throws {
        let keptRoot = try makeRootDirectory()
        let droppedRoot = try makeRootDirectory()
        try "kept".write(
            to: keptRoot.appendingPathComponent("data.txt"), atomically: true, encoding: .utf8)
        try "dropped".write(
            to: droppedRoot.appendingPathComponent("data.txt"), atomically: true, encoding: .utf8)
        let keptFile = keptRoot.appendingPathComponent("data.txt").path
        let droppedFile = droppedRoot.appendingPathComponent("data.txt").path
        let box = GrantBox(roots: [keptRoot.path, droppedRoot.path])
        let channel = makeChannel(box: box)

        let bothKept = await channel.send(request(.readFile, keptFile))
        XCTAssertEqual(dataText(bothKept), "kept")
        let bothDropped = await channel.send(request(.readFile, droppedFile))
        XCTAssertEqual(dataText(bothDropped), "dropped")

        box.setRoots([keptRoot.path])

        let keptAfterNarrowing = await channel.send(request(.readFile, keptFile))
        XCTAssertEqual(
            dataText(keptAfterNarrowing),
            "kept",
            "the surviving root must keep working after narrowing")
        let droppedAfterNarrowing = await channel.send(request(.readFile, droppedFile))
        XCTAssertNotNil(
            deniedMessage(droppedAfterNarrowing),
            "the removed root must deny on the next operation after narrowing")
    }

    // MARK: Broker: bounded transfers

    func testOversizedReadsAndPayloadsAreDenied() async throws {
        let root = try makeRootDirectory()
        let cap = AppFileReadLimits.maximumBytes
        let bigURL = root.appendingPathComponent("big.bin")
        XCTAssertTrue(
            FileManager.default.createFile(
                atPath: bigURL.path, contents: Data(repeating: 0, count: cap + 1)),
            "the oversized fixture must exist for the read-cap probe")
        let channel = makeChannel(box: GrantBox(roots: [root.path]))

        let oversizedRead = await channel.send(request(.readFile, "big.bin"))
        XCTAssertNotNil(
            deniedMessage(oversizedRead),
            "a read over the transfer cap must be denied, got: \(oversizedRead)")

        let oversizedPayload = await channel.send(
            request(.writeFile, "out.bin", payload: Data(repeating: 0, count: cap + 1)))
        XCTAssertNotNil(
            deniedMessage(oversizedPayload),
            "a payload over the transfer cap must be denied, got: \(oversizedPayload)")

        let longPath = await channel.send(
            request(.readFile, String(repeating: "a", count: 4_097)))
        XCTAssertNotNil(
            deniedMessage(longPath),
            "an over-long path must be denied, got: \(longPath)")
    }

    // MARK: Facade: the frozen repl.fs surface (5.6 conventions from 2.1)

    func testReplFsSurfaceIsFrozenWithExactlyTheSpecifiedMembers() {
        let harness = FileFacadeHarness { _ in .denied("unused") }

        let probes: [(String, String)] = [
            ("repl is an object", "typeof repl === 'object' && repl !== null"),
            ("repl object is frozen", "Object.isFrozen(repl)"),
            ("repl object is not extensible", "!Object.isExtensible(repl)"),
            ("repl exposes exactly fs", "Object.keys(repl).join(',') === 'fs'"),
            (
                "repl.fs is frozen and not extensible",
                "Object.isFrozen(repl.fs) && !Object.isExtensible(repl.fs)"),
            (
                "repl.fs exposes exactly the three operations",
                "Object.keys(repl.fs).sort().join(',') === 'list,readFile,writeFile'"),
            (
                "every repl.fs member is a frozen function with a non-writable "
                    + "non-configurable descriptor",
                """
                (() => {
                    const names = ['readFile', 'writeFile', 'list'];
                    return names.every(name => {
                        const descriptor = Object.getOwnPropertyDescriptor(repl.fs, name);
                        return descriptor !== undefined
                            && typeof descriptor.value === 'function'
                            && descriptor.writable === false
                            && descriptor.configurable === false
                            && Object.isFrozen(descriptor.value);
                    });
                })()
                """),
            (
                "the global repl binding is non-writable and non-configurable",
                """
                (() => {
                    const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'repl');
                    return descriptor !== undefined
                        && descriptor.writable === false
                        && descriptor.configurable === false;
                })()
                """),
            (
                "the installation bridge is not left on the global",
                """
                !Object.prototype.hasOwnProperty.call(globalThis, '__turbosparkFileRequest')
                """
            )
        ]

        for (description, script) in probes {
            let passed = harness.evaluate(script)?.toBool() ?? false
            XCTAssertTrue(passed, "expected true: \(description)")
        }
        XCTAssertTrue(harness.exceptions.isEmpty, "probes must not throw: \(harness.exceptions)")
    }

    func testTamperingWithTheFileFacadeThrowsAndOriginalMembersSurvive() {
        let harness = FileFacadeHarness { _ in .data(Data("intact".utf8)) }

        harness.evaluate(#""use strict"; repl.fs = {};"#)
        harness.evaluate(#""use strict"; globalThis.repl = {};"#)
        harness.evaluate(#""use strict"; repl.fs.readFile = () => 1;"#)
        harness.evaluate("delete repl.fs.list;")

        XCTAssertEqual(
            harness.exceptions.count, 3,
            "strict reassignment must throw (delete only fails silently), got: \(harness.exceptions)")

        harness.evaluate("""
        globalThis.outcome = "__pending__";
        repl.fs.readFile("anywhere.txt").then(
            value => { globalThis.outcome = "ok:" + value; },
            error => { globalThis.outcome = "err:" + error.message; });
        """)
        XCTAssertEqual(
            harness.awaitGlobal("outcome"),
            "ok:intact",
            "the frozen original member must still serve after tampering attempts")
    }

    func testReplFsForwardsToTheRequestChannelAndNeverReadsDisk() throws {
        let root = try makeRootDirectory()
        try "disk answer".write(
            to: root.appendingPathComponent("real.txt"), atomically: true, encoding: .utf8)
        let harness = FileFacadeHarness { request in
            switch request.operation {
            case .readFile: return .data(Data("channel answer".utf8))
            case .writeFile: return .written
            case .list: return .entries(["alpha", "beta"])
            }
        }

        harness.evaluate("""
        globalThis.outcome = "__pending__";
        repl.fs.readFile("real.txt").then(
            value => { globalThis.outcome = "ok:" + value; },
            error => { globalThis.outcome = "err:" + error.message; });
        """)
        XCTAssertEqual(
            harness.awaitGlobal("outcome"),
            "ok:channel answer",
            "the facade must surface the channel's answer, not the file on disk")

        harness.evaluate("""
        globalThis.writeOutcome = "__pending__";
        repl.fs.writeFile("out.txt", "hello").then(
            () => { globalThis.writeOutcome = "ok"; },
            error => { globalThis.writeOutcome = "err:" + error.message; });
        """)
        XCTAssertEqual(harness.awaitGlobal("writeOutcome"), "ok")
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: root.appendingPathComponent("out.txt").path),
            "a channel-only facade must not write files itself")

        harness.evaluate("""
        globalThis.listOutcome = "__pending__";
        repl.fs.list("real.txt").then(
            value => { globalThis.listOutcome = "ok:" + value.join(','); },
            error => { globalThis.listOutcome = "err:" + error.message; });
        """)
        XCTAssertEqual(harness.awaitGlobal("listOutcome"), "ok:alpha,beta")

        let requests = harness.channel.recordedRequests
        XCTAssertEqual(requests.map(\.operation), [.readFile, .writeFile, .list])
        XCTAssertEqual(requests.map(\.path), ["real.txt", "out.txt", "real.txt"])
        XCTAssertEqual(requests[1].payload, Data("hello".utf8))
        XCTAssertNil(requests[0].payload, "read requests carry no payload")
        XCTAssertEqual(
            Set(requests.map(\.id)).count, 3,
            "every request must carry its own identifier")
    }

    func testPermissionDenialRejectsThePromiseAsAnOrdinaryScriptError() {
        let harness = FileFacadeHarness { _ in
            .denied("REPL file access denied: no file root is granted for this session.")
        }

        harness.evaluate("""
        globalThis.outcome = "__pending__";
        repl.fs.readFile("secret.txt").then(
            value => { globalThis.outcome = "ok:" + value; },
            error => { globalThis.outcome = "err:" + error.message; });
        """)

        XCTAssertEqual(
            harness.awaitGlobal("outcome"),
            "err:REPL file access denied: no file root is granted for this session.",
            "a broker denial must surface as an ordinary rejected promise")
        XCTAssertTrue(harness.exceptions.isEmpty)
    }

    // MARK: Worker context: real evaluation through the channel seam

    func testGrantedFileAccessFlowsThroughTheWorkerEvaluationPath() async throws {
        let root = try makeRootDirectory()
        try "42".write(
            to: root.appendingPathComponent("data.txt"), atomically: true, encoding: .utf8)
        let box = GrantBox(roots: [root.path])
        let worker = REPLWorkerContext(
            fileChannel: makeChannel(box: box))

        let read = await worker.evaluate(code: """
        const payload = await repl.fs.readFile("data.txt");
        payload.trim() + "!"
        """)
        XCTAssertEqual(read.status, .completed, read.errorText ?? "")
        XCTAssertEqual(read.completionText, "42!")

        let write = await worker.evaluate(code: """
        await repl.fs.writeFile("made.txt", "made by the worker");
        "written"
        """)
        XCTAssertEqual(write.status, .completed, write.errorText ?? "")
        XCTAssertEqual(
            try String(contentsOf: root.appendingPathComponent("made.txt"), encoding: .utf8),
            "made by the worker")

        let list = await worker.evaluate(code: """
        const names = await repl.fs.list(".");
        names.sort().join("|")
        """)
        XCTAssertEqual(list.status, .completed, list.errorText ?? "")
        XCTAssertEqual(list.completionText, "data.txt|made.txt")
    }

    func testDeniedAccessFailsTheCallAndTheSessionSurvives() async throws {
        let root = try makeRootDirectory()
        try "kept".write(
            to: root.appendingPathComponent("data.txt"), atomically: true, encoding: .utf8)
        let outside = try makeOutsideFile(content: "outside secret")
        let worker = REPLWorkerContext(
            fileChannel: makeChannel(box: GrantBox(roots: [root.path])))

        let setup = await worker.evaluate(code: "let note = 'before'")
        XCTAssertEqual(setup.status, .completed, setup.errorText ?? "")

        let denied = await worker.evaluate(code: """
        await repl.fs.readFile("../\(outside.lastPathComponent)");
        """)
        XCTAssertEqual(denied.status, .failed, "a permission denial fails the call")
        XCTAssertTrue(
            denied.errorText?.lowercased().contains("denied") == true,
            "the failure text must read as a permission error, got: \(denied.errorText ?? "")")

        let followUp = await worker.evaluate(code: "note + '|alive'")
        XCTAssertEqual(
            followUp.status, .completed,
            "a permission denial must not end the session: \(followUp.errorText ?? "")")
        XCTAssertEqual(followUp.completionText, "before|alive")
    }

    func testRevocationTakesEffectInsideOneWorkerSession() async throws {
        let root = try makeRootDirectory()
        try "value".write(
            to: root.appendingPathComponent("data.txt"), atomically: true, encoding: .utf8)
        let box = GrantBox(roots: [root.path])
        let worker = REPLWorkerContext(fileChannel: makeChannel(box: box))

        let granted = await worker.evaluate(code: "await repl.fs.readFile('data.txt')")
        XCTAssertEqual(granted.status, .completed, granted.errorText ?? "")

        box.setRoots([])

        let revoked = await worker.evaluate(code: "await repl.fs.readFile('data.txt')")
        XCTAssertEqual(
            revoked.status, .failed,
            "revoking the grant must deny the next operation in the same session")
        XCTAssertTrue(
            revoked.errorText?.lowercased().contains("denied") == true,
            "expected a permission error, got: \(revoked.errorText ?? "")")

        box.setRoots([root.path])
        let regranted = await worker.evaluate(code: "await repl.fs.readFile('data.txt')")
        XCTAssertEqual(
            regranted.status, .completed,
            "re-granting must restore access on the next operation: \(regranted.errorText ?? "")")
    }

    func testWorkerContextWithoutAChannelFailsClosed() async {
        let worker = REPLWorkerContext()

        let result = await worker.evaluate(code: """
        globalThis.fileOutcome = "__pending__";
        repl.fs.readFile("anything.txt").then(
            value => { globalThis.fileOutcome = "ok:" + value; },
            error => { globalThis.fileOutcome = "err:" + error.message; });
        "started"
        """)
        XCTAssertEqual(result.status, .completed, result.errorText ?? "")

        var outcome: String?
        for _ in 0..<500 {
            let poll = await worker.evaluate(code: "globalThis.fileOutcome")
            outcome = poll.completionText
            if outcome != "__pending__" { break }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }

        XCTAssertEqual(
            outcome, "err:REPL file access denied: no file access is granted to this session.",
            "a worker without a bound channel must deny every file operation")
    }
}

// MARK: - Test doubles

/// Stands in for the app-owned project-permission store: the broker reads
/// it again for every operation, exactly as production reads the chat's
/// attached `AppProjectPermissions.replFileAccessRoots`.
private final class GrantBox: @unchecked Sendable {
    private let lock = NSLock()
    private var permissions: AppProjectPermissions

    init(roots: [String] = []) {
        self.permissions = AppProjectPermissions(replFileAccessRoots: roots)
    }

    func setRoots(_ roots: [String]) {
        lock.lock()
        permissions.replFileAccessRoots = roots
        lock.unlock()
    }

    private var currentRoots: [String] {
        lock.lock()
        defer { lock.unlock() }
        return permissions.replFileAccessRoots
    }

    var source: REPLFileGrantSource {
        let box = self
        return { _ in box.currentRoots }
    }
}

/// Records every request and answers from a canned responder, proving the
/// facade forwards through the channel rather than touching the filesystem.
private final class RecordingChannel: REPLFileRequestChannel, @unchecked Sendable {
    private let lock = NSLock()
    private var requests: [REPLFileAccessRequest] = []
    private let responder: (REPLFileAccessRequest) -> REPLFileAccessResult

    init(responder: @escaping (REPLFileAccessRequest) -> REPLFileAccessResult) {
        self.responder = responder
    }

    func send(_ request: REPLFileAccessRequest) async -> REPLFileAccessResult {
        record(request)
        let answer = responder(request)
        return answer
    }

    private func record(_ request: REPLFileAccessRequest) {
        lock.lock()
        requests.append(request)
        lock.unlock()
    }

    var recordedRequests: [REPLFileAccessRequest] {
        lock.lock()
        defer { lock.unlock() }
        return requests
    }
}

/// A bare JavaScriptCore context with the file facade installed against a
/// recording channel.
private final class FileFacadeHarness: @unchecked Sendable {
    let context: JSContext
    let channel: RecordingChannel
    private let lock = NSLock()
    private var exceptionStorage: [String] = []

    init(responder: @escaping (REPLFileAccessRequest) -> REPLFileAccessResult) {
        guard let context = JSContext() else {
            fatalError("JavaScriptCore could not create a file facade test context")
        }
        self.context = context
        self.channel = RecordingChannel(responder: responder)
        let harness = self
        context.exceptionHandler = { _, exception in
            harness.recordException(exception?.toString() ?? "unknown exception")
        }
        REPLHostFacade(output: { _ in })
            .installFileSystem(into: context, channel: channel)
    }

    @discardableResult
    func evaluate(_ script: String) -> JSValue? {
        context.evaluateScript(script)
    }

    /// Polls a global string until it leaves the pending sentinel, the same
    /// way a later REPL call observes promise settlement.
    func awaitGlobal(_ name: String) -> String? {
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            if let value = context.evaluateScript("globalThis.\(name)")?.toString(),
                value != "__pending__" {
                return value
            }
            Thread.sleep(forTimeInterval: 0.01)
        }
        return nil
    }

    var exceptions: [String] {
        lock.lock()
        defer { lock.unlock() }
        return exceptionStorage
    }

    private func recordException(_ message: String) {
        lock.lock()
        exceptionStorage.append(message)
        lock.unlock()
    }
}
