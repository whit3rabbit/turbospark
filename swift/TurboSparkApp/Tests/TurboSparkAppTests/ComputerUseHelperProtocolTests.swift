import ComputerUseKit
import XCTest

@testable import TurboSparkApp

final class ComputerUseHelperProtocolTests: XCTestCase {
    // MARK: - Peer policy

    private let app = "/Applications/TurboSpark.app"
    private let token = String(repeating: "a", count: 64)

    private func accepts(
        uid: uid_t = 501, path: String? = "/Applications/TurboSpark.app", valid: Bool = true,
        presented: String? = nil, expected: String? = nil, expectedPath: String? = "/Applications/TurboSpark.app"
    ) -> Bool {
        ComputerUsePeerPolicy.accepts(
            peer: ComputerUsePeer(uid: uid, bundlePath: path, signatureValid: valid),
            ownUID: 501,
            expectedBundlePath: expectedPath,
            presentedToken: presented ?? token,
            expectedToken: expected ?? token)
    }

    func testAcceptsOnlyTheSignedContainingAppWithTheToken() {
        XCTAssertTrue(accepts())
        XCTAssertFalse(accepts(uid: 502), "another user")
        XCTAssertFalse(accepts(valid: false), "invalid signature")
        XCTAssertFalse(accepts(path: "/tmp/Evil.app"), "another app")
        XCTAssertFalse(accepts(path: nil), "unknown peer path")
        XCTAssertFalse(accepts(expectedPath: nil), "helper not inside an app bundle")
        XCTAssertFalse(accepts(presented: String(repeating: "b", count: 64)), "wrong token")
        XCTAssertFalse(accepts(presented: "", expected: ""), "empty token")
    }

    func testContainingAppBundleNeedsTheHelpersLayout() {
        let helper = URL(fileURLWithPath: "/Applications/TurboSpark.app/Contents/Helpers/TurboSpark Computer Use.app")
        XCTAssertEqual(ComputerUsePeerPolicy.containingAppBundle(helperBundleURL: helper)?.path, app)
        let bare = URL(fileURLWithPath: "/repo/.build/debug/TurboSparkComputerUse")
        XCTAssertNil(ComputerUsePeerPolicy.containingAppBundle(helperBundleURL: bare))
        let wrongDir = URL(fileURLWithPath: "/Applications/TurboSpark.app/Contents/MacOS/Helper.app")
        XCTAssertNil(ComputerUsePeerPolicy.containingAppBundle(helperBundleURL: wrongDir))
    }

    func testNoHelperInsideTheTestBundle() {
        XCTAssertNil(HelperComputerUsePort.installedHelperURL())
    }

    // MARK: - Wire format

    func testRequestsAndResponsesRoundTrip() throws {
        let requests: [ComputerUseRequest] = [
            .hello(token: token, protocolVersion: 1),
            .requestPermission(.screenRecording),
            .command(.click(bundleID: "a.b", target: .point(x: 1.5, y: 2), button: .right, count: 2)),
            .command(.scroll(bundleID: "a.b", target: nil, direction: .down, amount: 3)),
            .resume,
        ]
        for request in requests {
            var buffer = try ComputerUseFraming.encode(request)
            let frames = try ComputerUseFraming.decodeFrames(&buffer)
            XCTAssertEqual(frames.count, 1)
            XCTAssertEqual(try JSONDecoder().decode(ComputerUseRequest.self, from: frames[0]), request)
            XCTAssertTrue(buffer.isEmpty)
        }
        let capture = ComputerUseResponse.result(.screenshot(ComputerUseCapture(
            pngData: Data([1, 2, 3]), windowFrame: ComputerUseRect(x: 0, y: 0, width: 10, height: 10))))
        XCTAssertEqual(
            try JSONDecoder().decode(ComputerUseResponse.self, from: JSONEncoder().encode(capture)), capture)
    }

    func testFramingWaitsForPartialFramesAndRejectsOversize() throws {
        let whole = try ComputerUseFraming.encode(ComputerUseRequest.status)
        var partial = whole.prefix(whole.count - 1)
        var data = Data(partial)
        XCTAssertTrue(try ComputerUseFraming.decodeFrames(&data).isEmpty)
        data.append(whole.last!)
        XCTAssertEqual(try ComputerUseFraming.decodeFrames(&data).count, 1)
        partial = Data([0x7F, 0xFF, 0xFF, 0xFF])
        var huge = Data(partial)
        XCTAssertThrowsError(try ComputerUseFraming.decodeFrames(&huge))
    }

    func testFramingOverARealSocketPair() throws {
        var fds: [Int32] = [0, 0]
        XCTAssertEqual(socketpair(AF_UNIX, SOCK_STREAM, 0, &fds), 0)
        defer { close(fds[0]); close(fds[1]) }
        try ComputerUseFraming.write(ComputerUseRequest.command(.listApps), to: fds[0])
        XCTAssertEqual(try ComputerUseFraming.read(ComputerUseRequest.self, from: fds[1]), .command(.listApps))
        close(fds[0])
        XCTAssertThrowsError(try ComputerUseFraming.read(ComputerUseRequest.self, from: fds[1]))
    }

    // MARK: - Key combos

    func testKeyCombos() {
        XCTAssertEqual(ComputerUseKeyCombo.parse("cmd+s"), ComputerUseKeyCombo(keyCode: 0x01, modifiers: .command))
        XCTAssertEqual(
            ComputerUseKeyCombo.parse("Ctrl + Shift + Tab"),
            ComputerUseKeyCombo(keyCode: 0x30, modifiers: [.control, .shift]))
        XCTAssertEqual(ComputerUseKeyCombo.parse("return")?.keyCode, 0x24)
        XCTAssertEqual(ComputerUseKeyCombo.parse("f12")?.keyCode, 0x6F)
        XCTAssertNil(ComputerUseKeyCombo.parse("hyper+s"))
        XCTAssertNil(ComputerUseKeyCombo.parse("cmd+"))
        XCTAssertNil(ComputerUseKeyCombo.parse("notakey"))
    }

    // MARK: - Kill switch through the executor

    func testKillSwitchRevokesGrantsAndAGrantResumes() async throws {
        let session = ComputerUseSession()
        _ = await session.grant(["com.apple.TextEdit"])
        let port = StoppedPort()
        let runtime = ComputerUseToolRuntime(
            availability: ComputerUseToolAvailability(
                isEnabled: true,
                manifest: ComputerUseBackendManifest(
                    backendIdentifier: "t", supportedCommands: Set(ComputerUseCommandKind.allCases)),
                mediaCapability: .imageBearingToolResults()),
            session: session, port: port)

        let typed = await ComputerUseToolExecutor.execute(
            call: AppToolCall(name: "computer_type", arguments: ["app": "com.apple.TextEdit", "text": "x"]),
            runtime: runtime)
        guard case .denied(let reason) = typed else { return XCTFail("\(typed)") }
        XCTAssertTrue(reason.contains("stopped"))
        let afterStop = await session.tier(for: "com.apple.TextEdit")
        XCTAssertNil(afterStop)

        _ = await ComputerUseToolExecutor.execute(
            call: AppToolCall(
                name: "computer_request_access", arguments: ["apps": "com.apple.TextEdit", "reason": "r"]),
            runtime: runtime)
        let resumed = await port.resumed
        XCTAssertEqual(resumed, 1)
    }
}

private actor StoppedPort: ComputerUseControlPort {
    private(set) var resumed = 0

    func perform(_ command: ComputerUseCommand) async throws -> ComputerUseResultValue {
        throw ComputerUsePortError(kind: .stoppedByUser, message: "The user stopped agent control.")
    }

    func resumeAfterStop() async {
        resumed += 1
    }
}
