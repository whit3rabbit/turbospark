import XCTest

@testable import TurboSparkApp

final class BrowserControlProtocolTests: XCTestCase {
    func testCommandVocabularyMapsEveryCommandToItsVersionedManifestEntry() throws {
        let kinds = Set(BrowserControlCommandKind.allCases)
        let supported = Set(BrowserControlCommandKind.allCases)
        let manifest = BrowserBackendManifest(
            backendIdentifier: StubBrowserControlPort.backendIdentifier,
            supportedCommands: supported
        )
        let reference = "element-1"
        let commands: [BrowserControlCommand] = [
            .navigate(url: "https://example.com", waitUntil: .finished, timeoutSeconds: 10),
            .click(reference: reference, timeoutSeconds: 5),
            .type(reference: reference, text: "value", submit: false, timeoutSeconds: 5),
            .pressKey(reference: reference, key: "Enter", timeoutSeconds: 5),
            .scroll(reference: reference, direction: .down, amount: 100),
            .screenshot(fullPage: false),
            .readState(scope: .interactiveElements),
            .waitFor(target: .loadState(.finished), timeoutSeconds: 10),
            .setViewport(width: 1280, height: 800, zoom: 1)
        ]

        XCTAssertEqual(Set(manifest.commandSupport.keys), kinds)
        XCTAssertEqual(Set(commands.map(\.kind)), kinds)
        XCTAssertEqual(commands.count, kinds.count)
        for kind in kinds {
            XCTAssertEqual(manifest.support(for: kind), .supported)
        }

        let v1Commands: Set<BrowserControlCommandKind> = [
            .navigate, .click, .type, .pressKey, .scroll,
            .screenshot, .readState, .waitFor, .setViewport
        ]
        let v1Snapshot = BrowserControlProtocol.commandKinds(forVersion: 1)
        let currentSnapshot = try XCTUnwrap(
            BrowserControlProtocol.commandKindsByVersion[BrowserControlProtocol.currentVersion],
            "The current manifest version must have an explicit command snapshot."
        )
        let latestSnapshotVersion = try XCTUnwrap(
            BrowserControlProtocol.commandKindsByVersion.keys.max()
        )

        XCTAssertEqual(Set(v1Snapshot), v1Commands, "The v1 vocabulary is a frozen contract.")
        XCTAssertEqual(v1Snapshot.count, v1Commands.count, "A snapshot must not duplicate commands.")
        XCTAssertEqual(Set(currentSnapshot), kinds, "Every enum case must be in the current snapshot.")
        XCTAssertEqual(currentSnapshot.count, kinds.count, "The current snapshot must not duplicate commands.")
        XCTAssertEqual(BrowserControlProtocol.currentVersion, latestSnapshotVersion)
        XCTAssertEqual(manifest.commandSurfaceVersion, BrowserControlProtocol.currentVersion)
        XCTAssertEqual(manifest.backendIdentifier, StubBrowserControlPort.backendIdentifier)
    }

    func testBackendManifestReportsSupportPerCommandKind() {
        let supported: Set<BrowserControlCommandKind> = [.navigate, .screenshot]
        let manifest = BrowserBackendManifest(
            backendIdentifier: "partial-test-backend",
            supportedCommands: supported
        )

        XCTAssertEqual(manifest.support(for: .navigate), .supported)
        XCTAssertEqual(manifest.support(for: .screenshot), .supported)
        XCTAssertEqual(manifest.support(for: .click), .unsupported)
    }

    func testTypedResultValuesCoverTheClosedCommandSet() {
        let values: [BrowserControlValue] = [
            .navigated(BrowserNavigationResult(url: "https://example.com", reachedState: .finished)),
            .clicked(BrowserElementActionResult(reference: "r1")),
            .typed(BrowserElementActionResult(reference: "r1")),
            .keyPressed(BrowserKeyPressResult(reference: "r1", key: "Enter")),
            .scrolled(BrowserScrollResult(reference: nil, direction: .down, amount: 10)),
            .screenshot(BrowserScreenshotResult(captureID: UUID(), width: 10, height: 10, fullPage: false)),
            .state(BrowserSnapshotResult(scope: .pageStructure, url: nil, title: nil, snapshot: "", truncated: false)),
            .waitCompleted(BrowserWaitResult(target: .loadState(.finished))),
            .viewportSet(BrowserViewportResult(width: 1280, height: 800, zoom: 1))
        ]

        XCTAssertEqual(Set(values.map(\.commandKind)), Set(BrowserControlCommandKind.allCases))
        XCTAssertEqual(values.count, BrowserControlCommandKind.allCases.count)
    }

    func testPortReturnsARequestBoundTypedResult() async throws {
        let requestID = UUID()
        let request = BrowserControlRequest(
            id: requestID,
            command: .navigate(url: "https://example.com", waitUntil: .finished, timeoutSeconds: nil)
        )
        let result = try await StubBrowserControlPort().perform(request)

        XCTAssertEqual(result.requestID, requestID)
        XCTAssertEqual(result.commandKind, .navigate)
        XCTAssertEqual(
            result.value,
            .navigated(BrowserNavigationResult(url: "https://example.com", reachedState: .finished))
        )
    }

    func testCancellationIsIdempotentForARequestID() async {
        let port = StubBrowserControlPort()
        let requestID = UUID()

        await port.cancel(commandID: requestID)
        await port.cancel(commandID: requestID)

        let isCancelled = await port.isCancelled(commandID: requestID)
        XCTAssertTrue(isCancelled)
    }

    func testProtocolErrorsAreTyped() {
        let errors: [BrowserControlError] = [
            .staleReference(reference: "ref-1"),
            .timeout,
            .unsupported(command: .screenshot),
            .blockedNavigation(origin: "https://blocked.example"),
            .navigationFailure(reason: .transport),
            .engineCrashed,
            .invalidInput(reason: .invalidKey)
        ]

        XCTAssertEqual(Set(errors.map(\.kind)), [
            .staleReference, .timeout, .unsupported, .blockedNavigation,
            .navigationFailure, .engineCrashed, .invalidInput
        ])
    }
}

private actor StubBrowserControlPort: BrowserControlPort {
    static let backendIdentifier = "test-backend"
    static let supportedCommands: Set<BrowserControlCommandKind> = [
        .navigate, .click, .type, .pressKey, .scroll,
        .screenshot, .readState, .waitFor, .setViewport
    ]

    private var cancelledCommandIDs: Set<UUID> = []

    func perform(_ request: BrowserControlRequest) async throws -> BrowserControlResult {
        let value: BrowserControlValue
        switch request.command {
        case let .navigate(url, waitUntil, _):
            value = .navigated(BrowserNavigationResult(url: url, reachedState: waitUntil))
        case let .click(reference, _):
            value = .clicked(BrowserElementActionResult(reference: reference))
        case let .type(reference, _, _, _):
            value = .typed(BrowserElementActionResult(reference: reference))
        case let .pressKey(reference, key, _):
            value = .keyPressed(BrowserKeyPressResult(reference: reference, key: key))
        case let .scroll(reference, direction, amount):
            value = .scrolled(BrowserScrollResult(reference: reference, direction: direction, amount: amount))
        case let .screenshot(fullPage):
            value = .screenshot(
                BrowserScreenshotResult(
                    captureID: UUID(),
                    width: 1,
                    height: 1,
                    fullPage: fullPage
                )
            )
        case let .readState(scope):
            value = .state(
                BrowserSnapshotResult(
                    scope: scope,
                    url: nil,
                    title: nil,
                    snapshot: "",
                    truncated: false
                )
            )
        case let .waitFor(target, _):
            value = .waitCompleted(BrowserWaitResult(target: target))
        case let .setViewport(width, height, zoom):
            value = .viewportSet(BrowserViewportResult(width: width, height: height, zoom: zoom))
        }
        return BrowserControlResult(request: request, value: value, durationMilliseconds: 1)
    }

    func cancel(commandID: UUID) async {
        cancelledCommandIDs.insert(commandID)
    }

    func isCancelled(commandID: UUID) -> Bool {
        cancelledCommandIDs.contains(commandID)
    }
}
