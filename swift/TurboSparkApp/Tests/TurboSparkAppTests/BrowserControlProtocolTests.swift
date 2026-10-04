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

final class BrowserAutomationSessionTests: XCTestCase {
    func testWaitPollsUntilConditionIsSatisfiedForTheAttachedTab() async throws {
        let store = BrowserTabStore()
        let visibleTab = store.createTab(owner: .user)
        let controlledTab = store.createTab(owner: .agent, select: false)
        let port = SessionTestBrowserControlPort()
        let session = BrowserAutomationSession(
            tabStore: store,
            portFactory: { _ in port },
            clock: SystemBrowserAutomationClock(),
            waitPollIntervalSeconds: 0.001
        )
        let token = try await session.attach(to: controlledTab)
        let target = BrowserWaitTarget.element(reference: "button-1", condition: .enabled)

        let result = try await session.perform(.waitFor(target: target, timeoutSeconds: 2), using: token)

        XCTAssertEqual(result.value, .waitCompleted(BrowserWaitResult(target: target)))
        let probeCount = await port.waitProbeCount()
        XCTAssertEqual(probeCount, 3)
        XCTAssertEqual(store.activeTabID, visibleTab)
        XCTAssertEqual(store.tab(id: visibleTab)?.owner, .user)
    }

    func testWaitPollsForLoadState() async throws {
        let store = BrowserTabStore()
        let controlledTab = store.createTab(owner: .agent)
        let port = SessionTestBrowserControlPort()
        await port.setWaitSatisfiedAfter(1)
        let session = BrowserAutomationSession(tabStore: store, portFactory: { _ in port })
        let token = try await session.attach(to: controlledTab)
        let target = BrowserWaitTarget.loadState(.finished)

        let result = try await session.perform(.waitFor(target: target, timeoutSeconds: 1), using: token)

        XCTAssertEqual(result.value, .waitCompleted(BrowserWaitResult(target: target)))
        let lastTarget = await port.lastWaitTarget()
        XCTAssertEqual(lastTarget, target)
    }

    func testTimeoutCancelsCommandAndIgnoresLateReply() async throws {
        let store = BrowserTabStore()
        let controlledTab = store.createTab(owner: .agent)
        let port = SessionTestBrowserControlPort()
        await port.setSuspendsReplies(true)
        let session = BrowserAutomationSession(tabStore: store, portFactory: { _ in port })
        let token = try await session.attach(to: controlledTab)
        let command = Task {
            try await session.perform(.readState(scope: .pageStructure), using: token, timeoutSeconds: 0.05)
        }
        while await port.requestCount() == 0 {
            await Task.yield()
        }

        do {
            _ = try await command.value
            XCTFail("The command should time out.")
        } catch let error as BrowserControlError {
            XCTAssertEqual(error, .timeout)
        }

        let requestIDs = await port.requestIDs()
        let requestID = try XCTUnwrap(requestIDs.first)
        let cancelled = await port.cancelledCommandIDs()
        XCTAssertTrue(cancelled.contains(requestID))

        await port.replyLate(to: requestID)
        await port.setSuspendsReplies(false)
        let nextResult = try await session.perform(
            .readState(scope: .pageStructure),
            using: token,
            timeoutSeconds: 1
        )
        XCTAssertEqual(nextResult.commandKind, .readState)
    }

    func testUserTakeoverCancelsInFlightCommandAndDropsSessionOwnership() async throws {
        let store = BrowserTabStore()
        let controlledTab = store.createTab(owner: .agent)
        let port = SessionTestBrowserControlPort()
        await port.setSuspendsReplies(true)
        let session = BrowserAutomationSession(tabStore: store, portFactory: { _ in port })
        let token = try await session.attach(to: controlledTab)
        let command = Task {
            try await session.perform(.readState(scope: .pageStructure), using: token)
        }
        while await port.requestCount() == 0 {
            await Task.yield()
        }

        try store.transferOwnershipToUser(of: controlledTab)
        await session.userTookOver(tabID: controlledTab)

        do {
            _ = try await command.value
            XCTFail("User takeover should cancel the command.")
        } catch let error as BrowserAutomationSessionError {
            XCTAssertEqual(error, .userTookOver)
        }

        let cancelled = await port.cancelledCommandIDs()
        XCTAssertEqual(cancelled.count, 1)
        let attachedTab = await session.attachedTabID()
        XCTAssertNil(attachedTab)
        XCTAssertEqual(store.tab(id: controlledTab)?.owner, .user)
        XCTAssertThrowsError(try store.agentControlledTab(for: token))

        let requestIDs = await port.requestIDs()
        if let requestID = requestIDs.first {
            await port.replyLate(to: requestID)
        }
    }

    func testOnlyOneCommandRunsAndDetachCancelsAndReleasesControl() async throws {
        let store = BrowserTabStore()
        let controlledTab = store.createTab(owner: .agent)
        let otherAgentTab = store.createTab(owner: .agent, select: false)
        let port = SessionTestBrowserControlPort()
        await port.setSuspendsReplies(true)
        let session = BrowserAutomationSession(tabStore: store, portFactory: { _ in port })
        let token = try await session.attach(to: controlledTab)
        do {
            _ = try await session.attach(to: otherAgentTab)
            XCTFail("A session must not attach to a second tab.")
        } catch let error as BrowserAutomationSessionError {
            XCTAssertEqual(error, .alreadyAttached(controlledTab))
        }
        let command = Task {
            try await session.perform(.readState(scope: .pageStructure), using: token)
        }
        while await port.requestCount() == 0 {
            await Task.yield()
        }

        do {
            _ = try await session.perform(.readState(scope: .pageStructure), using: token)
            XCTFail("A second command must not overlap the first.")
        } catch let error as BrowserAutomationSessionError {
            XCTAssertEqual(error, .commandInFlight)
        }

        await session.detach()
        do {
            _ = try await command.value
            XCTFail("Detaching should cancel the active command.")
        } catch let error as BrowserAutomationSessionError {
            XCTAssertEqual(error, .detached)
        }

        let cancelled = await port.cancelledCommandIDs()
        XCTAssertEqual(cancelled.count, 1)
        XCTAssertEqual(store.tab(id: controlledTab)?.owner, .user)
        XCTAssertThrowsError(try store.agentControlledTab(for: token))
        let requestIDs = await port.requestIDs()
        if let requestID = requestIDs.first {
            await port.replyLate(to: requestID)
        }
    }

    func testTabCloseCancelsActiveCommand() async throws {
        let store = BrowserTabStore()
        let controlledTab = store.createTab(owner: .agent)
        let port = SessionTestBrowserControlPort()
        await port.setSuspendsReplies(true)
        let session = BrowserAutomationSession(tabStore: store, portFactory: { _ in port })
        let token = try await session.attach(to: controlledTab)
        let command = Task {
            try await session.perform(.readState(scope: .pageStructure), using: token)
        }
        while await port.requestCount() == 0 {
            await Task.yield()
        }

        try store.closeTab(controlledTab)
        await session.tabClosed(tabID: controlledTab)

        do {
            _ = try await command.value
            XCTFail("Closing the controlled tab should cancel its command.")
        } catch let error as BrowserAutomationSessionError {
            XCTAssertEqual(error, .tabClosed(controlledTab))
        }

        let cancelled = await port.cancelledCommandIDs()
        XCTAssertEqual(cancelled.count, 1)
        let attachedTab = await session.attachedTabID()
        XCTAssertNil(attachedTab)
        let requestIDs = await port.requestIDs()
        if let requestID = requestIDs.first {
            await port.replyLate(to: requestID)
        }
    }

    func testWaitTimeoutReportsTheUnmetCondition() async throws {
        let store = BrowserTabStore()
        let controlledTab = store.createTab(owner: .agent)
        let port = SessionTestBrowserControlPort()
        await port.setWaitSatisfiedAfter(Int.max)
        let session = BrowserAutomationSession(
            tabStore: store,
            portFactory: { _ in port },
            waitPollIntervalSeconds: 0.001
        )
        let token = try await session.attach(to: controlledTab)
        let target = BrowserWaitTarget.element(reference: "submit-button", condition: .visible)

        do {
            _ = try await session.perform(
                .waitFor(target: target, timeoutSeconds: 0.05),
                using: token
            )
            XCTFail("The wait should stop at its deadline.")
        } catch let error as BrowserAutomationSessionError {
            XCTAssertEqual(error, .waitConditionNotMet(target))
            XCTAssertTrue(error.localizedDescription.contains("submit-button"))
        }

        let probeCount = await port.waitProbeCount()
        XCTAssertGreaterThan(probeCount, 0)
        let cancelled = await port.cancelledCommandIDs()
        XCTAssertEqual(cancelled.count, 1)
    }

    func testTimeoutPolicyUsesTenSecondDefaultAndSixtySecondCap() throws {
        let command = BrowserControlCommand.readState(scope: .pageStructure)

        XCTAssertEqual(
            try BrowserAutomationSession.effectiveTimeoutSeconds(for: command, overrideSeconds: nil),
            10
        )
        XCTAssertEqual(
            try BrowserAutomationSession.effectiveTimeoutSeconds(for: command, overrideSeconds: 3),
            3
        )
        XCTAssertEqual(
            try BrowserAutomationSession.effectiveTimeoutSeconds(for: command, overrideSeconds: 120),
            60
        )
        XCTAssertThrowsError(
            try BrowserAutomationSession.effectiveTimeoutSeconds(for: command, overrideSeconds: 0)
        ) { error in
            XCTAssertEqual(error as? BrowserControlError, .invalidInput(reason: .invalidTimeout))
        }
    }
}

private actor SessionTestBrowserControlPort: BrowserControlPort {
    private struct PendingReply {
        let request: BrowserControlRequest
        let continuation: CheckedContinuation<BrowserControlResult, Error>
    }

    static let backendIdentifier = "session-test-backend"
    static let supportedCommands = Set(BrowserControlCommandKind.allCases)

    private var suspendsReplies = false
    private var pendingReplies: [UUID: PendingReply] = [:]
    private var requests: [BrowserControlRequest] = []
    private var cancelledIDs: Set<UUID> = []
    private var waitProbes = 0
    private var lastProbedTarget: BrowserWaitTarget?

    private var waitSatisfiedAfter = 3

    func setSuspendsReplies(_ value: Bool) { suspendsReplies = value }
    func requestCount() -> Int { requests.count }
    func requestIDs() -> [UUID] { requests.map(\.id) }
    func cancelledCommandIDs() -> Set<UUID> { cancelledIDs }
    func setWaitSatisfiedAfter(_ count: Int) { waitSatisfiedAfter = count }
    func waitProbeCount() -> Int { waitProbes }
    func lastWaitTarget() -> BrowserWaitTarget? { lastProbedTarget }

    func perform(_ request: BrowserControlRequest) async throws -> BrowserControlResult {
        requests.append(request)
        if suspendsReplies {
            return try await withCheckedThrowingContinuation {
                (continuation: CheckedContinuation<BrowserControlResult, Error>) in
                pendingReplies[request.id] = PendingReply(request: request, continuation: continuation)
            }
        }
        return try await StubBrowserControlPort().perform(request)
    }

    func cancel(commandID: UUID) async { cancelledIDs.insert(commandID) }

    func isWaitConditionSatisfied(
        _ target: BrowserWaitTarget,
        commandID: UUID
    ) async throws -> Bool {
        waitProbes += 1
        lastProbedTarget = target
        return waitProbes >= waitSatisfiedAfter
    }

    func replyLate(to requestID: UUID) {
        guard let pending = pendingReplies.removeValue(forKey: requestID) else { return }
        let value = BrowserSnapshotResult(
            scope: .pageStructure, url: nil, title: nil, snapshot: "late", truncated: false
        )
        pending.continuation.resume(returning: BrowserControlResult(
            request: pending.request, value: .state(value), durationMilliseconds: 1
        ))
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

    func isWaitConditionSatisfied(
        _ target: BrowserWaitTarget,
        commandID: UUID
    ) async throws -> Bool {
        true
    }

    func isCancelled(commandID: UUID) -> Bool {
        cancelledCommandIDs.contains(commandID)
    }
}
