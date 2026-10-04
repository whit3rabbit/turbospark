import AppKit
import Foundation

/// Adapts the app-owned WebKit engine to the backend-neutral automation session.
public final class WebKitBrowserControlPort: BrowserControlPort, @unchecked Sendable {
    public static let backendIdentifier = "webkit-macos"
    public static let supportedCommands: Set<BrowserControlCommandKind> = [
        .navigate, .click, .type, .pressKey, .scroll, .screenshot, .readState, .waitFor,
    ]
    public static let manifest = BrowserBackendManifest(
        backendIdentifier: backendIdentifier,
        supportedCommands: supportedCommands
    )

    private let engine: WebKitBrowserEngine
    private let tabID: BrowserTabID
    private let lock = NSLock()
    private var activeCommands: [UUID: BrowserControlCommandKind] = [:]
    private var cancelledCommandIDs = Set<UUID>()

    public init(engine: WebKitBrowserEngine, tabID: BrowserTabID) {
        self.engine = engine
        self.tabID = tabID
    }

    public func perform(_ request: BrowserControlRequest) async throws -> BrowserControlResult {
        guard Self.supportedCommands.contains(request.command.kind) else {
            throw BrowserControlError.unsupported(command: request.command.kind)
        }
        try register(request)
        defer { finish(request.id) }

        return try await Task { @MainActor [weak self] in
            guard let self else { throw BrowserControlError.engineCrashed }
            return try await self.performOnMainActor(request)
        }.value
    }

    public func isWaitConditionSatisfied(
        _ target: BrowserWaitTarget,
        commandID: UUID
    ) async throws -> Bool {
        guard !isCancelled(commandID) else { throw CancellationError() }
        return try await Task { @MainActor [weak self] in
            guard let self else { throw BrowserControlError.engineCrashed }
            return try await self.waitConditionOnMainActor(target, commandID: commandID)
        }.value
    }

    public func cancel(commandID: UUID) async {
        let commandKind = markCancelled(commandID)
        let engine = self.engine
        let tabID = self.tabID
        await Task { @MainActor in
            for dialog in engine.pendingDialogs where dialog.tabID == tabID {
                engine.cancelPendingDialog(dialog.id, reason: .cancelled)
            }
            if commandKind == .navigate {
                engine.stopLoading(in: tabID)
            }
        }.value
    }

    @MainActor
    private func performOnMainActor(_ request: BrowserControlRequest) async throws -> BrowserControlResult {
        try checkNotCancelled(request.id)
        let start = ProcessInfo.processInfo.systemUptime
        let value: BrowserControlValue
        var screenshotPNGData: Data?

        do {
            switch request.command {
            case let .navigate(rawURL, waitUntil, timeoutSeconds):
                guard let url = URL(string: rawURL), BrowserNavigationDestination(url: url) != nil else {
                    throw BrowserControlError.invalidInput(reason: .invalidAddress)
                }
                try engine.navigate(to: url, in: tabID)
                try await waitForLoadState(
                    waitUntil,
                    timeoutSeconds: timeoutSeconds ?? BrowserAutomationSession.defaultTimeoutSeconds,
                    commandID: request.id
                )
                try checkNotCancelled(request.id)
                let actualAddress = engine.tabStates[tabID]?.address ?? rawURL
                value = .navigated(BrowserNavigationResult(url: actualAddress, reachedState: waitUntil))

            case let .click(reference, _):
                let service = try snapshotService()
                let generation = try await service.snapshot().generation
                try checkNotCancelled(request.id)
                let action = try await service.perform(.click, on: .reference(reference), generation: generation)
                value = .clicked(BrowserElementActionResult(reference: action.reference))

            case let .type(reference, text, submit, _):
                let service = try snapshotService()
                let generation = try await service.snapshot().generation
                try checkNotCancelled(request.id)
                let action = try await service.perform(
                    .type(text: text, submit: submit),
                    on: .reference(reference),
                    generation: generation
                )
                value = .typed(BrowserElementActionResult(reference: action.reference))

            case let .pressKey(reference, key, _):
                let service = try snapshotService()
                let generation = try await service.snapshot().generation
                try checkNotCancelled(request.id)
                let locator: DOMSnapshotLocator = reference.map(DOMSnapshotLocator.reference)
                    ?? .cssSelector("body")
                let action = try await service.perform(.pressKey(key: key), on: locator, generation: generation)
                value = .keyPressed(BrowserKeyPressResult(reference: reference ?? action.reference, key: key))

            case let .scroll(reference, direction, amount):
                let service = try snapshotService()
                let generation = try await service.snapshot().generation
                try checkNotCancelled(request.id)
                let locator: DOMSnapshotLocator = reference.map(DOMSnapshotLocator.reference)
                    ?? .cssSelector("body")
                let action = try await service.perform(
                    .scroll(direction: Self.snapshotDirection(direction), amount: amount),
                    on: locator,
                    generation: generation
                )
                value = .scrolled(BrowserScrollResult(
                    reference: reference ?? action.reference,
                    direction: direction,
                    amount: amount
                ))

            case let .screenshot(fullPage):
                guard !fullPage else {
                    throw BrowserControlError.unsupported(command: .screenshot)
                }
                let image = try await engine.captureScreenshot(for: tabID)
                try checkNotCancelled(request.id)
                guard let representation = image.tiffRepresentation.flatMap(NSBitmapImageRep.init(data:)),
                      let png = representation.representation(using: .png, properties: [:]) else {
                    throw BrowserControlError.engineCrashed
                }
                screenshotPNGData = png
                value = .screenshot(BrowserScreenshotResult(
                    captureID: UUID(),
                    width: representation.pixelsWide,
                    height: representation.pixelsHigh,
                    fullPage: false
                ))

            case let .readState(scope):
                let envelope = try await snapshotService().snapshot()
                try checkNotCancelled(request.id)
                let visibleNodes = scope == .interactiveElements
                    ? envelope.nodes.filter { $0.reference != nil }
                    : envelope.nodes
                let boundedEnvelope = DOMSnapshotEnvelope(
                    version: envelope.version,
                    generation: envelope.generation,
                    nodes: visibleNodes,
                    truncated: envelope.truncated
                )
                let data = try JSONEncoder().encode(boundedEnvelope)
                value = .state(BrowserSnapshotResult(
                    scope: scope,
                    url: engine.tabStates[tabID]?.address,
                    title: engine.tabStates[tabID]?.title,
                    snapshot: String(decoding: data, as: UTF8.self),
                    truncated: envelope.truncated
                ))

            case .waitFor:
                throw BrowserControlError.unsupported(command: .waitFor)

            case .setViewport:
                throw BrowserControlError.unsupported(command: .setViewport)
            }
        } catch let error as BrowserControlError {
            throw error
        } catch let error as DOMSnapshotServiceError {
            throw Self.controlError(for: error)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            if let navigationError = engine.tabStates[tabID]?.navigationError {
                throw navigationError
            }
            throw BrowserControlError.engineCrashed
        }

        try checkNotCancelled(request.id)
        return BrowserControlResult(
            request: request,
            value: value,
            durationMilliseconds: Self.elapsedMilliseconds(since: start),
            screenshotPNGData: screenshotPNGData
        )
    }

    @MainActor
    private func waitConditionOnMainActor(
        _ target: BrowserWaitTarget,
        commandID: UUID
    ) async throws -> Bool {
        try checkNotCancelled(commandID)
        switch target {
        case .loadState(.started):
            guard let state = engine.tabStates[tabID]?.loadState else { return false }
            return state != .idle
        case .loadState(.committed):
            return engine.snapshotService(for: tabID)?.canPickElement == true
        case .loadState(.finished):
            return engine.tabStates[tabID]?.loadState == .loaded
        case let .element(reference, condition):
            let service = try snapshotService()
            return try await service.isWaitConditionSatisfied(reference: reference, condition: condition)
        }
    }

    @MainActor
    private func waitForLoadState(
        _ target: BrowserLoadState,
        timeoutSeconds: TimeInterval,
        commandID: UUID
    ) async throws {
        let timeout = min(max(timeoutSeconds, 0.001), BrowserAutomationSession.maximumTimeoutSeconds)
        let deadline = ProcessInfo.processInfo.systemUptime + timeout
        while ProcessInfo.processInfo.systemUptime < deadline {
            try checkNotCancelled(commandID)
            if try await waitConditionOnMainActor(.loadState(target), commandID: commandID) {
                return
            }
            if let state = engine.tabStates[tabID] {
                switch state.navigationError {
                case .blockedNavigation(let origin): throw BrowserControlError.blockedNavigation(origin: origin)
                case .navigationFailure(let reason): throw BrowserControlError.navigationFailure(reason: reason)
                case .engineCrashed: throw BrowserControlError.engineCrashed
                case .staleReference(_), .timeout, .unsupported(_), .invalidInput(_), .none: break
                }
                if state.loadState == .crashed { throw BrowserControlError.engineCrashed }
                if case .failed = state.loadState {
                    throw BrowserControlError.navigationFailure(reason: .unknown)
                }
            }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        throw BrowserControlError.timeout
    }

    @MainActor
    private func snapshotService() throws -> DOMSnapshotService {
        guard let service = engine.snapshotService(for: tabID) else {
            throw BrowserControlError.navigationFailure(reason: .unknown)
        }
        return service
    }

    private func register(_ request: BrowserControlRequest) throws {
        lock.lock()
        defer { lock.unlock() }
        guard !cancelledCommandIDs.contains(request.id) else { throw CancellationError() }
        activeCommands[request.id] = request.command.kind
    }

    private func finish(_ commandID: UUID) {
        lock.lock()
        defer { lock.unlock() }
        activeCommands.removeValue(forKey: commandID)
    }

    private func isCancelled(_ commandID: UUID) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelledCommandIDs.contains(commandID)
    }

    private func checkNotCancelled(_ commandID: UUID) throws {
        if isCancelled(commandID) { throw CancellationError() }
        try Task.checkCancellation()
    }

    private func markCancelled(_ commandID: UUID) -> BrowserControlCommandKind? {
        lock.lock()
        defer { lock.unlock() }
        cancelledCommandIDs.insert(commandID)
        return activeCommands[commandID]
    }

    private static func controlError(for error: DOMSnapshotServiceError) -> BrowserControlError {
        switch error {
        case .invalidLocator, .targetNotFound, .locatorLimitExceeded, .invalidAction, .actionRejected:
            .invalidInput(reason: .missingReference)
        case .inactiveDocument, .rejectedMessage, .invalidSnapshot, .unavailable:
            .navigationFailure(reason: .unknown)
        }
    }

    private static func snapshotDirection(
        _ direction: BrowserScrollDirection
    ) -> DOMSnapshotScrollDirection {
        switch direction {
        case .up: .up
        case .down: .down
        case .left: .left
        case .right: .right
        }
    }

    private static func elapsedMilliseconds(since start: TimeInterval) -> Int {
        Int(max(0, ProcessInfo.processInfo.systemUptime - start) * 1_000)
    }
}
