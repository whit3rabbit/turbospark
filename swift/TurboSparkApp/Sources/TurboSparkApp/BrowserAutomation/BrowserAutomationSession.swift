import Foundation

public enum BrowserAutomationSessionError: Error, Equatable, Sendable, LocalizedError {
    case notAttached
    case alreadyAttached(BrowserTabID)
    case commandInFlight
    case invalidOwnershipToken
    case ownershipLost
    case userTookOver
    case detached
    case tabClosed(BrowserTabID)
    case unexpectedResponse

    case waitConditionNotMet(BrowserWaitTarget)

    public var errorDescription: String? {
        guard case .waitConditionNotMet(let target) = self else { return nil }
        return "The browser wait condition was not met before the action deadline: \(target)."
    }
}

public protocol BrowserAutomationClock: Sendable {
    func sleep(for interval: TimeInterval) async throws
}

public struct SystemBrowserAutomationClock: BrowserAutomationClock {
    public init() {}

    public func sleep(for interval: TimeInterval) async throws {
        guard interval > 0 else { return }
        try await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))
    }
}

public typealias BrowserControlPortFactory = @Sendable (BrowserTabID) -> any BrowserControlPort

/// Owns the single active tab lease and serializes all work sent through its tab-bound port.
public actor BrowserAutomationSession {
    public static let defaultTimeoutSeconds: TimeInterval = 10
    public static let maximumTimeoutSeconds: TimeInterval = 60

    private struct Attachment {
        let tabID: BrowserTabID
        let token: BrowserAgentControlToken
        let port: any BrowserControlPort
    }

    private final class ActiveCommand {
        let request: BrowserControlRequest
        let port: any BrowserControlPort
        let continuation: CheckedContinuation<BrowserControlResult, Error>
        let startedAt: TimeInterval
        var operationTask: Task<Void, Never>?
        var deadlineTask: Task<Void, Never>?

        init(
            request: BrowserControlRequest,
            port: any BrowserControlPort,
            continuation: CheckedContinuation<BrowserControlResult, Error>,
            startedAt: TimeInterval
        ) {
            self.request = request
            self.port = port
            self.continuation = continuation
            self.startedAt = startedAt
        }
    }

    private let tabStore: BrowserTabStore
    private let portFactory: BrowserControlPortFactory
    private let clock: any BrowserAutomationClock
    private let waitPollIntervalSeconds: TimeInterval
    private var attachment: Attachment?
    private var activeCommand: ActiveCommand?
    private var cancellationInProgress = false

    public init(
        tabStore: BrowserTabStore,
        portFactory: @escaping BrowserControlPortFactory,
        clock: any BrowserAutomationClock = SystemBrowserAutomationClock(),
        waitPollIntervalSeconds: TimeInterval = 0.1
    ) {
        self.tabStore = tabStore
        self.portFactory = portFactory
        self.clock = clock
        self.waitPollIntervalSeconds = max(waitPollIntervalSeconds, 0.001)
    }

    @discardableResult
    public func attach(to tabID: BrowserTabID) throws -> BrowserAgentControlToken {
        guard !cancellationInProgress else { throw BrowserAutomationSessionError.commandInFlight }
        // Defensive: an attachment to a tab the store no longer has can never
        // be used again and would block every new attach.
        if let stale = attachment, tabStore.tab(id: stale.tabID) == nil {
            attachment = nil
        }
        if let attachment {
            guard attachment.tabID == tabID else {
                throw BrowserAutomationSessionError.alreadyAttached(attachment.tabID)
            }
            return attachment.token
        }
        guard let tab = tabStore.tab(id: tabID) else {
            throw BrowserAutomationSessionError.tabClosed(tabID)
        }
        guard tab.owner == .agent else {
            throw BrowserTabStoreError.tabNotAgentOwned(tabID)
        }

        let token = try tabStore.acquireAgentControl(for: tabID)
        attachment = Attachment(tabID: tabID, token: token, port: portFactory(tabID))
        return token
    }

    public func attachedTabID() -> BrowserTabID? {
        attachment?.tabID
    }

    /// Direct user input has already transferred the tab lease in the engine.
    public func userTookOver(tabID: BrowserTabID) async {
        guard attachment?.tabID == tabID else { return }
        attachment = nil
        await cancelActive(with: BrowserAutomationSessionError.userTookOver)
    }

    /// The tab owner calls this when BrowserTabStore closes the attached tab.
    public func tabClosed(tabID: BrowserTabID) async {
        guard attachment?.tabID == tabID else { return }
        attachment = nil
        await cancelActive(with: BrowserAutomationSessionError.tabClosed(tabID))
    }

    /// Detaching returns the tab to direct user ownership and releases the store's lease.
    public func detach() async {
        guard let attachment else { return }
        self.attachment = nil
        await cancelActive(with: BrowserAutomationSessionError.detached)
        try? tabStore.transferOwnershipToUser(of: attachment.tabID)
    }

    public func cancelCurrentCommand() async {
        await cancelActive(with: CancellationError())
    }

    public func perform(
        _ command: BrowserControlCommand,
        using token: BrowserAgentControlToken,
        timeoutSeconds overrideSeconds: TimeInterval? = nil
    ) async throws -> BrowserControlResult {
        try Task.checkCancellation()
        guard let attachment else { throw BrowserAutomationSessionError.notAttached }
        guard attachment.token == token else {
            throw BrowserAutomationSessionError.invalidOwnershipToken
        }
        guard !cancellationInProgress, activeCommand == nil else {
            throw BrowserAutomationSessionError.commandInFlight
        }
        guard tabStore.tab(id: attachment.tabID) != nil else {
            self.attachment = nil
            throw BrowserAutomationSessionError.tabClosed(attachment.tabID)
        }
        do {
            guard try tabStore.agentControlledTab(for: token) == attachment.tabID else {
                throw BrowserAutomationSessionError.ownershipLost
            }
        } catch let error as BrowserAutomationSessionError {
            throw error
        } catch {
            self.attachment = nil
            throw BrowserAutomationSessionError.ownershipLost
        }

        let timeout = try Self.effectiveTimeoutSeconds(for: command, overrideSeconds: overrideSeconds)
        let request = BrowserControlRequest(command: command)
        let startedAt = ProcessInfo.processInfo.systemUptime

        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation {
                (continuation: CheckedContinuation<BrowserControlResult, Error>) in
                let active = ActiveCommand(
                    request: request,
                    port: attachment.port,
                    continuation: continuation,
                    startedAt: startedAt
                )
                activeCommand = active
                active.operationTask = Task { [weak self] in
                    guard let self else { return }
                    do {
                        let result = try await self.execute(request, on: attachment)
                        await self.complete(requestID: request.id, result: .success(result))
                    } catch {
                        await self.complete(requestID: request.id, result: .failure(error))
                    }
                }
                active.deadlineTask = Task { [weak self, clock] in
                    do {
                        try await clock.sleep(for: timeout)
                    } catch {
                        return
                    }
                    await self?.expire(requestID: request.id)
                }
            }
        } onCancel: {
            Task { await self.cancel(requestID: request.id, with: CancellationError()) }
        }
    }

    static func effectiveTimeoutSeconds(
        for command: BrowserControlCommand,
        overrideSeconds: TimeInterval?
    ) throws -> TimeInterval {
        let requested = overrideSeconds ?? command.timeoutSeconds ?? defaultTimeoutSeconds
        guard requested.isFinite, requested > 0 else {
            throw BrowserControlError.invalidInput(reason: .invalidTimeout)
        }
        return min(requested, maximumTimeoutSeconds)
    }

    private func execute(
        _ request: BrowserControlRequest,
        on attachment: Attachment
    ) async throws -> BrowserControlResult {
        guard case .waitFor(let target, _) = request.command else {
            return try await attachment.port.perform(request)
        }

        while true {
            try Task.checkCancellation()
            if try await attachment.port.isWaitConditionSatisfied(target, commandID: request.id) {
                let elapsed = max(
                    0,
                    ProcessInfo.processInfo.systemUptime
                        - (activeCommand?.startedAt ?? ProcessInfo.processInfo.systemUptime)
                )
                return BrowserControlResult(
                    request: request,
                    value: .waitCompleted(BrowserWaitResult(target: target)),
                    durationMilliseconds: Int(elapsed * 1_000)
                )
            }
            try Task.checkCancellation()
            try await clock.sleep(for: waitPollIntervalSeconds)
        }
    }

    private func complete(
        requestID: UUID,
        result: Result<BrowserControlResult, Error>
    ) {
        guard let active = activeCommand, active.request.id == requestID else {
            return
        }
        activeCommand = nil
        active.deadlineTask?.cancel()

        switch result {
        case .success(let value):
            guard value.requestID == requestID else {
                active.continuation.resume(throwing: BrowserAutomationSessionError.unexpectedResponse)
                return
            }
            active.continuation.resume(returning: value)
        case .failure(let error):
            active.continuation.resume(throwing: error)
        }
    }

    private func expire(requestID: UUID) async {
        guard let active = activeCommand, active.request.id == requestID else { return }
        let error: Error
        if case .waitFor(let target, _) = active.request.command {
            error = BrowserAutomationSessionError.waitConditionNotMet(target)
        } else {
            error = BrowserControlError.timeout
        }
        await cancelActive(with: error)
    }

    private func cancel(requestID: UUID, with error: Error) async {
        guard activeCommand?.request.id == requestID else { return }
        await cancelActive(with: error)
    }

    private func cancelActive(with error: Error) async {
        guard let active = activeCommand else { return }
        activeCommand = nil
        cancellationInProgress = true
        active.deadlineTask?.cancel()
        active.operationTask?.cancel()
        await active.port.cancel(commandID: active.request.id)
        cancellationInProgress = false
        active.continuation.resume(throwing: error)
    }
}

extension BrowserControlCommand {
    fileprivate var timeoutSeconds: TimeInterval? {
        switch self {
        case let .navigate(_, _, timeoutSeconds),
             let .click(_, timeoutSeconds),
             let .type(_, _, _, timeoutSeconds),
             let .pressKey(_, _, timeoutSeconds):
            timeoutSeconds
        case let .waitFor(_, timeoutSeconds):
            timeoutSeconds
        case .scroll, .screenshot, .readState, .setViewport:
            nil
        }
    }
}
