import Foundation
import WebKit

/// How long a permission bubble waits before the request is denied.
public let sitePermissionPromptTimeout: TimeInterval = 30

/// Maps what the WebKit delegate delivers to the classes the app governs.
/// The list is closed: an empty result means "not a class we classify", and
/// the policy denies it. Notifications, clipboard-read, screen share and
/// MIDI/HID/USB/serial/Bluetooth have no `WKUIDelegate` hook, so they never
/// reach this code and are not governed here.
enum SitePermissionClassifier {
    static func classes(forCapture type: WKMediaCaptureType) -> [SitePermissionClass] {
        switch type {
        case .camera: return [.camera]
        case .microphone: return [.microphone]
        case .cameraAndMicrophone: return [.camera, .microphone]
        @unknown default: return []
        }
    }

    static let geolocation: [SitePermissionClass] = [.geolocation]

    /// The requesting frame's origin from WebKit's security origin. `nil` for
    /// anything that is not an HTTP(S) origin, which the policy denies.
    static func origin(
        scheme: String, host: String, port: Int
    ) -> BrowserOrigin? {
        let serializedHost = host.contains(":") && !host.hasPrefix("[") ? "[\(host)]" : host
        let suffix = port > 0 ? ":\(port)" : ""
        return BrowserOrigin(origin: "\(scheme)://\(serializedHost)\(suffix)")
    }
}

/// Calls its handler at most once. A leaked wrapper (its owner dropped it
/// without answering) denies on deinit, so WebKit's decision handler is never
/// left uncalled. Not main-actor isolated on purpose: deinit has no isolation.
public final class SingleShotPermissionDecision: @unchecked Sendable {
    private let lock = NSLock()
    private var handler: (@Sendable (Bool) -> Void)?

    public init(_ handler: @escaping @Sendable (Bool) -> Void) {
        self.handler = handler
    }

    /// Returns true only for the call that actually delivered the answer.
    @discardableResult
    public func resolve(_ allow: Bool) -> Bool {
        lock.lock()
        let handler = self.handler
        self.handler = nil
        lock.unlock()
        guard let handler else { return false }
        handler(allow)
        return true
    }

    deinit {
        // Not `resolve`: self is already being destroyed.
        handler?(false)
    }
}

public struct SitePermissionRequest: Equatable, Sendable {
    public let tabID: BrowserTabID
    /// The frame that asked. Records are keyed by this, never by the top page.
    public let requestingOrigin: BrowserOrigin?
    public let topLevelOrigin: BrowserOrigin?
    /// Empty means the delegate delivered a class the app does not classify.
    public let classes: [SitePermissionClass]

    public init(
        tabID: BrowserTabID,
        requestingOrigin: BrowserOrigin?,
        topLevelOrigin: BrowserOrigin?,
        classes: [SitePermissionClass]
    ) {
        self.tabID = tabID
        self.requestingOrigin = requestingOrigin
        self.topLevelOrigin = topLevelOrigin
        self.classes = classes
    }
}

public enum SitePermissionAnswer: Equatable, Sendable {
    case allowOnce
    case allowSite
    case denyOnce
    case denySite
}

/// What the confirmation bubble shows. The origin lines are rendered by the
/// app from `canonicalString`, never from page text.
public struct SitePermissionPrompt: Identifiable, Equatable, Sendable {
    public let id: UUID
    public let tabID: BrowserTabID
    public let requestingOrigin: BrowserOrigin
    public let topLevelOrigin: BrowserOrigin?
    public let classes: [SitePermissionClass]
    public let expiresAt: Date
    public let duration: TimeInterval

    /// A cross-origin frame asked: the bubble names both origins.
    public var showsPageOrigin: Bool {
        guard let topLevelOrigin else { return false }
        return topLevelOrigin != requestingOrigin
    }
}

/// Schedules the timeout and returns a cancel closure.
public typealias SitePermissionScheduler = @MainActor (
    TimeInterval, @escaping @MainActor () -> Void
) -> @MainActor () -> Void

/// The decision logic for page permission requests. Default deny:
///
/// 1. An agent-owned tab is denied before any record is read, so a stored
///    allow from a user takeover can never reach an agent tab.
/// 2. A class the app does not classify, or an origin it cannot name, is denied.
/// 3. A request from a tab that is not the active one is denied.
/// 4. A stored record answers without asking; any stored deny wins.
/// 5. Otherwise a bubble asks, one pending per tab, and the countdown denies.
///
/// Every request's `SingleShotPermissionDecision` is resolved exactly once.
@MainActor
public final class SitePermissionPolicy {
    public static let defaultScheduler: SitePermissionScheduler = { delay, fire in
        let task = Task { @MainActor in
            do { try await Task.sleep(nanoseconds: UInt64(max(0, delay) * 1_000_000_000)) } catch { return }
            fire()
        }
        return { task.cancel() }
    }

    private struct Pending {
        var prompt: SitePermissionPrompt
        var waiters: [SingleShotPermissionDecision]
        let cancelTimer: @MainActor () -> Void
    }

    public let store: SitePermissionStore
    /// Fires after `pendingPrompts` changes, so the engine can publish it.
    public var onPendingPromptsChanged: (([SitePermissionPrompt]) -> Void)?

    private let ownerProvider: @MainActor (BrowserTabID) -> BrowserTabOwner?
    private let activeTabProvider: @MainActor () -> BrowserTabID?
    private let scheduler: SitePermissionScheduler
    private let now: () -> Date
    private let timeout: TimeInterval
    private var pending: [UUID: Pending] = [:]
    private var order: [UUID] = []

    public var pendingPrompts: [SitePermissionPrompt] {
        order.compactMap { pending[$0]?.prompt }
    }

    public init(
        store: SitePermissionStore,
        ownerProvider: @escaping @MainActor (BrowserTabID) -> BrowserTabOwner?,
        activeTabProvider: @escaping @MainActor () -> BrowserTabID?,
        scheduler: SitePermissionScheduler? = nil,
        now: @escaping () -> Date = Date.init,
        timeout: TimeInterval = sitePermissionPromptTimeout
    ) {
        self.store = store
        self.ownerProvider = ownerProvider
        self.activeTabProvider = activeTabProvider
        self.scheduler = scheduler ?? Self.defaultScheduler
        self.now = now
        self.timeout = timeout
    }

    // MARK: - Requests

    public func decide(_ request: SitePermissionRequest, decision: SingleShotPermissionDecision) {
        // 1. Ownership first, no lookup. A tab the policy cannot find is a
        // closed tab.
        switch ownerProvider(request.tabID) {
        case .agent:
            autoDeny(request, decision, .agentTab)
            return
        case nil:
            autoDeny(request, decision, .tabClosed)
            return
        case .user:
            break
        }
        // 2. Closed class list, nameable origin.
        guard !request.classes.isEmpty else {
            autoDeny(request, decision, .unclassified)
            return
        }
        guard let origin = request.requestingOrigin else {
            autoDeny(request, decision, .unknownOrigin)
            return
        }
        // 3. A background tab cannot draw a bubble over the page being viewed.
        guard activeTabProvider() == request.tabID else {
            autoDeny(request, decision, .background)
            return
        }

        // 4. Records. Any stored deny denies the whole request.
        var missing: [SitePermissionClass] = []
        var denied: [SitePermissionClass] = []
        for permissionClass in request.classes {
            switch store.verdict(for: origin, permissionClass: permissionClass) {
            case .deny: denied.append(permissionClass)
            case .allow: break
            case nil: missing.append(permissionClass)
            }
        }
        if !denied.isEmpty {
            for permissionClass in request.classes {
                if denied.contains(permissionClass) {
                    store.recordAudit(
                        permissionClass: permissionClass, decision: .denyRecord, origin: origin)
                } else {
                    store.recordAudit(
                        permissionClass: permissionClass, decision: .autoDeny,
                        origin: origin, reason: .otherClassDenied)
                }
            }
            decision.resolve(false)
            return
        }
        if missing.isEmpty {
            for permissionClass in request.classes {
                store.recordAudit(
                    permissionClass: permissionClass, decision: .allowRecord, origin: origin)
            }
            decision.resolve(true)
            return
        }

        // 5. Ask. One pending bubble per tab. An identical repeat coalesces
        // onto it; anything else is a flood and is denied.
        if let existingID = order.first(where: { pending[$0]?.prompt.tabID == request.tabID }),
           var existing = pending[existingID] {
            if existing.prompt.requestingOrigin == origin, existing.prompt.classes == missing {
                existing.waiters.append(decision)
                pending[existingID] = existing
            } else {
                autoDeny(request, decision, .tooManyRequests)
            }
            return
        }

        let id = UUID()
        let prompt = SitePermissionPrompt(
            id: id,
            tabID: request.tabID,
            requestingOrigin: origin,
            topLevelOrigin: request.topLevelOrigin,
            classes: missing,
            expiresAt: now().addingTimeInterval(timeout),
            duration: timeout
        )
        let cancel = scheduler(timeout) { [weak self] in
            self?.finish(id, outcome: .auto(.timeout))
        }
        pending[id] = Pending(prompt: prompt, waiters: [decision], cancelTimer: cancel)
        order.append(id)
        publish()
    }

    // MARK: - Answers

    /// Applies the user's answer from the bubble. False when the prompt is
    /// gone, which means it was already resolved (timeout, navigation, ...).
    @discardableResult
    public func answer(promptID: UUID, _ answer: SitePermissionAnswer) -> Bool {
        guard let entry = pending[promptID] else { return false }
        // Re-read ownership and visibility at answer time: an answer must not
        // grant a tab that is no longer a user-owned, active tab.
        let tabID = entry.prompt.tabID
        if ownerProvider(tabID) != .user {
            return finish(promptID, outcome: .auto(.ownershipChanged))
        }
        if activeTabProvider() != tabID {
            return finish(promptID, outcome: .auto(.background))
        }
        return finish(promptID, outcome: .answered(answer))
    }

    /// Denies every pending prompt of a tab (navigation, close, crash,
    /// takeover, tab switch).
    public func cancelPrompts(for tabID: BrowserTabID, reason: SitePermissionAutoDenyReason) {
        for id in order where pending[id]?.prompt.tabID == tabID {
            finish(id, outcome: .auto(reason))
        }
    }

    /// Denies the prompts of every tab except `activeTabID`.
    public func cancelPromptsOutside(activeTabID: BrowserTabID?) {
        for id in order {
            if let prompt = pending[id]?.prompt, prompt.tabID != activeTabID {
                finish(id, outcome: .auto(.background))
            }
        }
    }

    // MARK: - Internals

    private enum Outcome {
        case answered(SitePermissionAnswer)
        case auto(SitePermissionAutoDenyReason)
    }

    @discardableResult
    private func finish(_ id: UUID, outcome: Outcome) -> Bool {
        guard let entry = pending.removeValue(forKey: id) else { return false }
        order.removeAll { $0 == id }
        entry.cancelTimer()
        let prompt = entry.prompt
        let allow: Bool
        switch outcome {
        case .auto(let reason):
            for permissionClass in prompt.classes {
                store.recordAudit(
                    permissionClass: permissionClass, decision: .autoDeny,
                    origin: prompt.requestingOrigin, reason: reason)
            }
            allow = false
        case .answered(let answer):
            allow = apply(answer, to: prompt)
        }
        for waiter in entry.waiters { waiter.resolve(allow) }
        publish()
        return true
    }

    /// Writes records and audit rows for a user answer. A full store turns a
    /// "for site" answer into a once-only one rather than dropping the answer.
    private func apply(_ answer: SitePermissionAnswer, to prompt: SitePermissionPrompt) -> Bool {
        let origin = prompt.requestingOrigin
        for permissionClass in prompt.classes {
            let decision: SitePermissionAuditDecision
            switch answer {
            case .allowOnce:
                decision = .allowOnce
            case .denyOnce:
                decision = .denyOnce
            case .allowSite:
                let saved = store.setVerdict(.allow, origin: origin, permissionClass: permissionClass)
                decision = saved ? .allowSite : .allowOnce
            case .denySite:
                let saved = store.setVerdict(.deny, origin: origin, permissionClass: permissionClass)
                decision = saved ? .denySite : .denyOnce
            }
            store.recordAudit(permissionClass: permissionClass, decision: decision, origin: origin)
        }
        switch answer {
        case .allowOnce, .allowSite: return true
        case .denyOnce, .denySite: return false
        }
    }

    private func autoDeny(
        _ request: SitePermissionRequest,
        _ decision: SingleShotPermissionDecision,
        _ reason: SitePermissionAutoDenyReason
    ) {
        let classes: [SitePermissionClass?] = request.classes.isEmpty ? [nil] : request.classes
        for permissionClass in classes {
            store.recordAudit(
                permissionClass: permissionClass, decision: .autoDeny,
                origin: request.requestingOrigin, reason: reason)
        }
        decision.resolve(false)
    }

    private func publish() {
        onPendingPromptsChanged?(pendingPrompts)
    }
}
