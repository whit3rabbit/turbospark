import AppKit
import Foundation
import UserNotifications

/// Where macOS authorization for banners stands, as the Settings row shows it.
enum AttentionAuthorization: Equatable, Sendable {
    /// Not running from a signed .app bundle: `UNUserNotificationCenter`
    /// traps there, so nothing is read or posted.
    case unavailable
    /// The user has not been asked. Only the Allow button asks.
    case notDetermined
    case denied
    case authorized
}

/// One notification to post. The chat id travels in `userInfo` so a click can
/// select the chat; the identifier is the chat id too, so macOS itself
/// replaces an older banner for the same chat.
struct AttentionNotification: Equatable, Sendable {
    let chatID: UUID
    let title: String
    let body: String
}

/// The slice of `UNUserNotificationCenter` the app uses, so tests drive the
/// wrapper without posting anything.
@MainActor
protocol AttentionNotificationBackend: AnyObject {
    func authorizationStatus() async -> AttentionAuthorization
    /// Shows the system prompt. Reached only from `AppNotificationCenter`'s
    /// Allow path.
    func requestAuthorization() async -> Bool
    func post(_ notification: AttentionNotification) async throws
    func removeDelivered(chatIDs: [UUID])
}

/// What an agent push came to. Every case but `.posted` means nothing was
/// shown, and none of them is an error for the tool call.
enum AgentPushResult: Equatable, Sendable {
    case posted
    /// The "Allow agents to send notifications" switch is off.
    case optedOut
    /// macOS has not been asked or said no, or this is not the app bundle.
    case notAuthorized(AttentionAuthorization)
    /// A pending approval or question on the chat outranks it.
    case outranked
    /// The post itself failed.
    case failed
    /// A Ghost chat never reaches the system notification store.
    case ghostChat
}

/// OC-02: the one owner of the attention notifications: authorization state,
/// posting, sounds, and click routing. The decision of whether an event
/// deserves a banner is `AttentionEventPolicy`'s.
///
/// Authorization is user-initiated: `requestAuthorization()` is called only
/// from the Settings Allow button, never at launch and never from an event.
/// While it is not granted, events are dropped silently and everything else
/// in the app works unchanged.
///
/// The agent's `PushNotification` tool goes through `pushFromAgent`, the
/// Allow-only path: it reads the status without prompting, never calls
/// `requestAuthorization()`, and does nothing unless the user switched
/// "Allow agents to send notifications" on.
@MainActor
final class AppNotificationCenter: ObservableObject {
    @Published private(set) var authorization: AttentionAuthorization = .notDetermined

    private(set) var policy: AttentionEventPolicy
    private let backend: AttentionNotificationBackend
    private let playSound: (String) -> Void
    /// Selects the chat and brings the app forward. Set by `AppModel`.
    var openChat: (UUID) -> Void = { _ in }

    init(
        backend: AttentionNotificationBackend? = nil,
        isActive: @escaping () -> Bool = { NSApp?.isActive ?? true },
        playSound: @escaping (String) -> Void = { name in NSSound(named: name)?.play() }
    ) {
        let resolved = backend ?? SystemAttentionBackend()
        self.backend = resolved
        self.policy = AttentionEventPolicy(isActive: isActive)
        self.playSound = playSound
        (resolved as? SystemAttentionBackend)?.onClick = { [weak self] id in
            self?.handleClick(chatID: id)
        }
    }

    /// Reads the current status without prompting. Called when the Settings
    /// section appears and when the app becomes active (the user may have
    /// changed it in System Settings).
    func refreshAuthorization() async {
        authorization = await backend.authorizationStatus()
    }

    /// The Allow button. The only place the system prompt can start.
    func requestAuthorization() async {
        guard authorization == .notDetermined else { return }
        _ = await backend.requestAuthorization()
        await refreshAuthorization()
    }

    /// Posts `notification` for `event` when the policy says so, then plays
    /// the event's sound. The sound waits for a successful post and never
    /// delays it.
    func notify(
        _ event: AttentionEvent, notification: AttentionNotification,
        preferences: AttentionPreferences
    ) async {
        // Nothing is claimed or posted until the user has said yes.
        guard authorization == .authorized else { return }
        let decision = policy.decide(event, chat: notification.chatID, preferences: preferences)
        guard decision.shouldPost else { return }
        do {
            try await backend.post(notification)
        } catch {
            policy.release(chat: notification.chatID)
            return
        }
        if let name = decision.sound?.systemName { playSound(name) }
    }

    /// The agent's `PushNotification` tool. Opt-in first (off means nothing
    /// is read or posted), then macOS authorization read WITHOUT prompting,
    /// then the per-chat priority. The focus rule does not apply: the agent
    /// asked on purpose. Never starts the system prompt.
    func pushFromAgent(
        _ notification: AttentionNotification, preferences: AttentionPreferences
    ) async -> AgentPushResult {
        guard preferences.allowAgentPush else { return .optedOut }
        await refreshAuthorization()
        guard authorization == .authorized else { return .notAuthorized(authorization) }
        let decision = policy.decideAgentPush(chat: notification.chatID, preferences: preferences)
        guard decision.shouldPost else {
            return decision.outcome == .outranked ? .outranked : .optedOut
        }
        do {
            try await backend.post(notification)
        } catch {
            policy.release(chat: notification.chatID)
            return .failed
        }
        return .posted
    }

    /// The app is in front again: every outstanding cue is stale.
    func appDidBecomeActive() {
        let chats = policy.outstandingChats
        policy.releaseAll()
        backend.removeDelivered(chatIDs: chats)
        Task { await refreshAuthorization() }
    }

    /// A banner was clicked. The slot clears whether or not the chat still
    /// exists; `openChat` decides what to do with a deleted one.
    func handleClick(chatID: UUID) {
        policy.release(chat: chatID)
        openChat(chatID)
    }
}

// MARK: - The real center

/// `UNUserNotificationCenter` behind the protocol. Constructing it outside an
/// .app bundle is safe: every method answers `.unavailable` or does nothing.
@MainActor
final class SystemAttentionBackend: NSObject, AttentionNotificationBackend,
    UNUserNotificationCenterDelegate
{
    var onClick: (UUID) -> Void = { _ in }
    private let isBundled: Bool

    override init() {
        isBundled =
            Bundle.main.bundleURL.pathExtension == "app" && Bundle.main.bundleIdentifier != nil
        super.init()
        // Setting the delegate does not prompt; it is what lets a click on a
        // banner reach the app.
        if isBundled { UNUserNotificationCenter.current().delegate = self }
    }

    func authorizationStatus() async -> AttentionAuthorization {
        guard isBundled else { return .unavailable }
        let settings = await UNUserNotificationCenter.current().notificationSettings()
        switch settings.authorizationStatus {
        case .notDetermined: return .notDetermined
        case .denied: return .denied
        default: return .authorized
        }
    }

    func requestAuthorization() async -> Bool {
        guard isBundled else { return false }
        return (try? await UNUserNotificationCenter.current()
            .requestAuthorization(options: [.alert, .sound])) ?? false
    }

    func post(_ notification: AttentionNotification) async throws {
        guard isBundled else { return }
        let content = UNMutableNotificationContent()
        content.title = notification.title
        content.body = notification.body
        // The sound is played by the wrapper (NSSound), not by the banner.
        content.sound = nil
        content.userInfo = [Self.chatKey: notification.chatID.uuidString]
        let request = UNNotificationRequest(
            identifier: notification.chatID.uuidString, content: content, trigger: nil)
        try await UNUserNotificationCenter.current().add(request)
    }

    func removeDelivered(chatIDs: [UUID]) {
        guard isBundled, !chatIDs.isEmpty else { return }
        UNUserNotificationCenter.current()
            .removeDeliveredNotifications(withIdentifiers: chatIDs.map(\.uuidString))
    }

    static let chatKey = "turbospark.chatID"

    // The app only posts while inactive, so a banner arriving while active
    // is a race with activation: show nothing.
    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter, willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        completionHandler([])
    }

    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse,
        withCompletionHandler completionHandler: @escaping () -> Void
    ) {
        let raw = response.notification.request.content.userInfo[Self.chatKey] as? String
        completionHandler()
        guard let raw, let id = UUID(uuidString: raw) else { return }
        Task { @MainActor in self.onClick(id) }
    }
}
