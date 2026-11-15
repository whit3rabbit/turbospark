import Combine
import Foundation

/// OC-08: the explicit "I am talking to the assistant" signal.
///
/// Auto-send considers an utterance only when the user said it while Talk was
/// on. A text gate cannot tell a question for the assistant from the same
/// words said to someone in the room (the v3 holdout: precision about 0.72);
/// this switch is the input-side fact that can.
///
/// Two forms, one state:
///
/// - **Pointer hold.** `pointerDown` turns Talk on and `pointerUp` turns it
///   off if the press lasted at least `latchThreshold`. A shorter press is a
///   tap and latches Talk on (the toggle form) until the next tap or toggle.
/// - **Toggle.** `toggle()`: press once to start, press again to stop. The
///   keyboard chord and VoiceOver use this form. There is no key-up form on
///   purpose: a menu chord is key-down only, and a Command chord's key-up is
///   not reliably delivered to the app, so a hold would stick on or never
///   start.
///
/// `clear()` is the forced stop (mute, end, failure, approval, a turn
/// starting, the app losing focus). It also revokes any addressing already
/// captured for an utterance or a decision in flight, so those can only make
/// auto-send send less. A user release (`toggle`, `pointerUp`) does not
/// revoke: speech said while Talk was on stays addressed after Talk is let go.
@MainActor
final class VoiceCallTalkSwitch: ObservableObject {
    /// A press shorter than this is a tap (latch); longer is a hold.
    static let latchThreshold: TimeInterval = 0.35

    /// A LATCHED Talk (tap or toggle) clears itself after this long with no
    /// speech onset, so a forgotten Talk cannot silently restore the
    /// unaddressed-speech precision problem. A held pointer press is exempt.
    static let idleTimeout: TimeInterval = 45

    @Published private(set) var isTalking = false {
        didSet { if !isTalking { idleWatch?.cancel(); idleWatch = nil } }
    }

    /// Set by the controller. A switch nobody owns cannot be turned on, so
    /// Talk never lingers outside a live call with auto-send on.
    var canTalk: @MainActor () -> Bool = { false }
    /// Talk just turned on.
    var onStarted: (@MainActor () -> Void)?
    /// A forced clear happened (see the type comment).
    var onCleared: (@MainActor () -> Void)?

    private let now: () -> TimeInterval
    private let tick: @Sendable (Duration) async throws -> Void
    private var idleWatch: Task<Void, Never>?
    private var lastActivity: TimeInterval = 0
    private var pointerDownAt: TimeInterval?
    private var swallowNextPointerUp = false

    init(now: @escaping () -> TimeInterval = { ProcessInfo.processInfo.systemUptime },
         tick: @escaping @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) }) {
        self.now = now
        self.tick = tick
    }

    /// Talk is on and no pointer press is holding it.
    private var latched: Bool { isTalking && pointerDownAt == nil }

    /// A speech onset restarts the idle window of a latched Talk.
    func speechOnset() { if isTalking { lastActivity = now() } }

    /// Clears a latched Talk that has been idle for `idleTimeout`. The
    /// watcher calls it every second; tests call it on a fake clock.
    func checkIdle() {
        guard latched, now() - lastActivity >= Self.idleTimeout else { return }
        // A plain stop, like a user release: nothing was being said.
        isTalking = false
    }

    // MARK: Toggle form

    func toggle() {
        if isTalking { isTalking = false } else { _ = start() }
    }

    // MARK: Pointer form

    func pointerDown() {
        if isTalking {
            // Latched on: this press turns it off, and its release is not a hold.
            isTalking = false
            swallowNextPointerUp = true
            return
        }
        guard start() else { return }
        pointerDownAt = now()
    }

    func pointerUp() {
        if swallowNextPointerUp { swallowNextPointerUp = false; return }
        guard let down = pointerDownAt else { return }
        pointerDownAt = nil
        guard isTalking else { return }
        // A hold ends with the release. A tap stays on until toggled off.
        if now() - down >= Self.latchThreshold { isTalking = false } else { lastActivity = now() }
    }

    // MARK: Forced stop

    func clear() {
        pointerDownAt = nil
        swallowNextPointerUp = false
        if isTalking { isTalking = false }
        onCleared?()
    }

    private func start() -> Bool {
        guard !isTalking, canTalk() else { return false }
        isTalking = true
        lastActivity = now()
        let tick = self.tick
        idleWatch = Task { [weak self] in
            while !Task.isCancelled {
                try? await tick(.seconds(1))
                if Task.isCancelled { return }
                await self?.checkIdle()
            }
        }
        onStarted?()
        return true
    }
}
