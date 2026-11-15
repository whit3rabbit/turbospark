import AppKit
import Foundation

/// What the recovery panel shows (OC-14).
struct GenerationStallNotice: Equatable {
    /// Whole seconds since the engine last produced a stream event.
    var silenceSeconds: Int
}

/// Which local generate call a monitor watches (OC-14). Each source owns its
/// own monitor instance, so one source's arm, end or stall never touches
/// another's state.
enum StallSource: Hashable, CaseIterable {
    /// The main chat turn (`session.generate` in `executeGenerationTurn`).
    case chat
    /// The summarizer call of `/compact`, auto-compaction and the todo
    /// boundary compaction, on the local session.
    case compaction
}

extension AppModel {
    func stallWatchdogBegin(_ source: StallSource = .chat) {
        let watchdog = stallWatchdogs[source] ?? makeStallWatchdog(source)
        stallWatchdogs[source] = watchdog
        watchdog.begin(pollInterval: stallPollInterval)
    }

    func stallWatchdogNoteActivity(_ source: StallSource = .chat) {
        stallWatchdogs[source]?.noteActivity()
    }

    func stallWatchdogEnd(_ source: StallSource = .chat) {
        // `end()` fires `onRecovered` itself when this source's notice is up.
        // The notice is NOT cleared unconditionally: it may belong to
        // another source.
        stallWatchdogs[source]?.end()
    }

    /// Whether the call a source watches is still the live one, so a notice
    /// is never raised for a call that has already ended.
    private func stallSourceIsActive(_ source: StallSource) -> Bool {
        switch source {
        case .chat: return generating
        case .compaction: return isCompacting
        }
    }

    private func makeStallWatchdog(_ source: StallSource) -> GenerationStallWatchdog {
        GenerationStallWatchdog(
            clock: stallClock,
            onStall: { [weak self] seconds in
                guard let self, self.stallSourceIsActive(source) else { return }
                let whole = Int(seconds.rounded())
                self.generationStallNotice = GenerationStallNotice(silenceSeconds: whole)
                // The panel is a persistent state, but it appears unprompted,
                // so say so once (DESIGN.md: failures are announced).
                _ = AccessibilityNotification.Announcement.post(
                    .init(AppStrings.localized("The model has stopped responding")))
            },
            onRecovered: { [weak self] in self?.generationStallNotice = nil })
    }

    /// Keep Waiting: hide the panel and watch for another window.
    func keepWaitingAfterStall() {
        generationStallNotice = nil
        for watchdog in stallWatchdogs.values where watchdog.isShowingStall {
            watchdog.keepWaiting()
        }
    }

    /// Stop Turn: the ordinary cooperative cancel. It ends the turn here and
    /// keeps the partial reply, but it cannot free a wedged engine: the
    /// engine's serial queue stays in its blocking call and the next send
    /// waits behind it. The panel says so and points at Restart.
    func stopStalledTurn() {
        generationStallNotice = nil
        cancel()
    }

    func makeGuardedRelaunch() -> GuardedRelaunch {
        if let guardedRelaunchOverride { return guardedRelaunchOverride }
        return GuardedRelaunch(
            attemptGuard: RelaunchGuard(directory: AppStorageRoot.directory),
            seams: .live(
                flush: { [weak self] in self?.flushForRelaunch() },
                terminate: { NSApp.terminate(nil) },
                endSession: { CrashRecoveryStore.shared.endSession() }))
    }

    /// Restart TurboSpark: persist first, then relaunch (see `GuardedRelaunch`).
    /// No confirmation: the panel button is the user's explicit choice, and
    /// the chats are saved before anything quits. Refusals surface as a toast
    /// and leave the panel up, so Keep Waiting and Stop remain.
    func restartAfterStall() {
        // Kept so repeated clicks reach the same in-progress latch.
        let relaunch = guardedRelaunchOverride ?? stallRelaunch ?? makeGuardedRelaunch()
        stallRelaunch = relaunch
        switch relaunch.perform() {
        case .relaunching, .alreadyInProgress:
            break
        case .refusedTooManyAttempts:
            showToast(
                AppStrings.localized(
                    "TurboSpark already restarted twice in the last few minutes. Quit and reopen it yourself."),
                style: .warning, duration: 12)
        case .unavailable:
            showToast(
                AppStrings.localized(
                    "Restart is not available here. Quit and reopen TurboSpark yourself."),
                style: .warning, duration: 12)
        }
    }
}
