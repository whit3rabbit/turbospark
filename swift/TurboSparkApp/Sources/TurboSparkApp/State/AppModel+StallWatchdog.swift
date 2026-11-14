import AppKit
import Foundation

/// What the recovery panel shows (OC-14).
struct GenerationStallNotice: Equatable {
    /// Whole seconds since the engine last produced a stream event.
    var silenceSeconds: Int
}

extension AppModel {
    func stallWatchdogBegin() {
        let watchdog = stallWatchdog ?? makeStallWatchdog()
        stallWatchdog = watchdog
        watchdog.begin()
    }

    func stallWatchdogNoteActivity() {
        stallWatchdog?.noteActivity()
    }

    func stallWatchdogEnd() {
        stallWatchdog?.end()
        generationStallNotice = nil
    }

    private func makeStallWatchdog() -> GenerationStallWatchdog {
        GenerationStallWatchdog(
            onStall: { [weak self] seconds in
                guard let self, self.generating else { return }
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
        stallWatchdog?.keepWaiting()
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
