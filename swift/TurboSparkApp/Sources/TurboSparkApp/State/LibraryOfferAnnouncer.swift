import Combine
import Foundation

/// The marker ids already announced by a toast.
///
/// **ONE ANNOUNCEMENT PER OFFER ID PER PROCESS.** A vault lock discards the app
/// model and a later unlock builds another, with a fresh `LibraryState`. The ids
/// live here, outside both, so the same offer turning up in the new model does
/// not toast a second time (the banner still shows it). A test passes its own
/// ledger so the process-wide one never leaks between cases.
@MainActor
final class LibraryOfferAnnouncementLedger {
    /// The one the app uses.
    static let process = LibraryOfferAnnouncementLedger()

    private var announced: Set<String> = []

    init() {}

    /// Records `markerId`. True the first time, false for every repeat.
    func record(_ markerId: String) -> Bool {
        announced.insert(markerId).inserted
    }
}

/// Announces an add-library offer, once, when it becomes the offer on screen
/// (requirement 4.3: "announced once by an info toast").
///
/// **IT FIRES ON THE PENDING OFFER, NOT ON A PUBLISH.** `LibraryState.offers`
/// publishes on every change to the queue, including the ones that change
/// nothing (a drive replugged while its offer waits, a refresh that prunes
/// nothing), so each publish is checked against the ledger and only an id never
/// announced speaks. An offer that waits behind another is announced when it
/// becomes the pending one, which is when the banner first shows it. No offer
/// pending, no toast.
///
/// It takes a closure instead of the app model: `LibraryState` has no toast or
/// error reference by design, and this follows it. `AppModel` supplies the
/// closure that posts the info toast and never writes the app error.
@MainActor
final class LibraryOfferAnnouncer {
    private let libraryState: LibraryState
    private let ledger: LibraryOfferAnnouncementLedger
    private let language: @MainActor () -> AppLanguage
    private let announce: @MainActor (String) -> Void
    private var subscription: AnyCancellable?

    init(
        libraryState: LibraryState,
        ledger: LibraryOfferAnnouncementLedger = .process,
        language: @escaping @MainActor () -> AppLanguage = { AppStrings.currentLanguage },
        announce: @escaping @MainActor (String) -> Void
    ) {
        self.libraryState = libraryState
        self.ledger = ledger
        self.language = language
        self.announce = announce
    }

    var isRunning: Bool { subscription != nil }

    /// Starts watching the queue. The offer already pending, if any, is checked
    /// at once. Calling it again does nothing.
    func start() {
        guard subscription == nil else { return }
        // `$offers` emits the NEW value before it is stored, so the closure
        // reads what it is given, never `libraryState.offers`.
        subscription = libraryState.$offers.sink { [weak self] queue in
            self?.consider(queue.pending)
        }
    }

    func stop() {
        subscription?.cancel()
        subscription = nil
    }

    private func consider(_ pending: LibraryOffer?) {
        guard let pending, ledger.record(pending.markerId) else { return }
        announce(LibraryOfferBannerModel.announcement(for: pending, language: language()))
    }
}
