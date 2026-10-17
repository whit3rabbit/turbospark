import Combine
import Foundation
import SwiftUI

/// What the two buttons of the add-library banner do (requirements 4.3 and 4.4).
///
/// **ADD KEEPS THE OFFER WHEN IT FAILS.** A refusal is the answer to a question
/// the user just asked, so it is one localized sentence inside the banner, with a
/// glyph, and the offer stays so the user can try again or say Not now. It is
/// never written to `model.error` and never a toast (the same rule as the Model
/// libraries section): the controller does not hold the app model at all, only a
/// closure that says whether an install or a move is running.
///
/// **THE ENGINE CALL IS OFF THE MAIN ACTOR.** `LibraryState.accept` runs the
/// blocking `add` in a detached task and refreshes; this only awaits it.
///
/// **BOTH BUTTONS ARE DISABLED, NOT HIDDEN,** while an install or a move runs
/// (the engine would answer `busy`) and while the add is in flight. A controller
/// is made per offer, so a sentence about one library is never shown on the next.
@MainActor
final class LibraryOfferController: ObservableObject {
    /// The refusal on screen, one localized sentence. Cleared by the next
    /// action. Nil while there is nothing to say.
    @Published private(set) var failure: String?
    /// The add is running, so a second one is not started.
    @Published private(set) var isAdding = false

    let libraryState: LibraryState
    private let isBusy: @MainActor () -> Bool
    private let announce: @MainActor (String) -> Void
    private let language: @MainActor () -> AppLanguage

    /// - Parameters:
    ///   - isBusy: whether an install or a move is running
    ///     (`AppModel.isInstallOrMoveRunning`).
    ///   - announce: tells VoiceOver how the add ended. The sentence is also
    ///     what the banner shows, so a user who is not looking at it still hears
    ///     it (DESIGN.md: failure and completion are announced).
    init(
        libraryState: LibraryState,
        isBusy: @escaping @MainActor () -> Bool = { false },
        announce: @escaping @MainActor (String) -> Void = { _ = AccessibilityNotification.Announcement.post(.init($0)) },
        language: @escaping @MainActor () -> AppLanguage = { AppStrings.currentLanguage }
    ) {
        self.libraryState = libraryState
        self.isBusy = isBusy
        self.announce = announce
        self.language = language
    }

    /// An install or a move is running, or the add is in flight: the buttons
    /// are disabled meanwhile.
    var isChangeBlocked: Bool {
        isAdding || isBusy()
    }

    // MARK: - Add

    /// Adds the offered library. On a refusal the offer stays and the sentence is
    /// shown; on success the offer leaves (the next queued one takes its place)
    /// and VoiceOver is told.
    func add(_ offer: LibraryOffer) async {
        failure = nil
        guard !isAdding else { return }
        // The other placement may have answered it already.
        guard libraryState.offers.contains(markerId: offer.markerId) else { return }
        if isBusy() {
            fail(LibraryActionText.busy(language: language()))
            return
        }
        isAdding = true
        defer { isAdding = false }
        do {
            try await libraryState.accept(offer)
            // The same name the banner showed, so a blank marker name still
            // reads as something.
            let name = LibraryOfferBannerModel.displayName(
                of: offer, limit: LibraryOfferBannerModel.nameLimit, language: language())
            announce(LibraryActionText.added(name: name, language: language()))
        } catch {
            fail(LibraryActionText.sentence(for: error, language: language()))
        }
    }

    private func fail(_ sentence: String) {
        failure = sentence
        announce(sentence)
    }

    // MARK: - Not now

    /// Remembers that the user does not want this library offered, by marker id,
    /// and lets the next queued offer take its place. The library can still be
    /// added from settings.
    func notNow(_ offer: LibraryOffer) {
        failure = nil
        guard !isChangeBlocked else { return }
        guard libraryState.offers.contains(markerId: offer.markerId) else { return }
        libraryState.dismiss(offer)
    }
}
