import Foundation
import TurboSpark

/// The symbols the offer banner draws. A state is a glyph AND a sentence, never
/// a color alone (DESIGN.md section 4).
enum LibraryOfferGlyph {
    static let drive = "externaldrive.badge.plus"
    static let readOnly = "lock"
    static let failure = LibraryRowGlyph.warning
}

/// Everything the add-library banner says, as a value (requirement 4.3).
///
/// **IT IS PURE.** Every input is passed in, so a table test pins each wording
/// and each language without a view. The banner draws these strings and nothing
/// else, and VoiceOver reads `accessibilityLabel`, built from the same strings
/// in the order the banner reads.
///
/// **THE NAME COMES FROM A DRIVE.** A library's name is whatever its marker says,
/// so it goes through `UntrustedText.display` (control and direction characters
/// become visible escapes, the length is cut) before it is put in a sentence. A
/// blank name falls back to the volume's folder name, then to a plain word, so
/// the title never reads "Add the library ?".
struct LibraryOfferBannerModel: Equatable, Sendable {
    /// The longest library name a sentence carries, after escaping. The same
    /// limit the rest of the library sentences use.
    static let nameLimit = LibraryActionText.nameLimit
    /// The toast shows two lines, so its name is cut shorter: a long name must
    /// not push "Add it from ..." off the end.
    static let toastNameLimit = 40

    let markerId: String
    /// "Add the library T7?" with the name escaped and cut.
    let title: String
    /// "4 models". The same plural key the library rows use.
    let countText: String
    /// Why the library cannot be changed, when a newer TurboSpark wrote it.
    let readOnlyText: String?
    /// "2 more libraries found", when other offers wait behind this one.
    let queuedText: String?
    /// The banner's text as one VoiceOver element.
    let accessibilityLabel: String
    let language: AppLanguage

    /// - Parameter queuedCount: how many offers wait behind `offer`.
    init(offer: LibraryOffer, queuedCount: Int, language: AppLanguage = AppStrings.currentLanguage) {
        let name = Self.displayName(of: offer, limit: Self.nameLimit, language: language)
        let title = AppStrings.localized("Add the library \(name)?", language: language)
        let modelCount = max(0, offer.modelCount)
        let countText = AppStrings.localized("\(modelCount) model(s)", language: language)
        let readOnlyText = offer.readOnly
            ? AppStrings.localized(
                "A newer version of TurboSpark made this library. You can use its models, but you cannot change it.",
                language: language)
            : nil
        let waitingCount = max(0, queuedCount)
        let queuedText = waitingCount > 0
            ? AppStrings.localized("\(waitingCount) more libraries found", language: language) : nil

        self.markerId = offer.markerId
        self.title = title
        self.countText = countText
        self.readOnlyText = readOnlyText
        self.queuedText = queuedText
        self.accessibilityLabel = Self.spoken([title, countText, readOnlyText, queuedText].compactMap { $0 })
        self.language = language
    }

    /// The model for the offer on screen, nil when none is pending.
    @MainActor
    static func make(
        from state: LibraryState, language: AppLanguage = AppStrings.currentLanguage
    ) -> LibraryOfferBannerModel? {
        guard let offer = state.pendingOffer else { return nil }
        return LibraryOfferBannerModel(
            offer: offer, queuedCount: state.queuedOffers.count, language: language)
    }

    // MARK: - The name

    /// The library's name, escaped and cut. A name with nothing visible in it
    /// falls back to the folder the drive is mounted at, then to "Unknown
    /// library", so a sentence never has a hole in it.
    static func displayName(of offer: LibraryOffer, limit: Int, language: AppLanguage) -> String {
        if !offer.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return UntrustedText.display(offer.name, limit: limit)
        }
        let volume = URL(fileURLWithPath: offer.volumeRoot).lastPathComponent
        if !volume.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, volume != "/" {
            return UntrustedText.display(volume, limit: limit)
        }
        return AppStrings.localized("Unknown library", language: language)
    }

    // MARK: - Disabled buttons

    /// Why the buttons are disabled (DESIGN.md: a disabled control says why),
    /// nil when they are not. The add in flight is what the user just did, so it
    /// wins over a busy install.
    static func disabledReason(adding: Bool, busy: Bool, language: AppLanguage) -> String? {
        if adding { return AppStrings.localized("Adding the library\u{2026}", language: language) }
        if busy {
            return AppStrings.localized("Not available while a model is installing or moving.", language: language)
        }
        return nil
    }

    /// The tooltip and VoiceOver hint of Not now. It is honest about the label:
    /// the offer does not come back, and the library can still be added.
    static func notNowHelp(language: AppLanguage) -> String {
        AppStrings.localized(
            "TurboSpark will not offer this library again. You can add it later in Models & Storage settings.",
            language: language)
    }

    // MARK: - The toast

    /// The sentence of the ONE toast that announces an offer. The toast shows two
    /// lines at most, so the name is cut shorter than in the banner.
    static func announcement(for offer: LibraryOffer, language: AppLanguage) -> String {
        let name = displayName(of: offer, limit: toastNameLimit, language: language)
        return AppStrings.localized(
            "TurboSpark found the library \(name). Add it from Installed Models or Models & Storage.",
            language: language)
    }

    // MARK: - Speech

    /// Sentence ends that need no extra full stop before the next part.
    private static let terminators: Set<Character> = [
        ".", "?", "!", "\u{3002}", "\u{FF1F}", "\u{FF01}", "\u{061F}", "\u{0964}",
    ]

    /// Joins the parts of the spoken label: a part that already ends a sentence
    /// is followed by a space, any other by a full stop and a space, so the
    /// label never reads "Spare?. 4 models".
    static func spoken(_ parts: [String]) -> String {
        var out = ""
        for part in parts {
            if let last = out.last {
                out += terminators.contains(last) ? " " : ". "
            }
            out += part
        }
        return out
    }
}
