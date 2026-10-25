import Foundation
import TurboSpark

/// The one language a speech model handles, for the picker row and the
/// default model ranking. `auto` and `multilingual` are catalog markers, not
/// languages, so a row carrying them is never single-language.
enum AudioModelLanguage {
    static let markers: Set<String> = ["auto", "multilingual"]

    /// The row's only language code (`ja`, `zh`, `en-US`), nil for a
    /// multilingual row or one that declares no language.
    static func singleCode(for profile: AudioProfile) -> String? {
        let codes = profile.capabilities.languages.filter { !markers.contains($0.lowercased()) }
        return codes.count == 1 ? codes[0] : nil
    }

    /// The language subtag of a code, so `zh-Hans` and `zh` compare equal
    /// and `en-US` matches `en`.
    static func languageSubtag(_ code: String) -> String {
        Locale(identifier: code).language.languageCode?.identifier.lowercased() ?? code.lowercased()
    }

    /// Whether the row is the single-language final-pass model for `code`:
    /// the one a user whose UI is in that language is steered to. A row that
    /// only streams live text never becomes the default this way.
    static func isDedicated(_ profile: AudioProfile, to code: String?) -> Bool {
        guard let code, let own = singleCode(for: profile),
              profile.capabilities.operations.contains("transcribe") else { return false }
        return languageSubtag(own) == languageSubtag(code)
    }

    /// The language's name as `locale` spells it ("Japanese" under an English
    /// UI, "Japanisch" under a German one), falling back to the code.
    static func name(for code: String, locale: Locale) -> String {
        locale.localizedString(forLanguageCode: code) ?? locale.localizedString(forIdentifier: code) ?? code
    }

    /// The language subtag of the language the UI shows now. System follows
    /// the language System resolves to, as `AppStrings` does.
    static var uiLanguageCode: String {
        let language = AppStrings.currentLanguage
        let shown = language == .system ? AppLanguage.systemResolved : language
        return languageSubtag(shown.rawValue)
    }
}
