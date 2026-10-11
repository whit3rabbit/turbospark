import Foundation

/// Localized `String` values for the places a `Text` cannot go: toasts,
/// `NSAlert`/`NSOpenPanel` titles, status strings stored in a model, and a
/// chat draft seeded by a button.
///
/// `String(localized:bundle:)` resolves through the PROCESS's preferred
/// languages, while every `Text(_:bundle:)` follows the in-app language
/// picker through `\.locale`. With the picker set to French on an English
/// system, a status line built with `String(localized:)` sat in English
/// beside French labels. This resolves the catalog's compiled `.lproj` for
/// the picker's language (for `.system`, the language System resolves to),
/// and falls back to the module bundle for a missing table.
enum AppStrings {
    /// The language the UI is showing right now. Read from the same
    /// `AppStorage` key the scenes read, so it moves with the picker without
    /// a relaunch.
    static var currentLanguage: AppLanguage {
        AppLanguage.resolve(UserDefaults.standard.string(forKey: AppLanguage.storageKey) ?? "")
    }

    /// The bundle to resolve `language` from. `xcstringstool` writes
    /// lowercased `.lproj` names (`pt-br.lproj`), so both spellings are tried.
    /// System goes through `AppLanguage.systemResolved`, the language the
    /// window shows, because `.module` keeps the launch language after a
    /// mid-session switch back to System.
    static func bundle(for language: AppLanguage) -> Bundle {
        guard language != .system else { return bundle(for: AppLanguage.systemResolved) }
        for name in [language.rawValue.lowercased(), language.rawValue] {
            if let path = Bundle.module.path(forResource: name, ofType: "lproj"),
               let bundle = Bundle(path: path) {
                return bundle
            }
        }
        return .module
    }

    /// `key` from the catalog in the UI's current language. Interpolations
    /// become format specifiers exactly as in `Text`: a `String` is `%@`, an
    /// `Int` is `%lld`, so the catalog key must spell them that way.
    static func localized(_ key: String.LocalizationValue) -> String {
        localized(key, language: currentLanguage)
    }

    static func localized(_ key: String.LocalizationValue, language: AppLanguage) -> String {
        String(localized: key, bundle: bundle(for: language), locale: language.locale)
    }
}
