import Foundation
import SwiftUI

/// Supported application localization languages and text direction handling.
public enum AppLanguage: String, CaseIterable, Identifiable, Sendable {
    case system = "system"
    case english = "en"
    case spanish = "es"
    case french = "fr"
    case german = "de"
    case italian = "it"
    case portuguese = "pt-BR"
    case russian = "ru"
    case japanese = "ja"
    case korean = "ko"
    case simplifiedChinese = "zh-Hans"
    case traditionalChinese = "zh-Hant"
    case arabic = "ar"
    case hindi = "hi"
    case dutch = "nl"
    case polish = "pl"
    case turkish = "tr"
    case ukrainian = "uk"
    case swedish = "sv"
    case vietnamese = "vi"
    case indonesian = "id"
    case hebrew = "he"

    /// UserDefaults key where language preference is persisted.
    public static let storageKey = "TurboSpark.language"

    public var id: String { rawValue }

    /// Native / localized display label for language pickers.
    public var label: String {
        switch self {
        case .system: return "System Default"
        case .english: return "English"
        case .spanish: return "Español"
        case .french: return "Français"
        case .german: return "Deutsch"
        case .italian: return "Italiano"
        case .portuguese: return "Português (Brasil)"
        case .russian: return "Русский"
        case .japanese: return "日本語"
        case .korean: return "한국어"
        case .simplifiedChinese: return "简体中文"
        case .traditionalChinese: return "繁體中文"
        case .arabic: return "العربية (Arabic)"
        case .hindi: return "हिन्दी (Hindi)"
        case .dutch: return "Nederlands"
        case .polish: return "Polski"
        case .turkish: return "Türkçe"
        case .ukrainian: return "Українська"
        case .swedish: return "Svenska"
        case .vietnamese: return "Tiếng Việt"
        case .indonesian: return "Bahasa Indonesia"
        case .hebrew: return "עברית (Hebrew)"
        }
    }

    /// System Locale instance for the selected language.
    public var locale: Locale {
        switch self {
        case .system:
            return Self.systemLocale(
                resolved: Self.systemResolved, launchLocalization: Self.launchLocalization, current: .current)
        default:
            return Locale(identifier: rawValue)
        }
    }

    /// Whether the language uses right-to-left layout.
    public var isRTL: Bool {
        switch self {
        case .arabic, .hebrew:
            return true
        case .system:
            // Direction follows the catalog language System resolves to, not
            // the system locale: a system set to Persian with no Persian
            // catalog shows English and must stay LTR.
            return Self.systemResolved.isRTL
        default:
            return false
        }
    }

    /// The defaults key AppKit reads once, at launch, to choose the app's own
    /// localization: the menu bar, system panels and `Locale.current`.
    /// `\.locale` reaches window content only; `Commands` and AppKit never see it.
    static let processLanguagesKey = "AppleLanguages"

    /// Mirrors the picker into `processLanguagesKey` in this app's domain: a
    /// picked language alone, or no key for System so the global list applies.
    /// Takes effect at the next launch, and only for languages the main bundle
    /// declares (`Localization/Info.plist`).
    static func writeProcessLanguages(for language: AppLanguage, to defaults: UserDefaults = .standard) {
        if language == .system {
            defaults.removeObject(forKey: processLanguagesKey)
        } else {
            defaults.set([language.rawValue], forKey: processLanguagesKey)
        }
    }

    /// The catalog localization this process resolved at launch, which the menu
    /// bar and system panels keep for the whole run.
    static var launchLocalization: String {
        Bundle.module.preferredLocalizations.first ?? "en"
    }

    /// The offered language System stands for right now. Only meaningful while
    /// System is picked: a picked language sits in `processLanguagesKey`, so it
    /// would resolve to itself.
    ///
    /// Read from `Locale.preferredLanguages` because it follows a write or a
    /// removal of `processLanguagesKey` immediately, while `Locale.current` and
    /// `Bundle.preferredLocalizations` keep the launch language.
    static var systemResolved: AppLanguage {
        resolveSystem(preferences: Locale.preferredLanguages)
    }

    /// `current` keeps the user's own formats (24-hour time, custom date
    /// styles), but it is fixed at launch: after a switch from a picked
    /// language back to System it still speaks the picked one until relaunch.
    static func systemLocale(resolved: AppLanguage, launchLocalization: String, current: Locale) -> Locale {
        resolved == offered(launchLocalization) ? current : resolved.locale
    }

    /// `preferences` matched against the catalog, English when nothing matches.
    static func resolveSystem(
        preferences: [String], localizations: [String] = Bundle.module.localizations
    ) -> AppLanguage {
        Bundle.preferredLocalizations(from: localizations, forPreferences: preferences)
            .first.flatMap(offered) ?? .english
    }

    /// The offered language a localization code names. Compiled `.lproj`
    /// folders come out lowercased (`pt-br`), so the match ignores case.
    static func offered(_ code: String) -> AppLanguage? {
        allCases.first { $0 != .system && $0.rawValue.caseInsensitiveCompare(code) == .orderedSame }
    }

    /// Whether picking `language` leaves the menu bar and system panels in a
    /// different language from the one they show after a relaunch.
    static func needsRelaunch(for language: AppLanguage, launchLocalization: String = launchLocalization) -> Bool {
        let target = language == .system ? systemResolved : language
        return offered(launchLocalization) != target
    }

    /// Text and component layout direction.
    public var layoutDirection: LayoutDirection {
        isRTL ? .rightToLeft : .leftToRight
    }

    /// Resolves stored raw string value into an `AppLanguage` instance.
    public static func resolve(_ storedValue: String) -> Self {
        Self(rawValue: storedValue) ?? .system
    }
}
