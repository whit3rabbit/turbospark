import Foundation
import SwiftUI

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

    public static let storageKey = "TurboSpark.language"

    public var id: String { rawValue }

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
        }
    }



    public var locale: Locale {
        switch self {
        case .system:
            return Locale.current
        default:
            return Locale(identifier: rawValue)
        }
    }

    public var isRTL: Bool {
        switch self {
        case .arabic:
            return true
        case .system:
            if #available(macOS 13.0, *) {
                let code = Locale.current.language.languageCode?.identifier ?? "en"
                return Locale.Language(identifier: code).characterDirection == .rightToLeft
            } else {
                return false
            }
        default:
            return false
        }
    }


    public var layoutDirection: LayoutDirection {
        isRTL ? .rightToLeft : .leftToRight
    }

    public static func resolve(_ storedValue: String) -> Self {
        Self(rawValue: storedValue) ?? .system
    }
}
