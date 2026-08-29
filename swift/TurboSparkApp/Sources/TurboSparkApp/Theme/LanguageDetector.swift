import Carbon
import Foundation
import NaturalLanguage

public enum LanguageDetector {
    /// Detects the language code associated with the user's currently active macOS keyboard input source.
    /// Example return values: "es", "ar", "ru", "zh-Hans", "ja", "en", "de", "fr".
    public static func currentKeyboardLanguage() -> String? {
        guard let inputSource = TISCopyCurrentKeyboardInputSource()?.takeRetainedValue() else {
            return nil
        }
        guard let languagesPtr = TISGetInputSourceProperty(inputSource, kTISPropertyInputSourceLanguages) else {
            return nil
        }
        let languages = Unmanaged<CFArray>.fromOpaque(languagesPtr).takeUnretainedValue() as? [String]
        return languages?.first
    }

    /// Detects the human-readable name of the active keyboard input source.
    /// Example: "U.S.", "Spanish - ISO", "Russian", "Arabic", "Pinyin - Simplified".
    public static func currentKeyboardLayoutName() -> String? {
        guard let inputSource = TISCopyCurrentKeyboardInputSource()?.takeRetainedValue() else {
            return nil
        }
        guard let namePtr = TISGetInputSourceProperty(inputSource, kTISPropertyLocalizedName) else {
            return nil
        }
        return Unmanaged<CFString>.fromOpaque(namePtr).takeUnretainedValue() as String
    }

    /// Detects the dominant natural language of a user prompt or text snippet using Apple's on-device NLLanguageRecognizer.
    public static func detectTextLanguage(_ text: String) -> NLLanguage? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(trimmed)
        return recognizer.dominantLanguage
    }

    /// Checks if a language code or locale is Right-To-Left.
    public static func isRTL(languageCode: String) -> Bool {
        if #available(macOS 13.0, *) {
            return Locale.Language(identifier: languageCode).characterDirection == .rightToLeft
        } else {
            let prefix = languageCode.lowercased()
            return prefix.hasPrefix("ar") || prefix.hasPrefix("he") || prefix.hasPrefix("fa") || prefix.hasPrefix("ur")
        }
    }
}
