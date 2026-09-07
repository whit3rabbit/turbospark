import Carbon
import Foundation

/// Reads the active macOS keyboard input source, for the General pane's
/// informational "Active Keyboard Layout" row.
///
/// Three siblings this type used to carry -- a keyboard-language-code
/// reader, an on-device text-language detector, and a free-function RTL
/// check duplicating `AppLanguage.isRTL` -- had no production caller at all
/// (`docs/SWIFT_SETTINGS_AUDIT.md`): each looks like it was scaffolded for a
/// feature (auto-selecting the app language from the keyboard, flipping
/// text direction per message) that was never wired to a call site, and
/// their only callers were their own tests. Deleted rather than kept as
/// dead weight; recover them from git history if that feature gets built.
public enum LanguageDetector {
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
}
