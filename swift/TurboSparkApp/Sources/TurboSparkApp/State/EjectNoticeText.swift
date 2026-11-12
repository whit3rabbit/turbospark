import Foundation
import TurboSpark

/// The words for what the eject coordinator tells the user (requirements 6.1
/// and 6.3), in the language the UI is showing.
///
/// Four catalog keys, each with plain `%@` specifiers (the parity gate
/// compares specifiers literally, so positional ones are not used) and the
/// model names first:
///  - "Unloaded %@ and ejected the drive."
///  - "Unloaded %@, but the drive could not be ejected: %@"
///  - "Unloaded %@ because its drive was disconnected. Eject a drive before
///    disconnecting it."
///  - "Could not unload %@ because it is still in use, so the drive was not
///    ejected."
///
/// Each names only the models its claim is true of: the first three, models
/// that are really gone; the last, models that are still held.
///
/// **NAMES AND THE REASON ARE UNTRUSTED.** A model name comes from a folder on a
/// drive and the reason from the system, so each goes through
/// `UntrustedText.display` (control and direction characters become escapes, and
/// each is cut), the names are joined in the language's own list style, and the
/// whole list is capped, so a drive with a hundred models cannot fill the toast.
enum EjectNoticeText {
    /// The longest joined name list a toast carries, counted after escaping.
    static let nameListLimit = 160
    /// The longest system reason a toast carries.
    static let reasonLimit = 200

    static func message(for notice: EjectNotice, language: AppLanguage? = nil) -> String {
        let language = language ?? AppStrings.currentLanguage
        switch notice {
        case .ejected(let models):
            let names = nameList(models, language: language)
            return AppStrings.localized("Unloaded \(names) and ejected the drive.", language: language)
        case .ejectFailed(let models, let reason):
            let names = nameList(models, language: language)
            let why = UntrustedText.display(reason, limit: reasonLimit)
            return AppStrings.localized(
                "Unloaded \(names), but the drive could not be ejected: \(why)", language: language)
        case .removedWithoutEject(let models):
            let names = nameList(models, language: language)
            return AppStrings.localized(
                "Unloaded \(names) because its drive was disconnected. Eject a drive before disconnecting it.",
                language: language)
        case .unloadBlocked(let models):
            let names = nameList(models, language: language)
            return AppStrings.localized(
                "Could not unload \(names) because it is still in use, so the drive was not ejected.",
                language: language)
        }
    }

    static func style(for notice: EjectNotice) -> AppToast.Style {
        switch notice {
        case .ejected: return .success
        case .ejectFailed, .unloadBlocked: return .error
        case .removedWithoutEject: return .warning
        }
    }

    /// "A", "A and B", "A, B and C" in the language's own style, each name made
    /// safe to show, the whole list capped.
    static func nameList(_ names: [String], language: AppLanguage) -> String {
        let safe = names.map { UntrustedText.display($0) }
        let formatter = ListFormatter()
        formatter.locale = language.locale
        let joined = formatter.string(from: safe) ?? safe.joined(separator: ", ")
        return UntrustedText.display(joined, limit: nameListLimit)
    }
}
