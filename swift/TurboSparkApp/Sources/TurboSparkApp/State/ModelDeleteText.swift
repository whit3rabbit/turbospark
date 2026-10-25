import Foundation
import TurboSpark

/// The words for what happens when the user deletes a model or removes it from
/// the list (requirements 3.1, 3.7 and 5.3), in the language the UI is showing.
///
/// **ONE SENTENCE PER ENGINE CODE, KEYED ON THE PREFIX.** The engine answers a
/// refusal as `"<code>: <sentence>"` and a host localizes by the code, never by
/// the sentence (`LibraryFault`). A delete can answer `busy`, `not_connected`,
/// `unknown_install`, `io`, `library_read_only` and (never, now that every row
/// goes through its install id) `use_install_id`. `busy` is read from the
/// library fault and never from `TurboSparkError.Code.busy`, which also means
/// "the audio device is busy".
///
/// **THE SENTENCES ARE NEUTRAL ABOUT THE VERB.** A delete and a remove from the
/// list fail for the same reasons, and "was left as it is" is true of both.
///
/// **THE ONLY ENGINE TEXT SHOWN IS A NAME THE ROW ALREADY HAS.** The alias and
/// the library name come from a drive, so each goes through
/// `UntrustedText.display`; the engine's own sentence is never shown.
///
/// Every function takes the language as a parameter, so a test renders each
/// sentence through all 21 compiled bundles.
enum ModelDeleteText {
    /// The longest alias a sentence carries, after escaping.
    static let nameLimit = 100

    /// The sentence for a delete or a remove-from-list the engine refused or
    /// that failed. `libraryName` is the row's library (or the volume to
    /// connect), raw.
    static func failure(
        _ error: Error, alias: String, libraryName: String?, language: AppLanguage? = nil
    ) -> String {
        let language = language ?? AppStrings.currentLanguage
        let name = UntrustedText.display(alias, limit: nameLimit)
        guard let fault = (error as? TurboSparkError)?.libraryFault else {
            return generic(name: name, reason: reasonText(of: error), language: language)
        }
        switch fault.code {
        case .busy:
            return AppStrings.localized(
                "A model is installing or moving, so '\(name)' was left as it is. Try again when it finishes.",
                language: language)
        case .notConnected:
            return notConnected(alias: alias, libraryName: libraryName, language: language)
        case .unknownInstall:
            return AppStrings.localized("'\(name)' is no longer in the model list.", language: language)
        case .io:
            return AppStrings.localized(
                "'\(name)' could not be changed because of a file system error.", language: language)
        case .libraryReadOnly:
            return AppStrings.localized(
                "'\(name)' is in a read-only library, so it was left as it is.", language: language)
        case .useInstallId:
            // Not reachable once every engine row is deleted by its install id.
            return generic(name: name, reason: nil, language: language)
        default:
            // A code this build does not know: the engine's own sentence, cut
            // and escaped, is the only detail there is.
            return generic(name: name, reason: reasonText(of: error), language: language)
        }
    }

    /// The engine's message for a failure with no code of ours, escaped and cut.
    /// It can echo a name or a path from a drive.
    static let reasonLimit = 200

    private static func reasonText(of error: Error) -> String? {
        let message = (error as? TurboSparkError)?.message ?? error.localizedDescription
        let shown = UntrustedText.display(message, limit: reasonLimit)
        return shown.isEmpty ? nil : shown
    }

    /// "<Model> is on <Drive>, which is not connected." The one sentence the
    /// server favorite also uses, so the wording does not drift.
    static func notConnected(alias: String, libraryName: String?, language: AppLanguage? = nil) -> String {
        let language = language ?? AppStrings.currentLanguage
        let drive = (libraryName?.isEmpty == false)
            ? libraryName! : AppStrings.localized("Unknown library", language: language)
        return ServerFavoriteModel.notConnectedMessage(model: alias, library: drive, language: language)
    }

    /// The toast after a model was dropped from the list. Its files were not
    /// deleted.
    static func removedFromList(alias: String, language: AppLanguage? = nil) -> String {
        let name = UntrustedText.display(alias, limit: nameLimit)
        return AppStrings.localized(
            "Removed '\(name)' from the list.", language: language ?? AppStrings.currentLanguage)
    }

    private static func generic(name: String, reason: String?, language: AppLanguage) -> String {
        guard let reason else {
            return AppStrings.localized("TurboSpark could not change '\(name)'.", language: language)
        }
        return AppStrings.localized("TurboSpark could not change '\(name)': \(reason)", language: language)
    }
}
