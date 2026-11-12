import Foundation
import TurboSpark

/// The words for what the libraries section tells the user about an action it
/// ran (requirements 1.3 to 1.10 and 5.6), in the language the UI is showing.
///
/// **ONE SENTENCE PER ENGINE CODE.** The engine answers a refusal as
/// `"<code>: <sentence>"` and a host localizes by the code, never by the
/// sentence (`LibraryFault`). Every code the add, remove and default entry
/// points can answer has its own catalog key, so a refusal reads the same in
/// every language the app offers; a code this build does not know falls to one
/// generic sentence rather than to the engine's English.
///
/// **THE ONLY ENGINE TEXT SHOWN IS A LIBRARY NAME.** `nested` is the one
/// refusal that names something (the library that already covers the folder,
/// requirement 1.7), and the engine puts that name inside its English sentence.
/// It is cut out here and passed through `UntrustedText.display`, because a
/// marker on a drive can name its library anything, including text that reorders
/// what the user reads. Nothing else from the engine's sentence is displayed.
///
/// Every function takes the language as a parameter, so a test renders each
/// sentence through all 21 compiled bundles.
enum LibraryActionText {
    /// The longest library name a sentence carries, after escaping.
    static let nameLimit = 100
    /// The longest system reason a speed test failure carries.
    static let reasonLimit = 200

    // MARK: - Refusals and failures

    /// The sentence for a failed add, remove or default change.
    static func sentence(for error: Error, language: AppLanguage? = nil) -> String {
        let language = language ?? AppStrings.currentLanguage
        guard let fault = (error as? TurboSparkError)?.libraryFault else {
            return generic(language: language)
        }
        switch fault.code {
        case .refusal(let refusal):
            return sentence(for: refusal, sentence: fault.sentence, language: language)
        case .marker:
            return AppStrings.localized(
                "The library information on that drive could not be read or written.", language: language)
        case .io:
            return AppStrings.localized(
                "TurboSpark could not use that location because of a file system error.", language: language)
        case .busy:
            return busy(language: language)
        case .unknownLibrary:
            return AppStrings.localized("That library is no longer in the list.", language: language)
        case .internalLibrary:
            return internalLibrary(language: language)
        case .notConnected:
            return AppStrings.localized(
                "That drive is not connected. Connect it and try again.", language: language)
        default:
            return generic(language: language)
        }
    }

    private static func sentence(
        for refusal: LibraryRefusal, sentence engineSentence: String, language: AppLanguage
    ) -> String {
        switch refusal {
        case .sameDiskAsHome:
            return AppStrings.localized(
                "This Mac already covers that disk. Choose a folder on a different drive.", language: language)
        case .network:
            return AppStrings.localized(
                "Libraries must be on a directly connected drive, not on a network share.", language: language)
        case .smallFileLimit:
            return AppStrings.localized(
                "That drive's file system cannot store a single file of 4 GiB or more. Use a drive formatted as APFS or exFAT.",
                language: language)
        case .readOnly:
            return AppStrings.localized(
                "That drive is read-only. A library needs a drive TurboSpark can write to.", language: language)
        case .notADirectory:
            return AppStrings.localized(
                "That location is not a folder. Choose a folder or a drive.", language: language)
        case .nested:
            guard let covering = coveringLibraryName(in: engineSentence) else {
                return AppStrings.localized(
                    "That folder overlaps a library that is already listed.", language: language)
            }
            let name = UntrustedText.display(covering, limit: nameLimit)
            return AppStrings.localized(
                "That folder overlaps the library \(name), which already covers it.", language: language)
        case .insideHome:
            return AppStrings.localized(
                "That folder is inside the TurboSpark home folder, which This Mac already covers.",
                language: language)
        case .unknownVolume:
            return AppStrings.localized(
                "TurboSpark could not identify that drive. Choose a drive that is connected directly to this Mac.",
                language: language)
        }
    }

    /// The name of the library that covers a folder, from the engine's
    /// `"<place> overlaps the library <name>, which already covers it"`.
    ///
    /// The FIRST " overlaps the library " is the lead: the place before it is a
    /// path the user just chose, and a library named like the lead (a name is
    /// anything) still comes out whole after it. A sentence the engine cut at its
    /// length cap has no tail, which is not a reason to lose the name. Nil when
    /// the lead is missing or nothing follows it, so the caller falls back to a
    /// sentence that names no library.
    static func coveringLibraryName(in sentence: String) -> String? {
        guard let lead = sentence.range(of: " overlaps the library ") else { return nil }
        var name = String(sentence[lead.upperBound...])
        let tail = ", which already covers it"
        if name.hasSuffix(tail) { name.removeLast(tail.count) }
        name = name.trimmingCharacters(in: .whitespaces)
        return name.isEmpty ? nil : name
    }

    /// An install or a model move is running, so the engine refused a change to
    /// the library list (the library `busy` code), or the app knows one is.
    static func busy(language: AppLanguage? = nil) -> String {
        AppStrings.localized(
            "A model is installing or moving. Try again when it finishes.",
            language: language ?? AppStrings.currentLanguage)
    }

    /// This Mac is not a library that can be removed.
    static func internalLibrary(language: AppLanguage? = nil) -> String {
        AppStrings.localized("This Mac cannot be removed.", language: language ?? AppStrings.currentLanguage)
    }

    /// Any code this build does not map, and any error that is not the engine's.
    static func generic(language: AppLanguage? = nil) -> String {
        AppStrings.localized(
            "TurboSpark could not change the library list.", language: language ?? AppStrings.currentLanguage)
    }

    /// Stopping the wait for a moved store failed because its record could not
    /// be cleared (requirement 9.3).
    static func stopWaitingFailed(language: AppLanguage? = nil) -> String {
        AppStrings.localized(
            "TurboSpark could not stop waiting for this drive. Try again.",
            language: language ?? AppStrings.currentLanguage)
    }

    // MARK: - Outcomes

    static func added(name: String, language: AppLanguage? = nil) -> String {
        let name = UntrustedText.display(name, limit: nameLimit)
        return AppStrings.localized(
            "Added the library \(name).", language: language ?? AppStrings.currentLanguage)
    }

    static func removed(name: String, language: AppLanguage? = nil) -> String {
        let name = UntrustedText.display(name, limit: nameLimit)
        return AppStrings.localized(
            "Removed the library \(name).", language: language ?? AppStrings.currentLanguage)
    }

    /// `name` is the library's display name: already "This Mac" in the
    /// language for the internal library, and escaped here for any other.
    static func nowDefault(name: String, language: AppLanguage? = nil) -> String {
        let name = UntrustedText.display(name, limit: nameLimit)
        return AppStrings.localized(
            "\(name) is now the default library.", language: language ?? AppStrings.currentLanguage)
    }

    // MARK: - Speed test (requirement 5.6)

    /// Why a speed test did not produce a number. Cancellation is not one: the
    /// caller drops it without a word.
    static func speedTestSentence(for error: Error, language: AppLanguage? = nil) -> String {
        let language = language ?? AppStrings.currentLanguage
        let reason: String
        switch error as? DriveSpeedTestError {
        case .thisMac?:
            return AppStrings.localized("This Mac does not need a speed test.", language: language)
        case .noLargeFile?:
            return AppStrings.localized(
                "This library has no large model file to read yet. Install a model on it, then test again.",
                language: language)
        case .unreadable(let errnoText)?:
            reason = errnoText
        case nil:
            reason = "\(error)"
        }
        let why = UntrustedText.display(reason, limit: reasonLimit)
        return AppStrings.localized("The speed test could not read this drive: \(why)", language: language)
    }
}
