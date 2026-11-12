import Foundation
import TurboSpark

/// The words for an install the engine refused or lost (requirement 2.4), in the
/// language the UI is showing.
///
/// **ONE SENTENCE PER ENGINE CODE, KEYED ON THE PREFIX.** The engine answers a
/// refused or failed library install as `"<code>: <sentence>"` and a host
/// localizes by the code, never by the sentence (`LibraryFault`). An install can
/// answer `not_connected`, `library_disconnected`, `library_read_only`,
/// `unknown_library`, `alias_elsewhere` and `busy`. `busy` is read from the
/// library fault and never from `TurboSparkError.Code.busy`, which also means "the
/// audio device is busy".
///
/// **NIL MEANS "NOT MINE".** Any other error (a network failure, a cancel, a
/// location refusal whose spelling happens to be an ordinary word) returns nil, and
/// the caller keeps the failure surface it always had. Only a drive being away moves
/// off the app error: an unplugged drive is a state, not a fault of the app.
///
/// **THE ONLY ENGINE TEXT SHOWN IS A NAME WE ALREADY HOLD.** The library name and
/// the place come from a drive, so each goes through `UntrustedText.display`; the
/// engine's own sentence is never shown.
///
/// Every function takes the language as a parameter, so a test renders each
/// sentence through all 21 compiled bundles.
enum InstallFailureText {
    /// The longest library name a sentence carries, after escaping.
    static let nameLimit = InstallDestinationModel.nameLimit

    /// The sentence for a failed install, or nil when the error is not one of the
    /// install codes this file words.
    ///
    /// - Parameters:
    ///   - libraryName: the destination library's name, raw. For the codes that
    ///     name a drive.
    ///   - place: where the alias is installed already, for `alias_elsewhere`
    ///     (see `place(of:)`). Nil reads as "another place".
    static func sentence(
        for error: Error, alias: String, libraryName: String?, place: String?,
        language: AppLanguage? = nil
    ) -> String? {
        guard let fault = (error as? TurboSparkError)?.libraryFault else { return nil }
        let language = language ?? AppStrings.currentLanguage
        switch fault.code {
        case .libraryDisconnected:
            let name = displayName(libraryName, language: language)
            return AppStrings.localized(
                "\(name) was disconnected during the install. Nothing was added. Connect it and try again.",
                language: language)
        case .notConnected:
            return ModelInstallGate.notConnectedSentence(
                name: libraryName.flatMap { $0.isEmpty ? nil : $0 } ?? unknownName(language), language: language)
        case .libraryReadOnly:
            let name = displayName(libraryName, language: language)
            return AppStrings.localized(
                "\(name) is read-only, so nothing was installed there. Choose another library.",
                language: language)
        case .unknownLibrary:
            return AppStrings.localized(
                "That library is no longer in the list, so nothing was installed. Choose another library.",
                language: language)
        case .aliasElsewhere:
            guard let place, !place.isEmpty else {
                return AppStrings.localized(
                    "A model with this name is already installed in another place. Move it from there instead.",
                    language: language)
            }
            let shownPlace = UntrustedText.display(place, limit: nameLimit)
            return AppStrings.localized(
                "A model with this name is already installed on \(shownPlace). Move it from there instead.",
                language: language)
        case .busy:
            return LibraryActionText.busy(language: language)
        default:
            return nil
        }
    }

    /// Where the model with this alias is installed already, from the row that
    /// holds it: "This Mac" in the UI language, or the library's name. Nil for a
    /// custom folder and for no row, so the sentence says "another place".
    static func place(of row: InstalledModel?, language: AppLanguage? = nil) -> String? {
        guard let row else { return nil }
        let language = language ?? AppStrings.currentLanguage
        switch row.locationKind {
        case .thisMac:
            return AppStrings.localized("This Mac", language: language)
        case .library:
            guard let name = row.libraryName, !name.isEmpty else { return nil }
            return name
        case .custom, .unknown:
            return nil
        }
    }

    /// "Installed on T7, which is not connected. Connect the drive to use it."
    /// What the Model Hub and the audio pickers say instead of offering a download
    /// of a model that is installed on a drive that is away.
    static func offlineInstallStatement(libraryName: String, language: AppLanguage? = nil) -> String {
        let language = language ?? AppStrings.currentLanguage
        let name = displayName(libraryName, language: language)
        return AppStrings.localized(
            "Installed on \(name), which is not connected. Connect the drive to use it.",
            language: language)
    }

    private static func displayName(_ raw: String?, language: AppLanguage) -> String {
        guard let raw, !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return unknownName(language)
        }
        return UntrustedText.display(raw, limit: nameLimit)
    }

    private static func unknownName(_ language: AppLanguage) -> String {
        AppStrings.localized("Unknown library", language: language)
    }
}
