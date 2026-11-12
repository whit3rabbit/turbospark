import Foundation
import TurboSpark

/// What the audio catalog read in one pass: the pinned profiles, the install
/// report (with its `notConnected` bucket) and the family capabilities. The seam
/// the workspace reads through, so a test can answer without the engine.
struct AudioCatalogSnapshot: Sendable {
    var profiles: [AudioProfile]
    var installed: AudioInstalledReport
    var capabilities: [AudioFamilyCapability]

    /// The live engine. Blocking: call it off the main actor.
    static let live: @Sendable () throws -> AudioCatalogSnapshot = {
        AudioCatalogSnapshot(
            profiles: try AudioCatalog.profiles(), installed: try AudioCatalog.installed(),
            capabilities: try AudioCatalog.capabilities())
    }
}

/// Whether one audio profile can be used now (task 6.10, requirement 3.1).
///
/// **A DRIVE THAT IS AWAY IS A STATE, NOT A FAULT.** The audio report keeps an
/// install in a library that is not connected in its own `notConnected` bucket,
/// with no error: it is not `incompatible` and it is not missing. It cannot be
/// run, and its Download button is replaced by the statement, because a download
/// would end in `alias_elsewhere`.
enum AudioInstallStatus: Equatable {
    /// Installed and usable (the engine's list, or a checkpoint folder the user
    /// chose).
    case installed
    /// Installed in a library that is not connected. `libraryName` is raw.
    case notConnected(libraryName: String)
    case notInstalled

    /// - Parameters:
    ///   - installedPaths: the usable installs by alias.
    ///   - localModels: aliases the user pointed at a checkpoint folder.
    ///   - notConnected: the report's bucket.
    static func resolve(
        alias: String, installedPaths: [String: String], localModels: Set<String>,
        notConnected: [AudioNotConnectedInstall]
    ) -> AudioInstallStatus {
        // A usable install wins. A folder the user chose does not live on the
        // away drive, so it keeps the profile usable too.
        if installedPaths[alias] != nil || localModels.contains(alias) { return .installed }
        if let offline = notConnected.first(where: { $0.record.alias == alias }) {
            return .notConnected(libraryName: offline.libraryName)
        }
        return .notInstalled
    }

    var isInstalled: Bool { self == .installed }

    /// Only a profile installed nowhere offers a download.
    var offersDownload: Bool { self == .notInstalled }
}

/// The words and glyphs for an audio install on a drive that is away.
enum AudioOfflineInstalls {
    struct PickerLabel: Equatable {
        let title: String
        /// A shape for the state, so it is never color alone. Nil when usable.
        let glyph: String?
    }

    /// The model picker's row: the profile's name, and for an install on an away
    /// drive "Not connected" with the minus every such row draws.
    static func pickerLabel(
        displayName: String, status: AudioInstallStatus, language: AppLanguage? = nil
    ) -> PickerLabel {
        guard case .notConnected = status else { return PickerLabel(title: displayName, glyph: nil) }
        let word = AppStrings.localized("Not connected", language: language ?? AppStrings.currentLanguage)
        return PickerLabel(title: "\(displayName), \(word)", glyph: LibraryRowStatus.notConnected.glyph)
    }

    /// The statement that replaces the install button, or nil when there is none.
    static func statement(for status: AudioInstallStatus, language: AppLanguage? = nil) -> String? {
        guard case .notConnected(let name) = status else { return nil }
        return InstallFailureText.offlineInstallStatement(libraryName: name, language: language)
    }

    /// One offline speech install the API pane lists without offering it.
    struct Row: Equatable {
        let alias: String
        /// Raw: the pane shows it through `UntrustedText`.
        let libraryName: String
        var glyph: String { LibraryRowStatus.notConnected.glyph }

        func statement(language: AppLanguage? = nil) -> String {
            InstallFailureText.offlineInstallStatement(libraryName: libraryName, language: language)
        }
    }

    /// The speech families the API pane can serve.
    static let speechFamilies: Set<String> = ["qwen3_tts", "kokoro", "megatts3"]

    /// The report's away installs of those families, for the API pane: shown, never
    /// offered for attaching.
    static func offlineSpeechRows(in report: AudioInstalledReport) -> [Row] {
        report.notConnected
            .filter { speechFamilies.contains($0.record.family) }
            .map { Row(alias: $0.record.alias, libraryName: $0.libraryName) }
    }
}
