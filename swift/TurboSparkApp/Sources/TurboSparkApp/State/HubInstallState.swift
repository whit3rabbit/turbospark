import Foundation
import TurboSpark

/// What the Model Hub's detail pane offers for one catalog model (task 6.9).
///
/// **THE HUB READS A LIST THAT HIDES AN AWAY DRIVE.** `AppModel.installed` holds
/// available rows only, so a model installed on a drive that is not connected
/// looked absent and the pane offered "Download & Install". The download would end
/// in `alias_elsewhere`, because alias plus modality is unique across every
/// library. This asks the library state's full list instead and says where the
/// model is.
enum HubInstallState: Equatable {
    /// Installed and usable (This Mac, or a connected library).
    case installed
    /// Installed on a drive that is not connected. `libraryName` is raw.
    case installedOffline(libraryName: String)
    /// Installed nowhere: the pane offers the download.
    case download

    /// - Parameters:
    ///   - isInstalled: the pane's own reading of the available list.
    ///   - rows: `LibraryState.allInstalled`, including rows whose drive is away.
    static func resolve(alias: String, isInstalled: Bool, rows: [InstalledModel]) -> HubInstallState {
        // A usable install wins: the engine keeps one install per alias and
        // modality, so an away row beside it is a list caught mid-reconnect.
        if isInstalled { return .installed }
        // Text models only: this pane installs from the text catalog, and an
        // image or audio install with the same name is another model.
        guard let away = rows.first(where: {
            $0.alias == alias && $0.modality == "text" && $0.availability == .notConnected
        }) else { return .download }
        return .installedOffline(libraryName: away.connectName ?? away.libraryName ?? "")
    }

    var offersDownload: Bool {
        if case .download = self { return true }
        return false
    }

    /// The statement for an install on an away drive, nil otherwise.
    func statement(language: AppLanguage? = nil) -> String? {
        guard case .installedOffline(let name) = self else { return nil }
        return InstallFailureText.offlineInstallStatement(libraryName: name, language: language)
    }

    /// The minus every "Not connected" row draws, so status is a shape and a
    /// sentence and never a color alone.
    var glyph: String { LibraryRowStatus.notConnected.glyph }
}
