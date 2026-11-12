import Combine
import Foundation
import TurboSpark

/// The one list every picker, and every code path that starts a load or offers a
/// model to load, reads (requirement 3.3).
///
/// **WHY THERE IS A FILTER AT ALL.** `AppModel.installed` is the engine's
/// available-only list at the time of the last `refreshModels()`, plus the rows
/// the app scanned from other apps' folders (a placeholder id, always loadable).
/// It goes stale the moment a drive is unplugged, so a picker reading it can
/// offer a model whose files are gone. Two things fix that at the root, not in
/// each view: `installed` follows `libraryState.allInstalled` (below), and this
/// filter answers from the LIVE availability for the moment between an unplug and
/// the reload that follows.
///
/// **WHAT "LIVE" MEANS.** The row's `installId` is looked up in the library
/// state's full list. A row that is not in it, such as a scanned one or one read
/// before the first listing landed, counts as available: only a listing that says
/// otherwise can take a model out of a picker. A scanned row is never looked up
/// by id at all, because its placeholder id (`internal:text/<alias>`) can equal
/// the id of a This Mac row whose folder is gone, and they are two models.
enum LoadableModelList {
    /// The rows of `installed` whose live availability is `.available`, in
    /// `installed`'s order.
    static func filter(installed: [InstalledModel], live: [InstalledModel]) -> [InstalledModel] {
        let index = liveIndex(live)
        return installed.filter { availability(of: $0, index: index) == .available }
    }

    /// `row`'s availability as the full list says it now.
    static func availability(of row: InstalledModel, live: [InstalledModel]) -> LibraryAvailability {
        availability(of: row, index: liveIndex(live))
    }

    /// The full-list row for `model` when it has one, else `model` itself: the
    /// fresher answer about where it is and whether it can be opened.
    static func liveRow(for model: InstalledModel, in live: [InstalledModel]) -> InstalledModel {
        guard !ModelIdentityKey.isPlaceholder(model) else { return model }
        return live.first { !ModelIdentityKey.isPlaceholder($0) && $0.installId == model.installId } ?? model
    }

    private static func liveIndex(_ live: [InstalledModel]) -> [String: LibraryAvailability] {
        var index: [String: LibraryAvailability] = [:]
        for row in live where !ModelIdentityKey.isPlaceholder(row) && index[row.installId] == nil {
            index[row.installId] = row.availability
        }
        return index
    }

    private static func availability(
        of row: InstalledModel, index: [String: LibraryAvailability]
    ) -> LibraryAvailability {
        guard !ModelIdentityKey.isPlaceholder(row) else { return .available }
        return index[row.installId] ?? .available
    }
}

extension AppModel {
    /// The models a picker may offer and a load may start: `installed` minus the
    /// rows whose drive is away or whose folder is gone. Pure and cheap; read it
    /// from a view, not a copy of it.
    ///
    /// A view that reads it redraws when `installed` is published, which the
    /// follow-up below does on every change of the library list. `AppModel` does
    /// not forward `libraryState` itself, which would redraw every screen on a
    /// mount.
    var loadableModels: [InstalledModel] {
        LoadableModelList.filter(installed: installed, live: libraryState.allInstalled)
    }

    /// Whether `model` can be opened now, by the live library list.
    func liveAvailability(of model: InstalledModel) -> LibraryAvailability {
        LoadableModelList.availability(of: model, live: libraryState.allInstalled)
    }

    /// Models whose drive is not connected: they exist, and cannot be loaded
    /// until the drive is back. A model whose folder is gone does not count, since
    /// there is nothing to plug in.
    var notConnectedModels: [InstalledModel] {
        libraryState.allInstalled.filter { $0.availability == .notConnected }
    }

    /// "Connect T7 to load" for the first model whose drive is away, or nil when
    /// none is. What a picker with nothing to offer says instead of "No models
    /// installed": the user's models exist and are not connected, which is a
    /// statement and not an error.
    var notConnectedStatement: String? {
        guard let first = notConnectedModels.first else { return nil }
        return LoadGateText.statement(for: loadGate(for: first))
    }

    /// The same statement for the model installed under `alias`, or nil when that
    /// model is not on a drive that is away.
    func notConnectedStatement(forAlias alias: String) -> String? {
        guard let row = notConnectedModels.first(where: { $0.alias == alias }) else { return nil }
        return LoadGateText.statement(for: loadGate(for: row))
    }

    // MARK: - Following the library list

    /// Keeps `installed` in step with `libraryState.allInstalled`, so it follows
    /// mounts, unmounts and renames (requirement 3.3). Called once when library
    /// monitoring starts; calling it again does nothing.
    ///
    /// **NO LOOP.** The reload is `refreshModels()`, which reads the engine's
    /// available list and the scanned folders and never refreshes the library
    /// state, so the two cannot feed each other. It is also coalesced: any number
    /// of changes before the main actor next runs are one reload, and a listing
    /// that changed nothing publishes nothing (`LibraryState.apply` assigns only a
    /// different list).
    func startInstalledListFollowing() {
        guard installedListFollower == nil else { return }
        installedListFollower = libraryState.$allInstalled
            .dropFirst()
            .removeDuplicates()
            .sink { [weak self] _ in self?.scheduleInstalledReload() }
    }

    /// Stops following. `shutdown()` calls it, and a vault lock builds a new model
    /// that follows on its own.
    func stopInstalledListFollowing() {
        installedListFollower?.cancel()
        installedListFollower = nil
    }

    private func scheduleInstalledReload() {
        guard !installedReloadScheduled else { return }
        installedReloadScheduled = true
        // The next main-actor turn, not this one: the published value is set
        // after its subscribers are told, and the reload should find the new list.
        Task { @MainActor [weak self] in
            guard let self else { return }
            self.installedReloadScheduled = false
            guard self.installedListFollower != nil else { return }
            self.refreshModels()
        }
    }
}
