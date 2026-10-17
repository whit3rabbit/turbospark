import Foundation
import TurboSpark

/// The rows the Installed Models list shows (requirement 3.1): the rows
/// `AppModel.installed` holds, plus the rows of libraries that are not
/// connected and the rows whose folder is gone.
///
/// **TWO LISTS, ONE DISPLAY LIST.** `AppModel.installed` is the available-only
/// list every picker and load gate reads, and it also holds the rows the app
/// scanned from other apps' folders (LM Studio, custom scan folders), which have
/// a placeholder id and no library. `LibraryState.allInstalled` has every row
/// the engine lists, offline ones too, and none of the scanned ones. The manager
/// shows both, each model once, and changes neither list.
///
/// **THE RULES, IN ORDER**
/// 1. Every `installed` row keeps its place, so the list does not jump when a
///    drive is unplugged.
/// 2. When the library list says an `installed` row is no longer available (the
///    drive left and the picker list is not read again yet), the row shows the
///    library list's answer in the same place. The fresher answer wins; an
///    available answer changes nothing.
/// 3. Rows of the library list that are not available and not represented yet
///    follow, in the library list's order.
/// 4. A row of the library list that is available and absent from `installed`
///    is NOT added: the pickers do not offer it either, and the next refresh
///    brings it. (A drive that just came back is the case.)
///
/// **WHAT "REPRESENTED" MEANS.** Same install id, or the same standardized path.
/// A placeholder id (`internal:text/<alias>` on a row with no library, built by
/// the app for another app's folder) names no registry install and is never
/// matched by id: it can equal the id of a This Mac row whose folder is gone,
/// and they are two models. An empty path matches nothing.
enum InstalledModelDisplayList {
    static func merge(installed: [InstalledModel], libraryRows: [InstalledModel]) -> [InstalledModel] {
        var engineById: [String: InstalledModel] = [:]
        for row in libraryRows where !ModelIdentityKey.isPlaceholder(row) && engineById[row.installId] == nil {
            engineById[row.installId] = row
        }

        var shown: [InstalledModel] = []
        var seenIds = Set<String>()
        var seenPaths = Set<String>()

        func represented(_ row: InstalledModel) -> Bool {
            if !ModelIdentityKey.isPlaceholder(row), seenIds.contains(row.installId) { return true }
            let path = standardized(row.path)
            return !path.isEmpty && seenPaths.contains(path)
        }
        func remember(_ row: InstalledModel) {
            if !ModelIdentityKey.isPlaceholder(row) { seenIds.insert(row.installId) }
            let path = standardized(row.path)
            if !path.isEmpty { seenPaths.insert(path) }
        }

        for row in installed where !represented(row) {
            var displayed = row
            if !ModelIdentityKey.isPlaceholder(row), let fresh = engineById[row.installId],
                fresh.availability != .available
            {
                displayed = fresh
            }
            shown.append(displayed)
            remember(row)
            remember(displayed)
        }
        for row in libraryRows where row.availability != .available && !represented(row) {
            shown.append(row)
            remember(row)
        }
        return shown
    }

    private static func standardized(_ path: String) -> String {
        path.isEmpty ? "" : URL(fileURLWithPath: path).standardizedFileURL.path
    }
}

extension AppModel {
    /// What the Installed Models list shows: `installed` plus the offline and
    /// missing library rows. A view reading this must observe `libraryState`
    /// (`AppModel` does not publish it), or it will not redraw on a mount.
    var installedForDisplay: [InstalledModel] {
        InstalledModelDisplayList.merge(installed: installed, libraryRows: libraryState.allInstalled)
    }
}
