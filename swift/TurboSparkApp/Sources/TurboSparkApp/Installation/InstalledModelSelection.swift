import Foundation
import TurboSpark

/// The rules of multi-select delete in the Installed list, kept pure so each is
/// a table row in a test. The view only draws what this says.
///
/// A row is selectable only when Delete would be offered AND would remove files
/// the app owns right now:
/// - a model on a drive that is away has no file to reach (the engine refuses);
/// - a model whose folder is gone has nothing to delete;
/// - a row only a scan found is never the app's to delete (its "delete" only
///   stops listing it, `AppModel.startModelDelete`), so bulk "frees N GB" would
///   be a false claim;
/// - the loaded model is mapped; the single Delete unloads it, but a batch must
///   not silently eject the model the user is chatting with.
enum InstalledModelSelection {
    enum Block: Equatable {
        case notConnected
        case missing
        case scanned
        case loaded

        /// Why the row cannot be selected, in `language`. Shown as the tooltip and
        /// the VoiceOver hint of the disabled checkbox.
        func reason(language: AppLanguage = AppStrings.currentLanguage) -> String {
            switch self {
            case .notConnected:
                return AppStrings.localized("Cannot select: its drive is not connected.", language: language)
            case .missing:
                return AppStrings.localized("Cannot select: its files are missing.", language: language)
            case .scanned:
                return AppStrings.localized(
                    "Cannot select: TurboSpark does not delete files it only found by scanning.", language: language)
            case .loaded:
                return AppStrings.localized("Cannot select: it is loaded. Unload it first.", language: language)
            }
        }
    }

    /// Why `model` cannot be part of a bulk delete, nil when it can.
    static func block(for model: InstalledModel, status: InstalledRowStatus?, isLoaded: Bool) -> Block? {
        switch status {
        case .notConnected: return .notConnected
        case .missing: return .missing
        case nil: break
        }
        if ModelIdentityKey.isPlaceholder(model) { return .scanned }
        if isLoaded { return .loaded }
        return nil
    }

    /// The paths of the visible rows that may be selected.
    static func selectablePaths(
        in models: [InstalledModel], blockOf: (InstalledModel) -> Block?
    ) -> Set<String> {
        Set(models.filter { blockOf($0) == nil }.map(\.path))
    }

    /// The selection that still counts: a row hidden by a filter, deleted, or
    /// loaded since it was ticked drops out, so the bar never frees more than
    /// the user can see.
    static func effective(selection: Set<String>, selectable: Set<String>) -> Set<String> {
        selection.intersection(selectable)
    }

    /// Bytes the deletion would free.
    static func bytesFreed(of selection: Set<String>, in models: [InstalledModel]) -> UInt64 {
        models.filter { selection.contains($0.path) }.reduce(UInt64(0)) { $0 + $1.installBytes }
    }

    /// Select All when anything selectable is unticked, otherwise nothing (so the
    /// same control deselects).
    static func toggledAll(selection: Set<String>, selectable: Set<String>) -> Set<String> {
        selectable.isSubset(of: selection) && !selectable.isEmpty ? [] : selectable
    }

    /// The detail pane's model after `deleted` paths left: keep the current one,
    /// else the first remaining row in list order, else none.
    static func detailSelection(
        current: String?, deleted: Set<String>, remaining: [InstalledModel]
    ) -> String? {
        if let current, !deleted.contains(current), remaining.contains(where: { $0.path == current }) {
            return current
        }
        return remaining.first(where: { !deleted.contains($0.path) })?.path
    }
}

/// The words of the bulk-delete confirmation.
struct InstalledModelBulkDeleteDialog: Equatable {
    let count: Int
    let title: String
    let message: String
    let confirmTitle: String

    init(count: Int, bytes: UInt64, language: AppLanguage = AppStrings.currentLanguage) {
        self.count = count
        title = AppStrings.localized("Delete \(count) models?", language: language)
        message = AppStrings.localized(
            "This permanently deletes the model files and frees \(MetricFormat.storage(bytes)). This cannot be undone.",
            language: language)
        confirmTitle = AppStrings.localized("Delete \(count) Models", language: language)
    }
}

/// What a bulk delete did, decided from what is still listed afterwards rather
/// than from each call's return value (a refused delete, a scanned row and a
/// finished one all return nothing).
struct InstalledModelBulkDeleteOutcome: Equatable {
    let deleted: [String]
    let failed: [String]

    static func make(
        requested: [InstalledModel], remainingPaths: Set<String>
    ) -> InstalledModelBulkDeleteOutcome {
        InstalledModelBulkDeleteOutcome(
            deleted: requested.filter { !remainingPaths.contains($0.path) }.map(\.alias),
            failed: requested.filter { remainingPaths.contains($0.path) }.map(\.alias))
    }

    var hasFailures: Bool { !failed.isEmpty }

    /// The one announcement: what went, and which stayed. Names are data from
    /// disk, so they are escaped and the list is cut at three.
    func summary(language: AppLanguage = AppStrings.currentLanguage) -> String {
        let count = deleted.count
        guard hasFailures else {
            return AppStrings.localized("Deleted \(count) models", language: language)
        }
        let shown = failed.prefix(3).map { UntrustedText.display($0, limit: ModelDeleteText.nameLimit) }
        let names = shown.joined(separator: ", ") + (failed.count > 3 ? ", ..." : "")
        if count == 0 {
            return AppStrings.localized("Not deleted: \(names)", language: language)
        }
        return AppStrings.localized("Deleted \(count), not deleted: \(names)", language: language)
    }
}

extension AppModel {
    /// Deletes `models` one after another through `startModelDelete`, the same
    /// path the single Delete takes (its guards, the engine call, the notes
    /// cleanup). One at a time because the engine answers `busy` while another
    /// delete runs. The outcome is read from what is still listed.
    @MainActor
    func deleteModels(_ models: [InstalledModel]) async -> InstalledModelBulkDeleteOutcome {
        for model in models {
            if let task = startModelDelete(model) { await task.value }
        }
        let remaining = Set(installedForDisplay.map(\.path))
        let outcome = InstalledModelBulkDeleteOutcome.make(requested: models, remainingPaths: remaining)
        showToast(outcome.summary(), style: outcome.hasFailures ? .warning : .info, duration: 5.0)
        return outcome
    }
}
