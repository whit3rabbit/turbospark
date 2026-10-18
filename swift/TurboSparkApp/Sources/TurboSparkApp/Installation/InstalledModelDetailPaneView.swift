import AppKit
import SwiftUI
import TurboSpark

/// Right-column detail inspector for an installed model in the Model Manager.
struct InstalledModelDetailPaneView: View {
    @ObservedObject var model: AppModel
    /// Observed directly (`AppModel` does not publish it): the pane redraws when
    /// the model's drive is plugged or unplugged while it is open.
    @ObservedObject private var libraryState: LibraryState
    let installedModel: InstalledModel
    @ObservedObject private var orgStore = ModelOrganizationStore.shared
    /// Read so the strings built from `AppStrings` follow the language picker.
    @Environment(\.locale) private var locale

    @State private var showingDeleteConfirm = false
    @State private var showingRemoveConfirm = false

    init(model: AppModel, installedModel: InstalledModel) {
        self.model = model
        self.installedModel = installedModel
        _libraryState = ObservedObject(wrappedValue: model.libraryState)
    }

    private var language: AppLanguage {
        _ = locale
        return AppStrings.currentLanguage
    }

    /// Where the model is and whether it can be opened, from the library state.
    private var rowModel: InstalledModelRowModel {
        InstalledModelRowModel.make(
            for: installedModel, in: libraryState,
            isMoE: descriptor.routingType == .moe, language: language)
    }

    /// The library's name for the confirmations, as the library list has it now.
    private var libraryName: String? {
        libraryState.library(for: installedModel)?.name ?? installedModel.libraryName
    }

    private var descriptor: ModelFeatureDescriptor {
        // Pass the open session's own report when this model is the loaded
        // one, so the badges agree with the steering card's engine answer.
        ModelFeatureDescriptor.resolve(
            installedModel: installedModel,
            sessionInfo: isCurrentlyLoaded ? model.info : nil)
    }

    private var visuals: ModelFamilyVisuals {
        ModelFamilyVisuals.resolve(
            alias: installedModel.alias,
            family: installedModel.family,
            name: installedModel.alias
        )
    }

    /// This machine's answer for this install, or `nil` when the catalog has
    /// no row for it (a scanned or side-loaded directory).
    ///
    /// Loaded off the main actor by the `.task` below and cached in
    /// `@State`. It used to call the synchronous recommend pass from `body`,
    /// twice per evaluation, and this pane observes the whole AppModel.
    private var fit: ModelRecommendation? { loadedFit }
    @State private var loadedFit: ModelRecommendation? = nil

    /// The context and slot count the figures above are computed at.
    ///
    /// **Read off the SESSION when this model is the loaded one.**
    /// `SessionInfo` holds what was RESOLVED and under automatic sizing
    /// nothing was asked for (swift Gotcha 6), so the Inspector's setting is
    /// the request and `info` is the answer.
    private var resolvedSlots: Int? {
        if isCurrentlyLoaded, let slots = model.session?.info.expertCacheSlots, slots > 0 {
            return slots
        }
        return fit.map(\.slotCacheSlots)
    }

    /// What a longer window would cost this install, read off its own
    /// manifest. Loaded on appear rather than computed in `body`: it touches
    /// the filesystem and crosses the ABI, and a SwiftUI body runs far more
    /// often than an install changes.
    @State private var ladder: ContextLadder? = nil

    private var isCurrentlyLoaded: Bool {
        model.selected?.path == installedModel.path && model.session != nil
    }

    private var isCurrentlySelected: Bool {
        model.selected?.alias == installedModel.alias
    }

    private var isFavorite: Bool {
        orgStore.isFavorite(installedModel)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                InstalledModelHeroHeaderView(
                    installedModel: installedModel,
                    rowModel: rowModel,
                    visuals: visuals,
                    descriptor: descriptor,
                    isCurrentlyLoaded: isCurrentlyLoaded,
                    isCurrentlySelected: isCurrentlySelected,
                    isFavorite: isFavorite,
                    onToggleFavorite: {
                        orgStore.toggleFavorite(installedModel)
                    }
                )

                InstalledModelActionBarView(
                    installedModel: installedModel,
                    rowModel: rowModel,
                    isCurrentlyLoaded: isCurrentlyLoaded,
                    canUnloadModel: model.canUnloadModel,
                    canDeleteModel: model.canDeleteModel,
                    isChanging: model.modelsBeingChanged.contains(installedModel.installId),
                    onLoad: { model.selectModel(installedModel) },
                    // The same load as Load, minus the question: the warning is
                    // on screen beside the button (requirement 5.5).
                    onLoadAnyway: { model.loadAnyway(installedModel) },
                    onUnload: { model.unloadModel() },
                    onStartChat: { model.openChatWithModel(installedModel) },
                    onDelete: { showingDeleteConfirm = true },
                    onRemoveFromList: { showingRemoveConfirm = true },
                    onMove: {
                        // The sheet is hosted by the list, which outlives this
                        // pane: the pane is gone once the row has moved.
                        if let subject = MoveSubject.make(text: installedModel) {
                            model.presentMoveSheet(for: subject)
                        }
                    }
                )

                InstalledModelFeatureBadgesView(descriptor: descriptor)

                ModelSteeringCardView(
                    model: model,
                    catalogEntry: nil,
                    installedModel: installedModel
                )

                InstalledModelMemoryFitCardView(
                    fit: fit,
                    descriptor: descriptor,
                    installedModel: installedModel,
                    ladder: ladder,
                    activeFitContext: model.activeFitContext,
                    resolvedSlots: resolvedSlots
                )

                InstalledModelTechnicalSpecsView(
                    visuals: visuals,
                    descriptor: descriptor,
                    installedModel: installedModel
                )

                InstalledModelOrganizationCardView(installedModel: installedModel)

                InstalledModelDeveloperCommandsView(
                    modelPath: installedModel.path,
                    defaultSystemPrompt: model.defaultSystemPrompt,
                    onShowToast: { msg in
                        model.showToast(msg, style: .info)
                    }
                )
            }
            .padding(24)
        }
        // `.task(id:)` cancels the previous load when the key changes, so a
        // slower earlier read can never land after a newer one, and slot or
        // load-guard changes now reload the ladder too.
        .task(id: ladderKey) {
            await loadLadder()
        }
        .task(id: "\(installedModel.alias)|\(model.fitRecommendationConfigurationID)") {
            let alias = installedModel.alias
            if loadedFit?.alias != alias { loadedFit = nil }
            let rows = try? await model.loadFitRecommendations(probeIfNeeded: false)
            guard !Task.isCancelled else { return }
            loadedFit = rows?.first { $0.alias == alias }
        }
        .confirmationDialog(
            Text(verbatim: deleteDialog.title),
            isPresented: $showingDeleteConfirm,
            titleVisibility: .visible
        ) {
            Button(role: .destructive) {
                model.deleteModel(installedModel)
            } label: { Text(verbatim: deleteDialog.confirmTitle) }
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
        } message: {
            // A library model names its library and says the files on that
            // drive are deleted; a model on this Mac names its path.
            Text(verbatim: deleteDialog.message)
        }
        .confirmationDialog(
            Text(verbatim: removeDialog.title),
            isPresented: $showingRemoveConfirm,
            titleVisibility: .visible
        ) {
            Button(role: .destructive) {
                model.removeModelFromList(installedModel)
            } label: { Text(verbatim: removeDialog.confirmTitle) }
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
        } message: {
            Text(verbatim: removeDialog.message)
        }
    }

    /// The delete confirmation, in the picker's language.
    private var deleteDialog: ModelDeleteDialog {
        ModelDeleteDialog(model: installedModel, libraryName: libraryName, language: language)
    }

    /// "Remove from list" never deletes a file, whatever the row says: its
    /// confirmation is the list-only one.
    private var removeDialog: ModelDeleteDialog {
        ModelDeleteDialog(removingFromList: installedModel, language: language)
    }

    /// Everything the ladder depends on. `OpenOptions` values are not
    /// Equatable, so their descriptions stand in for them.
    private var ladderKey: String {
        "\(installedModel.path)|\(model.activeCacheSlots)|\(model.activeLoadGuard)"
    }

    /// Reads the ladder off this install's own manifest.
    ///
    /// Failure leaves it `nil` and the card simply omits the table. An
    /// install whose shape cannot be read has no ladder, and a ladder of
    /// zeros would be worse than none (swift Gotcha 23).
    ///
    /// The FFI call reads the manifest and walks the install directory, so it
    /// runs detached: this view is MainActor-inferred and a plain `Task`
    /// would inherit that and block the main thread on every selection.
    private func loadLadder() async {
        ladder = nil
        // A model that is away or missing has nothing at its recorded path to
        // read, and a drive of the same name may be mounted there.
        guard installedModel.availability == .available else { return }
        let path = installedModel.path
        let slots = model.activeCacheSlots
        let guard_ = model.activeLoadGuard
        let found = await Task.detached(priority: .userInitiated) {
            try? TurboSparkCatalog.contextLadder(
                modelPath: path, expertCacheSlots: slots, loadGuard: guard_)
        }.value
        // A changed key cancels this task; never publish a stale result.
        if !Task.isCancelled { ladder = found }
    }
}
