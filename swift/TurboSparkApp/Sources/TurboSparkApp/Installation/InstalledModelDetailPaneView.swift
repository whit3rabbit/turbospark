import AppKit
import SwiftUI
import TurboSpark

/// Right-column detail inspector for an installed model in the Model Manager.
struct InstalledModelDetailPaneView: View {
    @ObservedObject var model: AppModel
    let installedModel: InstalledModel
    @ObservedObject private var orgStore = ModelOrganizationStore.shared

    @State private var showingDeleteConfirm = false

    private var descriptor: ModelFeatureDescriptor {
        ModelFeatureDescriptor.resolve(installedModel: installedModel)
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
    private var fit: ModelRecommendation? {
        model.fit(for: installedModel.alias)
    }

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
        orgStore.isFavorite(alias: installedModel.alias, path: installedModel.path)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                InstalledModelHeroHeaderView(
                    installedModel: installedModel,
                    visuals: visuals,
                    descriptor: descriptor,
                    isCurrentlyLoaded: isCurrentlyLoaded,
                    isCurrentlySelected: isCurrentlySelected,
                    isFavorite: isFavorite,
                    onToggleFavorite: {
                        orgStore.toggleFavorite(alias: installedModel.alias, path: installedModel.path)
                    }
                )

                InstalledModelActionBarView(
                    installedModel: installedModel,
                    isCurrentlyLoaded: isCurrentlyLoaded,
                    canUnloadModel: model.canUnloadModel,
                    canDeleteModel: model.canDeleteModel,
                    onLoad: { model.selectModel(installedModel) },
                    onUnload: { model.unloadModel() },
                    onStartChat: { model.openChatWithModel(installedModel) },
                    onDelete: { showingDeleteConfirm = true }
                )

                InstalledModelFeatureBadgesView(descriptor: descriptor)

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
                    alias: installedModel.alias,
                    defaultSystemPrompt: model.defaultSystemPrompt,
                    onShowToast: { msg in
                        model.showToast(msg, style: .info)
                    }
                )
            }
            .padding(24)
        }
        .onAppear {
            loadLadder()
        }
        .onChange(of: installedModel.alias) {
            loadLadder()
        }
        .onChange(of: installedModel.path) {
            loadLadder()
        }
        .confirmationDialog(
            "Delete \(installedModel.alias)?",
            isPresented: $showingDeleteConfirm,
            titleVisibility: .visible
        ) {
            Button(role: .destructive) {
                model.deleteModel(installedModel)
            } label: { Text("Delete Model", bundle: .module) }
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
        } message: {
            Text("This will permanently remove the model files (\(MetricFormat.storage(installedModel.installBytes))) from disk at:\n\(installedModel.path)", bundle: .module)
        }
    }

    /// Reads the ladder off this install's own manifest.
    ///
    /// Failure leaves it `nil` and the card simply omits the table. An
    /// install whose shape cannot be read has no ladder, and a ladder of
    /// zeros would be worse than none (swift Gotcha 23).
    private func loadLadder() {
        ladder = nil
        let path = installedModel.path
        let slots = model.activeCacheSlots
        let guard_ = model.activeLoadGuard
        Task {
            let found = try? TurboSparkCatalog.contextLadder(
                modelPath: path, expertCacheSlots: slots, loadGuard: guard_)
            // The pane can be showing a DIFFERENT model by the time this
            // lands, which is the same captured-identity rule the agent loop
            // follows (state#17): publish only if the subject still matches.
            if installedModel.path == path { ladder = found }
        }
    }
}
