import Foundation
import TurboSpark

extension AppModel {
    /// Setup belongs to the image workspace, including when its catalog
    /// becomes available after the user has already opened that workspace.
    var shouldRecommendImageModel: Bool {
        activeSection == .images
            && !hasSupportedSelectedImageModel
            && !hasInstalledMLXImageModel
            && !recommendedImageModelSources.isEmpty
    }

    /// Image installs stay separate from the text catalog. The path remains a
    /// string because a user may still choose a valid side-loaded install.
    public var imageModelPath: String {
        let path = imageModelPathText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !path.isEmpty { return (path as NSString).expandingTildeInPath }
        return ""
    }

    public var canGenerateImage: Bool {
        !imageModelPath.isEmpty
            && hasSupportedSelectedImageModel
            && !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && canStartImageGeneration
    }

    public func selectImageModel(_ model: ImageInstalledModel) {
        imageModelPathText = model.path
    }

    /// True when an installed MLX checkpoint supported by the app is available.
    public var hasInstalledMLXImageModel: Bool {
        imageModels.contains {
            Self.supportsMLXImageModel(modelID: $0.modelID)
        }
    }

    var hasSupportedSelectedImageModel: Bool {
        guard let selectedImageModel else { return false }
        return Self.supportsMLXImageModel(modelID: selectedImageModel.modelID)
    }

    static func supportsMLXImageModel(modelID: String) -> Bool {
        [
            "andrevp/Z-Image-Turbo-MLX-2bit",
            "andrevp/Z-Image-Turbo-MLX-4bit",
            "andrevp/Z-Image-Turbo-MLX-8bit",
            "deepsweet/Z-Image-Turbo-6B-MLX-Q4",
            "Sawfwair/Qwen-Image-2.1-Turbo-MLX-Mixed-4bit",
        ].contains(modelID)
    }

    /// The MLX variants whose pinned install gates have passed. Native and
    /// unqualified sources stay out of every Swift app download surface.
    public static let testedImageModelAliases = [
        "z-image-turbo-mlx-2bit",
        "z-image-turbo-mlx-4bit",
        "z-image-turbo-mlx-8bit",
        "z-image-turbo-mlx-q4",
        "qwen-image-2.1-turbo-mlx-mixed-4bit",
    ]

    /// Returns MLX choices ranked for this machine; the first is the default.
    /// The ordering is intentional: the first row is the default suggestion,
    /// and the remaining rows give users a useful quality/footprint choice.
    ///
    /// The tiers are memory-fit gates, not speed claims. They retain only
    /// exports with a measured memory envelope for this app. Qwen-Image 2.1
    /// Turbo measured 11.2 GiB at 1024 x 1024 with staged residency and a
    /// bfloat16 decode; it sits in the 32 GiB tier behind the Z-Image 8-bit
    /// default, whose recorded 1024 x 1024 row in docs/BENCHMARKS.md is
    /// faster (not a paired run).
    public static func recommendedImageModelAliases(physicalMemoryBytes: UInt64) -> [String] {
        let gib = physicalMemoryBytes / (1024 * 1024 * 1024)
        if gib >= 32 {
            return [
                "z-image-turbo-mlx-8bit",
                "qwen-image-2.1-turbo-mlx-mixed-4bit",
                "z-image-turbo-mlx-4bit",
                "z-image-turbo-mlx-q4",
                "z-image-turbo-mlx-2bit",
            ]
        }
        if gib >= 16 {
            return [
                "z-image-turbo-mlx-4bit",
                "z-image-turbo-mlx-q4",
                "z-image-turbo-mlx-2bit",
            ]
        }
        return ["z-image-turbo-mlx-2bit"]
    }

    public var recommendedImageModelSources: [ImageCatalogEntry] {
        let memory = telemetry?.physicalMemoryBytes ?? 16 * 1024 * 1024 * 1024
        let byAlias = Dictionary(imageCatalog.map { ($0.alias, $0) }, uniquingKeysWith: { first, _ in first })
        return Self.recommendedImageModelAliases(physicalMemoryBytes: memory)
            .compactMap { byAlias[$0] }
    }

    /// Every setup surface uses the same tested, memory-ranked download order.
    var imageDownloadChoices: [ImageCatalogEntry] {
        let tested = imageCatalog.filter { Self.testedImageModelAliases.contains($0.alias) }
        let preferred = recommendedImageModelSources.map(\.alias)
        return tested.sorted {
            let left = preferred.firstIndex(of: $0.alias) ?? preferred.count
            let right = preferred.firstIndex(of: $1.alias) ?? preferred.count
            return left == right ? $0.alias < $1.alias : left < right
        }
    }

    /// Downloads a supported MLX source through the catalog install ABI.
    /// The source is packed and verified before it becomes selectable.
    public func installImageModel(_ source: ImageCatalogEntry) {
        guard Self.testedImageModelAliases.contains(source.alias),
              Self.supportsMLXImageModel(modelID: source.modelID) else {
            showToast(
                AppStrings.localized("Only supported MLX image models can be installed in the app."),
                style: .warning)
            return
        }
        guard !imageModels.contains(where: { $0.alias == source.alias }) else {
            if let installed = imageModels.first(where: { $0.alias == source.alias }) {
                selectImageModel(installed)
            }
            return
        }
        enqueueModelDownload(.image(alias: source.alias))
    }

    func startImageModelInstall(alias: String) {
        Self.modelInstallOwner = self
        isInstallingImageModel = true
        installingAlias = alias
        imageInstallAlias = alias
        installStageText = AppStrings.localized("Preparing image source...")
        imageInstallStage = installStageText
        installProgressFraction = nil
        installDownloadedBytes = nil
        installTotalBytes = nil
        installETAText = nil
        imageInstallProgressFraction = nil
        imageInstallTask = Task { [weak self] in
            guard let self else { return }
            defer {
                if Self.modelInstallOwner === self { Self.modelInstallOwner = nil }
                self.imageInstallTask = nil
                self.startNextModelDownloadIfPossible()
            }
            do {
                for try await event in TurboSparkCatalog.installImage(alias) {
                    if self.modelDownloadsShuttingDown {
                        TurboSparkCatalog.cancelInstall()
                        continue
                    }
                    if self.isCancellingModelInstall {
                        if case .finished = event {
                            self.setModelDownloadStatus(.completed)
                            self.refreshModels()
                        } else {
                            TurboSparkCatalog.cancelInstall()
                        }
                        continue
                    }
                    switch event {
                    case let .stage(stage):
                        self.recordModelDownloadStage(stage)
                        self.imageInstallStage = stage
                        self.installStageText = stage
                    case let .bytes(done, total):
                        self.recordModelDownloadProgress(done: done, total: total)
                        if total > 0 {
                            self.imageInstallProgressFraction =
                                min(Double(done) / Double(total), 1.0)
                        }
                    case let .finished(model):
                        self.setModelDownloadStatus(.completed)
                        self.refreshModels()
                        self.selectImageModel(model)
                        self.showToast(
                            AppStrings.localized("Installed image model '\(model.alias)'."),
                            style: .success, duration: 4.0)
                    }
                }
            } catch is CancellationError {
                if !self.modelDownloadsShuttingDown {
                    self.setModelDownloadStatus(.cancelled)
                    self.showToast(AppStrings.localized("Image model install stopped."), style: .info)
                }
            } catch {
                if !self.modelDownloadsShuttingDown && !self.isCancellingModelInstall {
                    self.setModelDownloadStatus(.failed, failure: error.localizedDescription)
                    self.showToast(
                        AppStrings.localized("Image model install failed: \(error.localizedDescription)"),
                        style: .error, duration: 6.0)
                }
            }
            guard !self.modelDownloadsShuttingDown else { return }
            self.finishModelInstallCancellation()
            self.isInstallingImageModel = false
            self.installingAlias = nil
            self.installStageText = nil
            self.installProgressFraction = nil
            self.installDownloadedBytes = nil
            self.installTotalBytes = nil
            self.installETAText = nil
            self.imageInstallAlias = nil
            self.imageInstallStage = nil
            self.imageInstallProgressFraction = nil
            self.imageInstallTask = nil
        }
    }

    /// Cancels the native image install walk and waits for its stream to close.
    /// Dropping the Swift consumer alone would leave the Rust packer writing.
    public func cancelImageInstall() {
        guard isInstallingImageModel, !isCancellingModelInstall else { return }
        isCancellingModelInstall = true
        setModelDownloadStatus(.cancelling)
        installETAText = nil
        imageInstallStage = AppStrings.localized("Stopping image install...")
        installStageText = imageInstallStage
        TurboSparkCatalog.cancelInstall()
    }

    /// Deletes a curated image install. The native catalog validates the
    /// manifest and resolves the path, so the app never removes an arbitrary
    /// folder selected by a user.
    public func deleteImageModel(_ image: ImageInstalledModel) {
        _ = startImageModelDelete(image)
    }

    /// `deleteImageModel`, returning the work it started so a test can wait.
    ///
    /// **THE ENGINE OWNS A ROW THAT CARRIES AN INSTALL ID** (task 6.4): it is
    /// deleted by that id, off the main actor, and the image session is kept
    /// until the engine has said yes (a `busy` answer changes nothing). A folder
    /// found by a scan has no id and is only excluded from the list.
    @discardableResult
    func startImageModelDelete(_ image: ImageInstalledModel) -> Task<Void, Never>? {
        guard !isInstallingImageModel, !generating,
              serverImageAttachedModel?.path != image.path else { return nil }
        let managedByPath = isManagedImageModel(image)
        // A model found in an additional folder is excluded, not deleted: the
        // next scan would find it again, and the bytes are not ours to remove.
        // The same rule the text models follow. Restore is under Settings >
        // Models, in the list of excluded paths.
        guard image.installId != nil || managedByPath else {
            let wasSelected = imageModelPath == image.path
            if imageSessionPath == image.path {
                imageSession?.cancel()
                imageSession = nil
                imageSessionPath = nil
            }
            ModelOrganizationStore.shared.excludeFromScan(path: image.path)
            if wasSelected { imageModelPathText = "" }
            refreshModels()
            showToast(
                AppStrings.localized(
                    "Removed '\(image.alias)' from TurboSpark. The files were left on disk at \(image.path); restore it under Settings > Models."),
                style: .info, duration: 5.0)
            return nil
        }
        let changing = image.installId ?? image.path
        guard !modelsBeingChanged.contains(changing) else { return nil }
        modelsBeingChanged.insert(changing)

        let service = libraryState.service
        let installId = image.installId
        let path = image.path
        return Task { [weak self] in
            let outcome = await Task.detached(priority: .userInitiated) {
                Result {
                    if let installId {
                        try service.deleteImageInstall(installId: installId)
                    } else {
                        // A managed row from an engine that sent no id: the
                        // engine's own validated delete by path (This Mac only).
                        try TurboSparkCatalog.deleteImage(path)
                    }
                }
            }.value
            await self?.finishImageModelDelete(image, changing: changing, outcome: outcome)
        }
    }

    private func finishImageModelDelete(
        _ image: ImageInstalledModel, changing: String, outcome: Result<Void, Error>
    ) async {
        modelsBeingChanged.remove(changing)
        switch outcome {
        case .failure(let error):
            showToast(
                ModelDeleteText.failure(error, alias: image.alias, libraryName: image.libraryName),
                style: .warning, duration: 6.0)
        case .success:
            if imageSessionPath == image.path {
                imageSession?.cancel()
                imageSession = nil
                imageSessionPath = nil
            }
            if imageModelPath == image.path { imageModelPathText = "" }
            refreshModels()
            await libraryState.refresh()
            showToast(AppStrings.localized("Deleted image model '\(image.alias)'."), style: .info)
        }
    }

    public var selectedImageModel: ImageInstalledModel? {
        imageModels.first { $0.path == imageModelPath }
    }

    public var imageSchedulerSteps: UInt32 {
        selectedImageModel?.schedulerSteps ?? 9
    }

    public var imageSizeLabel: String {
        imageResolution.label
    }

    /// Drops the resident image session to reclaim physical memory.
    public func unloadImageModel() {
        guard canUnloadImageModel, serverImageTasks.isEmpty else { return }
        let old = imageSession
        imageSession = nil
        imageSessionPath = nil
        // Cancel, wait for the producer, then release the weights and the
        // MLX buffer cache so the memory actually returns to the system.
        Task { await old?.unload() }
        showToast(AppStrings.localized("Image model unloaded"), style: .info)
    }
}
