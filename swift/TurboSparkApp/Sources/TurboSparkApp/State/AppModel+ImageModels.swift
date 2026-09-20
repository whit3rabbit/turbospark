import Foundation
import TurboSpark

extension AppModel {
    /// Image installs stay separate from the text catalog. The path remains a
    /// string because a user may still choose a valid side-loaded install.
    public var imageModelPath: String {
        let path = imageModelPathText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !path.isEmpty { return (path as NSString).expandingTildeInPath }
        return ""
    }

    public var canGenerateImage: Bool {
        !imageModelPath.isEmpty
            && !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && canStartImageGeneration
    }

    public func selectImageModel(_ model: ImageInstalledModel) {
        imageModelPathText = model.path
    }

    /// True when at least one curated Z-Image install is available locally.
    /// Side-loaded folders remain selectable by path, but do not count as a
    /// curated install until the native image catalog can validate them.
    public var hasInstalledZImageModel: Bool {
        imageModels.contains {
            $0.alias.lowercased().contains("z-image")
                || $0.modelID.lowercased().contains("z-image")
        }
    }

    /// The MLX variants whose pinned install gates have passed. FP16 is kept
    /// in the catalog for explicit users, but is not a first-run suggestion
    /// while its install gate remains open.
    public static let testedZImageAliases = [
        "z-image-turbo",
        "z-image-turbo-mlx-2bit",
        "z-image-turbo-mlx-4bit",
        "z-image-turbo-mlx-8bit",
    ]

    /// Returns one or more curated MLX choices appropriate for this machine.
    /// The ordering is intentional: the first row is the default suggestion,
    /// and the remaining rows give users a useful quality/footprint choice.
    public static func recommendedZImageAliases(physicalMemoryBytes: UInt64) -> [String] {
        let gib = physicalMemoryBytes / (1024 * 1024 * 1024)
        if gib >= 32 {
            return [
                "z-image-turbo-mlx-8bit",
                "z-image-turbo-mlx-4bit",
                "z-image-turbo-mlx-2bit",
            ]
        }
        if gib >= 16 {
            return ["z-image-turbo-mlx-4bit", "z-image-turbo-mlx-2bit"]
        }
        return ["z-image-turbo-mlx-2bit"]
    }

    public var recommendedZImageSources: [ImageCatalogEntry] {
        let memory = telemetry?.physicalMemoryBytes ?? 16 * 1024 * 1024 * 1024
        let byAlias = Dictionary(imageCatalog.map { ($0.alias, $0) }, uniquingKeysWith: { first, _ in first })
        return Self.recommendedZImageAliases(physicalMemoryBytes: memory)
            .compactMap { byAlias[$0] }
    }

    /// Every setup surface uses the same tested, memory-ranked download order.
    var imageDownloadChoices: [ImageCatalogEntry] {
        let tested = imageCatalog.filter { Self.testedZImageAliases.contains($0.alias) }
        let preferred = recommendedZImageSources.map(\.alias)
        return tested.sorted {
            let left = preferred.firstIndex(of: $0.alias) ?? preferred.count
            let right = preferred.firstIndex(of: $1.alias) ?? preferred.count
            return left == right ? $0.alias < $1.alias : left < right
        }
    }

    /// Downloads a curated image source through the native image-install ABI.
    /// The source is packed and verified before it becomes selectable.
    public func installImageModel(_ source: ImageCatalogEntry) {
        guard !isInstallingImageModel, !isInstallingModel, !generating else { return }
        guard !imageModels.contains(where: { $0.alias == source.alias }) else {
            if let installed = imageModels.first(where: { $0.alias == source.alias }) {
                selectImageModel(installed)
            }
            return
        }
        isInstallingImageModel = true
        imageInstallAlias = source.alias
        imageInstallStage = "Preparing image source..."
        imageInstallProgressFraction = nil
        imageInstallTask = Task { [weak self] in
            guard let self else { return }
            do {
                for try await event in TurboSparkCatalog.installImage(source.alias) {
                    switch event {
                    case let .stage(stage):
                        self.imageInstallStage = stage
                    case let .bytes(done, total):
                        if total > 0 {
                            self.imageInstallProgressFraction =
                                min(Double(done) / Double(total), 1.0)
                        }
                    case let .finished(model):
                        self.refreshModels()
                        self.selectImageModel(model)
                        self.showToast(
                            "Installed image model '\(model.alias)'.",
                            style: .success, duration: 4.0)
                    }
                }
            } catch is CancellationError {
                self.showToast("Image model install stopped.", style: .info)
            } catch {
                self.showToast(
                    "Image model install failed: \(error.localizedDescription)",
                    style: .error, duration: 6.0)
            }
            self.isInstallingImageModel = false
            self.imageInstallAlias = nil
            self.imageInstallStage = nil
            self.imageInstallProgressFraction = nil
            self.imageInstallTask = nil
        }
    }

    /// Cancels the native image install walk and waits for its stream to close.
    /// Dropping the Swift consumer alone would leave the Rust packer writing.
    public func cancelImageInstall() {
        guard isInstallingImageModel else { return }
        if TurboSparkCatalog.cancelInstall() {
            imageInstallStage = "Stopping image install..."
        }
    }

    /// Deletes a curated image install. The native catalog validates the
    /// manifest and resolves the path, so the app never removes an arbitrary
    /// folder selected by a user.
    public func deleteImageModel(_ image: ImageInstalledModel) {
        guard !isInstallingImageModel, !generating else { return }
        let wasSelected = imageModelPath == image.path
        if imageSessionPath == image.path {
            imageSession?.cancel()
            imageSession = nil
            imageSessionPath = nil
        }
        do {
            try TurboSparkCatalog.deleteImage(image.path)
            if wasSelected {
                imageModelPathText = ""
            }
            refreshModels()
            showToast("Deleted image model '\(image.alias)'.", style: .info)
        } catch {
            showToast(
                "Could not delete image model: \(error.localizedDescription)",
                style: .error, duration: 6.0)
        }
    }

    public var selectedImageModel: ImageInstalledModel? {
        imageModels.first { $0.path == imageModelPath }
    }

    /// Sizes are owned by the selected install. A side-loaded current
    /// Z-Image install uses the same single supported envelope.
    public var imageSupportedSize: (width: UInt32, height: UInt32)? {
        if let selectedImageModel {
            return (selectedImageModel.width, selectedImageModel.height)
        }
        return imageModelPath.isEmpty ? nil : (1024, 1024)
    }

    public var imageSchedulerSteps: UInt32 {
        selectedImageModel?.schedulerSteps ?? 9
    }

    public var imageSizeLabel: String {
        guard let size = imageSupportedSize else { return "Select an image model" }
        return "\(size.width) x \(size.height)"
    }

}
