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
            try TurboSparkCatalog.deleteImage(image.alias)
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

    public var savedImageArtifacts: [AppArtifact] {
        chats
            .flatMap(\.artifacts)
            .filter { $0.origin == .imageGeneration && $0.existsOnDisk }
            .sorted { $0.createdAt > $1.createdAt }
    }

    private var canStartImageGeneration: Bool {
        !generating && !submitting && !opening && !isInstallingModel
            && !isInstallingImageModel
            && pendingToolCall == nil && imageGenerationTask == nil
    }

    public var imageProgressFraction: Double? {
        guard let job = imageJob, job.total > 0 else { return nil }
        return min(1, Double(job.completed) / Double(job.total))
    }

    /// Starts a direct-prompt image turn. Image prompts intentionally bypass
    /// text hooks, tool calls, and the text transcript pipeline: the native
    /// image runtime has its own verified request envelope.
    public func generateImage() {
        let prompt = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            showToast("Enter an image prompt first.", style: .warning)
            return
        }
        guard !imageModelPath.isEmpty else {
            showToast("Select an image .gturbo install first.", style: .warning)
            return
        }
        guard !isInGhostChat else {
            showToast("Image generation is unavailable in a temporary chat.", style: .warning)
            return
        }
        guard canStartImageGeneration else { return }
        let seedText = imageSeedText.trimmingCharacters(in: .whitespacesAndNewlines)
        let seed: UInt64
        if seedText.isEmpty {
            seed = UInt64.random(in: 0...UInt64.max)
        } else if let parsed = UInt64(seedText) {
            seed = parsed
        } else {
            showToast("Seed must be an unsigned integer or blank for random.", style: .warning)
            return
        }
        guard let size = imageSupportedSize else {
            showToast("Select an image model with a supported size first.", style: .warning)
            return
        }
        let options = ImageGenerateOptions(
            prompt: prompt,
            seed: seed,
            width: size.width,
            height: size.height,
            steps: imageSchedulerSteps)
        let chatID = selectedChatID
        materializeDraftChatIfNeeded()
        if selectedChatIndex == nil {
            chats.insert(AppChat(id: chatID, projectID: selectedProjectID), at: 0)
        }
        if let index = chats.firstIndex(where: { $0.id == chatID }) {
            chats[index].messages.append(AppChatMessage(role: .user, content: prompt))
            chats[index].draft = ""
            chats[index].updatedAt = Date()
            persistChats()
        }
        startImageGeneration(options, chatID: chatID)
    }

    public func regenerateImage() {
        guard let job = imageJob else { return }
        guard !imageModelPath.isEmpty else {
            showToast("Select an image .gturbo install first.", style: .warning)
            return
        }
        guard canStartImageGeneration else { return }
        startImageGeneration(job.options, chatID: job.chatID)
    }

    public func regenerateImage(from artifact: AppArtifact) {
        guard let request = artifact.imageRequest else {
            showToast("This saved image has no request metadata to regenerate.", style: .warning)
            return
        }
        guard !imageModelPath.isEmpty else {
            showToast("Select an image .gturbo install first.", style: .warning)
            return
        }
        guard canStartImageGeneration else { return }
        startImageGeneration(request.options, chatID: artifact.chatID)
    }

    public func cancelImageGeneration() {
        imageSession?.cancel()
        imageGenerationTask?.cancel()
    }

    public func saveImage() {
        guard var job = imageJob, let result = job.result else { return }
        guard job.savedPath == nil else { return }
        let directory = AppStorageRoot.subdirectory("image-artifacts")
        let url = directory.appendingPathComponent("\(job.id.uuidString).png")
        do {
            try result.png.write(to: url, options: .atomic)
            job.savedPath = url.standardizedFileURL.path
            imageJob = job
            registerSavedImage(job: job, path: url)
            showToast("Image saved to this profile.", style: .success)
        } catch {
            showToast("Could not save image: \(error.localizedDescription)", style: .error)
        }
    }

    private func startImageGeneration(_ options: ImageGenerateOptions, chatID: UUID) {
        guard imageGenerationTask == nil else { return }
        let job = AppImageJob(chatID: chatID, options: options)
        imageJob = job
        generating = true
        isCancellationPending = false
        let modelPath = imageModelPath
        imageGenerationTask = Task { [weak self] in
            let coordinator = ImageJobCoordinator.shared
            let acquired = await coordinator.acquire()
            guard let self else {
                if acquired { await coordinator.release() }
                return
            }
            guard acquired else {
                if var current = self.imageJob {
                    current.status = .cancelled
                    self.imageJob = current
                }
                self.generating = false
                self.isCancellationPending = false
                self.imageGenerationTask = nil
                return
            }
            do {
                try Task.checkCancellation()
                if self.imageSession == nil || self.imageSessionPath != modelPath {
                    self.imageSession = try await TurboSparkImageSession(modelPath: modelPath)
                    self.imageSessionPath = modelPath
                }
                guard let session = self.imageSession else {
                    throw TurboSparkError(code: .open, message: "image session did not open")
                }
                if var current = self.imageJob {
                    current.status = .generating
                    self.imageJob = current
                }
                for try await event in session.generate(options) {
                    try Task.checkCancellation()
                    switch event {
                    case let .stage(name, completed, total):
                        guard var current = self.imageJob else { continue }
                        current.stage = name
                        current.completed = completed
                        current.total = total
                        self.imageJob = current
                    case let .finished(result):
                        guard var current = self.imageJob else { continue }
                        current.status = .completed
                        current.result = result
                        self.imageJob = current
                    case .cancelled:
                        if var current = self.imageJob {
                            current.status = .cancelled
                            self.imageJob = current
                        }
                    }
                }
            } catch is CancellationError {
                if var current = self.imageJob {
                    current.status = .cancelled
                    self.imageJob = current
                }
            } catch {
                if var current = self.imageJob {
                    current.status = .failed
                    self.imageJob = current
                }
                self.showToast("Image generation failed: \(error.localizedDescription)", style: .error)
            }
            self.generating = false
            self.isCancellationPending = false
            self.imageGenerationTask = nil
            await coordinator.release()
        }
    }

    private func registerSavedImage(job: AppImageJob, path: URL) {
        let now = Date()
        guard let index = chats.firstIndex(where: { $0.id == job.chatID }) else { return }
        let artifact = AppArtifact(
            chatID: job.chatID,
            path: path.path,
            title: "Generated image",
            origin: .imageGeneration,
            createdAt: now,
            updatedAt: now,
            lastKnownByteSize: (try? FileManager.default.attributesOfItem(atPath: path.path)[.size] as? Int),
            lastKnownModified: (try? FileManager.default.attributesOfItem(atPath: path.path)[.modificationDate] as? Date),
            imageRequest: AppImageRequest(options: job.options)
        )
        let row = AppArtifact.upsert(artifact, into: &chats[index].artifacts)
        let storedPath = "image-artifacts/\(job.id.uuidString).png"
        let assistant = AppChatMessage(
            role: .assistant,
            content: "Generated image, seed \(job.options.seed).",
            imagePaths: [storedPath])
        chats[index].messages.append(assistant)
        chats[index].updatedAt = now
        persistChats()
        maybeAutoOpenArtifact(row)
    }
}
