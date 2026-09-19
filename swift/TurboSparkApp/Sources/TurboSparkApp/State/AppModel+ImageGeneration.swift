import Foundation
import TurboSpark

extension AppModel {
    public var savedImageArtifacts: [AppArtifact] {
        chats
            .flatMap(\.artifacts)
            .filter { $0.origin == .imageGeneration }
            .sorted { $0.createdAt > $1.createdAt }
    }

    var canStartImageGeneration: Bool {
        !generating && !submitting && !opening && !isInstallingModel
            && !isInstallingImageModel
            && pendingToolCall == nil && imageGenerationTask == nil
            && !(imageJob?.result != nil && imageJob?.savedPath == nil)
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
        startImageGeneration(options, chatID: chatID, count: imageCount)
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
        isCancellationPending = true
        imageSession?.cancel()
        imageGenerationTask?.cancel()
    }

    @discardableResult
    public func saveImage() -> Bool {
        guard var job = imageJob, let result = job.result else { return false }
        guard job.savedPath == nil else { return true }
        guard chats.contains(where: { $0.id == job.chatID }) else { return false }
        do {
            let asset = try ManagedAssetStore.shared.store(
                data: result.png,
                fileName: "\(job.id.uuidString).png",
                mimeType: "image/png")
            job.savedPath = asset.storedReference
            imageJob = job
            registerSavedImage(job: job, asset: asset)
            return true
        } catch {
            showToast("Could not save image: \(error.localizedDescription)", style: .error)
            return false
        }
    }

    private func startImageGeneration(
        _ options: ImageGenerateOptions, chatID: UUID, count: Int = 1
    ) {
        guard imageGenerationTask == nil else { return }
        let requests = ImageGenerationSequence.requests(options: options, count: count)
        imageBatchIndex = 1
        imageBatchCount = requests.count
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
                try await ImageGenerationSequence.run(requests) { index, request in
                    self.imageBatchIndex = index + 1
                    self.imageJob = AppImageJob(
                        chatID: chatID, options: request, status: .generating)
                    var saved = false
                    for try await event in session.generate(request) {
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
                            // Save before advancing so a later cancellation cannot lose this output.
                            guard self.saveImage() else {
                                throw TurboSparkError(code: .generate, message: "Could not save image")
                            }
                            saved = true
                        case .cancelled:
                            throw CancellationError()
                        }
                    }
                    guard saved else {
                        throw TurboSparkError(code: .generate, message: "No image was returned")
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

    private func registerSavedImage(job: AppImageJob, asset: ManagedAssetDescriptor) {
        let now = Date()
        guard let index = chats.firstIndex(where: { $0.id == job.chatID }) else { return }
        let artifact = AppArtifact(
            chatID: job.chatID,
            path: asset.storedReference,
            title: "Generated image",
            origin: .imageGeneration,
            createdAt: now,
            updatedAt: now,
            lastKnownByteSize: Int(asset.byteCount),
            lastKnownModified: now,
            imageRequest: AppImageRequest(options: job.options)
        )
        AppArtifact.upsert(artifact, into: &chats[index].artifacts)
        let assistant = AppChatMessage(
            role: .assistant,
            content: "Generated image, seed \(job.options.seed).",
            imagePaths: [asset.storedReference])
        chats[index].messages.append(assistant)
        chats[index].updatedAt = now
        persistChats()
    }
}
