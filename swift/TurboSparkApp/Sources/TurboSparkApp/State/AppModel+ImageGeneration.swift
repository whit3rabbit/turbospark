import Foundation
import TurboSpark

extension AppModel {
    func imageGenerateOptions(prompt: String, seed: UInt64) -> ImageGenerateOptions {
        ImageGenerateOptions(
            prompt: prompt,
            seed: seed,
            width: imageResolution.width,
            height: imageResolution.height,
            steps: imageSchedulerSteps)
    }

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
        guard let job = imageJob else { return nil }
        let singleProgress: Double
        if job.status == .completed && job.savedPath != nil {
            singleProgress = 1.0
        } else if let stage = job.stage {
            switch stage {
            case "downloading_model", "transformer":
                guard job.total > 0 else { return nil }
                singleProgress = min(1.0, max(0.0, Double(job.completed) / Double(job.total)))
            case "loading_model", "loading_tokenizer", "loading_text_encoder",
                 "loading_transformer", "loading_vae", "text_encoder", "vae_decoder", "png_encode":
                return nil
            default:
                return nil
            }
        } else {
            return nil
        }

        return min(1.0, max(0.0, singleProgress))
    }

    /// Starts a direct-prompt image turn through the supported MLX pipeline.
    /// Image prompts bypass text hooks, tool calls, and text generation.
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
        guard hasSupportedSelectedImageModel else {
            showToast("Select a supported MLX Z-Image model from the curated list.", style: .warning)
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
        let options = imageGenerateOptions(prompt: prompt, seed: seed)
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
        // The draft no longer carries the style text, so a later menu action
        // must not try to strip it. `imageStyleID` stays: re-selecting the
        // same style re-inserts its text into the empty draft.
        appliedImageStyleID = nil
        startImageGeneration(options, chatID: chatID, count: imageCount)
    }

    /// Applies (or with nil clears) the image style and populates the prompt
    /// field with its text.
    ///
    /// The style text is APPENDED after anything the user already wrote,
    /// separated by a comma, so picking a style never discards a subject
    /// description. Switching styles swaps the injected text rather than
    /// stacking copies: the previously applied style's text is stripped first.
    /// The write goes through `writePromptTextDirectly` because style text is
    /// a programmatic insertion of hundreds of characters, which the
    /// large-paste policy would otherwise treat as a pasted attachment.
    public func setImageStyle(id: String?) {
        let current = promptText
        let base: String
        if let appliedID = appliedImageStyleID,
            let applied = AppImageStyleCatalog.style(id: appliedID) {
            base = Self.removingAppliedStyleText(applied.prompt, from: current)
        } else {
            base = current
        }
        let trimmedBase = base.trimmingCharacters(in: .whitespacesAndNewlines)
        if let id, let style = AppImageStyleCatalog.style(id: id) {
            writePromptTextDirectly(
                trimmedBase.isEmpty ? style.prompt : trimmedBase + ", " + style.prompt)
            imageStyleID = id
            appliedImageStyleID = id
        } else {
            if current != base {
                writePromptTextDirectly(trimmedBase)
            }
            imageStyleID = nil
            appliedImageStyleID = nil
        }
    }

    /// Removes a previously injected style text from the draft, returning the
    /// user's own text. Composition always appends the style last, so the
    /// two removable shapes are the exact match and the ", " suffix. Anything
    /// else means the user rewrote the draft: the whole text is theirs.
    static func removingAppliedStyleText(_ styleText: String, from prompt: String) -> String {
        if prompt == styleText { return "" }
        if prompt.hasSuffix(", " + styleText) {
            return String(prompt.dropLast(styleText.count + 2))
        }
        return prompt
    }

    public func regenerateImage() {
        guard let job = imageJob else { return }
        guard !imageModelPath.isEmpty else {
            showToast("Select an image .gturbo install first.", style: .warning)
            return
        }
        guard hasSupportedSelectedImageModel else {
            showToast("Select a supported MLX Z-Image model from the curated list.", style: .warning)
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
        guard hasSupportedSelectedImageModel else {
            showToast("Select a supported MLX Z-Image model from the curated list.", style: .warning)
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

    /// An API test stays in memory until the user adds it to the image gallery.
    @discardableResult
    public func saveAPIImageToGallery(
        png: Data, prompt: String, seed: UInt64, width: UInt32, height: UInt32
    ) -> Bool {
        guard !isInGhostChat else {
            showToast("Saving is unavailable in a temporary chat.", style: .warning)
            return false
        }
        let chatID = selectedChatID
        materializeDraftChatIfNeeded()
        if selectedChatIndex == nil {
            chats.insert(AppChat(id: chatID, projectID: selectedProjectID), at: 0)
        }
        let options = ImageGenerateOptions(prompt: prompt, seed: seed, width: width, height: height)
        let job = AppImageJob(chatID: chatID, options: options)
        do {
            let asset = try ManagedAssetStore.shared.store(
                data: png, fileName: "\(job.id.uuidString).png", mimeType: "image/png")
            registerSavedImage(job: job, asset: asset)
            showToast("Image saved to gallery", style: .success)
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
        guard hasSupportedSelectedImageModel else {
            showToast("Select a supported MLX Z-Image model from the curated list.", style: .warning)
            return
        }
        let requests = ImageGenerationSequence.requests(options: options, count: count)
        imageBatchIndex = 1
        imageBatchCount = requests.count
        let job = AppImageJob(chatID: chatID, options: options)
        imageJob = job
        generating = true
        isCancellationPending = false
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
            var usedSession: (any ImageGenerationSession)?
            do {
                try Task.checkCancellation()
                guard let selected = self.selectedImageModel,
                      Self.supportsMLXImageModel(modelID: selected.modelID) else {
                    throw TurboSparkError(
                        code: .open,
                        message: "Select a supported MLX image model from the curated list.")
                }
                let session = try self.sharedImageSession(for: selected)
                usedSession = session
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
            // A cancelled stream ends the consumer at once, but the detached
            // MLX producer keeps going until its next cancellation point.
            // Hold the permit (and the UI's generating gate) until it has
            // really stopped, or the next job runs on the same pipeline.
            await usedSession?.waitUntilIdle()
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
