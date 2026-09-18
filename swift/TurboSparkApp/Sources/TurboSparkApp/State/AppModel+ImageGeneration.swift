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

    private var canStartImageGeneration: Bool {
        !generating && !submitting && !opening && !isInstallingModel
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
        let options = ImageGenerateOptions(prompt: prompt, seed: UInt64.random(in: 0...UInt64.max))
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
            lastKnownModified: (try? FileManager.default.attributesOfItem(atPath: path.path)[.modificationDate] as? Date)
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
