import Foundation
import TurboSpark

@MainActor
struct AudioRecordingSetupDependencies {
    var requestMicrophonePermission: () async -> Bool = { await AudioCapture.requestMicrophonePermission() }
    var microphones: () throws -> [AudioCaptureMicrophone] = {
        try AudioCapture.microphones().map { AudioCaptureMicrophone(id: String($0.id), name: $0.name) }
    }
    var applications: () async throws -> [AudioCaptureApplication] = {
        try await AudioCapture.applications().map { AudioCaptureApplication(id: $0.id, name: $0.name) }
    }
    var start: (AudioCapture, AudioCapture.Configuration) async throws -> Void = { capture, configuration in
        try await capture.start(configuration: configuration)
    }
}

extension AudioWorkspaceController {
    private var microphoneSetupMessage: String {
        String(localized: "Allow microphone access in the macOS prompt to continue.", bundle: .module)
    }
    private var systemSetupMessage: String {
        String(localized: "Allow screen and system audio recording in System Settings if prompted.", bundle: .module)
    }
    private var setupCancelledMessage: String {
        String(localized: "Recording setup cancelled. Start again when you are ready.", bundle: .module)
    }
    private func showRecordingSetup(_ message: String) {
        recordingSetupStatus = message
        status = message
    }
    private func isCurrentRecordingSetup(_ attempt: UUID, token: UUID) -> Bool {
        valid(token) && recordingSetupAttempt == attempt && !Task.isCancelled
    }
    private func finishRecordingSetup(_ attempt: UUID) {
        guard recordingSetupAttempt == attempt else { return }
        recordingSetupAttempt = nil; recordingSetupStatus = nil; recordingSetupTask = nil
    }

    func prepareRecording() {
        guard !hasRecordingActivity, !isShutDown else { return }
        guard captureMicrophone || captureSystemAudio else {
            error = AudioCapture.CaptureError.noSource.localizedDescription
            return
        }
        let token = epoch; let attempt = UUID()
        let checksMicrophone = captureMicrophone; let checksSystemAudio = captureSystemAudio
        let dependencies = recordingSetupDependencies
        recordingSetupAttempt = attempt; error = nil
        showRecordingSetup(checksMicrophone ? microphoneSetupMessage : systemSetupMessage)
        recordingSetupTask = Task { [weak self] in
            guard let self else { return }
            defer { self.finishRecordingSetup(attempt) }
            do {
                if checksMicrophone {
                    let permitted = await dependencies.requestMicrophonePermission()
                    guard self.isCurrentRecordingSetup(attempt, token: token) else { return }
                    guard permitted else { throw AudioCapture.CaptureError.permission }
                    self.microphones = try dependencies.microphones()
                }
                if checksSystemAudio {
                    self.showRecordingSetup(self.systemSetupMessage)
                    let applications = try await dependencies.applications()
                    guard self.isCurrentRecordingSetup(attempt, token: token) else { return }
                    self.audioApplications = applications
                }
                self.status = String(localized: "Recording devices are ready.", bundle: .module)
            } catch {
                guard self.isCurrentRecordingSetup(attempt, token: token) else { return }
                self.error = error.localizedDescription; self.status = error.localizedDescription
            }
        }
    }

    func cancelRecordingSetup() {
        guard recordingSetupAttempt != nil else { return }
        if capture != nil { stopRecording(interrupted: true); return }
        recordingSetupTask?.cancel(); recordingSetupTask = nil
        recordingSetupAttempt = nil; recordingSetupStatus = nil
        status = setupCancelledMessage
    }

    func startRecording() {
        guard !hasRecordingActivity, !isShutDown, capture == nil else { return }
        guard captureMicrophone || captureSystemAudio else {
            error = AudioCapture.CaptureError.noSource.localizedDescription
            return
        }
        stopPlayback(); error = nil
        var item = AudioLibraryItem(title: String(localized: "Meeting", bundle: .module) + " " + Date().formatted(date: .abbreviated, time: .shortened), kind: .recording)
        item.status = .draft; item.recipe = recipe; item.recipe.task = AudioTask.speechToText.rawValue
        if let profile = selectedProfile, profile.identity.task == .speechToText {
            item.modelFamily = profile.family
            item.localModelBookmark = preferences.localModels[profile.identity.alias]
            if item.localModelBookmark == nil { item.modelIdentity = try? JSONEncoder().encode(profile.identity) }
        }
        do { try save(item) } catch { fail(error); return }
        recordingID = item.id; selectedID = item.id; draftThrough = 0; draftRequestedThrough = 0
        nextDraftAttemptAt = .distantPast
        let id = item.id; let token = epoch; let library = self.library; let assets = self.assets; let profileID = self.profileID
        let capture = AudioCapture(onChunk: { [weak self] chunk in
            guard ProfileVaultStore.shared.session?.profileID == profileID else { throw CancellationError() }
            let asset = try assets.store(data: chunk.wav, fileName: "recording-\(chunk.source)-\(UUID().uuidString).wav", mimeType: "audio/wav")
            let saved = try library.update(id) {
                $0.clips.append(AudioLibraryClip(assetReference: asset.storedReference, sampleRate: chunk.sampleRate, channels: chunk.channels, frameCount: chunk.frameCount, offset: chunk.offset, source: chunk.source))
                $0.updatedAt = Date()
            }
            Task { @MainActor in
                guard let self, self.publishCaptureUpdate(id, token: token) else { return }
                self.draftRequestedThrough = saved.duration
                self.transcriptBacklog = max(0, Int((saved.duration - self.draftThrough) / 15))
                self.startNextDraft()
            }
        }, onEvent: { [weak self] event in
            Task { @MainActor in
                guard let self, self.valid(token), self.recordingID == id else { return }
                switch event {
                case .level(_, let value): self.inputLevel = value
                case .failure(let message), .interruption(let message):
                    self.error = message
                    self.stopRecording(interrupted: true)
                }
            }
        })
        self.capture = capture
        let config = AudioCapture.Configuration(microphoneID: microphoneID.flatMap(UInt32.init), capturesMicrophone: captureMicrophone, systemSource: captureSystemAudio ? selectedApplicationID.map(AudioCapture.SystemSource.application) ?? .all : .none)
        let attempt = UUID(); let dependencies = recordingSetupDependencies
        recordingSetupAttempt = attempt
        recordingSeconds = 0; isPaused = false; inputLevel = 0
        showRecordingSetup(config.capturesMicrophone ? microphoneSetupMessage : systemSetupMessage)
        recordingSetupTask = Task { [weak self] in
            guard let self else { return }
            defer { self.finishRecordingSetup(attempt) }
            do {
                if config.capturesMicrophone {
                    let permitted = await dependencies.requestMicrophonePermission()
                    guard self.isCurrentRecordingSetup(attempt, token: token), self.capture === capture else { return }
                    guard permitted else { throw AudioCapture.CaptureError.permission }
                }
                guard self.isCurrentRecordingSetup(attempt, token: token), self.capture === capture else { return }
                self.showRecordingSetup(config.systemSource == .none
                    ? String(localized: "Preparing recording...", bundle: .module) : self.systemSetupMessage)
                try await dependencies.start(capture, config)
                guard self.isCurrentRecordingSetup(attempt, token: token), self.capture === capture, self.recordingID == id else { _ = await capture.stop(); return }
                self.isRecording = true; self.isPaused = false
                self.update(id) { $0.status = .recording }
                self.status = String(localized: "Recording and autosaving", bundle: .module)
            } catch {
                guard self.isCurrentRecordingSetup(attempt, token: token), self.capture === capture, self.recordingID == id else { _ = await capture.stop(); return }
                _ = capture.stopAcceptingAndDrain()
                Task { _ = await capture.stop() }
                self.isRecording = false; self.capture = nil; self.recordingID = nil
                self.update(id) { $0.status = .interrupted; $0.failure = error.localizedDescription }
                self.error = error.localizedDescription; self.status = error.localizedDescription
            }
        }
    }
    @discardableResult func publishCaptureUpdate(_ id: UUID, token: UUID) -> Bool {
        guard valid(token), recordingID == id else { return false }
        // Stop and transcript edits may have committed after the chunk callback.
        do { if let latest = try library.item(id) { publish(latest) } }
        catch { fail(error); return false }
        return true
    }
    func pauseRecording() { guard isRecording else { return }; capture?.pause(); isPaused = true }
    func resumeRecording() { guard isRecording else { return }; capture?.resume(); isPaused = false }
    func addMarker() {
        guard isRecording, let id = recordingID, let capture else { return }
        let seconds = capture.currentElapsed()
        update(id) { $0.markers.append(AudioRecordingMarker(time: seconds, title: String(localized: "Marker", bundle: .module) + " \($0.markers.count + 1)")) }
    }
    func stopRecording() { stopRecording(interrupted: false) }
    func stopRecording(interrupted: Bool) {
        guard let capture else { cancelRecordingSetup(); return }
        let id = recordingID; let token = epoch
        let started = isRecording
        recordingSetupTask?.cancel(); recordingSetupTask = nil
        recordingSetupAttempt = nil; recordingSetupStatus = nil
        recordingSeconds = capture.stopAcceptingAndDrain()
        let captureFailure = capture.failureMessage
        var interrupted = interrupted || captureFailure != nil || !started
        var failure = captureFailure ?? (started ? nil : setupCancelledMessage)
        if let captureFailure { error = captureFailure }
        self.capture = nil; recordingID = nil; isRecording = false; isPaused = false; inputLevel = 0
        if let id {
            update(id) {
                if !$0.clips.contains(where: { $0.frameCount > 0 }), started {
                    interrupted = true
                    failure = failure ?? String(localized: "Recording did not capture audio. Check the source and try again.", bundle: .module)
                }
                $0.status = interrupted ? .interrupted : .completed; $0.failure = failure; $0.updatedAt = Date()
            }
            if liveTranscription && !interrupted { pendingRefinementIDs.append(id) }
        }
        if let failure { status = failure }
        Task { _ = await capture.stop() }
        guard valid(token) else { return }
        if isBusy { if isDraftJob { cancel() } } else { startNextDraft() }
    }
    func startNextDraft() {
        guard !isBusy, !isShutDown, Date() >= nextDraftAttemptAt else { return }
        if !pendingRefinementIDs.isEmpty {
            let id = pendingRefinementIDs.removeFirst()
            refineRecording(id)
            return
        }
        guard isRecording, liveTranscription, draftRequestedThrough - draftThrough >= 15,
              let id = recordingID, let item = items.first(where: { $0.id == id }),
              let profile = profiles.first(where: { $0.identity.alias == item.recipe.modelID && $0.identity.task == .speechToText }),
              profile.capabilities.canRun,
              installedPaths[profile.identity.alias] != nil || preferences.localModels[profile.identity.alias] != nil else { return }
        let offset = max(0, draftThrough - 1)
        let duration = min(30, draftRequestedThrough - offset)
        let through = offset + duration; let token = epoch; let assets = self.assets; let lifetime = self.lifetime
        isBusy = true; isDraftJob = true
        jobTask = Task { [weak self] in
            guard let self else { return }
            let admitted = await ImageJobCoordinator.shared.acquire()
            defer { if admitted { Task { await ImageJobCoordinator.shared.release() } } }
            defer {
                if self.valid(token) { self.isBusy = false; self.isDraftJob = false; self.scheduleUnload(); self.startNextDraft() }
            }
            do {
                guard admitted else { throw CancellationError() }
                let session = try await self.residentSession(for: profile, source: item)
                let pcm = try await Task.detached { try AudioMediaIO.readMono16k(item.clips, store: assets, start: offset, duration: duration, cancel: lifetime.check) }.value
                let result = try await session.execute(self.request(from: item.recipe, task: .speechToText), pcm: pcm)
                try Task.checkCancellation()
                guard self.valid(token) else { return }
                if let transcript = result.transcript {
                    let native = try JSONEncoder().encode(transcript)
                    self.publish(try self.library.update(id) { current in
                        var segments = current.transcripts.last(where: { $0.source == .draft })?.segments ?? []
                        var next = Self.librarySegments(transcript, offset: offset)
                        Self.reconcileBoundary(previous: &segments, next: &next)
                        segments.append(contentsOf: next)
                        current.addTranscript(segments, source: .draft, language: transcript.language, nativeResult: native)
                    })
                }
                self.draftThrough = through
                self.transcriptBacklog = max(0, Int((self.draftRequestedThrough - through) / 15))
            } catch {
                guard self.valid(token) else { return }
                if !Task.isCancelled && Self.isDeviceBusy(error) {
                    // Capture keeps writing while another workload owns the device.
                    self.nextDraftAttemptAt = Date().addingTimeInterval(2)
                    self.status = error.localizedDescription
                } else if !Task.isCancelled {
                    self.error = error.localizedDescription
                    // Do not retry a failing model forever while capture is active.
                    self.liveTranscription = false
                }
            }
        }
    }
    func refineRecording(_ id: UUID) {
        guard let item = items.first(where: { $0.id == id }), !item.clips.isEmpty,
              let profile = profiles.first(where: { $0.identity.alias == item.recipe.modelID && $0.identity.task == .speechToText }),
              profile.capabilities.canRun,
              installedPaths[profile.identity.alias] != nil || preferences.localModels[profile.identity.alias] != nil else { return }
        let token = epoch
        isBusy = true
        update(id) { $0.status = .processing }
        jobTask = Task { [weak self] in
            guard let self else { return }
            let admitted = await ImageJobCoordinator.shared.acquire()
            defer { if admitted { Task { await ImageJobCoordinator.shared.release() } } }
            defer { if self.valid(token) { self.isBusy = false; self.progress = nil; self.scheduleUnload() } }
            do {
                guard admitted else { throw CancellationError() }
                let session = try await self.residentSession(for: profile, source: item)
                try await self.transcribe(item: item, into: id, session: session, request: self.request(from: item.recipe, task: .speechToText), source: .final, token: token)
                guard self.valid(token) else { return }
                self.publish(try self.library.update(id) { $0.status = .completed; $0.updatedAt = Date() })
                self.transcriptBacklog = 0
            } catch {
                guard self.valid(token) else { return }
                if !Task.isCancelled && Self.isDeviceBusy(error) {
                    self.pendingRefinementIDs.insert(id, at: 0)
                    self.nextDraftAttemptAt = Date().addingTimeInterval(2)
                    self.status = error.localizedDescription
                } else {
                    self.update(id) { $0.status = .interrupted; $0.failure = error.localizedDescription }
                    if !Task.isCancelled { self.error = error.localizedDescription }
                }
            }
        }
    }
}
