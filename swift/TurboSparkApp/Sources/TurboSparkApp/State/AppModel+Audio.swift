import Combine
import Foundation
import TurboSpark

extension AppModel {
    /// The top bar's eject reads `audioWorkspace.canUnloadModel` through
    /// `canUnloadAnyModel` but observes only AppModel, so it stayed dimmed
    /// after an audio model loaded until something else republished.
    /// Forward JUST that flag, not every meter or playback tick, which would
    /// redraw every AppModel view. `objectWillChange` fires before the
    /// property changes, so the flag is read after a main-queue hop.
    func forwardAudioUnloadChanges(from workspace: AudioWorkspaceController) {
        audioUnloadObserver = workspace.objectWillChange
            .receive(on: DispatchQueue.main)
            .compactMap { [weak workspace] in workspace?.canUnloadModel }
            .prepend(workspace.canUnloadModel)
            .removeDuplicates()
            .dropFirst()
            .sink { [weak self] _ in self?.objectWillChange.send() }
    }

    func openAudio(_ page: AudioWorkspacePage) {
        audioWorkspace.selectPage(page)
        activeSection = .audio
    }

    func audioDownload(for profile: AudioProfile) -> ModelDownload? {
        modelDownloads.first { $0.request == .audio(identity: profile.identity) }
    }

    /// Why an audio install cannot be deleted now, or nil. A served model has a
    /// second holder the audio pane cannot see: the server keeps it resident and
    /// keeps answering for it, so removing its files leaves a server serving a
    /// model whose bytes are gone. Text and image installs refuse the same way.
    func audioDeleteBlockedReason(alias: String, path: String?) -> String? {
        let attached = serverSpeechAttachedModel?.alias == alias
            || path.map { path in
                serverAudioModelPaths.values.contains { Self.normalizedModelPath($0) == Self.normalizedModelPath(path) }
            } == true
        return attached
            ? AppStrings.localized("'\(alias)' is attached to the running server. Detach it there first.") : nil
    }

    /// An audio install was removed, forgotten or adopted: Libraries lists it by
    /// modality, so it reads the registry again, as a finished download does.
    func audioInstalledModelsDidChange() {
        refreshLibraryStateInBackground()
    }

    func cancelQueuedAudioDownload(_ download: ModelDownload) {
        guard case .audio = download.request,
              let index = modelDownloads.firstIndex(where: { $0.id == download.id }),
              modelDownloads[index].status == .queued else { return }
        // Queued work has no native transfer handle. Cancelling it must not
        // signal the active transfer, which may belong to a different model.
        modelDownloads[index].status = .cancelled
        modelDownloads[index].updatedAt = Date()
        persistModelDownloads()
    }

    var canSummarizeAudio: Bool {
        session != nil && !opening && !generating && !submitting &&
            imageGenerationTask == nil && titleGenerationTask == nil &&
            pendingToolCall == nil && toolExecutionTask == nil
    }

    /// Meeting notes from a take's transcript, streamed through `progress`.
    /// Writes nothing itself: the caller saves the returned summary into
    /// `item.summary` and nowhere else, so transcripts stay ground truth.
    func summarizeAudioTranscript(_ item: AudioLibraryItem, targetModel: InstalledModel? = nil,
                                  progress: @escaping @MainActor (AudioSummaryProgress) -> Void = { _ in }) async throws -> AudioMeetingSummary {
        if let targetModel, loadedModelPath != targetModel.path {
            await open(targetModel)
        }
        guard canSummarizeAudio, let session, let transcript = item.preferredTranscript else {
            throw AudioWorkspaceError.message(AppStrings.localized("Load a local chat model to summarize this transcript."))
        }
        audioSummaryInFlight = true; generating = true
        let modelAlias = targetModel?.alias ?? selected?.alias
        defer { audioSummaryInFlight = false; generating = false }
        // The take's speaker names, so the summary can attribute owners.
        let chunks = AudioMeetingSummary.chunks(AudioSpeakerRoster(item).labeled(transcript.segments))
        guard !chunks.isEmpty else { throw AudioWorkspaceError.message(AppStrings.localized("The local model returned an empty summary. Try again.")) }
        let notes = item.userNotes

        func run(_ prompt: (system: String, user: String), streams: Bool) async throws -> String {
            var options = GenerateOptions()
            options.reasoning = .off; options.temperature = 0.2; options.maxNewTokens = 1024
            var text = ""
            for try await event in session.generate([.system(prompt.system), .user(prompt.user)], options: options) {
                try Task.checkCancellation()
                if case .content(let delta) = event {
                    text += delta
                    if streams { progress(AudioSummaryProgress(stage: nil, text: text)) }
                }
            }
            return text.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        let final: String
        if chunks.count == 1 {
            final = try await run(AudioMeetingNotesPrompt.single(transcript: chunks[0].text, userNotes: notes), streams: true)
        } else {
            // Map: terse notes per part. Reduce: merge level by level; only
            // the last merge writes the final format and streams to the view.
            var parts: [String] = []
            for (index, chunk) in chunks.enumerated() {
                try Task.checkCancellation()
                progress(AudioSummaryProgress(stage: AppStrings.localized("Reading part \(index + 1) of \(chunks.count)"), text: ""))
                let part = try await run(AudioMeetingNotesPrompt.part(transcript: chunk.text, start: chunk.start, index: index, count: chunks.count), streams: false)
                if !part.isEmpty { parts.append(part) }
            }
            var wroteFinal = false
            while parts.count > 1 {
                let isFinal = parts.count <= AudioMeetingNotesPrompt.mergeFanIn
                if !isFinal { progress(AudioSummaryProgress(stage: AppStrings.localized("Combining notes"), text: "")) }
                var merged: [String] = []
                for start in stride(from: 0, to: parts.count, by: AudioMeetingNotesPrompt.mergeFanIn) {
                    let group = Array(parts[start..<min(start + AudioMeetingNotesPrompt.mergeFanIn, parts.count)])
                    merged.append(try await run(AudioMeetingNotesPrompt.merge(parts: group, userNotes: notes, final: isFinal), streams: isFinal))
                }
                parts = merged.filter { !$0.isEmpty }
                if isFinal { wroteFinal = true; break }
            }
            // Only one part had anything: it still needs the final format.
            if !wroteFinal, let only = parts.first {
                parts = [try await run(AudioMeetingNotesPrompt.merge(parts: [only], userNotes: notes, final: true), streams: true)]
            }
            final = parts.first ?? ""
        }
        guard !final.isEmpty else { throw AudioWorkspaceError.message(AppStrings.localized("The local model returned an empty summary. Try again.")) }
        return AudioMeetingSummary(text: final, sourceTimes: chunks.map(\.start), transcriptRevisionID: transcript.id, modelAlias: modelAlias)
    }

    /// One answer from the chat model for an Audio feature (Ask). Gated and
    /// flagged like meeting notes, so chat cannot start a turn underneath it
    /// and Stop cancels it.
    func runAudioPrompt(system: String, user: String, targetModel: InstalledModel?,
                        progress: @escaping @MainActor (String) -> Void) async throws -> String {
        if let targetModel, loadedModelPath != targetModel.path {
            await open(targetModel)
        }
        guard canSummarizeAudio, let session else {
            throw AudioWorkspaceError.message(AppStrings.localized("Load a local chat model to ask about this meeting."))
        }
        audioSummaryInFlight = true; generating = true
        defer { audioSummaryInFlight = false; generating = false }
        var options = GenerateOptions()
        options.reasoning = .off; options.temperature = 0.2; options.maxNewTokens = 768
        var text = ""
        for try await event in session.generate([.system(system), .user(user)], options: options) {
            try Task.checkCancellation()
            if case .content(let delta) = event { text += delta; progress(text) }
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
