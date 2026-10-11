import AppKit
import AVFoundation
import Foundation
import UniformTypeIdentifiers

extension AudioWorkspaceController {
    func scratchURL(extension suffix: String) throws -> URL {
        guard let profileID, !isShutDown else { throw CancellationError() }
        let directory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            .appendingPathComponent("TurboSpark/\(profileID)/decrypted/audio-workspace", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
        let url = directory.appendingPathComponent(UUID().uuidString).appendingPathExtension(suffix)
        scratchURLs.insert(url)
        return url
    }
    func importAudio(asVoiceReference: Bool = false) {
        guard importTask == nil, !isShutDown, !hasRecordingActivity else { return }
        let token = epoch
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.audio]; panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let source = panel.url, valid(token), !hasRecordingActivity else { return }
        var item = AudioLibraryItem(title: source.deletingPathExtension().lastPathComponent, kind: .imported)
        item.recipe = recipe; item.status = .processing
        do { try save(item) } catch { fail(error); return }
        if !asVoiceReference { selectedID = item.id }
        let id = item.id; let assets = self.assets; let library = self.library; let lifetime = self.lifetime
        status = AppStrings.localized("Importing audio")
        importTask = Task { [weak self] in
            let outcome = await Task.detached { Result {
                try AudioMediaIO.importFile(source, cancel: lifetime.check) { chunk in
                    try lifetime.check()
                    let descriptor = try assets.store(data: chunk.wav, fileName: "import-\(UUID().uuidString).wav", mimeType: "audio/wav")
                    try lifetime.check()
                    try library.update(id) {
                        $0.clips.append(AudioLibraryClip(assetReference: descriptor.storedReference, sampleRate: chunk.sampleRate, channels: chunk.channels, frameCount: chunk.frameCount, offset: chunk.offset, source: chunk.source))
                        $0.updatedAt = Date()
                    }
                }
                return try library.update(id) { $0.status = .completed; $0.updatedAt = Date() }
            }}.value
            guard let self, self.valid(token) else { return }
            self.importTask = nil
            switch outcome {
            case .success(let saved):
                self.publish(saved)
                if asVoiceReference { self.setVoiceReference(saved) }
            case .failure(let error):
                self.update(id) { $0.status = .interrupted; $0.failure = error.localizedDescription }
                self.error = error.localizedDescription
            }
        }
    }
    func stopPlayback() {
        playbackLifetime?.end(); playbackLifetime = nil
        playbackTask?.cancel(); playbackTask = nil
        player?.stop(); player = nil; isPlaying = false; playbackTime = 0
        if let playbackURL { try? FileManager.default.removeItem(at: playbackURL); scratchURLs.remove(playbackURL) }
        playbackURL = nil
    }
    func playPause() {
        guard !hasRecordingActivity, !isShutDown else { return }
        if let player {
            if player.isPlaying { player.pause(); isPlaying = false }
            else { isPlaying = player.play() }
            return
        }
        guard let item = selectedItem else { return }
        play(item, at: 0)
    }
    func seek(to seconds: Double) {
        guard seconds.isFinite, !hasRecordingActivity, !isShutDown else { return }
        if let player { player.currentTime = min(player.duration, max(0, seconds)); playbackTime = player.currentTime }
        else if let selectedItem { play(selectedItem, at: max(0, seconds), autoplay: false) }
    }
    func compare(with item: AudioLibraryItem) { play(item, at: playbackTime) }
    func play(_ item: AudioLibraryItem, at seconds: Double, autoplay: Bool = true) {
        guard !item.clips.isEmpty, !isShutDown, !hasRecordingActivity else { return }
        stopPlayback(); stopReferencePreview()
        let token = epoch; let assets = self.assets; let lifetime = self.lifetime
        do {
            let destination = try scratchURL(extension: "wav")
            playbackURL = destination
            let rendering = AudioOperationLifetime()
            playbackLifetime = rendering
            playbackTask = Task { [weak self] in
                let outcome = await Task.detached { Result {
                    try AudioMediaIO.render(item.clips, store: assets, to: destination, cancel: {
                        try lifetime.check(); try rendering.check()
                    })
                } }.value
                guard let self else { try? FileManager.default.removeItem(at: destination); return }
                self.finishPreparingPlayback(outcome, destination: destination, at: seconds, autoplay: autoplay, token: token)
            }
        } catch { fail(error) }
    }
    func finishPreparingPlayback(_ outcome: Result<Void, Error>, destination: URL, at seconds: Double, autoplay: Bool, token: UUID) {
        // Permission setup can begin while decryption/rendering is suspended.
        guard valid(token), playbackURL == destination,
              !hasRecordingActivity, !Task.isCancelled else {
            if playbackURL == destination { stopPlayback() }
            try? FileManager.default.removeItem(at: destination)
            scratchURLs.remove(destination)
            return
        }
        playbackTask = nil
        do {
            try outcome.get()
            let player = try AVAudioPlayer(contentsOf: destination)
            player.currentTime = min(seconds, player.duration)
            self.player = player; playbackTime = player.currentTime
            isPlaying = autoplay && player.play()
        } catch { self.error = error.localizedDescription; stopPlayback() }
    }
    func exportAudio(_ item: AudioLibraryItem, preset: AudioExportPreset = .nativeWAV) {
        export(clips: item.clips, title: item.title, preset: preset)
    }
    func exportClip(_ clip: AudioLibraryClip, preset: AudioExportPreset) {
        var clip = clip; clip.offset = 0
        export(clips: [clip], title: (selectedItem?.title ?? "Audio") + " - " + clip.source, preset: preset)
    }
    func export(clips: [AudioLibraryClip], title: String, preset: AudioExportPreset) {
        guard !clips.isEmpty, !isExporting, exportTask == nil, !isShutDown, !hasRecordingActivity else { return }
        let token = epoch
        let chooser = AudioExportPanel(title: title, preset: preset)
        isExporting = true
        guard chooser.panel.runModal() == .OK, let target = chooser.panel.url, valid(token), !hasRecordingActivity else {
            isExporting = false
            return
        }
        let preset = chooser.preset
        let suffix = preset.fileExtension
        let assets = self.assets; let lifetime = self.lifetime
        do {
            let temporary = try scratchURL(extension: "wav")
            let encoded = try scratchURL(extension: suffix)
            if !isBusy { status = AppStrings.localized("Exporting audio") }
            exportTask = Task { [weak self] in
                let scoped = target.startAccessingSecurityScopedResource()
                defer {
                    if scoped { target.stopAccessingSecurityScopedResource() }
                    try? FileManager.default.removeItem(at: temporary)
                    try? FileManager.default.removeItem(at: encoded)
                    self?.scratchURLs.remove(temporary)
                    self?.scratchURLs.remove(encoded)
                    self?.exportTask = nil
                    self?.isExporting = false
                }
                do {
                    try await Task.detached { try AudioMediaIO.render(clips, store: assets, to: temporary, sampleRate: preset == .videoWAV ? 48_000 : nil, cancel: lifetime.check) }.value
                    try Task.checkCancellation()
                    switch preset {
                    case .m4a: try await AudioMediaIO.encodeM4A(from: temporary, to: encoded)
                    case .flac, .aiff:
                        try await Task.detached { try AudioMediaIO.encodeLossless(from: temporary, to: encoded, preset: preset, cancel: lifetime.check) }.value
                    case .nativeWAV, .videoWAV: break
                    }
                    guard let self, self.valid(token), !Task.isCancelled else { return }
                    let source = preset == .nativeWAV || preset == .videoWAV ? temporary : encoded
                    try AudioMediaIO.commitExport(from: source, to: target)
                    if !self.isBusy { self.status = AppStrings.localized("Audio exported") }
                } catch {
                    if let self, self.valid(token) { self.error = error.localizedDescription }
                }
            }
        } catch { isExporting = false; fail(error) }
    }
    func exportTranscript(_ item: AudioLibraryItem, format: AudioTranscriptExportFormat) {
        guard !isShutDown, !hasRecordingActivity, let transcript = item.preferredTranscript else { return }
        let token = epoch
        let panel = NSSavePanel(); panel.canCreateDirectories = true
        let suffix = format == .text ? "txt" : format.rawValue
        panel.allowedContentTypes = [UTType(filenameExtension: suffix) ?? .plainText]
        panel.nameFieldStringValue = ProfileBackup.sanitizedFileName(item.title, fallback: "Transcript") + "." + suffix
        guard panel.runModal() == .OK, let url = panel.url, valid(token), !hasRecordingActivity else { return }
        do {
            let data: Data
            switch format {
            case .text: data = Data(AudioTranscriptExport.text(transcript.segments).utf8)
            case .srt: data = Data(AudioTranscriptExport.srt(transcript.segments).utf8)
            case .vtt: data = Data(AudioTranscriptExport.vtt(transcript.segments).utf8)
            case .json:
                let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                var portable = item
                portable.localModelBookmark = nil
                data = try encoder.encode(portable)
            }
            try data.write(to: url, options: .atomic)
            status = AppStrings.localized("Transcript exported")
        } catch { self.error = error.localizedDescription }
    }
}

final class AudioOperationLifetime: @unchecked Sendable {
    private let lock = NSLock()
    private var active = true
    func end() { lock.lock(); active = false; lock.unlock() }
    func check() throws { lock.lock(); defer { lock.unlock() }; if !active { throw CancellationError() } }
}
