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
    func importAudio() {
        guard importTask == nil, !isShutDown, !hasRecordingActivity else { return }
        let token = epoch
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.audio]; panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let source = panel.url, valid(token), !hasRecordingActivity else { return }
        var item = AudioLibraryItem(title: source.deletingPathExtension().lastPathComponent, kind: .imported)
        item.recipe = recipe; item.status = .processing
        do { try save(item) } catch { fail(error); return }
        selectedID = item.id
        let id = item.id; let assets = self.assets; let library = self.library; let lifetime = self.lifetime
        status = String(localized: "Importing audio", bundle: .module)
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
            case .success(let saved): self.publish(saved)
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
        stopPlayback()
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
    func exportAudio(preset: AudioExportPreset) {
        guard let item = selectedItem else { return }
        export(clips: item.clips, title: item.title, preset: preset)
    }
    func exportClip(_ clip: AudioLibraryClip, preset: AudioExportPreset) {
        var clip = clip; clip.offset = 0
        export(clips: [clip], title: (selectedItem?.title ?? "Audio") + " - " + clip.source, preset: preset)
    }
    func export(clips: [AudioLibraryClip], title: String, preset: AudioExportPreset) {
        guard !clips.isEmpty, exportTask == nil, !isShutDown, !hasRecordingActivity else { return }
        let token = epoch
        let panel = NSSavePanel(); panel.canCreateDirectories = true
        let suffix = preset == .m4a ? "m4a" : "wav"
        panel.allowedContentTypes = [UTType(filenameExtension: suffix) ?? .audio]
        panel.nameFieldStringValue = ProfileBackup.sanitizedFileName(title, fallback: "Audio") + "." + suffix
        guard panel.runModal() == .OK, let target = panel.url, valid(token), !hasRecordingActivity else { return }
        let assets = self.assets; let lifetime = self.lifetime
        do {
            let temporary = try scratchURL(extension: "wav")
            let encoded = try scratchURL(extension: suffix)
            status = String(localized: "Exporting audio", bundle: .module)
            exportTask = Task { [weak self] in
                defer { try? FileManager.default.removeItem(at: temporary); try? FileManager.default.removeItem(at: encoded) }
                do {
                    try await Task.detached { try AudioMediaIO.render(clips, store: assets, to: temporary, sampleRate: preset == .videoWAV ? 48_000 : nil, cancel: lifetime.check) }.value
                    if preset == .m4a { try await AudioMediaIO.encodeM4A(from: temporary, to: encoded) }
                    guard let self, self.valid(token), !Task.isCancelled else { return }
                    let source = preset == .m4a ? encoded : temporary
                    // A failed encode never damages an existing export.
                    let sibling = target.deletingLastPathComponent().appendingPathComponent(".\(UUID().uuidString).\(suffix)")
                    defer { try? FileManager.default.removeItem(at: sibling) }
                    try FileManager.default.copyItem(at: source, to: sibling)
                    if FileManager.default.fileExists(atPath: target.path) { _ = try FileManager.default.replaceItemAt(target, withItemAt: sibling) }
                    else { try FileManager.default.moveItem(at: sibling, to: target) }
                    self.status = String(localized: "Audio exported", bundle: .module)
                } catch {
                    if let self, self.valid(token) { self.error = error.localizedDescription }
                }
                if let self, self.valid(token) { self.exportTask = nil }
            }
        } catch { fail(error) }
    }
    func exportTranscript(format: AudioTranscriptExportFormat) {
        guard !isShutDown, !hasRecordingActivity, let item = selectedItem, let transcript = item.preferredTranscript else { return }
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
            status = String(localized: "Transcript exported", bundle: .module)
        } catch { self.error = error.localizedDescription }
    }
}

final class AudioOperationLifetime: @unchecked Sendable {
    private let lock = NSLock()
    private var active = true
    func end() { lock.lock(); active = false; lock.unlock() }
    func check() throws { lock.lock(); defer { lock.unlock() }; if !active { throw CancellationError() } }
}
