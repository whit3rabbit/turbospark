import AVFoundation
import Foundation

/// The app's one audio player.
///
/// ONE because two clips playing over each other, or a clip over read-aloud,
/// is never what a user meant. Starting playback stops speech and speech
/// stops playback (`AppSpeechSynthesizer.speak`), so at most one source is
/// audible. Views identify a clip by `key` (an attachment ID or a stored
/// path) and read `isPlaying(key:)` / `progress(for:)`.
@MainActor
final class AudioPlaybackController: NSObject, ObservableObject, AVAudioPlayerDelegate {
    static let shared = AudioPlaybackController()

    @Published private(set) var activeKey: String?
    @Published private(set) var isPlaying = false
    @Published private(set) var currentTime: TimeInterval = 0
    @Published private(set) var duration: TimeInterval = 0
    @Published private(set) var rate: Float = 1

    static let rates: [Float] = [0.75, 1, 1.5, 2]
    /// The key engine read-aloud plays under. Playback with this key IS
    /// speech, so it must not stop the synthesizer that started it.
    static let readAloudKey = "TurboSpark.readAloud.engine"

    /// Called with the finished clip's key. Read aloud uses it to clear its
    /// speaking state when an engine-rendered reply ends.
    var onFinish: ((String) -> Void)?

    private var player: AVAudioPlayer?
    private var ticker: Task<Void, Never>?

    func isPlaying(key: String) -> Bool { isPlaying && activeKey == key }

    /// 0...1 for the active clip, nil for any other.
    func progress(for key: String) -> Double? {
        guard activeKey == key, duration > 0 else { return nil }
        return currentTime / duration
    }

    /// Play/pause for `key`, loading `url` when another clip was active.
    func toggle(url: URL, key: String) {
        if activeKey == key, let player {
            if player.isPlaying { pause() } else { resume() }
        } else {
            play(url: url, key: key)
        }
    }

    func play(url: URL, key: String, from fraction: Double = 0) {
        stop()
        if key != Self.readAloudKey { AppSpeechSynthesizer.shared.stop() }
        do {
            let player = try AVAudioPlayer(contentsOf: url)
            player.delegate = self
            player.enableRate = true
            player.rate = rate
            player.prepareToPlay()
            player.currentTime = player.duration * max(0, min(1, fraction))
            self.player = player
            activeKey = key
            duration = player.duration
            currentTime = player.currentTime
            resume()
        } catch {
            activeKey = nil
        }
    }

    func resume() {
        guard let player else { return }
        if activeKey != Self.readAloudKey { AppSpeechSynthesizer.shared.stop() }
        player.play()
        isPlaying = true
        startTicker()
    }

    func pause() {
        player?.pause()
        isPlaying = false
        ticker?.cancel()
    }

    func stop() {
        player?.stop()
        player = nil
        ticker?.cancel()
        isPlaying = false
        activeKey = nil
        currentTime = 0
        duration = 0
    }

    /// Seeks the active clip; loads and seeks when `key` is not active.
    func seek(to fraction: Double, url: URL, key: String) {
        guard activeKey == key, let player else {
            play(url: url, key: key, from: fraction)
            pause()
            return
        }
        player.currentTime = player.duration * max(0, min(1, fraction))
        currentTime = player.currentTime
    }

    func skip(by seconds: TimeInterval) {
        guard let player else { return }
        player.currentTime = max(0, min(player.duration, player.currentTime + seconds))
        currentTime = player.currentTime
    }

    func setRate(_ newRate: Float) {
        rate = newRate
        player?.rate = newRate
    }

    /// 20 Hz is smooth for a playhead and cheap; it stops with the clip.
    private func startTicker() {
        ticker?.cancel()
        ticker = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 50_000_000)
                guard let self, let player = self.player else { return }
                self.currentTime = player.currentTime
            }
        }
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in
            self.ticker?.cancel()
            self.isPlaying = false
            self.currentTime = 0
            self.player?.currentTime = 0
            if let key = self.activeKey { self.onFinish?(key) }
        }
    }
}
