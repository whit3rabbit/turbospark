import Foundation
import TurboSpark

/// The calls live text needs from a streaming model, so tests can drive the
/// transcriber without one.
protocol AudioLiveStreaming: AnyObject, Sendable {
    func append(_ pcm: [Float]) throws -> AudioStreamUpdate
    func finish() throws -> AudioStreamUpdate
    func cancel()
    func close()
}
extension AudioStreamSession: AudioLiveStreaming {}

/// Live text for one recording. A serial queue drains the capture feed
/// through the mixer into a streaming model and reports what it commits.
///
/// The model opens on its own thread because the first Neural Engine load
/// can take minutes. Mixed audio waits in a bounded backlog meanwhile and is
/// caught up once the model is ready (it runs many times faster than real
/// time); past the bound, live text gives up for this take rather than grow
/// without limit. The saved recording and the final pass never depend on it.
final class AudioLiveTranscriber: @unchecked Sendable {
    enum Event: Sendable, Equatable {
        /// The model is loading; text starts when it is ready.
        case preparing
        case update(AudioStreamUpdate)
        /// Live text stopped for this take; the reason is user-facing.
        case stopped(String)
    }

    private let queue = DispatchQueue(label: "app.turbospark.audio.live", qos: .userInitiated)
    private let feed: AudioLiveFeed
    private var mixer = AudioLiveMixer(makeResampler: { AudioLiveConverter() })
    private let open: @Sendable () throws -> AudioLiveStreaming
    private let onEvent: @Sendable (Event) -> Void
    private let backlogLimit: Int
    /// Audio handed to the model per append: big enough to catch up fast
    /// after a slow first load, small enough that the queue keeps draining
    /// the feed between appends (the feed drops what it cannot hold).
    private let sliceSamples = 5 * 16_000
    private var stream: AudioLiveStreaming? {
        didSet { cancelLock.lock(); cancelTarget = stream; cancelLock.unlock() }
    }
    /// The stream as seen from outside the queue, only to cancel it.
    private let cancelLock = NSLock()
    private var cancelTarget: AudioLiveStreaming?
    private var backlog: [Float] = []
    private var timer: DispatchSourceTimer?
    private var ended = false
    /// Everything committed so far, readable from any thread. The saved
    /// draft is built from this, not from events, so it cannot miss words
    /// whose events the main actor has not run yet.
    private let committedLock = NSLock()
    private var committedText = ""
    private var committedWords: [AudioStreamWord] = []

    /// - Parameter backlogSeconds: mixed audio kept while the model loads.
    init(feed: AudioLiveFeed, backlogSeconds: Double = 600,
         open: @escaping @Sendable () throws -> AudioLiveStreaming,
         onEvent: @escaping @Sendable (Event) -> Void) {
        self.feed = feed; self.open = open; self.onEvent = onEvent
        backlogLimit = Int(backlogSeconds * 16_000)
    }

    func start(pumpInterval: DispatchTimeInterval = .milliseconds(100)) {
        queue.async { [self] in
            onEvent(.preparing)
            let timer = DispatchSource.makeTimerSource(queue: queue)
            timer.schedule(deadline: .now() + pumpInterval, repeating: pumpInterval)
            timer.setEventHandler { [weak self] in self?.pump(flush: false) }
            timer.resume()
            self.timer = timer
        }
        Thread.detachNewThread { [self] in
            let result = Result { try open() }
            queue.async { [self] in opened(result) }
        }
    }

    private func opened(_ result: Result<AudioLiveStreaming, Error>) {
        switch result {
        case .success(let s):
            // Stopped while loading: nothing more to transcribe.
            if ended { s.close(); return }
            stream = s
            pump(flush: false)
        case .failure(let error):
            stop(reason: AppStrings.localized("Live text is unavailable for this recording: \(error.localizedDescription)"))
        }
    }

    private func pump(flush: Bool) {
        guard !ended || flush else { return }
        let (packets, _) = feed.drain()
        for packet in packets { mixer.add(packet) }
        backlog.append(contentsOf: mixer.drain(flush: flush))
        guard let stream else {
            if backlog.count > backlogLimit {
                stop(reason: AppStrings.localized("Live text could not keep up. The full transcript appears when you stop."))
            }
            return
        }
        guard !backlog.isEmpty else { return }
        do {
            repeat {
                let take = min(backlog.count, sliceSamples)
                let update = try stream.append(Array(backlog.prefix(take)))
                backlog.removeFirst(take)
                publish(update)
            } while flush && !backlog.isEmpty
            // Still catching up: come back right away instead of waiting a tick.
            if !backlog.isEmpty { queue.async { [weak self] in self?.pump(flush: false) } }
        } catch {
            stop(reason: AppStrings.localized("Live text stopped: \(error.localizedDescription)"))
        }
    }

    private func publish(_ update: AudioStreamUpdate) {
        guard !update.appendedText.isEmpty || !update.closedWords.isEmpty || update.isFinal else { return }
        committedLock.lock()
        committedText += update.appendedText
        committedWords += update.closedWords
        committedLock.unlock()
        onEvent(.update(update))
    }

    /// The transcript and finished words committed so far.
    func snapshot() -> (text: String, words: [AudioStreamWord]) {
        committedLock.lock(); defer { committedLock.unlock() }
        return (committedText, committedWords)
    }

    private func stop(reason: String) {
        guard !ended else { return }
        ended = true
        timer?.cancel(); timer = nil
        backlog = []
        stream?.close(); stream = nil
        onEvent(.stopped(reason))
    }

    /// Transcribes what is left, commits the tail, closes the model, and
    /// returns the whole committed transcript. Returns only once the model
    /// is closed, so the final pass can start.
    func finishAndClose() async -> (text: String, words: [AudioStreamWord]) {
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            queue.async { [self] in
                timer?.cancel(); timer = nil
                if !ended, stream != nil {
                    pump(flush: true)
                    if let stream, !ended, let update = try? stream.finish() { publish(update) }
                }
                ended = true
                stream?.close(); stream = nil
                backlog = []
                done.resume()
            }
        }
        return snapshot()
    }

    /// Stops without transcribing the rest, for shutdown and profile lock.
    /// A running append is cancelled rather than waited for.
    func stopAndDrain() {
        cancelLock.lock(); let running = cancelTarget; cancelLock.unlock()
        running?.cancel()
        queue.sync {
            ended = true
            timer?.cancel(); timer = nil
            stream?.close(); stream = nil
            backlog = []
        }
    }

    #if DEBUG
    /// Test seam: runs one pump on the queue and waits for it.
    func pumpForTesting() { queue.sync { pump(flush: false) } }
    var isOpenForTesting: Bool { queue.sync { stream != nil } }
    #endif
}
