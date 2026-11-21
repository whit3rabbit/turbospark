import Foundation
import TurboSpark

/// Vary re-renders a saved request with a new seed. Regenerate keeps the
/// saved seed; the two must never produce the same image.
enum ImageVariation {
    /// `random` is the same draw a blank-seed fresh generation makes
    /// (`generateImage`), so a variation is as different as a new image.
    /// Only the saved seed is refused, because it would turn Vary into
    /// Regenerate.
    static func seed(replacing saved: UInt64, random: () -> UInt64) -> UInt64 {
        let candidate = random()
        return candidate == saved ? saved &+ 1 : candidate
    }

    static func options(from request: AppImageRequest, random: () -> UInt64) -> ImageGenerateOptions {
        ImageGenerateOptions(
            prompt: request.prompt, seed: seed(replacing: request.seed, random: random),
            width: request.width, height: request.height, steps: request.steps)
    }
}

/// The images of the most recent multi-image request, for "pick one".
///
/// Artifacts carry no batch id (adding one would change the saved chat
/// format), so a run is recognized from what `ImageGenerationSequence`
/// writes: same chat, prompt, size and steps, seeds that step by one, and
/// saves close together. Anything that does not fit returns nil rather
/// than guessing, so the row only appears for a real batch.
enum ImageResultRun {
    /// Longest pause between two saves of one run. A large model at full
    /// size can take minutes per image; an hour apart is a new request.
    static let maxGap: TimeInterval = 15 * 60

    /// `artifacts` newest first (as `savedImageArtifacts`). Returns the
    /// latest run in render order, or nil for a single image.
    static func latest(in artifacts: [AppArtifact]) -> [AppArtifact]? {
        guard let newest = artifacts.first, let newestRequest = newest.imageRequest else { return nil }
        var run = [newest]
        var previous = newest
        var previousRequest = newestRequest
        for candidate in artifacts.dropFirst() {
            guard run.count < ImageGenerationSequence.maxCount,
                  let request = candidate.imageRequest,
                  candidate.chatID == newest.chatID,
                  request.prompt == newestRequest.prompt,
                  request.width == newestRequest.width,
                  request.height == newestRequest.height,
                  request.steps == newestRequest.steps,
                  request.seed &+ 1 == previousRequest.seed,
                  candidate.createdAt <= previous.createdAt,
                  previous.createdAt.timeIntervalSince(candidate.createdAt) <= maxGap
            else { break }
            run.append(candidate)
            previous = candidate
            previousRequest = request
        }
        return run.count > 1 ? run.reversed() : nil
    }
}
