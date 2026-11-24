import Foundation

/// Denoise progress as the MLX pipelines report it during the "transformer"
/// stage: `completed` counts finished steps, emitted after each step.
///
/// The pipelines send only step counts, no intermediate latents or previews,
/// so the progress card shows "Step N of M" over the placeholder and never a
/// fabricated image. Nil when the stage is not denoising or the total is
/// unknown, so the caller shows the indeterminate stage label instead.
struct ImageDenoiseProgress: Equatable {
    let step: Int
    let total: Int

    var fraction: Double { Double(step) / Double(total) }

    init?(stage: String?, completed: Int, total: Int) {
        // Other stages reuse these fields for file counts or a 0-of-1
        // placeholder, which are not denoise steps.
        guard stage == "transformer", total > 0 else { return nil }
        self.step = min(max(completed, 0), total)
        self.total = total
    }
}
