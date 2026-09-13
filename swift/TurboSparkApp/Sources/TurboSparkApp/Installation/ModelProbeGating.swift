import Foundation
import TurboSpark

/// Repository sanitization and install gating helpers for `ModelProbeSheet`.
enum ModelProbeGating {

    /// Strips leading URLs, hostnames, and slashes so users can paste a Hugging Face web URL
    /// or shorthand into the repository field.
    static func sanitizeRepo(_ input: String) -> String {
        var s = input.trimmingCharacters(in: .whitespacesAndNewlines)
        if let range = s.range(of: "^https?://", options: .regularExpression) {
            s.removeSubrange(range)
        }
        for prefix in ["huggingface.co/", "hf.co/", "www.huggingface.co/"] {
            if s.lowercased().hasPrefix(prefix) {
                s.removeFirst(prefix.count)
                break
            }
        }
        while s.hasSuffix("/") {
            s.removeLast()
        }
        return s
    }

    /// Evaluates whether the probe sheet allows installing what is currently configured.
    static func installDecision(
        repo: String,
        probeError: String?,
        report: ProbeReport?,
        variants: RepoVariants?,
        ggufFile: String,
        selectedVariant: RepoVariant?,
        freeDiskBytes: UInt64?
    ) -> ModelInstallDecision {
        if let err = probeError {
            return .blocked(err)
        }
        if let listed = variants, listed.variants.count > 1 && ggufFile.isEmpty {
            return .blocked(
                String(
                    localized: "This repository offers multiple GGUF variants. Select a specific quantization variant before installing.",
                    bundle: .module
                )
            )
        }
        guard let report else {
            return .blocked(
                String(
                    localized: "Run probe to verify model compatibility before installing.",
                    bundle: .module
                )
            )
        }
        if !report.runnable {
            return .blocked(
                report.refusedBecause
                    ?? String(
                        localized: "This port has no decode flow for this checkpoint.",
                        bundle: .module
                    )
            )
        }
        return ModelInstallGate.decide(
            probeRunnable: report.runnable,
            refusedBecause: report.refusedBecause,
            verdict: report.fit?.verdict,
            installBytes: report.downloadBytes ?? selectedVariant?.bytes,
            freeDiskBytes: freeDiskBytes
        )
    }
}
