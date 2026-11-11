import Foundation
import TurboSpark

/// What installing a Hugging Face search result is allowed to do.
///
/// The counterpart of `ModelProbeGating.installDecision` for a `HubCheck`:
/// the state of the check decides first, and only a runnable repository goes
/// on to `ModelInstallGate`, which owns the memory and disk rules. Two gates
/// for two sources is acceptable; two copies of the disk arithmetic is not.
enum HubInstallGating {
    static func decision(
        check: HubCheck,
        selectedVariant: RepoVariant?,
        destination: InstallGateDestination
    ) -> ModelInstallDecision {
        switch check.state {
        case .needsAuth:
            return .blocked(
                AppStrings.localized(
                    "This repository is gated or needs a Hugging Face sign-in. Add a token in Settings, then check again."))
        case .unavailable:
            return .blocked(
                AppStrings.localized(
                    "Could not check this repository: \(check.reason ?? AppStrings.localized("no answer"))"))
        case .refused:
            return .blocked(
                check.reason
                    ?? AppStrings.localized("This port has no decode flow for this checkpoint."))
        case .runnable:
            break
        }

        // The check read ONE file's header. Installing a quantization whose
        // name declares a type with no kernels would fail at the install's own
        // probe, minutes later, so refuse it now.
        if let variant = selectedVariant, !variant.executable {
            return .blocked(
                AppStrings.localized(
                    "This port has no kernels for the type this filename names. Pick another quantization."))
        }

        let bytes = selectedVariant?.bytes ?? check.downloadBytes
        return ModelInstallGate.decide(
            probeRunnable: true,
            refusedBecause: nil,
            verdict: check.fit?.verdict,
            installBytes: bytes,
            downloadBytes: bytes,
            to: destination)
    }
}
