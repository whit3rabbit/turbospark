import SwiftUI
import TurboSpark

/// What Auto residency will do for one install on this Mac, and why, for a
/// model family with a resident flow. Draws nothing for any other model.
///
/// The answer comes from the engine's own pre-open resolver
/// (`TurboSparkCapabilities.expertResidency(forInstallAt:requested:)`), so it
/// cannot disagree with what the next load does. It is read off the main
/// actor because the resolver reads the install's expert layout.
struct ExpertResidencyDetectionRow: View {
    let path: String?
    @State private var preflight: ExpertResidencyPreflight?

    var body: some View {
        Group {
            if let preflight, let need = preflight.residentBytes,
               let available = preflight.workingSetBytes
            {
                if preflight.resolved == "resident" {
                    Text(
                        "Auto runs this model with Resident experts: they need \(Self.bytes(need)) of the \(Self.bytes(available)) this Mac's GPU can use.",
                        bundle: .module)
                } else {
                    Text(
                        "Auto keeps this model's experts streamed: Resident needs \(Self.bytes(need)) plus a \(Self.bytes(ExpertResidencyPreflight.residencyReserveBytes)) reserve, and this Mac's GPU can use \(Self.bytes(available)).",
                        bundle: .module)
                }
            }
        }
        .themedFont(.tiny)
        .foregroundStyle(.appSecondary)
        .fixedSize(horizontal: false, vertical: true)
        .task(id: path) {
            guard let path, !path.isEmpty else {
                preflight = nil
                return
            }
            preflight = await Task.detached(priority: .utility) {
                TurboSparkCapabilities.expertResidency(forInstallAt: path, requested: .auto)
            }.value
        }
    }

    private static func bytes(_ value: UInt64) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(clamping: value), countStyle: .memory)
    }
}
