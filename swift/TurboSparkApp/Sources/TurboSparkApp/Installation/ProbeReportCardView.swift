import SwiftUI
import TurboSpark

/// A probe report, rendered.
///
/// It used to be the raw JSON string in a monospace `Text`, which carried
/// every fact below and asked the reader to parse it.
@MainActor
struct ProbeReportCardView: View {
    let report: ProbeReport

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            verdictRow
            if let why = report.refusedBecause {
                calloutBox(why, tone: .red, icon: "xmark.octagon.fill")
            }
            identityRows
            if let fit = report.fit { fitSection(fit) }
            if !report.types.isEmpty { blockTypeSection }
            sidecarSection
            ForEach(Array(report.warnings.enumerated()), id: \.offset) { _, w in
                calloutBox(w, tone: .orange, icon: "exclamationmark.triangle.fill")
            }
        }
        .padding(14)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Verdict and identity

    private var verdictRow: some View {
        HStack(spacing: 8) {
            Label(
                report.runnable ? "Would run here" : "Would not run here",
                systemImage: report.runnable ? "checkmark.seal.fill" : "xmark.octagon.fill"
            )
            .themedFont(.small, weight: .semibold)
            .foregroundStyle(report.runnable ? Color.green : Color.red)
            Spacer()
            // The fit pill is a SECOND question and only shown when something
            // answered it. A runnable checkpoint whose header yielded no
            // shape gets no pill rather than a green one.
            if let fit = report.fit {
                ModelFitVerdictPill(verdict: fit.verdict)
            }
        }
    }

    private var identityRows: some View {
        VStack(alignment: .leading, spacing: 4) {
            if let file = report.file { specRow("File", file) }
            if let arch = report.architecture { specRow("Declares", arch) }
            // `family` nil beside a non-nil `architecture` is the interesting
            // case and the reason both are shown: it means this port read the
            // architecture and has no flow for it.
            specRow("Family here", report.family ?? "none")
            if let bytes = report.downloadBytes {
                specRow("Download", MetricFormat.storage(bytes))
            }
            if let affine = report.affine {
                specRow("Quantization", "MLX affine, \(affine.bits)-bit at group \(affine.groupSize)")
            }
            if let stride = report.expertStride {
                specRow("One expert", MetricFormat.storage(stride))
            }
            if let template = report.chatTemplate {
                specRow("Chat template", template)
            } else {
                specRow("Chat template", "none found")
            }
        }
    }

    // MARK: - Fit

    @ViewBuilder
    private func fitSection(_ fit: ProbeFit) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Memory at \(fit.context.formatted()) tokens, \(fit.slotCacheSlots) slots", bundle: .module)
                .themedFont(.small, weight: .semibold)
                .foregroundStyle(.appSecondary)

            // Guarded on the SOURCE, not on the number. A row nothing has
            // sized reports zeros, and a zero rendered as a figure reads as
            // "fits easily" (swift Gotcha 23).
            if fit.countedSource == .unknown {
                Text("Nothing has read this checkpoint's shape, so its footprint is unknown.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            } else {
                specRow("Allocates", MetricFormat.storage(fit.countedBytes))
                if fit.slotCacheBytes > 0 {
                    specRow("Expert slot cache", MetricFormat.storage(fit.slotCacheBytes))
                }
                if fit.kvBytes > 0 { specRow("KV cache", MetricFormat.storage(fit.kvBytes)) }
                if fit.largestContext > 0 {
                    specRow("Largest context", "\(fit.largestContext.formatted()) tokens")
                }
            }

            // What a LONGER window would cost. Rendered from the engine's
            // own rungs and never multiplied here: KV is not linear in the
            // window (a sliding-window layer is a ring and stops growing),
            // so a UI doing its own arithmetic is 3.5x high on Gemma.
            if !fit.contextLadder.isEmpty {
                Divider().padding(.vertical, 2)
                ContextLadderView(
                    rungs: fit.contextLadder,
                    trainedContext: report.trainedContext,
                    currentContext: fit.context)
            }

            // The mapped figure is the PUBLISHED CHECKPOINT and not the
            // install this port writes. Saying so is the whole point of
            // `mappedSource`; presenting it as an install size would be a
            // number the user could check and find wrong.
            if fit.mappedBytes > 0 {
                specRow(
                    "Reads",
                    fit.mappedSource == "download"
                        ? "\(MetricFormat.storage(fit.mappedBytes)) as published"
                        : MetricFormat.storage(fit.mappedBytes))
            }
        }
    }

    // MARK: - Block types

    private var blockTypeSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Block types", bundle: .module)
                .themedFont(.small, weight: .semibold)
                .foregroundStyle(.appSecondary)
            ForEach(report.types, id: \.name) { t in
                HStack(spacing: 6) {
                    Text(t.name)
                        .themedCode(.small)
                        .frame(width: 74, alignment: .leading)
                    Text("\(t.tensors) tensors", bundle: .module)
                        .themedFont(.small).monospacedDigit()
                        .foregroundStyle(.appSecondary)
                        .frame(width: 90, alignment: .leading)
                    // **UNSIZED, NEVER 0 BYTES.** A type this port cannot
                    // size is usually the one carrying the model, and a zero
                    // sorts to the bottom of a share column, which is the
                    // inverse of its real rank.
                    Text(t.bytes.map(MetricFormat.storage) ?? "UNSIZED")
                        .themedFont(.small).monospacedDigit()
                        .foregroundStyle(t.bytes == nil ? Color.orange : .secondary)
                        .frame(width: 84, alignment: .leading)
                    Text(t.executable ? "has kernels" : "no kernels")
                        .themedFont(.small)
                        .foregroundStyle(t.executable ? Color.green : Color.red)
                    Spacer()
                }
            }
        }
    }

    // MARK: - Sidecars

    private var sidecarSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            if !report.sidecarsPresent.isEmpty {
                specRow("Sidecars found", report.sidecarsPresent.joined(separator: ", "))
            }
            if !report.sidecarsMissing.isEmpty {
                // Missing sidecars cost SECONDS here and a whole re-stream if
                // they are discovered after the walk (catalog Gotcha 1), so
                // this is a finding rather than a footnote.
                specRow("Sidecars missing", report.sidecarsMissing.joined(separator: ", "))
                    .foregroundStyle(Color.orange)
            }
        }
    }

    // MARK: - Bits

    private func specRow(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(label)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .frame(width: 118, alignment: .leading)
            Text(value)
                .themedFont(.small).monospacedDigit()
                .textSelection(.enabled)
            Spacer()
        }
    }

    private func calloutBox(_ text: String, tone: Color, icon: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: icon)
                .themedFont(.small)
                .accessibilityHidden(true)
            Text(text).themedFont(.small)
        }
        .foregroundStyle(tone)
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(tone.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
    }
}
