import SwiftUI

/// The numbered chips under a reply for the sources its `[n]` markers cite
/// (component audit, inline-citation). Each chip names its number and
/// domain and reveals the tool call that found the source, the same path
/// the sources popover's reveal button takes. The inline marker itself is a
/// link in the prose, which SwiftUI text cannot give a hover card or its
/// own VoiceOver name, so these chips carry both.
///
/// A chip says which source the model POINTED at, not that the claim came
/// from that page; the wording stays "Source n", never "verified".
struct InlineCitationChipsView: View {
    @ObservedObject var model: AppModel
    let sources: TurnSources
    /// Cited source numbers, 1-based and already validated against
    /// `sources.sources.count` by `InlineCitations`.
    let numbers: [Int]

    var body: some View {
        FlowLayout(spacing: 6, lineSpacing: 6) {
            ForEach(numbers, id: \.self) { number in
                if sources.sources.indices.contains(number - 1) {
                    chip(number: number, source: sources.sources[number - 1])
                }
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(Text("Sources cited in this reply", bundle: .module))
    }

    private func chip(number: Int, source: TurnSource) -> some View {
        Button {
            model.revealToolCard(for: source)
        } label: {
            HStack(spacing: 4) {
                Text(verbatim: "\(number)")
                    .themedFont(.tiny, weight: .semibold)
                    .monospacedDigit()
                    .foregroundStyle(.appAccentText)
                Text(verbatim: source.domain)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                    .stroke(.appBorder, lineWidth: 1)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.96))
        .disabled(source.origin == nil)
        .help(Text("Source \(number): \(source.title)", bundle: .module))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text("Source \(number): \(source.title)", bundle: .module))
        .accessibilityHint(Text("Reveal in transcript", bundle: .module))
    }
}
