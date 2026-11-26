import SwiftUI
import TurboSpark

/// Download choices for pages that use a curated set of audio models.
@MainActor
struct AudioModelGateView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController
    let gate: AudioWorkspaceController.ModelGate
    var inline = false

    @Environment(\.appTheme) private var theme

    private var scale: CGFloat { AppMetrics.textScale(theme) }

    /// The memory the audio models may use, as every loader counts it. Read once:
    /// it asks Metal for the device.
    private static let memoryBudgetBytes = MusicMemoryAdvisor.evaluate(for: nil).budgetBytes

    var body: some View {
        if inline {
            card
        } else {
            ScrollView {
                card
                    .frame(maxWidth: (640 * scale).rounded())
                    .padding(24)
                    .frame(maxWidth: .infinity)
            }
            .accessibilityElement(children: .contain)
        }
    }

    private var card: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: gate.page.symbol)
                    .themedFont(.title3).foregroundStyle(.appAccentText).accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 4) {
                    title.themedFont(.title3, weight: .semibold).accessibilityAddTraits(.isHeader)
                    message.foregroundStyle(.appSecondary).themedFont(.small)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            ForEach(Array(gate.entries.enumerated()), id: \.element.identity.alias) { index, profile in
                if index > 0 { Divider() }
                row(profile)
            }
        }
        .appCardSurface(padding: 16)
    }

    private var title: Text {
        switch gate.page {
        case .music:
            return Text("Download the music model", bundle: .module)
        case .voiceDesign:
            return Text("Voice Design", bundle: .module)
        default:
            return Text("Choose a voice cloning model", bundle: .module)
        }
    }

    private var message: Text {
        switch gate.page {
        case .music:
            return Text("Music generation needs its model on this Mac. Download it to continue.", bundle: .module)
        case .voiceDesign:
            return Text("Describe a voice and generate an audio clip. Reuse the clip as a cloning reference for a consistent voice.", bundle: .module)
        default:
            return Text("Voice cloning needs one of these models on this Mac. Pick one to download.", bundle: .module)
        }
    }

    private func row(_ profile: AudioProfile) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(profile.displayName).themedFont(.base, weight: .medium)
                .fixedSize(horizontal: false, vertical: true)
            Text(LocalizedStringKey(AudioEngineLabel(profile).title), bundle: .module)
                .themedFont(.small).foregroundStyle(.appSecondary)
            AudioModelFacts(profile: profile)
            if profile.residentMemoryBytes > Self.memoryBudgetBytes {
                Label { Text("May not fit this Mac", bundle: .module) }
                    icon: { Image(systemName: "exclamationmark.triangle.fill").accessibilityHidden(true) }
                    .themedFont(.small).foregroundStyle(AppAttention.color)
            }
            AudioModelDownloadControls(
                model: model, controller: controller, profile: profile, checksSpace: true,
                onStart: { controller.selectModel(profile.identity.alias) })
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .contain)
    }
}
