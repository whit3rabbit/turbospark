import SwiftUI
import TurboSpark

/// The card laid over Music or Voice Cloning while none of the page's curated
/// models is on this Mac. It lists each model with its download size and the
/// memory it needs, and downloads the one the user picks. The page under it is
/// disabled, so nothing runs without a model. The sidebar and the page header are
/// outside the overlay: the way out that is not a download is the Audio library.
@MainActor
struct AudioModelGateView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController
    let gate: AudioWorkspaceController.ModelGate

    @Environment(\.appTheme) private var theme

    private var scale: CGFloat { AppMetrics.textScale(theme) }

    /// The memory the audio models may use, as every loader counts it. Read once:
    /// it asks Metal for the device.
    private static let memoryBudgetBytes = MusicMemoryAdvisor.evaluate(for: nil).budgetBytes

    var body: some View {
        ZStack {
            // Opaque enough to read the card over, and it takes the clicks the
            // disabled page would otherwise leave to the scroll view under it.
            Rectangle().fill(.appPage).opacity(theme.surfaceOpacity(0.86))
                .contentShape(Rectangle())
                .onTapGesture {}
                .accessibilityHidden(true)
            ScrollView {
                card
                    .frame(maxWidth: (560 * scale).rounded())
                    .padding(24)
                    .frame(maxWidth: .infinity)
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityAddTraits(.isModal)
    }

    private var card: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 6) {
                title.themedFont(.title3, weight: .semibold).accessibilityAddTraits(.isHeader)
                message.foregroundStyle(.appSecondary).themedFont(.small)
                    .fixedSize(horizontal: false, vertical: true)
            }
            ForEach(Array(gate.entries.enumerated()), id: \.element.identity.alias) { index, profile in
                if index > 0 { Divider() }
                row(profile)
            }
            Divider()
            Button { controller.selectPage(.library) } label: {
                Label { Text("Audio library", bundle: .module) } icon: { Image(systemName: "books.vertical") }
            }
            .appAction(.quiet, size: .small)
        }
        .appCardSurface(padding: 20)
    }

    private var title: Text {
        gate.page == .music
            ? Text("Download the music model", bundle: .module)
            : Text("Choose a voice cloning model", bundle: .module)
    }

    private var message: Text {
        gate.page == .music
            ? Text("Music generation needs its model on this Mac. Download it to continue.", bundle: .module)
            : Text("Voice cloning needs one of these models on this Mac. Pick one to download.", bundle: .module)
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
