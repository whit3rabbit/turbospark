import SwiftUI

// OC-07 step bodies. Static copy and shared controls only: no step starts a
// download, asks macOS for access, or touches the network. Choosing a model
// is the existing Discover section, reached by the person's own click.

struct FirstRunWelcomeStep: View {
    @Environment(\.appTheme) private var theme

    var body: some View {
        VStack(spacing: 14) {
            TSIdlingSparkView(size: 92)
            Text("Welcome to TurboSpark", bundle: .module)
                .font(theme.ui(.display, weight: .regular, systemDesign: .serif))
                .foregroundStyle(.appText)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityAddTraits(.isHeader)
            Text("Everything here runs on your Mac. Nothing leaves it.", bundle: .module)
                .themedFont(.callout)
                .foregroundStyle(.appSecondary)
                .multilineTextAlignment(.center)
            Text("A short tour gets you to your first chat. Every step is optional.", bundle: .module)
                .themedFont(.base)
                .foregroundStyle(.appText)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

struct FirstRunModelStep: View {
    let browseModels: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            AppEmptyState(
                systemImage: "shippingbox",
                title: Text("Choose a model", bundle: .module),
                message: Text(
                    "A model is the assistant that answers you. It downloads once, then runs on this Mac. Nothing downloads until you choose one.",
                    bundle: .module)
            ) {
                Button(action: browseModels) {
                    Text("Browse Models", bundle: .module)
                }
                .appAction(.secondary)
            }
        }
    }
}

struct FirstRunPrimerStep: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(spacing: 6) {
                Text("What TurboSpark will ask for", bundle: .module)
                    .themedFont(.title3, weight: .semibold)
                    .foregroundStyle(.appText)
                    .accessibilityAddTraits(.isHeader)
                Text(
                    "Nothing is requested during this setup. macOS shows a privacy prompt only after you press Allow where a feature needs it.",
                    bundle: .module
                )
                .themedFont(.callout)
                .foregroundStyle(.appSecondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity)

            VStack(alignment: .leading, spacing: 12) {
                row(
                    symbol: "mic",
                    title: Text("Microphone", bundle: .module),
                    detail: Text(
                        "Only when you press Allow before recording your voice in Audio.",
                        bundle: .module))
                row(
                    symbol: "rectangle.on.rectangle",
                    title: Text("Screen and system audio recording", bundle: .module),
                    detail: Text(
                        "Only when you press Allow before capturing a meeting or system audio.",
                        bundle: .module))
                row(
                    symbol: "bell",
                    title: Text("Notifications", bundle: .module),
                    detail: Text(
                        "Only if you turn on alerts in Settings and then press Allow.",
                        bundle: .module))
            }
            .appCardSurface()
        }
    }

    private func row(symbol: String, title: Text, detail: Text) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: symbol)
                .themedFont(.base)
                .foregroundStyle(.appAccentText)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                title.themedFont(.base, weight: .medium).foregroundStyle(.appText)
                detail
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .accessibilityElement(children: .combine)
    }
}
