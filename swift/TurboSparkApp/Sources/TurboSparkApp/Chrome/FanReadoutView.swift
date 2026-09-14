import SwiftUI

/// Status bar fan readout and control popover, backed by `FanController`'s
/// bridge to the `thermalforge` CLI. Renders nothing (divider included)
/// when the binary is not installed, so a machine without ThermalForge
/// shows exactly the status bar it showed before.
@MainActor
struct FanReadoutView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject private var fans: FanController = .shared
    @ObservedObject var model: AppModel
    /// False when this readout is FIRST in the strip. The divider travels
    /// with the readout so an absent fan controller leaves no dangling mark,
    /// which inverts once nothing precedes it: memory and CPU moved to the
    /// top bar, so the left group can now start here.
    var showsLeadingDivider: Bool = true
    @State private var showsControls = false

    var body: some View {
        if fans.isAvailable {
            HStack(spacing: 0) {
                if showsLeadingDivider {
                    Rectangle()
                        .fill(.appBorder)
                        .frame(width: 0.5, height: 11)
                        .padding(.horizontal, 9)
                }
                readout
            }
            .popover(isPresented: $showsControls, arrowEdge: .bottom) { controls }
        }
    }

    private var isPinned: Bool { fans.status?.anyFanHeld == true }

    /// Peak RPM across fans: on a multi-fan machine the highest reading is
    /// the one that matters for cooling headroom, and the popover lists
    /// each fan individually.
    private var rpmText: String {
        guard let fanList = fans.status?.fans, let peak = fanList.map(\.actualRPM).max() else {
            return "\u{2014}"
        }
        return "\(peak.formatted()) RPM"
    }

    private var readout: some View {
        Button {
            showsControls = true
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "fanblades")
                    .font(theme.ui(.tiny))
                    .foregroundStyle(isPinned ? Color.orange : Color.secondary)
                    .accessibilityHidden(true)
                Text(rpmText)
                    .monospacedDigit()
                    .foregroundStyle(.appText)
                if isPinned {
                    Text("MAX", bundle: .module)
                        .font(theme.ui(.micro, weight: .semibold))
                        .foregroundStyle(Color.orange)
                }
            }
        }
        .buttonStyle(.plain)
        .help("Fan speed via ThermalForge. Click to pin max or restore the automatic curve.")
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Fan speed")
        .accessibilityValue(rpmText + (isPinned ? ", pinned to maximum" : ""))
        .appPointerCursor()
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label { Text("Fans", bundle: .module) } icon: { Image(systemName: "fanblades") }
                    .font(theme.ui(.callout, weight: .semibold))
                Spacer()
                if fans.isBusy {
                    ProgressView()
                        .controlSize(.small)
                }
            }

            if let status = fans.status {
                ForEach(status.fans) { fan in
                    HStack {
                        Text(verbatim: "Fan \(fan.index)")
                            .foregroundStyle(.appSecondary)
                        Spacer()
                        Text(verbatim: "\(fan.actualRPM.formatted()) RPM")
                            .monospacedDigit()
                            .foregroundStyle(.appText)
                        Text(fan.isHeld ? "held" : "auto")
                            .foregroundStyle(fan.isHeld ? Color.orange : Color.secondary)
                    }
                    .font(theme.ui(.tiny))
                }
                if let sensor = status.hottestSensor {
                    Text("Hottest sensor \(sensor.name): \(String(format: "%.1f", sensor.celsius)) C", bundle: .module)
                        .font(theme.ui(.tiny))
                        .foregroundStyle(.tertiary)
                }
            } else {
                Text(fans.statusError ?? "Reading fan status...")
                    .font(theme.ui(.tiny))
                    .foregroundStyle(.tertiary)
            }

            Divider()

            HStack(spacing: 8) {
                Button {
                    Task { await fans.pinMax() }
                } label: { Text("Pin Max", bundle: .module) }
                .disabled(fans.isBusy)
                Button {
                    Task { await fans.restoreAuto() }
                } label: { Text("Restore Auto", bundle: .module) }
                .disabled(fans.isBusy)
            }
            .font(theme.ui(.tiny))

            Toggle(isOn: Binding(
                get: { fans.keepFansPinnedOnQuit },
                set: { newValue in
                    // The controller's copy is what the quit path reads;
                    // the AppModel copy is what persists. Both, always.
                    fans.keepFansPinnedOnQuit = newValue
                    model.keepFansPinnedOnQuit = newValue
                    model.persistSettingsDebounced()
                }
            )) {
                Text("Keep fans pinned when TurboSpark quits", bundle: .module)
                    .font(theme.ui(.tiny))
            }
            .controlSize(.small)

            if let error = fans.lastError ?? fans.statusError {
                Text(error)
                    .font(theme.ui(.tiny))
                    .foregroundStyle(.red)
            }

            Text("A pinned hold outlives this app: quitting restores Apple's curve unless the toggle above is on. Fans can always be released with `thermalforge auto`.", bundle: .module)
            .font(theme.ui(.tiny))
            .foregroundStyle(.tertiary)
            .fixedSize(horizontal: false, vertical: true)
        }
        .padding(14)
        .frame(width: 300, alignment: .leading)
    }
}
