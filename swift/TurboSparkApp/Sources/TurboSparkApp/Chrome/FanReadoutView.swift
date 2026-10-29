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
                    .foregroundStyle(isPinned ? theme.warning : theme.secondaryText)
                    .accessibilityHidden(true)
                Text(verbatim: rpmText)
                    .monospacedDigit()
                    .foregroundStyle(.appText)
                if isPinned {
                    Text("MAX", bundle: .module)
                        .font(theme.ui(.micro, weight: .semibold))
                        .foregroundStyle(AppAttention.color)
                }
            }
        }
        .buttonStyle(.plain)
        .help(Text("Fan speed via ThermalForge. Click to pin max or restore the automatic curve.", bundle: .module))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text("Fan speed", bundle: .module))
        .accessibilityValue(
            isPinned ? Text("\(rpmText), pinned to maximum", bundle: .module) : Text(verbatim: rpmText))
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
                        Text("Fan \(fan.index)", bundle: .module)
                            .foregroundStyle(.appSecondary)
                        Spacer()
                        Text(verbatim: "\(fan.actualRPM.formatted()) RPM")
                            .monospacedDigit()
                            .foregroundStyle(.appText)
                        (fan.isHeld ? Text("Held", bundle: .module) : Text("Auto", bundle: .module))
                            .foregroundStyle(fan.isHeld ? theme.warning : theme.secondaryText)
                    }
                    .font(theme.ui(.tiny))
                }
                if let sensor = status.hottestSensor {
                    Text("Hottest sensor \(sensor.name): \(String(format: "%.1f", sensor.celsius)) C", bundle: .module)
                        .font(theme.ui(.tiny))
                        .foregroundStyle(.appSecondary)
                }
            } else {
                // The poll's own failure text comes from `FanController`.
                Group {
                    if let statusError = fans.statusError {
                        Text(verbatim: statusError)
                    } else {
                        Text("Reading fan status...", bundle: .module)
                    }
                }
                .font(theme.ui(.tiny))
                .foregroundStyle(.appSecondary)
            }

            Divider()

            HStack(spacing: 8) {
                Button {
                    Task { await fans.pinMax() }
                } label: { Text("Pin Max", bundle: .module) }
                .appAction(.secondary, size: .small)
                .disabled(fans.isBusy)
                Button {
                    Task { await fans.restoreAuto() }
                } label: { Text("Restore Auto", bundle: .module) }
                .appAction(.secondary, size: .small)
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
                Text(verbatim: error)
                    .font(theme.ui(.tiny))
                    .foregroundStyle(AppAttention.color)
            }

            Text("A pinned hold outlives this app: quitting restores Apple's curve unless the toggle above is on. Fans can always be released with `thermalforge auto`.", bundle: .module)
            .font(theme.ui(.tiny))
            .foregroundStyle(.appSecondary)
            .fixedSize(horizontal: false, vertical: true)
        }
        .padding(14)
        .frame(width: 300, alignment: .leading)
    }
}
