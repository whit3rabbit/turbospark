import SwiftUI

/// A chip in the composer's vocabulary (`SearchToggleButton`): radius 8, a
/// faint fill that deepens on hover, and an accent tint while on.
struct HumanizeChip: View {
    let title: LocalizedStringKey
    let systemImage: String
    var isOn = false
    var iconOnly = false
    let action: () -> Void

    @Environment(\.appTheme) private var theme
    @Environment(\.isEnabled) private var isEnabled
    @State private var hovered = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 5) {
                Image(systemName: systemImage).font(theme.ui(.tiny, weight: .medium))
                if !iconOnly { Text(title, bundle: .module).font(theme.ui(.small, weight: .medium)) }
            }
            .lineLimit(1)
            .padding(.horizontal, 8).padding(.vertical, 5)
            .foregroundStyle(isOn ? Color.primary : Color.secondary)
            .background(
                isOn ? theme.accent.opacity(0.14) : Color.primary.opacity(hovered && isEnabled ? 0.08 : 0.04),
                in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay {
                if isOn {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(theme.accent.opacity(0.3), lineWidth: 0.5)
                }
            }
            .opacity(isEnabled ? 1 : 0.45)
            .contentShape(Rectangle())
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.96))
        .onHover { hovered = $0 }
        .animation(TSMotion.hover, value: isOn)
        .fixedSize()
        .help(Text(title, bundle: .module))
        .accessibilityLabel(Text(title, bundle: .module))
    }
}

/// The composer's surface (`PromptComposerView`): radius 20, the composer
/// fill, an accent edge while focused, and a soft shadow. A fixed header and
/// footer around a flexible body keep two cards aligned row for row.
struct HumanizeCard<Header: View, Content: View, Footer: View>: View {
    var focused = false
    @ViewBuilder var header: Header
    @ViewBuilder var content: Content
    @ViewBuilder var footer: Footer

    @Environment(\.appTheme) private var theme

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) { header }
                .padding(.horizontal, 16).frame(height: 46)
            Divider().opacity(0.6)
            content.frame(maxWidth: .infinity, maxHeight: .infinity)
            Divider().opacity(0.6)
            HStack(spacing: 8) { footer }
                .padding(.horizontal, 12).frame(height: 46)
        }
        .background {
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .fill(theme.composerBackground)
                .overlay {
                    RoundedRectangle(cornerRadius: 20, style: .continuous)
                        .stroke(
                            focused ? theme.accent.opacity(0.55) : Color.primary.opacity(theme.isDark ? 0.18 : 0.14),
                            lineWidth: focused ? 1.5 : 1.0)
                }
                .shadow(color: Color.black.opacity(theme.isDark ? 0.28 : 0.08), radius: focused ? 10 : 6, x: 0, y: 2)
        }
        .animation(.easeInOut(duration: 0.15), value: focused)
    }
}

/// The chat model bar (`ModelLoaderControl`) for the humanizer: the same
/// capsule, status dot, two-line label, and Load or Eject action. Its menu
/// holds the destructive and navigational actions so the bar stays one line.
struct HumanizeModelBar: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: HumanizeController
    let onDelete: () -> Void

    @Environment(\.appTheme) private var theme
    @ScaledMetric private var barHeight: CGFloat = 34

    private var downloadPending: Bool {
        model.isModelDownloadPending(.catalog(alias: HumanizeController.modelAlias))
    }

    private var state: ModelLoaderControl.LoaderState {
        if downloadPending { return .installing }
        if controller.phase == .loadingModel { return .opening }
        if !controller.isInstalled { return .noneInstalled }
        return controller.engineLoaded ? .ready : .notLoaded
    }

    var body: some View {
        HStack(spacing: 0) {
            chooser
            Rectangle().fill(.appBorder).frame(width: 0.5, height: barHeight * 0.55)
            action
        }
        .frame(height: barHeight)
        .background(TurboSparkTheme.surfaceColor, in: Capsule())
        .overlay {
            Capsule().stroke(
                controller.engineLoaded ? theme.accent.opacity(0.32) : TurboSparkTheme.hairlineColor, lineWidth: 0.5)
        }
        .animation(TSMotion.select, value: controller.engineLoaded)
        .animation(TSMotion.hover, value: state)
        .fixedSize()
    }

    private var chooser: some View {
        Menu {
            if controller.isInstalled {
                Button(role: .destructive, action: onDelete) {
                    Label { Text("Delete Model", bundle: .module) } icon: { Image(systemName: "trash") }
                }
                .disabled(controller.isBusy || model.isRunning || model.opening)
                Divider()
            }
            Button { model.activeSection = .modelManager } label: {
                Label { Text("Manage installed models…", bundle: .module) } icon: { Image(systemName: "internaldrive") }
            }
        } label: {
            HStack(spacing: 8) {
                ModelStatusDot(state: state, accent: theme.accent)
                VStack(alignment: .leading, spacing: 1) {
                    Text(verbatim: HumanizeController.modelAlias)
                        .font(theme.ui(.small, weight: .semibold)).foregroundStyle(.appText)
                    secondaryText.font(theme.ui(.micro)).foregroundStyle(.appSecondary)
                }
                .lineLimit(1)
                if state == .opening {
                    ModelLoadProgressBar(fraction: nil, tint: theme.accent).frame(width: 42)
                } else {
                    Image(systemName: "chevron.up.chevron.down")
                        .font(theme.ui(.micro, weight: .bold)).foregroundStyle(.tertiary)
                        .accessibilityHidden(true)
                }
            }
            .padding(.leading, 10).padding(.trailing, 9)
            .frame(maxHeight: .infinity)
            .contentShape(Rectangle())
        }
        // `.button`, not `.borderlessButton`: the latter flattens a custom
        // label to its first Text on the macOS 14 SDK (see ModelLoaderControl).
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .accessibilityElement(children: .combine)
    }

    @ViewBuilder private var secondaryText: some View {
        switch state {
        case .installing: Text("Installing", bundle: .module)
        case .opening: Text("Loading", bundle: .module)
        case .ready:
            Text("Ready", bundle: .module) + Text(verbatim: " -- \(HumanizePlanner.window.formatted()) ctx")
        case .notLoaded: Text("Not loaded", bundle: .module)
        case .noneInstalled, .noneSelected: Text("Not downloaded", bundle: .module)
        }
    }

    @ViewBuilder private var action: some View {
        if controller.engineLoaded {
            barButton("Eject", systemImage: "eject.fill", tint: nil) { controller.unload() }
                .disabled(controller.isBusy)
        } else if controller.isInstalled {
            barButton("Load", systemImage: "play.fill", tint: theme.accent) { controller.load() }
                .disabled(controller.isBusy)
        } else if downloadPending {
            barButton("Downloads", systemImage: "arrow.down.circle", tint: nil) {
                model.isDownloadManagerExpanded = true
            }
        } else {
            barButton("Download", systemImage: "arrow.down.circle.fill", tint: theme.accent) {
                controller.installHandler()
            }
            .disabled(!model.canQueueModelDownload(.catalog(alias: HumanizeController.modelAlias)))
        }
    }

    private func barButton(
        _ title: LocalizedStringKey, systemImage: String, tint: Color?, action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Label { Text(title, bundle: .module) } icon: { Image(systemName: systemImage) }
                .font(theme.ui(.tiny, weight: .semibold))
                .labelStyle(.titleAndIcon)
                .imageScale(.small)
                .foregroundStyle(tint.map { AnyShapeStyle($0) } ?? AnyShapeStyle(.appSecondary))
                .padding(.horizontal, 11)
                .frame(maxHeight: .infinity)
                .contentShape(Rectangle())
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.94))
    }
}

/// The run button between the cards. Idle and ready, a slow ripple invites
/// the first run; while reading the draft the ring fills with prefill; while
/// writing an arc spins. Every motion is state-driven and stops with reduce
/// motion on.
struct HumanizeRunButton: View {
    let busy: Bool
    let ready: Bool
    let prefill: Double?
    let arrow: String
    var invite = false
    let action: () -> Void

    @Environment(\.appTheme) private var theme

    var body: some View {
        TSMotionContext { reduceMotion in
            let ink = ready ? theme.accent.contrastForeground : Color.secondary.opacity(0.5)
            Button(action: action) {
                ZStack {
                    if invite, ready, !busy, !reduceMotion { RunRipple(color: theme.accent) }
                    Circle().fill(ready ? AnyShapeStyle(theme.accent.gradient) : AnyShapeStyle(Color.primary.opacity(0.08)))
                    if busy {
                        Circle().stroke(ink.opacity(0.22), lineWidth: 2.5).padding(4)
                        if let prefill {
                            Circle().trim(from: 0, to: prefill)
                                .stroke(ink, style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
                                .rotationEffect(.degrees(-90)).padding(4)
                                .animation(TSMotion.hover, value: prefill)
                        } else if !reduceMotion {
                            // Torn down with the view, unlike a repeatForever
                            // animation that keeps a display link alive.
                            TimelineView(.animation) { context in
                                let turn = context.date.timeIntervalSinceReferenceDate
                                    .truncatingRemainder(dividingBy: 1.1) / 1.1
                                Circle().trim(from: 0, to: 0.28)
                                    .stroke(ink, style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
                                    .rotationEffect(.degrees(turn * 360 - 90)).padding(4)
                            }
                        }
                    }
                    Image(systemName: busy ? "stop.fill" : arrow)
                        .font(theme.ui(busy ? .callout : .large, weight: .bold))
                        .foregroundStyle(ink)
                        .contentTransition(.symbolEffect(.replace))
                }
                .frame(width: 50, height: 50)
                .shadow(color: ready && !busy ? theme.accent.opacity(0.4) : .clear, radius: 8, y: 2)
                .contentShape(Circle())
            }
            .buttonStyle(TSPressScaleStyle(scale: 0.92))
        }
    }
}

private struct RunRipple: View {
    let color: Color

    var body: some View {
        Circle().stroke(color.opacity(0.55), lineWidth: 2)
            .phaseAnimator([false, true]) { view, expanded in
                view.scaleEffect(expanded ? 1.45 : 1).opacity(expanded ? 0 : 0.9)
            } animation: { expanded in
                expanded ? .easeOut(duration: 1.8) : .linear(duration: 0.01)
            }
            .accessibilityHidden(true)
    }
}

/// Placeholder lines with a passing shimmer, shown before the first words of
/// a rewrite arrive.
struct HumanizeSkeleton: View {
    @Environment(\.appTheme) private var theme
    private static let widths: [CGFloat] = [0.94, 0.86, 0.97, 0.58, 0.9, 0.76, 0.42]

    var body: some View {
        TSMotionContext { reduceMotion in
            GeometryReader { geometry in
                let width = geometry.size.width
                ZStack(alignment: .topLeading) {
                    bars(width).foregroundStyle(Color.primary.opacity(0.07))
                    if !reduceMotion {
                        Color.clear
                            .overlay(alignment: .leading) {
                                LinearGradient(
                                    colors: [.clear, theme.accent.opacity(0.25), .clear],
                                    startPoint: .leading, endPoint: .trailing)
                                .frame(width: width * 0.4)
                                .phaseAnimator([false, true]) { view, moved in
                                    view.offset(x: moved ? width : -width * 0.4)
                                } animation: { moved in
                                    moved ? .easeInOut(duration: 1.4) : .linear(duration: 0.01)
                                }
                            }
                            .mask(alignment: .topLeading) { bars(width) }
                    }
                }
            }
        }
        .frame(height: CGFloat(Self.widths.count) * 21)
        .accessibilityHidden(true)
    }

    private func bars(_ width: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            ForEach(Array(Self.widths.enumerated()), id: \.offset) { _, fraction in
                Capsule().frame(width: width * fraction, height: 9)
            }
        }
    }
}
