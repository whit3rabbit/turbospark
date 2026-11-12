import SwiftUI

/// Frames in the Humanize section that hold text grow with the user's text
/// size instead of clipping it. Fonts scale with the base UI size and the
/// text-size setting, so a fixed 46pt card row or a 300pt panel stayed put
/// while the text inside them grew.
enum HumanizeMetrics {
    /// How far the UI font sits above the default. One source with
    /// `AppMetrics.textScale`, which other workspaces use too.
    static func textScale(_ theme: ResolvedAppTheme) -> CGFloat {
        AppMetrics.textScale(theme)
    }

    /// A card's header and footer row.
    static func rowHeight(_ theme: ResolvedAppTheme) -> CGFloat {
        (46 * textScale(theme)).rounded()
    }

    /// The narrowest the Advanced panel gets before its text only wraps.
    static let advancedMinWidth: CGFloat = 240

    /// The Advanced panel's preferred width. Capped so very large text wraps
    /// inside the panel instead of squeezing the two cards.
    static func advancedWidth(_ theme: ResolvedAppTheme) -> CGFloat {
        min((300 * textScale(theme)).rounded(), 420)
    }
}

/// The composer's surface (`appComposerSurface`, as `PromptComposerView`
/// draws it): radius 20, the composer fill, an accent edge while focused,
/// and a soft shadow. A fixed header and footer around a flexible body keep
/// two cards aligned row for row.
struct HumanizeCard<Header: View, Content: View, Footer: View>: View {
    var focused = false
    @ViewBuilder var header: Header
    @ViewBuilder var content: Content
    @ViewBuilder var footer: Footer

    @Environment(\.appTheme) private var theme

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) { header }
                .padding(.horizontal, 16).frame(height: HumanizeMetrics.rowHeight(theme))
            // Softened seams, but never under Increase Contrast, which raises
            // borders instead of letting a surface lower them.
            Divider().opacity(theme.isHighContrast ? 1 : 0.6)
            content.frame(maxWidth: .infinity, maxHeight: .infinity)
            Divider().opacity(theme.isHighContrast ? 1 : 0.6)
            HStack(spacing: 8) { footer }
                .padding(.horizontal, 12).frame(height: HumanizeMetrics.rowHeight(theme))
        }
        .appComposerSurface(focused: focused, halo: false)
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
        if !controller.isInstalled {
            // On a drive that is away is not "not downloaded": downloading it
            // again would only be refused, and the drive is what to connect.
            return notConnectedStatement == nil ? .noneInstalled : .notConnected
        }
        return controller.engineLoaded ? .ready : .notLoaded
    }

    private var notConnectedStatement: String? { controller.notConnectedStatement() }

    var body: some View {
        HStack(spacing: 0) {
            chooser
            if state != .notConnected {
                Rectangle().fill(.appBorder).frame(width: 0.5, height: barHeight * 0.55)
                action
            }
        }
        .frame(height: barHeight)
        .background(.appSurface, in: Capsule())
        .overlay {
            Capsule().stroke(
                controller.engineLoaded ? (theme.isHighContrast ? theme.accent : theme.accent.opacity(0.32)) : theme.border, lineWidth: 0.5)
        }
        .tsAnimation(TSMotion.select, value: controller.engineLoaded)
        .tsAnimation(TSMotion.hover, value: state)
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
                Label { Text("Manage installed models\u{2026}", bundle: .module) } icon: { Image(systemName: "internaldrive") }
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
                        .font(theme.ui(.micro, weight: .bold)).foregroundStyle(.appSecondary)
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
        case .notConnected: Text(verbatim: notConnectedStatement ?? "")
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
        HumanizeBarButton(title: title, systemImage: systemImage, tint: tint, action: action)
    }
}

private struct HumanizeBarButton: View {
    let title: LocalizedStringKey
    let systemImage: String
    let tint: Color?
    let action: () -> Void

    @Environment(\.appTheme) private var theme
    @Environment(\.isEnabled) private var isEnabled
    @State private var hovered = false

    var body: some View {
        Button(action: action) {
            Label { Text(title, bundle: .module) } icon: { Image(systemName: systemImage) }
                .font(theme.ui(.tiny, weight: .semibold))
                .labelStyle(.titleAndIcon)
                .imageScale(.small)
                .foregroundStyle(tint.map { AnyShapeStyle($0) } ?? AnyShapeStyle(.appSecondary))
                .padding(.horizontal, 11)
                .frame(maxHeight: .infinity)
                .background(hovered && isEnabled ? (tint.map { AnyShapeStyle($0.opacity(0.12)) } ?? AnyShapeStyle(.appSecondary.opacity(0.12))) : AnyShapeStyle(Color.clear))
                .contentShape(Rectangle())
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.94))
        .onHover { hovered = $0 }
        .tsAnimation(TSMotion.hover, value: hovered)
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
    @State private var hovered = false

    var body: some View {
        TSMotionContext { reduceMotion in
            let scale = HumanizeMetrics.textScale(theme)
            let buttonSize = (50 * scale).rounded()
            let inkColor = ready ? theme.accent.contrastForeground : theme.metadataForeground
            let inkStyle: AnyShapeStyle = ready ? AnyShapeStyle(theme.accent.contrastForeground) : AnyShapeStyle(.appSecondary)
            Button(action: action) {
                ZStack {
                    if invite, ready, !busy, !reduceMotion { RunRipple(color: theme.accent) }
                    Circle()
                        .fill(ready ? AnyShapeStyle(theme.accent.gradient) : AnyShapeStyle(.appSurface))
                    if theme.isHighContrast {
                        Circle()
                            .stroke(ready ? theme.accent : theme.border, lineWidth: 1.5)
                    }
                    if busy {
                        Circle()
                            .stroke(inkColor.opacity(0.22), lineWidth: 2.5)
                            .padding(4)
                            .tsPulse(active: !reduceMotion)
                        if let prefill {
                            Circle().trim(from: 0, to: prefill)
                                .stroke(inkColor, style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
                                .rotationEffect(.degrees(-90)).padding(4)
                                .animation(reduceMotion ? nil : TSMotion.hover, value: prefill)
                        } else if !reduceMotion {
                            // Torn down with the view, unlike a repeatForever
                            // animation that keeps a display link alive.
                            TimelineView(.animation) { context in
                                let turn = context.date.timeIntervalSinceReferenceDate
                                    .truncatingRemainder(dividingBy: 1.1) / 1.1
                                Circle().trim(from: 0, to: 0.28)
                                    .stroke(inkColor, style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
                                    .rotationEffect(.degrees(turn * 360 - 90)).padding(4)
                            }
                        }
                    }
                    Image(systemName: busy ? "stop.fill" : arrow)
                        .font(theme.ui(busy ? .callout : .large, weight: .bold))
                        .foregroundStyle(inkStyle)
                        .contentTransition(.symbolEffect(.replace))
                }
                .frame(width: buttonSize, height: buttonSize)
                .scaleEffect(hovered && ready && !busy && !reduceMotion ? 1.05 : 1.0)
                .appPrimaryGlow(ready && !busy)
                .contentShape(Circle())
            }
            .buttonStyle(TSPressScaleStyle(scale: 0.92))
            .onHover { isHovered in
                hovered = isHovered
            }
            .tsAnimation(TSMotion.hover, value: hovered)
            .tsAnimation(TSMotion.hover, value: ready)
            .tsAnimation(TSMotion.select, value: busy)
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
                expanded ? TSMotion.breathe : TSMotion.snap
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
                    bars(width).foregroundStyle(.appSecondary.opacity(0.15))
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
                                    moved ? TSMotion.sweep : TSMotion.snap
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
