import SwiftUI

/// One place for the app's motion vocabulary.
///
/// Every animated site in the app used to spell its own duration inline
/// (`.easeInOut(duration: 0.12)`, `.smooth(duration: 0.15)`,
/// `.easeInOut(duration: 0.2)`), which is why hover, selection and pane
/// transitions all felt slightly out of step with each other. These are the
/// four curves the redesign uses; a call site picks the one that names what
/// it is doing rather than inventing a number.
///
/// All four are cheap: nothing here animates layout on a per-token basis, so
/// they are safe to apply to rows in a `LazyVStack`.
public enum TSMotion {
    /// Hover in/out, chip highlight, focus ring. Should feel instant.
    public static let hover: Animation = .easeOut(duration: 0.13)

    /// Selection moving between rows, segment pills, disclosure open/close.
    public static let select: Animation = .spring(response: 0.32, dampingFraction: 0.82)

    /// Press-down feedback on a button. Faster in than out.
    public static let press: Animation = .spring(response: 0.22, dampingFraction: 0.7)

    /// Panes sliding, greeting entrance, anything that moves a whole block.
    public static let pane: Animation = .spring(response: 0.44, dampingFraction: 0.86)
}

/// Resolves the app's reduce-motion preference over the system one and hands
/// the answer to a closure, so an animated view does not have to repeat the
/// two-source check (`AppearanceManager` first, `\.accessibilityReduceMotion`
/// as its `.system` fallback).
@MainActor
public struct TSMotionContext<Content: View>: View {
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion
    private let content: (Bool) -> Content

    public init(@ViewBuilder content: @escaping (Bool) -> Content) {
        self.content = content
    }

    public var body: some View {
        content(appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion))
    }
}

/// Hover treatment for a row or chip: a touch of lift, no layout change.
///
/// Scale rather than padding on purpose. A hover that changes a row's frame
/// re-lays-out its `LazyVStack` neighbours, which is what made the old chat
/// list feel loose when the pointer crossed it.
private struct TSHoverLift: ViewModifier {
    let scale: CGFloat
    @State private var isHovered = false
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion

    private var reduceMotion: Bool {
        appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
    }

    func body(content: Content) -> some View {
        content
            .scaleEffect(isHovered && !reduceMotion ? scale : 1.0)
            .animation(reduceMotion ? nil : TSMotion.hover, value: isHovered)
            .onHover { isHovered = $0 }
    }
}

/// Fades and lifts a block into place once, on appear.
///
/// Used for the welcome hero's three rows so the empty state assembles rather
/// than snapping in whole. `delay` staggers siblings; keep it under ~0.2s in
/// total or the app feels slow to open a new chat.
private struct TSEntrance: ViewModifier {
    let delay: Double
    @State private var hasAppeared = false
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion

    private var reduceMotion: Bool {
        appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
    }

    func body(content: Content) -> some View {
        content
            .opacity(hasAppeared || reduceMotion ? 1 : 0)
            .offset(y: hasAppeared || reduceMotion ? 0 : 10)
            .onAppear {
                guard !reduceMotion else { return }
                withAnimation(TSMotion.pane.delay(delay)) { hasAppeared = true }
            }
    }
}

public extension View {
    /// Hover lift for rows, cards and chips. Default is deliberately subtle.
    func tsHoverLift(scale: CGFloat = 1.012) -> some View {
        modifier(TSHoverLift(scale: scale))
    }

    /// One-shot entrance. `delay` staggers siblings.
    func tsEntrance(delay: Double = 0) -> some View {
        modifier(TSEntrance(delay: delay))
    }
}

public extension ButtonStyle where Self == TSPressScaleStyle {
    /// `.buttonStyle(.tsPress)` -- plain button plus press feedback.
    static var tsPress: TSPressScaleStyle { TSPressScaleStyle(scale: 0.96) }
}

/// Public wrapper so the `.tsPress` shorthand above can name a concrete type.
public struct TSPressScaleStyle: ButtonStyle {
    let scale: CGFloat
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion

    public init(scale: CGFloat = 0.96) {
        self.scale = scale
    }

    public func makeBody(configuration: Configuration) -> some View {
        let reduce = appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
        return configuration.label
            .scaleEffect(configuration.isPressed && !reduce ? scale : 1.0)
            .animation(reduce ? nil : TSMotion.press, value: configuration.isPressed)
    }
}

/// The mascot, idling.
///
/// A slow bob plus a breathing glow. `phaseAnimator` rather than a repeating
/// `withAnimation`, so the cycle is owned by the view and stops cleanly when
/// it leaves the hierarchy -- a repeatForever on the welcome screen kept a
/// display link alive behind the transcript in the old build.
@MainActor
public struct TSIdlingSparkView: View {
    public let size: CGFloat
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion

    public init(size: CGFloat = 92) {
        self.size = size
    }

    private var reduceMotion: Bool {
        appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
    }

    public var body: some View {
        ZStack {
            glow
            spark
        }
        .frame(width: size, height: size)
        .accessibilityHidden(true)
    }

    @ViewBuilder
    private var glow: some View {
        let base = RadialGradient(
            colors: [TurboSparkTheme.accentColor.opacity(0.30), .clear],
            center: .center,
            startRadius: 0,
            endRadius: size * 0.62)

        if reduceMotion {
            Circle().fill(base)
        } else {
            Circle()
                .fill(base)
                .phaseAnimator([0.45, 0.85]) { circle, opacity in
                    circle.opacity(opacity)
                } animation: { _ in
                    .easeInOut(duration: 1.7)
                }
        }
    }

    @ViewBuilder
    private var spark: some View {
        let image = Group {
            if let nsImage = WelcomeCharacterAssetCache.shared.welcomeImage {
                Image(nsImage: nsImage)
                    .resizable()
                    .interpolation(.high)
                    .scaledToFit()
            } else {
                // Fallback only for a build whose resource bundle did not
                // come along; the asset is the product's character.
                Image(systemName: "flame.fill")
                    .themedFont(fitting: size * 0.5)
                    .foregroundStyle(TurboSparkTheme.accentColor)
            }
        }
        .frame(width: size, height: size)
        .shadow(color: .black.opacity(0.32), radius: size * 0.09, y: size * 0.07)

        if reduceMotion {
            image
        } else {
            image.phaseAnimator([0.0, 1.0]) { view, phase in
                view
                    .offset(y: phase == 0 ? 0 : -size * 0.075)
                    .rotationEffect(.degrees(phase == 0 ? -1.5 : 1.5))
            } animation: { _ in
                .easeInOut(duration: 2.3)
            }
        }
    }
}
