import AppKit
import SwiftUI

/// Thread-safe asset loader for the bundled waving flame welcome character.
public final class WelcomeCharacterAssetCache {
    public static let shared = WelcomeCharacterAssetCache()
    private var cachedImage: NSImage?

    private init() {
        if let url = Bundle.module.url(forResource: "WelcomeCharacter", withExtension: "png"),
           let img = NSImage(contentsOf: url) {
            self.cachedImage = img
        }
    }

    public var welcomeImage: NSImage? {
        cachedImage
    }
}

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The empty-state hero: mascot, greeting, composer, and model loader.
///
/// Redesign notes:
///
/// - The rows stagger in (`tsEntrance`) rather than appearing at once,
///   so opening a chat reads as the app waking up.
/// - The mascot idles (`TSIdlingSparkView`) with a glow that picks up the
///   theme accent, which is what ties the character to the rest of the
///   chrome instead of leaving it a floating sticker.
/// - The subhead names the product's actual promise ("Everything here runs on
///   your Mac") instead of asking "How can I help you today?", which is the
///   one line every assistant already says.
///
/// Everything here still respects reduce-motion through the `TS*` helpers.
@MainActor
public struct WelcomeHeroView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    public let size: CGFloat

    public init(model: AppModel, size: CGFloat = 92) {
        self.model = model
        self.size = size
    }

    public var body: some View {
        VStack(spacing: 26) {
            greetingHeader
                .tsEntrance()

            VStack(spacing: 12) {
                PromptComposerView(model: model)
                ErrorBanner(model: model)
                // Appended in both chat hosts rather than placed at a fixed
                // index: the two order the error banner differently, and last
                // is the only position that keeps the pill at the floor of the
                // chrome in both without wedging between a composer and its
                // error.
                ModelLoaderControl(model: model, density: .compact)
            }
            .frame(maxWidth: 700)
            .tsEntrance(delay: 0.08)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 28)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        // A soft accent wash behind the hero, plus a scattering of tiny
        // dots in dark mode for a "night sky" backdrop. This is the only
        // place in the app with a gradient; it exists to stop the empty
        // state from reading as a blank window, and it stays subtle enough
        // that text on top of it still clears contrast.
        .background {
            ZStack {
                starfieldOverlay
                RadialGradient(
                    colors: [theme.accent.opacity(theme.isDark ? 0.10 : 0.07), .clear],
                    center: UnitPoint(x: 0.5, y: 0.45),
                    startRadius: 0,
                    endRadius: 460)
            }
            .allowsHitTesting(false)
        }
    }

    /// Fixed sparkle positions for the dark-mode backdrop: (x fraction, y
    /// fraction, diameter, opacity). Fixed rather than randomized per render
    /// so the scatter doesn't reshuffle on every layout pass, and kept out
    /// of light mode entirely since the same low-opacity dots read as dust
    /// on a white background rather than as stars.
    private static let starfieldDots: [(x: CGFloat, y: CGFloat, diameter: CGFloat, opacity: Double)] = [
        (0.08, 0.12, 2.5, 0.35), (0.18, 0.32, 1.5, 0.25), (0.27, 0.08, 2.0, 0.30),
        (0.35, 0.42, 1.5, 0.20), (0.46, 0.18, 2.5, 0.40), (0.58, 0.06, 1.5, 0.28),
        (0.63, 0.35, 2.0, 0.22), (0.72, 0.14, 1.5, 0.32), (0.81, 0.28, 2.5, 0.24),
        (0.89, 0.10, 1.5, 0.30), (0.93, 0.38, 2.0, 0.18), (0.14, 0.48, 1.5, 0.16),
        (0.5, 0.5, 2.0, 0.14), (0.7, 0.46, 1.5, 0.18)
    ]

    @ViewBuilder
    private var starfieldOverlay: some View {
        if theme.isDark {
            GeometryReader { proxy in
                ForEach(Array(Self.starfieldDots.enumerated()), id: \.offset) { _, dot in
                    Circle()
                        .fill(theme.accent.opacity(dot.opacity))
                        .frame(width: dot.diameter, height: dot.diameter)
                        .position(x: proxy.size.width * dot.x, y: proxy.size.height * dot.y)
                }
            }
        }
    }

    @AppStorage(AppLanguage.storageKey) private var languageRawValue = AppLanguage.system.rawValue
    @State private var currentGreeting: Greeting?

    private var activeLanguage: AppLanguage {
        AppLanguage.resolve(languageRawValue)
    }

    private var greetingHeader: some View {
        VStack(spacing: 14) {
            TSIdlingSparkView(size: size)

            Button(action: cycleGreeting) {
                Text(displayGreeting)
                    .font(theme.ui(.display, weight: .regular, systemDesign: .serif))
                    .foregroundStyle(.appText)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                    // The greeting is re-rolled on click, so the swap is a
                    // real content change and gets a crossfade.
                    .contentTransition(.opacity)
                    .contentShape(.rect)
            }
            .buttonStyle(TSPressScaleStyle(scale: 0.985))
            .help(Text("Click to change greeting phrase", bundle: .module))
            .accessibilityLabel(displayGreeting)
            .accessibilityHint("Changes the greeting phrase")

            Text("Everything here runs on your Mac. Nothing leaves it.", bundle: .module)
                .font(theme.ui(.callout))
                .foregroundStyle(.appSecondary)
        }
        .onAppear {
            if currentGreeting == nil {
                currentGreeting = GreetingProvider.shared.availableGreetings().randomElement()
            }
        }
    }

    /// The active greeting string translated into the currently selected language.
    private var displayGreeting: String {
        if let greeting = currentGreeting {
            return greeting.text(for: activeLanguage)
        }
        let initial = GreetingProvider.shared.availableGreetings().randomElement()
        return initial?.text(for: activeLanguage) ?? "Welcome!"
    }

    /// Cycles to another greeting from the currently valid time-of-day candidates.
    private func cycleGreeting() {
        let pool = GreetingProvider.shared.availableGreetings()
        let alternatives = pool.filter { $0.id != currentGreeting?.id }
        withAnimation(TSMotion.select) {
            currentGreeting = alternatives.randomElement() ?? pool.randomElement()
        }
    }
}
