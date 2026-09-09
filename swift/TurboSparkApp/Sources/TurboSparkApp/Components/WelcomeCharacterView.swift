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

/// One suggestion on the empty state.
///
/// The old version put four raw prompt strings on the screen ("Write a
/// high-performance Rust function") as bare capsules. A first-time user
/// cannot tell from those whether the app can read their files, whether it
/// needs a model loaded, or what a good prompt looks like -- so each card now
/// carries the invitation AND what it will actually do. `prompt` is what
/// lands in the composer; `title` and `detail` are what the user reads.
struct WelcomeSuggestion: Identifiable {
    let id = UUID()
    let systemImage: String
    let title: String
    let detail: String
    let prompt: String
}

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The empty-state hero: mascot, greeting, composer, and three ways in.
///
/// Redesign notes:
///
/// - The three rows stagger in (`tsEntrance`) rather than appearing at once,
///   so opening a chat reads as the app waking up. Total stagger is 0.16s;
///   past about 0.25s it starts to feel like a loading screen.
/// - The mascot idles (`TSIdlingSparkView`) with a glow that picks up the
///   theme accent, which is what ties the character to the rest of the
///   chrome instead of leaving it a floating sticker.
/// - The subhead names the product's actual promise ("Everything here runs on
///   your Mac") instead of asking "How can I help you today?", which is the
///   one line every assistant already says.
/// - Suggestions are cards with an icon and a description. Same four ideas,
///   trimmed to three so the row does not wrap at the sidebar-open width.
///
/// Everything here still respects reduce-motion through the `TS*` helpers.
@MainActor
public struct WelcomeHeroView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    public let size: CGFloat

    private static let suggestions: [WelcomeSuggestion] = [
        WelcomeSuggestion(
            systemImage: "chevron.left.forwardslash.chevron.right",
            title: "Explain a codebase",
            detail: "Point it at a folder and ask what lives where.",
            prompt: "Explain how this project works"),
        WelcomeSuggestion(
            systemImage: "wand.and.stars",
            title: "Write a function",
            detail: "Give it a signature; get an implementation.",
            prompt: "Write a high-performance Rust function"),
        WelcomeSuggestion(
            systemImage: "rectangle.on.rectangle",
            title: "Review a design",
            detail: "Paste a SwiftUI view and ask for a critique.",
            prompt: "Design a clean SwiftUI component"),
    ]

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
            }
            .frame(maxWidth: 700)
            .tsEntrance(delay: 0.08)

            if model.promptText.isEmpty {
                suggestionCards
                    .tsEntrance(delay: 0.16)
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 24)
        .padding(.vertical, 28)
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
                    center: UnitPoint(x: 0.5, y: 0.26),
                    startRadius: 0,
                    endRadius: 460)
            }
            .allowsHitTesting(false)
        }
        .animation(TSMotion.pane, value: model.promptText.isEmpty)
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
                    .foregroundStyle(.primary)
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
                .foregroundStyle(.secondary)
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

    private var suggestionCards: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("START HERE", bundle: .module)
                .font(theme.ui(.tiny, weight: .semibold))
                .tracking(1.0)
                .foregroundStyle(.tertiary)
                .padding(.leading, 2)
                .accessibilityHidden(true)

            // Grid rather than FlowLayout: three equal cards should stay
            // equal, and `FlowLayout` sizes each child to its content, which
            // made the old pills a ragged row of three different widths.
            LazyVGrid(
                columns: Array(repeating: GridItem(.flexible(), spacing: 9), count: 3),
                spacing: 9
            ) {
                ForEach(Self.suggestions) { suggestion in
                    suggestionCard(suggestion)
                }
            }
        }
        .frame(maxWidth: 700)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Suggested prompts")
    }

    private func suggestionCard(_ suggestion: WelcomeSuggestion) -> some View {
        Button {
            withAnimation(TSMotion.select) {
                model.promptText = suggestion.prompt
            }
        } label: {
            VStack(alignment: .leading, spacing: 7) {
                Image(systemName: suggestion.systemImage)
                    .font(theme.ui(.small, weight: .semibold))
                    .foregroundStyle(theme.accent)
                    .frame(width: 24, height: 24)
                    .background(theme.accent.opacity(0.14), in: RoundedRectangle(cornerRadius: 7, style: .continuous))
                    .accessibilityHidden(true)

                Text(suggestion.title)
                    .font(theme.ui(.small, weight: .semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)

                Text(suggestion.detail)
                    .font(theme.ui(.tiny))
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(12)
            .background {
                RoundedRectangle(cornerRadius: 13, style: .continuous)
                    .fill(TurboSparkTheme.surfaceColor)
                    .overlay {
                        RoundedRectangle(cornerRadius: 13, style: .continuous)
                            .stroke(Color.primary.opacity(0.08), lineWidth: 1)
                    }
            }
            .contentShape(.rect(cornerRadius: 13))
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.975))
        .tsHoverLift(scale: 1.02)
        .help("Insert this prompt into the composer")
        .accessibilityLabel("\(suggestion.title): \(suggestion.detail)")
        .accessibilityHint("Puts this prompt in the composer so you can edit it")
    }
}
