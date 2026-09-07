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
/// Animated welcome hero view displayed on app launch and empty chat transcripts.
/// Displays the Claude-inspired centered greeting with embedded floating prompt composer.
@MainActor
public struct WelcomeHeroView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    public let size: CGFloat

    private static let samplePrompts: [String] = [
        "Explain how this project works",
        "Write a high-performance Rust function",
        "Design a clean SwiftUI component",
        "Review architecture and suggest improvements"
    ]

    public init(model: AppModel, size: CGFloat = 44) {
        self.model = model
        self.size = size
    }

    public var body: some View {
        VStack(spacing: 28) {
            greetingHeader

            VStack(spacing: 12) {
                PromptComposerView(model: model)
                ErrorBanner(model: model)
            }
            .frame(maxWidth: 680)

            if model.promptText.isEmpty {
                samplePromptPills
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 24)
        .padding(.vertical, 28)
        .animation(.smooth(duration: 0.22), value: model.promptText.isEmpty)
    }

    @AppStorage(AppLanguage.storageKey) private var languageRawValue = AppLanguage.system.rawValue
    @State private var currentGreeting: Greeting?

    private var activeLanguage: AppLanguage {
        AppLanguage.resolve(languageRawValue)
    }

    private var greetingHeader: some View {
        VStack(spacing: 8) {
            Button(action: cycleGreeting) {
                HStack(spacing: 10) {
                    welcomeCharacter

                    Text(displayGreeting)
                        .font(theme.ui(points: 28, weight: .medium, systemDesign: .serif))
                        .foregroundStyle(.primary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .buttonStyle(.plain)
            .help(Text("Click to change greeting phrase", bundle: .module))

            Text("How can I help you today?", bundle: .module)
                .font(theme.ui(points: 15))
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
        currentGreeting = alternatives.randomElement() ?? pool.randomElement()
    }

    /// The bundled waving spark, with the SF symbol only as a fallback for a
    /// build whose resource bundle did not come along. The asset is the
    /// product's character; a sun here reads as a different app.
    @ViewBuilder
    private var welcomeCharacter: some View {
        if let image = WelcomeCharacterAssetCache.shared.welcomeImage {
            Image(nsImage: image)
                .resizable()
                .interpolation(.high)
                .scaledToFit()
                .frame(width: size, height: size)
                .accessibilityHidden(true)
        } else {
            Image(systemName: "flame.fill")
                .font(theme.ui(points: 22))
                .foregroundStyle(TurboSparkTheme.accentColor)
                .accessibilityHidden(true)
        }
    }

    private var samplePromptPills: some View {
        FlowLayout(spacing: 8, lineSpacing: 8, alignment: .center) {
            ForEach(Self.samplePrompts, id: \.self) { prompt in
                Button(action: {
                    model.promptText = prompt
                }) {
                    Text(LocalizedStringKey(prompt))
                        .font(theme.ui(points: 13))
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color(nsColor: .controlBackgroundColor).opacity(0.8))
                        .overlay(
                            RoundedRectangle(cornerRadius: 14)
                                .stroke(Color.primary.opacity(0.1), lineWidth: 1)
                        )
                        .clipShape(RoundedRectangle(cornerRadius: 14))
                }
                .buttonStyle(.plain)
                .help("Insert prompt into editor")
            }
        }
        .frame(maxWidth: 680)
        .padding(.horizontal, 12)
    }
}
