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

/// Animated welcome hero view displayed on app launch and empty chat transcripts.
/// Displays the Claude-inspired centered greeting with embedded floating prompt composer.
public struct WelcomeHeroView: View {
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

            samplePromptPills
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 24)
        .padding(.vertical, 28)
    }

    private var greetingHeader: some View {
        VStack(spacing: 8) {
            HStack(spacing: 10) {
                Image(systemName: "sun.max.fill")
                    .font(.title)
                    .foregroundStyle(TurboSparkTheme.accentColor)
                    .accessibilityHidden(true)

                Text(timeBasedGreeting)
                    .font(.system(size: 28, weight: .medium, design: .serif))
                    .foregroundStyle(.primary)
            }

            Text("How can I help you today?")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
    }

    private var timeBasedGreeting: String {
        let hour = Calendar.current.component(.hour, from: Date())
        let name = NSFullUserName().components(separatedBy: " ").first ?? ""
        let prefix: String
        if hour < 12 {
            prefix = "Good morning"
        } else if hour < 17 {
            prefix = "Good afternoon"
        } else {
            prefix = "Good evening"
        }
        return name.isEmpty ? "\(prefix)" : "\(prefix), \(name)"
    }

    private var samplePromptPills: some View {
        FlowLayout(spacing: 8) {
            ForEach(Self.samplePrompts, id: \.self) { prompt in
                Button(action: {
                    model.promptText = prompt
                }) {
                    Text(LocalizedStringKey(prompt))
                        .font(.caption)
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

/// Flexible multi-line flow layout for sample prompt suggestion chips.
private struct FlowLayout: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let maxWidth = proposal.width ?? 500
        var height: CGFloat = 0
        var currentX: CGFloat = 0
        var currentY: CGFloat = 0
        var lineHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if currentX + size.width > maxWidth, currentX > 0 {
                currentX = 0
                currentY += lineHeight + spacing
                lineHeight = 0
            }
            currentX += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
        height = currentY + lineHeight
        return CGSize(width: maxWidth, height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var currentX = bounds.minX
        var currentY = bounds.minY
        var lineHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if currentX + size.width > bounds.maxX, currentX > bounds.minX {
                currentX = bounds.minX
                currentY += lineHeight + spacing
                lineHeight = 0
            }
            subview.place(at: CGPoint(x: currentX, y: currentY), proposal: .unspecified)
            currentX += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
    }
}
