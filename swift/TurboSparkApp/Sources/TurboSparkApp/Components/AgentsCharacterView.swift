import AppKit
import SwiftUI

/// Thread-safe asset loader for the bundled AGENTS mascot character.
public final class AgentsCharacterAssetCache {
    public static let shared = AgentsCharacterAssetCache()
    private var cachedImage: NSImage?

    private init() {
        if let url = Bundle.module.url(forResource: "AgentsCharacter", withExtension: "png"),
           let img = NSImage(contentsOf: url) {
            self.cachedImage = img
        }
    }

    public var agentsImage: NSImage? {
        cachedImage
    }
}

/// The AGENTS mascot character in a tuxedo holding the AGENTS.md sign,
/// idling with a breathing glow and subtle bobbing animation.
///
/// Matches the visual language of `TSIdlingSparkView`, `TSIdlingMemorySparkView`,
/// and `TSIdlingSoulSparkView`, respecting system reduce-motion preferences.
@MainActor
public struct TSIdlingAgentsSparkView: View {
    public let size: CGFloat
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion

    public init(size: CGFloat = 80) {
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
            colors: [TurboSparkTheme.accentColor.opacity(0.28), .clear],
            center: .center,
            startRadius: 0,
            endRadius: size * 0.62)

        if reduceMotion {
            Circle().fill(base)
        } else {
            Circle()
                .fill(base)
                .phaseAnimator([0.40, 0.82]) { circle, opacity in
                    circle.opacity(opacity)
                } animation: { _ in
                    .easeInOut(duration: 1.8)
                }
        }
    }

    @ViewBuilder
    private var spark: some View {
        let image = Group {
            if let nsImage = AgentsCharacterAssetCache.shared.agentsImage {
                Image(nsImage: nsImage)
                    .resizable()
                    .interpolation(.high)
                    .scaledToFit()
            } else {
                Image(systemName: "person.2.badge.gearshape")
                    .themedFont(fitting: size * 0.5)
                    .foregroundStyle(TurboSparkTheme.accentColor)
            }
        }
        .frame(width: size, height: size)
        .shadow(color: .black.opacity(0.28), radius: size * 0.08, y: size * 0.06)

        if reduceMotion {
            image
        } else {
            image.phaseAnimator([0.0, 1.0]) { view, phase in
                view
                    .offset(y: phase == 0 ? 0 : -size * 0.065)
                    .rotationEffect(.degrees(phase == 0 ? -1.2 : 1.2))
            } animation: { _ in
                .easeInOut(duration: 2.4)
            }
        }
    }
}
