import AppKit
import SwiftUI

/// Thread-safe asset loader for the bundled Memory mascot character.
public final class MemoryCharacterAssetCache {
    public static let shared = MemoryCharacterAssetCache()
    private var cachedImage: NSImage?

    private init() {
        if let url = Bundle.module.url(forResource: "MemoryCharacter", withExtension: "png"),
           let img = NSImage(contentsOf: url) {
            self.cachedImage = img
        }
    }

    public var memoryImage: NSImage? {
        cachedImage
    }
}

/// The Memory mascot character holding the MEMORY.md sign, idling with
/// a breathing glow and subtle bobbing animation.
///
/// Matches the visual language of `TSIdlingSparkView` on the welcome screen,
/// respecting system reduce-motion preferences.
@MainActor
public struct TSIdlingMemorySparkView: View {
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
            if let nsImage = MemoryCharacterAssetCache.shared.memoryImage {
                Image(nsImage: nsImage)
                    .resizable()
                    .interpolation(.high)
                    .scaledToFit()
            } else {
                Image(systemName: "brain")
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
