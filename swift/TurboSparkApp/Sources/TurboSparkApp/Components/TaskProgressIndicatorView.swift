import AppKit
import SwiftUI

/// Animated progress indicator displaying the TurboSpark flame character bobbing up and down.
/// Similar to Claude's task indicator, it provides smooth visual feedback that a task is actively running.
public struct TaskProgressIndicatorView: View {
    public let text: String?
    public let size: CGFloat
    public let isRunning: Bool

    @State private var isFloating = false

    public init(text: String? = nil, size: CGFloat = 16, isRunning: Bool = true) {
        self.text = text
        self.size = size
        self.isRunning = isRunning
    }

    public var body: some View {
        if isRunning {
            HStack(spacing: 6) {
                TaskProgressFlameIcon(size: size, isFloating: $isFloating)

                // The caller owns the text's localization. The old
                // `LocalizedStringKey` lookup ran against the main bundle,
                // which holds no strings in a SwiftPM build.
                if let text = text, !text.isEmpty {
                    Text(verbatim: text)
                        .themedFont(fitting: max(11, size * 0.8), weight: .regular)
                        .foregroundStyle(.appSecondary)
                }
            }
            .onAppear {
                startFloatingAnimation()
            }
            .accessibilityElement(children: .combine)
            .accessibilityLabel(text.map { Text(verbatim: $0) } ?? Text("Task in progress", bundle: .module))
            .accessibilityHint(Text("Animates while task is executing", bundle: .module))
            .transition(.opacity.combined(with: .scale(scale: 0.95)))
        }
    }

    // The binding drives the icon's bob, so this has to honor Reduce
    // Motion itself: the icon's own guard only covers its internal state.
    @MainActor
    private func startFloatingAnimation() {
        TSMotion.animate(
            .easeInOut(duration: 0.9)
            .repeatForever(autoreverses: true)
        ) {
            isFloating = true
        }
    }
}

/// Floating character icon with subtle vertical bobbing animation.
public struct TaskProgressFlameIcon: View {
    public let size: CGFloat
    @Binding var isFloating: Bool

    @State private var internalFloating = false
    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion

    public init(size: CGFloat = 16, isFloating: Binding<Bool>? = nil) {
        self.size = size
        if let binding = isFloating {
            self._isFloating = binding
        } else {
            self._isFloating = .constant(false)
        }
    }

    private var activeFloating: Bool {
        isFloating || internalFloating
    }

    public var body: some View {
        Group {
            if let nsImage = TaskProgressAssetCache.shared.progressImage {
                Image(nsImage: nsImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: size, height: size)
            } else {
                Image(systemName: "flame.fill")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .foregroundStyle(Color.orange)
                    .frame(width: size, height: size)
            }
        }
        .offset(y: activeFloating ? -2.5 : 2.5)
        .onAppear {
            // The bob repeats forever, so honor Reduce motion by not starting it.
            guard !appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion) else { return }
            withAnimation(
                .easeInOut(duration: 0.9)
                .repeatForever(autoreverses: true)
            ) {
                internalFloating = true
            }
        }
        .accessibilityHidden(true)
    }
}

/// Convenience aliases for the progress character icon
public typealias ProgressCharacterIcon = TaskProgressFlameIcon
public typealias TaskProgressCharacterIcon = TaskProgressFlameIcon

/// Thread-safe asset cache for loading the bundled progress character PNG.
public final class TaskProgressAssetCache {
    public static let shared = TaskProgressAssetCache()
    private var cachedImage: NSImage?

    private init() {
        if let url = Bundle.module.url(forResource: "progress_character", withExtension: "png") ??
                     Bundle.module.url(forResource: "TaskProgress", withExtension: "png"),
           let img = NSImage(contentsOf: url) {
            self.cachedImage = img
        }
    }

    public var progressImage: NSImage? {
        if cachedImage == nil {
            if let url = Bundle.module.url(forResource: "progress_character", withExtension: "png") ??
                         Bundle.module.url(forResource: "TaskProgress", withExtension: "png"),
               let img = NSImage(contentsOf: url) {
                self.cachedImage = img
            }
        }
        return cachedImage
    }
}

