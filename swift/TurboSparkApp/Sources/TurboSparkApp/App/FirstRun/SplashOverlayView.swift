import AppKit
import SwiftUI

/// OC-07: when the splash may draw and leave. Pure, so the rules are tested
/// without a clock or a view.
///
/// The splash covers SLOW loads only:
/// - It draws only if state is still not ready after `appearDelay`. A launch
///   that is ready sooner never shows it, so there is no artificial delay.
/// - Once drawn it stays at least `minimumOnScreen`, so it never flashes.
/// - Reduce Motion never draws it.
/// - `maximumDisplay` is a hard cap on the whole wait: a ready signal that
///   never arrives must not leave the window covered.
enum SplashPolicy {
    static let appearDelay: TimeInterval = 0.4
    static let minimumOnScreen: TimeInterval = 0.4
    static let maximumDisplay: TimeInterval = 3.0

    /// Whether the mark should be on screen at `elapsed` since launch.
    static func shouldDraw(
        elapsed: TimeInterval, stateReady: Bool, reduceMotion: Bool
    ) -> Bool {
        !reduceMotion && !stateReady && elapsed >= appearDelay && elapsed < maximumDisplay
    }

    /// Seconds to wait before the draw decision, while state is not ready.
    static func drawDelay(elapsed: TimeInterval) -> TimeInterval {
        max(0, appearDelay - elapsed)
    }

    /// Seconds still to wait before dismissing. Zero means dismiss now.
    /// `shownFor` is how long the mark has been on screen, nil if it never
    /// drew.
    static func dismissWait(
        elapsed: TimeInterval, shownFor: TimeInterval?, stateReady: Bool
    ) -> TimeInterval {
        if elapsed >= maximumDisplay { return 0 }
        guard stateReady else { return maximumDisplay - elapsed }
        guard let shownFor else { return 0 }
        return max(0, minimumOnScreen - shownFor)
    }
}

/// OC-07: the in-window launch splash. An overlay on the one main window, not
/// a second window, so the vault gate keeps its place beneath it. Theme
/// colors only; the pulse is decorative and static under Reduce Motion.
struct SplashOverlayView: View {
    @Binding var isPresented: Bool
    /// False only while an unlock is in flight. A locked vault is ready: the
    /// unlock screen is what the splash hands off to.
    let stateReady: Bool

    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion
    @State private var launchedAt = Date()
    @State private var shownAt: Date?

    private var reduceMotion: Bool {
        appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
    }

    var body: some View {
        ZStack {
            if isPresented && shownAt != nil {
                ZStack {
                    Rectangle().fill(.appPage)
                    Text(verbatim: "TurboSpark")
                        .themedFont(.display, weight: .semibold)
                        .foregroundStyle(.appText)
                        .tsPulse(active: true)
                }
                .ignoresSafeArea()
                .transition(.opacity)
                // Swallows clicks while it is up.
                .contentShape(Rectangle())
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(Text("Starting TurboSpark", bundle: .module))
            }
        }
        .task(id: stateReady) {
            let launchedAt = self.launchedAt
            func elapsed() -> TimeInterval { Date().timeIntervalSince(launchedAt) }
            func sleep(_ seconds: TimeInterval) async {
                if seconds > 0 { try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000)) }
            }
            if !stateReady {
                await sleep(SplashPolicy.drawDelay(elapsed: elapsed()))
                if Task.isCancelled { return }
                if SplashPolicy.shouldDraw(
                    elapsed: elapsed(), stateReady: false, reduceMotion: reduceMotion)
                {
                    shownAt = Date()
                }
            }
            let shownFor = shownAt.map { Date().timeIntervalSince($0) }
            await sleep(SplashPolicy.dismissWait(
                elapsed: elapsed(), shownFor: shownFor, stateReady: stateReady))
            if Task.isCancelled { return }
            await MainActor.run {
                TSMotion.animate(TSMotion.pane) { isPresented = false }
            }
        }
    }
}
