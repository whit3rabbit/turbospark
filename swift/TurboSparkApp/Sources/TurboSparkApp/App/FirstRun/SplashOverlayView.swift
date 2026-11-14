import AppKit
import SwiftUI

/// OC-07: when the splash may leave. Pure, so the three rules are tested
/// without a clock or a view.
///
/// - The mark stays at least `minimumDisplay` once state is ready, so it does
///   not flash.
/// - Reduce Motion dismisses the moment state is ready: no pulse, no floor.
/// - `maximumDisplay` is a hard cap: a ready signal that never arrives must
///   not leave the window covered.
enum SplashPolicy {
    static let minimumDisplay: TimeInterval = 0.4
    static let maximumDisplay: TimeInterval = 3.0

    /// Seconds still to wait before dismissing. Zero means dismiss now.
    static func remaining(
        elapsed: TimeInterval, stateReady: Bool, reduceMotion: Bool
    ) -> TimeInterval {
        if elapsed >= maximumDisplay { return 0 }
        guard stateReady else { return maximumDisplay - elapsed }
        if reduceMotion { return 0 }
        return max(0, minimumDisplay - elapsed)
    }

    /// Whether the overlay should draw at all on a given frame: under Reduce
    /// Motion with state already ready it never does, so there is no flash.
    static func drawsMark(stateReady: Bool, reduceMotion: Bool) -> Bool {
        !(stateReady && reduceMotion)
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
    @State private var shownAt = Date()

    private var reduceMotion: Bool {
        appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
    }

    var body: some View {
        ZStack {
            if isPresented && SplashPolicy.drawsMark(stateReady: stateReady, reduceMotion: reduceMotion) {
                ZStack {
                    Rectangle().fill(.appPage)
                    Text(verbatim: "TurboSpark")
                        .themedFont(.display, weight: .semibold)
                        .foregroundStyle(.appText)
                        .tsPulse(active: true)
                }
                .ignoresSafeArea()
                .transition(.opacity)
                // Swallows clicks for the fraction of a second it is up, so
                // nothing underneath is pressed before it has drawn.
                .contentShape(Rectangle())
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(Text("Starting TurboSpark", bundle: .module))
            }
        }
        .task(id: stateReady) {
            let wait = SplashPolicy.remaining(
                elapsed: Date().timeIntervalSince(shownAt), stateReady: stateReady,
                reduceMotion: reduceMotion)
            if wait > 0 {
                try? await Task.sleep(nanoseconds: UInt64(wait * 1_000_000_000))
                if Task.isCancelled { return }
            }
            await MainActor.run {
                TSMotion.animate(TSMotion.pane) { isPresented = false }
            }
        }
    }
}
