import SwiftUI

/// OC-07: mounts the first-run flow over the main window and applies its exit.
///
/// A modifier, not RootView body code, so the window shell's already large
/// body does not grow. It sits inside RootView, which only exists once the
/// vault gate has opened a profile, so the flow never draws over the unlock
/// screen and the app-level flag is read after the gate.
struct FirstRunHost: ViewModifier {
    @ObservedObject var model: AppModel
    @StateObject private var coordinator = FirstRunCoordinator()

    func body(content: Content) -> some View {
        content
            .overlay {
                if coordinator.isActive {
                    FirstRunFlowView(coordinator: coordinator)
                        .transition(.opacity)
                }
            }
            .onAppear { [weak model] in
                guard let model else { return }
                coordinator.onExit = { [weak model] exit in
                    guard let model else { return }
                    switch exit {
                    case .newChat:
                        model.createChat()
                        // One turn later so the composer has been shown by
                        // the time it is asked to take focus.
                        DispatchQueue.main.async {
                            NotificationCenter.default.post(name: .focusPrompt, object: nil)
                        }
                    case .modelHub:
                        model.activeSection = .modelHub
                    case .stay:
                        break
                    }
                }
                // The tour no longer auto-runs on first launch. It opens only
                // from Help > Show Welcome Tour (the onReceive below).
            }
            .onReceive(NotificationCenter.default.publisher(for: .showWelcomeTour)) { _ in
                coordinator.replay(hasInstalledModel: !model.installed.isEmpty)
            }
            .onChange(of: model.installed.isEmpty) { _, isEmpty in
                coordinator.installedModelChanged(!isEmpty)
            }
    }
}
