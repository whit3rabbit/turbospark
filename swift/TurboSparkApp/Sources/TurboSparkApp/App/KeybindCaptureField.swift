import AppKit
import SwiftUI

/// Records one chord from a local key-down monitor (OC-01).
///
/// SwiftUI's text-input APIs cannot see a bare chord on macOS 14 (Cmd+Shift+
/// anything is a menu key equivalent, and arrows and Tab move focus), so the
/// recorder watches `NSEvent` key-downs for the window while a row is
/// recording. The monitor RETURNS NIL for every key-down while it is
/// installed: the chord is captured instead of also firing its menu item, so
/// recording Cmd+N does not make a new chat.
///
/// Lifecycle is the risk this type exists to contain: one monitor at most,
/// removed on every exit path (a captured chord, Escape, the row going away,
/// the app losing focus), because a leaked monitor would keep swallowing
/// every key the user types.
@MainActor
final class KeyChordRecorder {
    private var monitor: Any?
    private var resignObserver: NSObjectProtocol?

    var isActive: Bool { monitor != nil }

    func start(
        onCapture: @escaping (KeyChord.Capture) -> Void, onInterrupted: @escaping () -> Void
    ) {
        stop()
        monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            // A local monitor runs on the main thread, in event dispatch.
            let capture = MainActor.assumeIsolated {
                KeyChord.capture(
                    keyCode: event.keyCode,
                    unmodifiedCharacters: event.characters(byApplyingModifiers: []),
                    flags: event.modifierFlags)
            }
            MainActor.assumeIsolated { onCapture(capture) }
            return nil
        }
        // Switching to another app mid-recording would otherwise leave the
        // monitor armed for whatever the user types when they come back.
        resignObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.didResignActiveNotification, object: nil, queue: .main
        ) { _ in
            MainActor.assumeIsolated { onInterrupted() }
        }
    }

    func stop() {
        if let monitor { NSEvent.removeMonitor(monitor) }
        monitor = nil
        if let resignObserver { NotificationCenter.default.removeObserver(resignObserver) }
        resignObserver = nil
    }
}

/// The record control for one Shortcuts pane row: a button that reads as the
/// chord badge. Return or Space on the focused button starts recording
/// (native button activation, so the control is keyboard-operable), the next
/// chord is captured, and Escape cancels.
///
/// VoiceOver: the button is named for the command, its value is the chord (or
/// "Recording" while capture is armed), and a hint says how to use it.
@MainActor
struct KeybindCaptureField: View {
    @Environment(\.appTheme) private var theme

    /// The command's name, as the row's label.
    let name: Text
    /// The chord in effect, as glyphs.
    let keys: String
    /// A fixed hidden alternate chord, read as part of the value.
    let alternateKeys: String?
    @Binding var isRecording: Bool
    let onCapture: (KeyChord) -> Void
    /// A key that cannot be a chord (Escape with a modifier, a function key).
    let onUnsupported: () -> Void

    @State private var recorder = KeyChordRecorder()

    var body: some View {
        Button {
            isRecording.toggle()
        } label: {
            Group {
                if isRecording {
                    Text("Press keys...", bundle: .module)
                } else {
                    Text(verbatim: keys)
                }
            }
            .font(theme.code(.small))
            .foregroundStyle(tint(.appText))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(.appSurface)
            .clipShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                    .stroke(tint(.appBorder), lineWidth: isRecording ? 2 : 1)
            )
        }
        .buttonStyle(.plain)
        .fixedSize()
        .accessibilityLabel(name)
        .accessibilityValue(spokenValue)
        .accessibilityHint(
            Text("Starts recording a new shortcut. Escape cancels.", bundle: .module))
        .onChange(of: isRecording) { _, recording in
            if recording { arm() } else { recorder.stop() }
        }
        .onAppear { if isRecording { arm() } }
        .onDisappear {
            recorder.stop()
            if isRecording { isRecording = false }
        }
    }

    /// The accent while recording, so the armed state shows without relying on
    /// the label change alone.
    private func tint(_ idle: AppThemeColor) -> AppThemeColor {
        isRecording ? .appAccentText : idle
    }

    private var spokenValue: Text {
        if isRecording {
            return Text("Recording. Press the new keys, or Escape to cancel.", bundle: .module)
        }
        let chord = Text(verbatim: keys)
        guard let alternateKeys else { return chord }
        return chord + Text(verbatim: ", ") + Text("also \(alternateKeys)", bundle: .module)
    }

    private func arm() {
        recorder.start(
            onCapture: { capture in
                switch capture {
                case .cancel:
                    isRecording = false
                case .chord(let chord):
                    isRecording = false
                    onCapture(chord)
                case .unsupported:
                    onUnsupported()
                }
            },
            onInterrupted: { isRecording = false })
    }
}

extension KeybindRefusal {
    /// The sentence shown and announced for a refusal. Names the holder, so
    /// the user knows which command to move first.
    func message(for chord: KeyChord) -> String {
        switch self {
        case .notRebindable:
            return AppStrings.localized(
                "Shared with a control in the window, so it cannot be changed.")
        case .needsCommandOrControl:
            return AppStrings.localized("Shortcuts need Command or Control.")
        case .reserved:
            return AppStrings.localized(
                "\(chord.display) is reserved by macOS or text editing and cannot be assigned.")
        case .heldBy(let id, let isAlternate):
            let holder = KeyboardShortcutCatalog.command(for: id)
                .map { AppStrings.localized(String.LocalizationValue($0.label)) } ?? id.rawValue
            if isAlternate {
                return AppStrings.localized(
                    "\(chord.display) is already an alternate shortcut for \(holder).")
            }
            return AppStrings.localized("\(chord.display) is already used by \(holder).")
        }
    }
}
