import SwiftUI

/// The bindings used by the General Settings controls also give the settings
/// seam a small, direct test surface for both directions of the UI contract.
@MainActor
struct ChatRuntimeSettingsBindings {
    let chatToolsEnabled: Binding<Bool>
    let microcompactEnabled: Binding<Bool>
    let autoContinuationEnabled: Binding<Bool>
    let nativeToolCallingEnabled: Binding<Bool>
    let nativeToolCallingIgnoresSpeed: Binding<Bool>
    let instructionPinningEnabled: Binding<Bool>
    let microcompactMinimumSavingsTokens: Binding<Int>
    let instructionPinTokenCeiling: Binding<Int>

    init(model: AppModel) {
        chatToolsEnabled = Binding(
            get: { model.chatToolsEnabled },
            set: { value in
                model.chatToolsEnabled = value
                model.persistSettingsDebounced()
            }
        )
        microcompactEnabled = Binding(
            get: { model.microcompactEnabled },
            set: { value in
                model.microcompactEnabled = value
                model.persistSettingsDebounced()
            }
        )
        autoContinuationEnabled = Binding(
            get: { model.autoContinuationEnabled },
            set: { value in
                model.autoContinuationEnabled = value
                model.persistSettingsDebounced()
            }
        )
        nativeToolCallingEnabled = Binding(
            get: { model.nativeToolCallingEnabled },
            set: { value in
                model.nativeToolCallingEnabled = value
                model.persistSettingsDebounced()
            }
        )
        nativeToolCallingIgnoresSpeed = Binding(
            get: { model.nativeToolCallingIgnoresSpeed },
            set: { value in
                model.nativeToolCallingIgnoresSpeed = value
                model.persistSettingsDebounced()
            }
        )
        instructionPinningEnabled = Binding(
            get: { model.instructionPinningEnabled },
            set: { value in
                model.instructionPinningEnabled = value
                model.persistSettingsDebounced()
            }
        )
        microcompactMinimumSavingsTokens = Binding(
            get: { model.microcompactMinimumSavingsTokens },
            set: { value in
                model.microcompactMinimumSavingsTokens = MacAppSettings
                    .clampMicrocompactMinimumSavingsTokens(value)
                model.persistSettingsDebounced()
            }
        )
        instructionPinTokenCeiling = Binding(
            get: { model.instructionPinTokenCeiling },
            set: { value in
                model.instructionPinTokenCeiling = MacAppSettings
                    .clampInstructionPinTokenCeiling(value)
                model.persistSettingsDebounced()
            }
        )
    }
}
