import SwiftUI

/// The ONE confirmation before a warned model is loaded (requirement 5.5, task
/// 6.7): the warning sentence, "Load anyway" and "Cancel".
///
/// Mounted once at the window root, so every interactive entry point (the pickers,
/// the Load button, the menu bar, the open panel, the server pane) reaches the same
/// dialog through `AppModel.requestLoad`, and no picker carries a dialog of its
/// own. Advice never prevents a load: the dialog only asks, and Cancel leaves
/// everything as it was.
///
/// The words are built when the dialog is shown, in the picker's language
/// (`@Environment(\.locale)` is read so a language change redraws it).
struct LoadAnywayConfirmation: ViewModifier {
    @ObservedObject var model: AppModel
    @Environment(\.locale) private var locale

    private var language: AppLanguage {
        _ = locale
        return AppStrings.currentLanguage
    }

    func body(content: Content) -> some View {
        content.confirmationDialog(
            Text(verbatim: model.pendingLoadConfirmation?.title(language: language) ?? ""),
            isPresented: Binding(
                get: { model.pendingLoadConfirmation != nil },
                // Dismissing the dialog any other way (Escape, a click outside)
                // is Cancel.
                set: { presented in
                    if !presented { model.cancelPendingLoad() }
                }),
            titleVisibility: .visible,
            presenting: model.pendingLoadConfirmation
        ) { pending in
            Button {
                model.confirmLoad(pending)
            } label: { Text("Load anyway", bundle: .module) }
            Button(role: .cancel) {
                model.cancelPendingLoad()
            } label: { Text("Cancel", bundle: .module) }
        } message: { pending in
            Text(verbatim: pending.message(language: language))
        }
    }
}
