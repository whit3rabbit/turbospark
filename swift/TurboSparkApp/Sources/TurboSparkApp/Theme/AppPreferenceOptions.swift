import SwiftUI

/// The option rows for the three display preferences, shared by every
/// surface that offers them: the Appearance menu, the sidebar profile menu,
/// General settings and Appearance settings.
///
/// The UI audit of 2026-10-11 found the same three pickers written out
/// three times, with labels localized in one place and verbatim English in
/// the others. One definition keeps the options, their order and their
/// translations identical wherever the setting is reachable.
struct AppearanceModeOptions: View {
    var body: some View {
        ForEach(AppAppearance.allCases) { option in
            Label { Text(LocalizedStringKey(option.label), bundle: .module) } icon: {
                Image(systemName: option.systemImage)
            }
            .tag(option)
        }
    }
}

struct TextSizeOptions: View {
    var body: some View {
        ForEach(AppTextSize.allCases) { size in
            Text(LocalizedStringKey(size.label), bundle: .module).tag(size)
        }
    }
}

/// Languages are listed by their native names, which are not translated on
/// purpose (a reader looks for their own language in its own script). Only
/// the follow-the-system entry is a UI string.
struct LanguageOptions: View {
    var body: some View {
        ForEach(AppLanguage.allCases) { language in
            Group {
                if language == .system {
                    Text("System", bundle: .module)
                } else {
                    Text(verbatim: language.label)
                }
            }
            .tag(language.rawValue)
        }
    }
}
