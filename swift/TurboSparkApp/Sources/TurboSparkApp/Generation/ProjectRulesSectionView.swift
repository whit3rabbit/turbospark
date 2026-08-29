import AppKit
import SwiftUI

/// Settings section for managing project rules, AGENTS.md / CLAUDE.md auto-detection, and custom instructions.
struct ProjectRulesSectionView: View {
    let rootDirectoryPath: String
    @Binding var rulePreference: AppRulePreference
    @Binding var customInstructions: String
    @Binding var rulesAutoDetectedMessage: String?
    let onAutoDetect: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Project Rules & Instructions")
                    .font(.subheadline.weight(.semibold))
                    .accessibilityAddTraits(.isHeader)
                Spacer()
                if !rootDirectoryPath.isEmpty {
                    Button("Detect CLAUDE.md / AGENTS.md") {
                        onAutoDetect()
                    }
                    .font(.caption)
                    .buttonStyle(.borderless)
                    .help("Detect project rules from AGENTS.md or CLAUDE.md")
                }
            }

            HStack {
                Text("Conflict Preference")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Picker("Conflict Preference", selection: $rulePreference) {
                    ForEach(AppRulePreference.allCases) { pref in
                        Text(pref.label).tag(pref)
                    }
                }
                .pickerStyle(.menu)
                .frame(width: 180)
                .accessibilityLabel("Rules conflict preference")
            }

            if let rulesAutoDetectedMessage {
                Text(rulesAutoDetectedMessage)
                    .font(.caption)
                    .foregroundStyle(TurboSparkTheme.accentColor)
            }

            TextEditor(text: $customInstructions)
                .font(.callout.monospaced())
                .frame(height: 100)
                .padding(4)
                .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.2), lineWidth: 0.5))
                .accessibilityLabel("Project rules and instructions")
                .accessibilityHint("Free-form text sent to the model as project-specific guidance")
        }
    }
}
