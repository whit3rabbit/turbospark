import SwiftUI

/// Review gate for conversation capture. The source is deliberately reduced
/// to the human-readable transcript before it reaches this sheet, so system
/// prompts and tool payloads are never written to profile memory.
struct ProfileMemoryCaptureSheet: View {
    @ObservedObject var model: AppModel
    @Binding var isPresented: Bool
    @State private var draft: String
    @State private var isSummarizing = true

    init(model: AppModel, isPresented: Binding<Bool>, source: String) {
        self.model = model
        self._isPresented = isPresented
        let lines = source
            .components(separatedBy: .newlines)
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        self._draft = State(initialValue: lines.prefix(80).joined(separator: "\n"))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Add conversation to memory", bundle: .module)
                .themedFont(.title3, weight: .semibold)
                .accessibilityAddTraits(.isHeader)
            Text("Review and edit what will be appended to this user's profile MEMORY.md. Nothing is saved until you confirm.", bundle: .module)
                .foregroundStyle(.appSecondary)
            TextEditor(text: $draft)
                .themedFont(.base)
                .appEditorSurface(minHeight: 260)
            HStack {
                if isSummarizing {
                    AppProgressBar(
                        value: nil,
                        label: Text("Extracting durable memories\u{2026}", bundle: .module),
                        size: .compact
                    )
                    .frame(width: 80)
                    Text("Extracting durable memories\u{2026}", bundle: .module)
                        .foregroundStyle(.appSecondary)
                }
                Spacer()
                Button { isPresented = false } label: {
                    Text("Cancel", bundle: .module)
                }
                .keyboardShortcut(.cancelAction)
                Button {
                    model.handleProfileMemorySave(draft)
                    isPresented = false
                } label: {
                    Text("Save to Memory", bundle: .module)
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(isSummarizing || draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(20)
        .frame(minWidth: 600)
        .task {
            let startingDraft = draft
            if let summary = await model.summarizeConversationForProfileMemory(),
                draft == startingDraft  // never overwrite edits made while summarizing
            {
                draft = summary
            }
            isSummarizing = false
        }
    }
}
