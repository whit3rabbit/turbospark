import SwiftUI

@MainActor
struct ConversationPaneView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        if model.hasOutputTranscript {
            VStack(spacing: 0) {
                OutputPaneView(model: model)
                    .frame(maxHeight: .infinity)
                conversationChrome
                    .fixedSize(horizontal: false, vertical: true)
            }
        } else {
            OutputPaneView(model: model)
        }
    }

    private var conversationChrome: some View {
        VStack(spacing: 8) {
            ErrorBanner(model: model)
            PromptComposerView(model: model)
            ModelLoaderControl(model: model, density: .compact)
        }
        .conversationColumn()
        .padding(.top, 8)
        .padding(.bottom, 12)
        .background(.appPage)
    }
}
