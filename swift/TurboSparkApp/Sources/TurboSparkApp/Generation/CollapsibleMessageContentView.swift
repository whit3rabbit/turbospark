import SwiftUI

/// Container view that clamps long text/markdown to a maximum height with a smooth bottom fade
/// and a "Show more" / "Show less" toggle, matching Claude-style message containment.
public struct CollapsibleMessageContentView: View {
    @Environment(\.appTheme) private var theme
    public let text: String
    public var isUser: Bool = false
    public var maxHeight: CGFloat = 220
    /// Forwarded to the markdown renderer's ```html fences. Nil hides the
    /// Preview buttons, which is right everywhere except the transcript.
    public var onPreviewHTML: ((String) -> Void)?

    @State private var isExpanded: Bool = false
    @State private var contentHeight: CGFloat = 0

    public init(
        text: String,
        isUser: Bool = false,
        maxHeight: CGFloat = 220,
        onPreviewHTML: ((String) -> Void)? = nil
    ) {
        self.text = text
        self.isUser = isUser
        self.maxHeight = maxHeight
        self.onPreviewHTML = onPreviewHTML
    }

    private var isClamped: Bool {
        contentHeight > maxHeight + 10
    }

    public var body: some View {
        VStack(alignment: isUser ? .trailing : .leading, spacing: 6) {
            ZStack(alignment: .topLeading) {
                contentBody
                    .background(
                        GeometryReader { geo in
                            Color.clear.preference(
                                key: ContentHeightPreferenceKey.self,
                                value: geo.size.height
                            )
                        }
                    )
                    .frame(maxHeight: isClamped && !isExpanded ? maxHeight : nil, alignment: .topLeading)
                    .clipped()
                    .mask(
                        VStack(spacing: 0) {
                            if isClamped && !isExpanded {
                                Rectangle()
                                    .fill(Color.black)
                                    .frame(maxHeight: max(0, maxHeight - 48))
                                LinearGradient(
                                    colors: [Color.black, Color.black.opacity(0)],
                                    startPoint: .top,
                                    endPoint: .bottom
                                )
                                .frame(height: 48)
                            } else {
                                Rectangle().fill(Color.black)
                            }
                        }
                    )
            }
            .onPreferenceChange(ContentHeightPreferenceKey.self) { measured in
                if measured > 0 {
                    contentHeight = measured
                }
            }

            if isClamped {
                Button {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isExpanded.toggle()
                    }
                } label: {
                    HStack(spacing: 4) {
                        Text(isExpanded ? "Show less" : "Show more")
                            .themedFont(.small, weight: .medium)
                        Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                            .themedFont(.micro, weight: .bold)
                    }
                    .foregroundStyle(.appSecondary)
                    .padding(.top, 2)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help(isExpanded ? "Collapse message" : "Expand full message")
                .accessibilityLabel(isExpanded ? "Show less of message" : "Show more of message")
            }
        }
    }

    @ViewBuilder
    private var contentBody: some View {
        if isUser {
            Text(text)
                .font(theme.uiFont)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            // Interactive tables (qwen-code parity): a message carrying a
            // pipe table renders that segment as a sortable, copyable grid;
            // everything else is the ordinary markdown renderer.
            MarkdownContentWithTablesView(text: text, onPreviewHTML: onPreviewHTML)
        }
    }
}

private struct ContentHeightPreferenceKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}
