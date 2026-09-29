import SwiftUI

enum APIWorkspaceTab: Int, CaseIterable, Identifiable {
    case text, image, typeSafe

    var id: Int { rawValue }
    var title: String {
        switch self {
        case .text: return String(localized: "Text", bundle: .module)
        case .image: return String(localized: "Image", bundle: .module)
        case .typeSafe: return String(localized: "TypeSafe", bundle: .module)
        }
    }

    func moved(_ offset: Int) -> Self {
        Self(rawValue: min(max(rawValue + offset, 0), 2)) ?? self
    }
}

struct APIWorkspaceSelector: View {
    @Binding var selection: APIWorkspaceTab

    var body: some View {
        HStack(spacing: 2) {
            ForEach(APIWorkspaceTab.allCases) { tab in
                Button {
                    selection = tab
                } label: {
                    Text(tab.title)
                        .frame(minWidth: 88)
                }
                .buttonStyle(.bordered)
                .tint(selection == tab ? .accentColor : .clear)
                .accessibilityAddTraits(selection == tab ? .isSelected : [])
                .accessibilityLabel(tab.title)
            }
        }
        .frame(maxWidth: .infinity)
        .contentShape(Rectangle())
        .focusable()
        .gesture(DragGesture(minimumDistance: 25).onEnded { value in
            guard abs(value.translation.width) > abs(value.translation.height) else { return }
            selection = selection.moved(value.translation.width < 0 ? 1 : -1)
        })
        .onMoveCommand { direction in
            switch direction {
            case .left: selection = selection.moved(-1)
            case .right: selection = selection.moved(1)
            default: break
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(Text("API mode", bundle: .module))
        .accessibilityAdjustableAction { direction in
            switch direction {
            case .increment: selection = selection.moved(1)
            case .decrement: selection = selection.moved(-1)
            @unknown default: break
            }
        }
    }
}
