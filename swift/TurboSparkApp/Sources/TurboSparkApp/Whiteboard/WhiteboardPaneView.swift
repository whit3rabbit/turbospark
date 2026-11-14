import AppKit
import SwiftUI

/// The whiteboard side pane (ZP-10): a board switcher, a tool strip and a
/// PencilKit canvas, with one action that sends the board to the composer.
///
/// Why a custom strip and not `PKToolPicker`: the system palette is a floating
/// panel the app cannot theme, localize or place, it adds a window the
/// right-column slot knows nothing about, and the app's own header controls
/// (`AppPillPicker`, `AppIconButton`) already carry the keyboard and VoiceOver
/// contract DESIGN.md requires. The canvas still draws with PencilKit tools.
@MainActor
struct WhiteboardPaneView: View {
    @ObservedObject var model: AppModel
    @StateObject private var controller = WhiteboardCanvasController()
    @State private var selectedBoardID: UUID?
    @State private var isRenaming = false
    @State private var renameText = ""
    @State private var isConfirmingDelete = false

    @Environment(\.appTheme) private var theme

    private var store: WhiteboardStore { model.whiteboardStore }
    private var key: String { model.whiteboardWorkspaceKey }

    var body: some View {
        // The store is a separate object: observing it here keeps the board
        // list and titles current without AppModel republishing.
        WhiteboardPaneContent(
            model: model, store: model.whiteboardStore, controller: controller,
            key: key, selectedBoardID: $selectedBoardID, isRenaming: $isRenaming,
            renameText: $renameText, isConfirmingDelete: $isConfirmingDelete)
    }
}

@MainActor
private struct WhiteboardPaneContent: View {
    @ObservedObject var model: AppModel
    @ObservedObject var store: WhiteboardStore
    @ObservedObject var controller: WhiteboardCanvasController
    let key: String
    @Binding var selectedBoardID: UUID?
    @Binding var isRenaming: Bool
    @Binding var renameText: String
    @Binding var isConfirmingDelete: Bool

    @Environment(\.appTheme) private var theme

    private var boards: [WhiteboardBoard] { store.boards(in: key) }
    private var selected: WhiteboardBoard? { selectedBoardID.flatMap { store.board(id: $0) } }

    var body: some View {
        VStack(spacing: 0) {
            boardHeader
                .padding(.horizontal, 12)
                .padding(.top, 10)
                .padding(.bottom, 6)
            if selected != nil {
                toolHeader
                    .padding(.horizontal, 12)
                    .padding(.bottom, 8)
            }
            if let issue = store.lastIssue {
                issueNote(issue)
            }
            Divider()
            if let board = selected {
                canvasArea(board)
            } else {
                AppEmptyState(
                    systemImage: "scribble",
                    title: Text("No boards yet", bundle: .module),
                    message: Text("Create a board to sketch on. It is saved for this workspace.", bundle: .module)
                ) {
                    Button { newBoard() } label: { Text("New board", bundle: .module) }
                        .appAction(.primary)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.appPage)
        .onAppear { ensureSelection() }
        .onChange(of: key) { _, _ in ensureSelection(force: true) }
        .onChange(of: selectedBoardID) { _, id in
            if let id { controller.open(boardID: id, store: store) } else { controller.closeBoard() }
        }
        .onDisappear { store.flush() }
        .alert(Text("Rename board", bundle: .module), isPresented: $isRenaming) {
            TextField(text: $renameText, prompt: Text("Board name", bundle: .module)) {
                Text("Board name", bundle: .module)
            }
            Button { commitRename() } label: { Text("Rename", bundle: .module) }
                .keyboardShortcut(.defaultAction)
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
        }
        .confirmationDialog(
            Text("Delete this board?", bundle: .module), isPresented: $isConfirmingDelete,
            titleVisibility: .visible
        ) {
            Button(role: .destructive) { deleteSelected() } label: { Text("Delete board", bundle: .module) }
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
        } message: {
            Text("The board and its strokes are removed. This cannot be undone.", bundle: .module)
        }
    }

    // MARK: - Headers

    private var boardHeader: some View {
        AppAdaptiveHeader {
            Menu {
                ForEach(boards) { board in
                    Button {
                        selectedBoardID = board.id
                    } label: {
                        if board.id == selectedBoardID {
                            Label { Text(verbatim: board.title) } icon: { Image(systemName: "checkmark") }
                        } else {
                            Text(verbatim: board.title)
                        }
                    }
                }
            } label: {
                Label {
                    Text(verbatim: selected?.title ?? "")
                } icon: {
                    Image(systemName: "scribble")
                }
            }
            .appMenuAction()
            .accessibilityLabel(Text("Boards", bundle: .module))
            .accessibilityValue(Text(verbatim: selected?.title ?? ""))
            .disabled(boards.isEmpty)
        } actions: {
            AppIconButton("plus", label: Text("New board", bundle: .module)) { newBoard() }
            AppIconButton("pencil", label: Text("Rename board", bundle: .module)) {
                renameText = selected?.title ?? ""
                isRenaming = true
            }
            .disabled(selected == nil)
            AppIconButton("trash", label: Text("Delete board", bundle: .module), role: .destructive) {
                isConfirmingDelete = true
            }
            .disabled(selected == nil)
        }
    }

    private var toolHeader: some View {
        AppAdaptiveHeader {
            HStack(spacing: 10) {
                AppPillPicker(
                    selection: $controller.tool,
                    items: [
                        AppPillItem(value: .pen, label: Text("Pen", bundle: .module), systemImage: "pencil.tip"),
                        AppPillItem(value: .eraser, label: Text("Eraser", bundle: .module), systemImage: "eraser"),
                    ],
                    label: Text("Whiteboard tool", bundle: .module))
                swatches
                widthSlider
            }
        } actions: {
            AppIconButton("arrow.uturn.backward", label: Text("Undo", bundle: .module)) { controller.undo() }
                .disabled(!controller.canUndo)
            AppIconButton("arrow.uturn.forward", label: Text("Redo", bundle: .module)) { controller.redo() }
                .disabled(!controller.canRedo)
            AppIconButton("xmark.bin", label: Text("Clear board", bundle: .module), role: .destructive) {
                controller.clear()
            }
            .disabled(controller.strokeCount == 0)
            exportButton
        }
    }

    private var swatches: some View {
        HStack(spacing: 6) {
            ForEach(WhiteboardSwatch.allCases) { swatch in
                let isOn = controller.swatch == swatch
                Button {
                    controller.swatch = swatch
                    controller.tool = .pen
                } label: {
                    Circle()
                        .fill(Color(nsColor: swatch.color))
                        .frame(width: 16, height: 16)
                        .padding(3)
                        .overlay(Circle().stroke(.appText, lineWidth: isOn ? 2 : 0))
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
                .help(Text(LocalizedStringKey(swatch.name), bundle: .module))
                .accessibilityLabel(Text(LocalizedStringKey(swatch.name), bundle: .module))
                .accessibilityAddTraits(isOn && controller.tool == .pen ? .isSelected : [])
            }
        }
        .disabled(controller.tool == .eraser)
        .opacity(controller.tool == .eraser ? 0.4 : 1)
    }

    private var widthSlider: some View {
        Slider(value: $controller.width, in: WhiteboardCanvasController.widthRange, step: 1) {
            Text("Stroke width", bundle: .module)
        }
        .labelsHidden()
        .frame(minWidth: 60, idealWidth: 90, maxWidth: 120)
        .accessibilityValue(Text(verbatim: "\(Int(controller.width))"))
        .help(Text("Stroke width", bundle: .module))
        .disabled(controller.tool == .eraser)
    }

    private var exportButton: some View {
        let hasStrokes = controller.strokeCount > 0
        return Button {
            exportToChat()
        } label: {
            Label { Text("Export to chat", bundle: .module) } icon: { Image(systemName: "paperplane") }
        }
        .appAction(.primary)
        .disabled(!hasStrokes)
        .help(hasStrokes
            ? Text("Attach the board as an image to the prompt", bundle: .module)
            : Text("Draw something first", bundle: .module))
    }

    private func issueNote(_ issue: String) -> some View {
        Label {
            Text(verbatim: issue).themedFont(.small)
        } icon: {
            Image(systemName: "exclamationmark.triangle")
        }
        .foregroundStyle(.appWarning)
        .padding(.horizontal, 12)
        .padding(.bottom, 6)
        .frame(maxWidth: .infinity, alignment: .leading)
        .onTapGesture { store.clearIssue() }
        .accessibilityAddTraits(.isButton)
        .accessibilityHint(Text("Dismisses this message", bundle: .module))
    }

    // MARK: - Canvas

    private func canvasArea(_ board: WhiteboardBoard) -> some View {
        WhiteboardCanvasView(controller: controller)
            .accessibilityLabel(Text("\(board.title). Strokes: \(controller.strokeCount)", bundle: .module))
            .accessibilityAddTraits(.allowsDirectInteraction)
    }

    // MARK: - Actions

    private func ensureSelection(force: Bool = false) {
        let current = boards
        if !force, let selectedBoardID, current.contains(where: { $0.id == selectedBoardID }) { return }
        if let first = current.first {
            selectedBoardID = first.id
            // `onChange` does not fire when the id is unchanged (a first
            // appear on the same board), so open explicitly.
            if controller.boardID != first.id { controller.open(boardID: first.id, store: store) }
        } else {
            // First open in a workspace: a board to draw on, not an empty list.
            newBoard()
        }
    }

    private func newBoard() {
        let board = store.createBoard(workspaceKey: key)
        selectedBoardID = board.id
    }

    private func commitRename() {
        guard let id = selectedBoardID else { return }
        store.rename(id: id, to: renameText)
    }

    private func deleteSelected() {
        guard let id = selectedBoardID else { return }
        controller.closeBoard()
        store.delete(id: id)
        selectedBoardID = boards.first?.id
    }

    private func exportToChat() {
        guard let board = selected else { return }
        do {
            let png = try controller.exportPNG()
            model.attachWhiteboardExport(png, boardTitle: board.title)
        } catch {
            model.showToast(error.localizedDescription, style: .warning)
        }
    }
}
