import AppKit
import PencilKit
import SwiftUI

enum WhiteboardTool: String, CaseIterable, Hashable {
    case pen, eraser
}

/// Undo and redo over whole-drawing snapshots.
///
/// Snapshots rather than `NSUndoManager`: the canvas's own undo registers on
/// the WINDOW's manager, which depends on which view is first responder, and
/// the toolbar buttons must work (and be testable) either way. One entry per
/// completed stroke or erase.
struct WhiteboardHistory: Equatable {
    private(set) var current: Data
    private var undoStack: [Data] = []
    private var redoStack: [Data] = []

    static let maxUndoDepth = 100

    init(base: Data) { current = base }

    var canUndo: Bool { !undoStack.isEmpty }
    var canRedo: Bool { !redoStack.isEmpty }

    mutating func record(_ data: Data) {
        guard data != current else { return }
        undoStack.append(current)
        if undoStack.count > Self.maxUndoDepth { undoStack.removeFirst() }
        redoStack.removeAll()
        current = data
    }

    mutating func undo() -> Data? {
        guard let previous = undoStack.popLast() else { return nil }
        redoStack.append(current)
        current = previous
        return previous
    }

    mutating func redo() -> Data? {
        guard let next = redoStack.popLast() else { return nil }
        undoStack.append(current)
        current = next
        return next
    }
}

/// Owns the canvas view for the whiteboard pane: tool state, history and
/// the change callback that feeds the store.
@MainActor
final class WhiteboardCanvasController: NSObject, ObservableObject {
    static let widthRange: ClosedRange<CGFloat> = 1...24

    let canvas = WhiteboardCanvasNSView()

    @Published var tool: WhiteboardTool = .pen { didSet { applyTool() } }
    @Published var swatch: WhiteboardSwatch = .black { didSet { applyTool() } }
    @Published var width: CGFloat = 4 { didSet { applyTool() } }
    @Published private(set) var canUndo = false
    @Published private(set) var canRedo = false
    @Published private(set) var strokeCount = 0

    /// The board on the canvas and the store its changes are queued in. Both
    /// nil in a test that only exercises tools and history.
    private(set) var boardID: UUID?
    private weak var store: WhiteboardStore?

    private var history = WhiteboardHistory(base: PKDrawing().dataRepresentation())
    override init() {
        super.init()
        canvas.onDrawingCommitted = { [weak self] in self?.commit() }
        applyTool()
    }

    private func applyTool() {
        canvas.tool = tool
        canvas.inkColor = swatch.color
        canvas.inkWidth = width
    }

    /// Shows `id` from `store`. A change still queued for the board being left
    /// is flushed first, so switching never loses a stroke.
    func open(boardID id: UUID, store: WhiteboardStore) {
        store.flush()
        self.store = store
        boardID = id
        switch store.loadDrawing(id: id) {
        case .drawing(let data): load(data)
        case .empty, .recoveredFromCorruption: load(nil)
        }
    }

    /// Leaves the current board (deleted, or the pane is closing).
    func closeBoard() {
        store?.flush()
        boardID = nil
        load(nil)
    }

    /// Replaces the canvas content and resets history. `nil` is a blank board.
    func load(_ data: Data?) {
        let drawing = data.flatMap { try? PKDrawing(data: $0) } ?? PKDrawing()
        apply(drawing)
        history = WhiteboardHistory(base: drawing.dataRepresentation())
        refreshState()
    }

    private func apply(_ drawing: PKDrawing) {
        canvas.drawing = drawing
    }

    private func refreshState() {
        canUndo = history.canUndo
        canRedo = history.canRedo
        strokeCount = canvas.drawing.strokes.count
    }

    // MARK: - Commands

    func undo() {
        guard let data = history.undo(), let drawing = try? PKDrawing(data: data) else { return }
        apply(drawing)
        refreshState()
        persist(data)
    }

    func redo() {
        guard let data = history.redo(), let drawing = try? PKDrawing(data: data) else { return }
        apply(drawing)
        refreshState()
        persist(data)
    }

    /// A new empty drawing, as one undoable step.
    func clear() {
        guard !canvas.drawing.strokes.isEmpty else { return }
        let empty = PKDrawing()
        apply(empty)
        history.record(empty.dataRepresentation())
        refreshState()
        persist(history.current)
    }

    /// Adds a stroke without a pointer, for tests and the export path's
    /// fixtures. Goes through the same history as a drawn stroke.
    func addStroke(_ stroke: PKStroke) {
        var drawing = canvas.drawing
        drawing.strokes.append(stroke)
        apply(drawing)
        commit()
    }

    private func persist(_ data: Data) {
        guard let boardID else { return }
        store?.scheduleSave(id: boardID, data: data)
    }

    private func commit() {
        history.record(canvas.drawing.dataRepresentation())
        refreshState()
        persist(history.current)
    }

    func exportPNG() throws -> Data {
        try WhiteboardExporter.png(for: canvas.drawing)
    }
}

/// The SwiftUI wrapper. The controller owns the view so the pane's toolbar
/// and the canvas share one instance.
struct WhiteboardCanvasView: NSViewRepresentable {
    let controller: WhiteboardCanvasController

    func makeNSView(context: Context) -> WhiteboardCanvasNSView { controller.canvas }
    func updateNSView(_ nsView: WhiteboardCanvasNSView, context: Context) {}
}
