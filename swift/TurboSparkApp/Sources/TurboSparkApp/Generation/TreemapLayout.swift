import CoreGraphics
import Foundation

/// One weighted input to the treemap. `id` is the stable key the caller maps
/// the result back with, and the final tie-break, so equal weights always lay
/// out the same way.
struct TreemapItem: Equatable {
    let id: String
    let weight: Double
}

/// One placed rectangle, in the same coordinate space as the bounds passed
/// to `TreemapLayout.layout`.
struct TreemapRect: Equatable {
    let id: String
    let rect: CGRect
}

/// Squarified treemap layout (Bruls, Huizing and van Wijk, 2000).
///
/// Pure and deterministic: no view state, no randomness, and input order does
/// not matter, because items are sorted by weight descending and then by id
/// ascending before placement. Items whose weight is not a positive finite
/// number take no area and are omitted from the result (the caller reports
/// them separately). The areas of the returned rectangles sum to the area of
/// `bounds`, each proportional to its weight, up to floating-point error.
enum TreemapLayout {
    static func layout(items: [TreemapItem], in bounds: CGRect) -> [TreemapRect] {
        guard bounds.width > 0, bounds.height > 0 else { return [] }
        let sorted = items
            .filter { $0.weight.isFinite && $0.weight > 0 }
            .sorted { lhs, rhs in
                lhs.weight != rhs.weight ? lhs.weight > rhs.weight : lhs.id < rhs.id
            }
        guard !sorted.isEmpty else { return [] }

        let total = sorted.reduce(0.0) { $0 + $1.weight }
        let scale = Double(bounds.width * bounds.height) / total

        var placed: [TreemapRect] = []
        placed.reserveCapacity(sorted.count)
        var remaining = bounds
        var row: [(item: TreemapItem, area: Double)] = []

        func worst(_ candidate: [(item: TreemapItem, area: Double)], side: Double) -> Double {
            let sum = candidate.reduce(0.0) { $0 + $1.area }
            guard let maxArea = candidate.map(\.area).max(),
                  let minArea = candidate.map(\.area).min(),
                  sum > 0, minArea > 0 else { return .infinity }
            let s2 = side * side
            return max(s2 * maxArea / (sum * sum), (sum * sum) / (s2 * minArea))
        }

        // Lays the row along the shorter side of what is left, then shrinks
        // the remainder by the strip it used.
        func flush() {
            guard !row.isEmpty else { return }
            let sum = row.reduce(0.0) { $0 + $1.area }
            if remaining.width >= remaining.height {
                let stripWidth = CGFloat(sum) / remaining.height
                var y = remaining.minY
                for entry in row {
                    let h = CGFloat(entry.area) / stripWidth
                    placed.append(TreemapRect(
                        id: entry.item.id,
                        rect: CGRect(x: remaining.minX, y: y, width: stripWidth, height: h)))
                    y += h
                }
                remaining = CGRect(
                    x: remaining.minX + stripWidth, y: remaining.minY,
                    width: max(0, remaining.width - stripWidth), height: remaining.height)
            } else {
                let stripHeight = CGFloat(sum) / remaining.width
                var x = remaining.minX
                for entry in row {
                    let w = CGFloat(entry.area) / stripHeight
                    placed.append(TreemapRect(
                        id: entry.item.id,
                        rect: CGRect(x: x, y: remaining.minY, width: w, height: stripHeight)))
                    x += w
                }
                remaining = CGRect(
                    x: remaining.minX, y: remaining.minY + stripHeight,
                    width: remaining.width, height: max(0, remaining.height - stripHeight))
            }
            row.removeAll(keepingCapacity: true)
        }

        for item in sorted {
            let entry = (item: item, area: item.weight * scale)
            let side = Double(min(remaining.width, remaining.height))
            if row.isEmpty || worst(row + [entry], side: side) <= worst(row, side: side) {
                row.append(entry)
            } else {
                flush()
                row.append(entry)
            }
        }
        flush()
        return placed
    }
}
