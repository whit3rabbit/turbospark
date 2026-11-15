import CoreGraphics
import XCTest
@testable import TurboSparkApp

final class TreemapLayoutTests: XCTestCase {
    private let bounds = CGRect(x: 10, y: 20, width: 400, height: 300)

    private func items(_ weights: [Double]) -> [TreemapItem] {
        weights.enumerated().map { TreemapItem(id: String(format: "item-%03d", $0.offset), weight: $0.element) }
    }

    /// Area of the overlap of two rects, zero when they only touch.
    private func overlap(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let i = a.intersection(b)
        return i.isNull ? 0 : i.width * i.height
    }

    private func assertInvariants(
        _ placed: [TreemapRect], weights: [String: Double], in bounds: CGRect,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let total = weights.values.reduce(0, +)
        let area = Double(bounds.width * bounds.height)
        let slack = bounds.insetBy(dx: -0.001, dy: -0.001)
        for entry in placed {
            // Containment.
            XCTAssertTrue(slack.contains(entry.rect), "\(entry.id) escapes the bounds", file: file, line: line)
            // Area proportional to weight.
            let expected = area * (weights[entry.id] ?? 0) / total
            XCTAssertEqual(Double(entry.rect.width * entry.rect.height), expected,
                           accuracy: max(0.01, expected * 1e-6), entry.id, file: file, line: line)
        }
        // No overlaps.
        for i in 0..<placed.count {
            for j in (i + 1)..<placed.count {
                XCTAssertLessThan(overlap(placed[i].rect, placed[j].rect), 0.001,
                                  "\(placed[i].id) overlaps \(placed[j].id)", file: file, line: line)
            }
        }
        // The cells tile the whole bounds.
        let sum = placed.reduce(0.0) { $0 + Double($1.rect.width * $1.rect.height) }
        XCTAssertEqual(sum, area, accuracy: area * 1e-6, file: file, line: line)
    }

    func testAreasAreProportionalAndTileWithoutOverlap() {
        let input = items([60, 30, 20, 10, 10, 5, 3, 1])
        let placed = TreemapLayout.layout(items: input, in: bounds)
        XCTAssertEqual(placed.count, input.count)
        assertInvariants(placed, weights: Dictionary(uniqueKeysWithValues: input.map { ($0.id, $0.weight) }), in: bounds)
    }

    func testLargeInputKeepsInvariants() {
        var seed: UInt64 = 42
        let weights = (0..<400).map { _ -> Double in
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Double((seed >> 33) % 500 + 1)
        }
        let input = items(weights)
        let placed = TreemapLayout.layout(items: input, in: bounds)
        XCTAssertEqual(placed.count, 400)
        assertInvariants(placed, weights: Dictionary(uniqueKeysWithValues: input.map { ($0.id, $0.weight) }), in: bounds)
    }

    func testSingleItemFillsBounds() {
        let placed = TreemapLayout.layout(items: [TreemapItem(id: "only", weight: 7)], in: bounds)
        XCTAssertEqual(placed.map(\.id), ["only"])
        let r = placed[0].rect
        XCTAssertEqual(r.minX, bounds.minX, accuracy: 0.001)
        XCTAssertEqual(r.minY, bounds.minY, accuracy: 0.001)
        XCTAssertEqual(r.width, bounds.width, accuracy: 0.001)
        XCTAssertEqual(r.height, bounds.height, accuracy: 0.001)
    }

    func testZeroNegativeAndNonFiniteWeightsTakeNoArea() {
        let input = [
            TreemapItem(id: "a", weight: 5), TreemapItem(id: "zero", weight: 0),
            TreemapItem(id: "neg", weight: -3), TreemapItem(id: "nan", weight: .nan),
            TreemapItem(id: "inf", weight: .infinity),
        ]
        let placed = TreemapLayout.layout(items: input, in: bounds)
        XCTAssertEqual(placed.map(\.id), ["a"])
        XCTAssertTrue(TreemapLayout.layout(items: [TreemapItem(id: "z", weight: 0)], in: bounds).isEmpty)
        XCTAssertTrue(TreemapLayout.layout(items: [], in: bounds).isEmpty)
    }

    func testEmptyBoundsPlaceNothing() {
        let input = items([1, 2, 3])
        XCTAssertTrue(TreemapLayout.layout(items: input, in: .zero).isEmpty)
        XCTAssertTrue(TreemapLayout.layout(items: input, in: CGRect(x: 0, y: 0, width: 0, height: 50)).isEmpty)
    }

    func testInputOrderDoesNotChangeTheLayout() {
        // Many equal weights make the tie-break the only thing that decides
        // placement, so this fails if ties fall back to input order.
        let input = items([5, 5, 5, 5, 5, 5, 3, 3, 3, 1, 1])
        let baseline = TreemapLayout.layout(items: input, in: bounds)
        var rng = SystemRandomNumberGenerator()
        for _ in 0..<20 {
            let shuffled = input.shuffled(using: &rng)
            XCTAssertEqual(TreemapLayout.layout(items: shuffled, in: bounds), baseline)
        }
    }

    func testTiesBreakByIdAscending() {
        let input = [
            TreemapItem(id: "c", weight: 1), TreemapItem(id: "a", weight: 1), TreemapItem(id: "b", weight: 1),
        ]
        XCTAssertEqual(TreemapLayout.layout(items: input, in: bounds).map(\.id), ["a", "b", "c"])
    }

    func testHeavierItemsArePlacedFirst() {
        let input = [TreemapItem(id: "small", weight: 1), TreemapItem(id: "big", weight: 9)]
        let placed = TreemapLayout.layout(items: input, in: bounds)
        XCTAssertEqual(placed.first?.id, "big")
        XCTAssertGreaterThan(
            placed[0].rect.width * placed[0].rect.height, placed[1].rect.width * placed[1].rect.height)
    }

    func testCellsStayNearSquare() {
        // Squarifying is the point: equal weights in a wide box must not
        // degenerate into slivers (a slice-and-dice layout gives 16:1).
        let input = items(Array(repeating: 1, count: 16))
        let placed = TreemapLayout.layout(items: input, in: CGRect(x: 0, y: 0, width: 400, height: 400))
        for cell in placed {
            let ratio = max(cell.rect.width / cell.rect.height, cell.rect.height / cell.rect.width)
            XCTAssertLessThan(ratio, 2, cell.id)
        }
    }
}
