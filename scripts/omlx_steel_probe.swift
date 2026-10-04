// Isolated synthetic Metal probe, not linked into TurboSpark. See MLX_KERNELS.md.
import Foundation
import Metal

enum ProbeError: Error { case failed(String) }

func require(_ ok: Bool, _ message: String) throws {
    if !ok { throw ProbeError.failed(message) }
}

func emit(_ record: [String: Any]) throws {
    let data = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
    print(String(decoding: data, as: UTF8.self))
    fflush(stdout)
}

func buffer<T>(_ device: MTLDevice, _ values: [T]) -> MTLBuffer {
    values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
}

func bf16(_ x: Float) -> UInt16 { UInt16(truncatingIfNeeded: (x.bitPattern + 0x7fff + ((x.bitPattern >> 16) & 1)) >> 16) }
func bfValue(_ bits: UInt16) -> Double { Double(Float(bitPattern: UInt32(bits) << 16)) }

// Match the existing MMA parity metric: cancellation makes per-element
// relative error misleading. Bound error by each output row's range.
func rowRangeError(_ actual: [Double], _ expected: [Double], n: Int) throws -> Double {
    try require(actual.count == expected.count && expected.count % n == 0, "parity dimensions differ")
    var worst = 0.0
    for start in stride(from: 0, to: expected.count, by: n) {
        let range = expected[start..<(start+n)].map { abs($0) }.max()!
        try require(range > 0 && actual[start..<(start+n)].allSatisfy { $0.isFinite }, "invalid parity fixture/output")
        for i in start..<(start+n) { worst = max(worst, abs(actual[i]-expected[i])/range) }
    }
    return worst
}

struct Fixture {
    let n: Int
    let k: Int
    let packed: [UInt8]
    let scales: [UInt16]
    let biases: [UInt16]
    let x: [Float16]
    let buffers: [MTLBuffer]
    init(_ device: MTLDevice, n: Int, k: Int, m: Int) {
        self.n = n; self.k = k
        // Vary groups, rows, nibbles and sign. Dyadic-only fixtures conceal
        // reassociation and dtype errors, so use ragged BF16 companions.
        packed = (0..<(n*k/2)).map { UInt8(truncatingIfNeeded: $0 &* 73 &+ ($0 / 19) &+ 11) }
        scales = (0..<(n*k/64)).map { bf16(0.0067 + Float($0 % 31) * 0.00037) }
        biases = scales.enumerated().map { bf16(-Float(bfValue($0.element)) * (5.13 + Float($0.offset % 11) * 0.17)) }
        x = (0..<(m*k)).map { Float16(Float(($0 &* 37 &+ $0 / 7) % 131 - 65) / 51.3) }
        buffers = [buffer(device, packed), buffer(device, scales), buffer(device, biases), buffer(device, x),
                   device.makeBuffer(length: m*n*2, options: .storageModeShared)!]
    }
    func reference(m: Int) -> [Double] {
        (0..<(m*n)).map { index in
            let token = index / n, row = index % n
            var sum = 0.0
            for col in 0..<k {
                let byte = packed[(row*k + col)/2]
                let q = Int(col % 2 == 0 ? byte & 15 : byte >> 4)
                let group = (row*k + col)/64
                sum += (Double(q) * bfValue(scales[group]) + bfValue(biases[group])) * Double(x[token*k + col])
            }
            return sum
        }
    }
    func output(m: Int) -> [Double] {
        let ptr = buffers[4].contents().bindMemory(to: Float16.self, capacity: m*n)
        return (0..<(m*n)).map { Double(ptr[$0]) }
    }
}

struct Arm {
    let name: String
    let pipeline: MTLComputePipelineState
    let groups: MTLSize
    let threads: Int
    let steel: Bool
}

func arms(_ device: MTLDevice, control: MTLLibrary, steel: MTLLibrary, n: Int, k: Int, m: Int) throws -> [Arm] {
    try require(m > 0 && (m <= 16 || m % 16 == 0), "control requires complete 16-row chunks beyond its cap")
    let rowBlock = m <= 1 ? 1 : (m <= 4 ? 2 : 4)
    let constants = MTLFunctionConstantValues()
    for (index, value) in [(20,0), (21,0), (100,n), (101,k), (102,min(m,16)), (104,rowBlock)] {
        var v = UInt32(value); constants.setConstantValue(&v, type: .uint, index: index)
    }
    var off = false, on = true
    constants.setConstantValue(&off, type: .bool, index: 22)
    constants.setConstantValue(&on, type: .bool, index: 103)
    let fn = try control.makeFunction(name: "dequant_int4_gemm_simd", constantValues: constants)
    let pipeline = try device.makeComputePipelineState(function: fn)
    var result = [Arm(name: "control", pipeline: pipeline,
                      groups: MTLSize(width: (n+8*rowBlock-1)/(8*rowBlock), height: 1, depth: 1),
                      threads: 256, steel: false)]
    for (bm,bk,bn) in [(32,32,32), (32,64,64), (64,32,64)] {
        try require(n % bn == 0 && k % bk == 0, "unsupported Steel shape")
        let name = "ts_steel_\(bm)_\(bk)_\(bn)"
        let fn = steel.makeFunction(name: name)!
        let pipeline = try device.makeComputePipelineState(function: fn)
        result.append(Arm(name: name, pipeline: pipeline,
                          groups: MTLSize(width: n/bn, height: (m+bm-1)/bm, depth: 1),
                          threads: 128, steel: true))
    }
    for arm in result {
        try require(arm.threads <= arm.pipeline.maxTotalThreadsPerThreadgroup, "thread count exceeds pipeline limit")
        try require(arm.pipeline.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength, "threadgroup memory exceeds device limit")
    }
    return result
}

func dispatch(_ queue: MTLCommandQueue, arm: Arm, fixture: Fixture, m: Int, reps: Int) throws -> Double {
    try autoreleasepool {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(arm.pipeline)
        // The control supports at most 16 rows. For M=32, encode two
        // production-width calls, including the second input/output offsets.
        let chunks = arm.steel ? [(0,m)] : stride(from: 0, to: m, by: 16).map { ($0,min(16,m-$0)) }
        for _ in 0..<reps {
            for (start,count) in chunks {
                for (i,b) in fixture.buffers.enumerated() {
                    let offset = i == 3 ? start*fixture.k*2 : (i == 4 ? start*fixture.n*2 : 0)
                    enc.setBuffer(b, offset: offset, index: i)
                }
                // Steel uses K,N,M; our shader uses N,K,M at indices 5,6,7.
                let dimensions = arm.steel ? [fixture.k,fixture.n,count] : [fixture.n,fixture.k,count]
                for (i,v) in dimensions.enumerated() {
                    var value = UInt32(v); enc.setBytes(&value, length: 4, index: i+5)
                }
                enc.dispatchThreadgroups(arm.groups, threadsPerThreadgroup: MTLSize(width: arm.threads, height: 1, depth: 1))
            }
        }
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        if cb.status != .completed { throw cb.error ?? ProbeError.failed("Metal command failed") }
        let elapsed = cb.gpuEndTime-cb.gpuStartTime
        try require(elapsed > 0, "GPU timestamps unavailable")
        return elapsed / Double(reps)
    }
}

func run() throws {
    let root = URL(fileURLWithPath: CommandLine.arguments[1])
    let device = MTLCreateSystemDefaultDevice()!
    let queue = device.makeCommandQueue()!
    let options = MTLCompileOptions()
    options.languageVersion = .version3_1
    if #available(macOS 15.0, *) { options.mathMode = .fast }
    else { options.fastMathEnabled = true }
    let control = try device.makeLibrary(source: String(contentsOf: root.appendingPathComponent("control.metal"), encoding: .utf8), options: options)
    let steel = try device.makeLibrary(source: String(contentsOf: root.appendingPathComponent("steel.metal"), encoding: .utf8), options: options)
    try emit(["kind":"host", "device":device.name, "os":ProcessInfo.processInfo.operatingSystemVersionString,
              "thermal":ProcessInfo.processInfo.thermalState.rawValue, "metal":"3.1", "fast_math":true])
    for m in [1,2,3,7,16,32] {
        let fixture = Fixture(device, n: 64, k: 192, m: m)
        let expected = fixture.reference(m: m)
        for arm in try arms(device, control: control, steel: steel, n: 64, k: 192, m: m) {
            _ = try dispatch(queue, arm: arm, fixture: fixture, m: m, reps: 1)
            let actual = fixture.output(m: m)
            let error = zip(actual,expected).map { abs($0-$1) }.max()!
            let normalized = try rowRangeError(actual, expected, n: 64)
            try emit(["kind":"parity", "arm":arm.name, "m":m, "n":64, "k":192,
                      "max_abs_cpu":error, "max_row_range_error_cpu":normalized])
            // Existing dequant_int4_mma_parity's bound, unchanged. This is
            // synthetic screening and does not qualify model logits.
            try require(normalized <= 0.005, "synthetic CPU screening failed")
        }
    }
    for (label,n,k) in [("qwen36_shared",512,2048), ("qwen36_qkv",8192,2048), ("qwen38_gate",17408,5120)] {
        for m in [2,8,16,32] {
            let fixture = Fixture(device, n: n, k: k, m: m)
            let variants = try arms(device, control: control, steel: steel, n: n, k: k, m: m)
            for arm in variants { _ = try dispatch(queue, arm: arm, fixture: fixture, m: m, reps: 3) }
            var repetitions: [String: Int] = [:]
            for arm in variants {
                let seconds = try dispatch(queue, arm: arm, fixture: fixture, m: m, reps: 16)
                // Match steady device duration across arms instead of pricing
                // a few launches while the GPU clocks are still ramping.
                repetitions[arm.name] = max(16,min(32768,Int(0.05/seconds)))
                _ = try dispatch(queue, arm: arm, fixture: fixture, m: m, reps: repetitions[arm.name]!)
            }
            _ = try dispatch(queue, arm: variants[0], fixture: fixture, m: m, reps: 1)
            let expected = fixture.output(m: m)
            for arm in variants.dropFirst() {
                _ = try dispatch(queue, arm: arm, fixture: fixture, m: m, reps: 1)
                let actual = fixture.output(m: m)
                let difference = zip(actual,expected).map { abs($0-$1) }.max()!
                let squaredError = zip(actual,expected).reduce(0.0) { $0 + ($1.0-$1.1)*($1.0-$1.1) }
                let squaredReference = expected.reduce(0.0) { $0 + $1*$1 }
                let normalized = try rowRangeError(actual, expected, n: n)
                try emit(["kind":"full_shape_parity", "shape":label, "m":m, "arm":arm.name,
                          "max_abs_control":difference, "relative_l2_control":sqrt(squaredError/squaredReference),
                          "max_row_range_error_control":normalized])
                try require(normalized <= 0.005, "synthetic GPU screening failed")
            }
            for round in 0..<6 {
                let order = round % 2 == 0 ? variants : Array(variants.reversed())
                for arm in order {
                    let reps = repetitions[arm.name]!
                    let seconds = try dispatch(queue, arm: arm, fixture: fixture, m: m, reps: reps)
                    try emit(["kind":"timing", "shape":label, "n":n, "k":k, "m":m,
                              "round":round, "arm":arm.name, "gpu_ms":seconds*1000,
                              "thermal":ProcessInfo.processInfo.thermalState.rawValue, "reps":reps])
                }
            }
        }
    }
}

do { try run() }
catch { fputs("\(error)\n", stderr); exit(1) }
