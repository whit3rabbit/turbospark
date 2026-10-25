import AVFoundation
import CoreML
import FluidAudio
import Foundation

// usage: FAProbe <models_root> <audio_dir> <out_dir>
// models_root holds vad/, diarizer/, kokoro/ laid out as planned installs.
let args = CommandLine.arguments
let modelsRoot = URL(fileURLWithPath: args[1])
let audioDir = URL(fileURLWithPath: args[2])
let outDir = URL(fileURLWithPath: args[3])
let fm = FileManager.default

func log(_ s: String) { print(s); fflush(stdout) }

func snapshot(_ dir: URL) -> [String: String] {
    var out: [String: String] = [:]
    guard let e = fm.enumerator(at: dir, includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey]) else { return out }
    for case let url as URL in e {
        let v = try? url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey, .isDirectoryKey])
        if v?.isDirectory == true { continue }
        let rel = String(url.path.dropFirst(dir.path.count))
        out[rel] = "\(v?.fileSize ?? -1)@\(v?.contentModificationDate?.timeIntervalSince1970 ?? 0)"
    }
    return out
}

func readMono16k(_ name: String) throws -> [Float] {
    let file = try AVAudioFile(forReading: audioDir.appendingPathComponent(name))
    let buf = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
    try file.read(into: buf)
    precondition(file.processingFormat.sampleRate == 16_000 && file.processingFormat.channelCount == 1)
    return Array(UnsafeBufferPointer(start: buf.floatChannelData![0], count: Int(buf.frameLength)))
}

func timed<T>(_ label: String, _ body: () async throws -> T) async rethrows -> T {
    let t0 = Date()
    let r = try await body()
    log(String(format: "  %@: %.3f s", label, Date().timeIntervalSince(t0)))
    return r
}

func config(_ units: MLComputeUnits) -> MLModelConfiguration {
    let c = MLModelConfiguration()
    c.computeUnits = units
    return c
}

let suite = args.count > 4 ? args[4] : "small"
let installs = (suite == "big" ? ["sensevoice", "paraformer", "parakeet-ja", "pocket"] : ["vad", "diarizer", "kokoro"])
    .map { modelsRoot.appendingPathComponent($0) }
let before = installs.map(snapshot)
let cacheDirs = [
    fm.homeDirectoryForCurrentUser.appendingPathComponent(".cache/fluidaudio"),
    fm.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/FluidAudio"),
]

// A3: offline switches, set before any other FluidAudio use.
ModelHub.offlineMode = true
ModelRegistry.baseURL = "turbospark-offline:"
log("offlineMode=\(ModelHub.offlineMode) baseURL=\(ModelRegistry.baseURL)")

if suite == "big" {
    await runBig(modelsRoot: modelsRoot, audioDir: audioDir, outDir: outDir)
} else {

let vadFile = "silero-vad-unified-256ms-v6.2.1.mlmodelc"

// A3a: missing model in offline mode must throw modelMissing, not fetch.
do {
    let empty = fm.temporaryDirectory.appendingPathComponent("faprobe-empty-\(UUID().uuidString)")
    try fm.createDirectory(at: empty, withIntermediateDirectories: true)
    _ = try await ModelHub.loadModels(.vad, modelNames: [vadFile], directory: empty)
    log("A3a FAIL: load from empty dir succeeded")
} catch {
    log("A3a missing model -> \(error)")
}

// A3b: corrupt model in offline mode must throw and leave the folder in place.
do {
    let parent = fm.temporaryDirectory.appendingPathComponent("faprobe-corrupt-\(UUID().uuidString)")
    let repoDir = parent.appendingPathComponent(Repo.vad.folderName)
    try fm.createDirectory(at: repoDir, withIntermediateDirectories: true)
    try fm.copyItem(at: modelsRoot.appendingPathComponent("vad/\(vadFile)"), to: repoDir.appendingPathComponent(vadFile))
    let weight = repoDir.appendingPathComponent("\(vadFile)/weights/weight.bin")
    let h = try FileHandle(forWritingTo: weight)
    try h.truncate(atOffset: 1024)
    try h.close()
    do {
        _ = try await ModelHub.loadModels(.vad, modelNames: [vadFile], directory: parent)
        log("A3b FAIL: corrupt load succeeded")
    } catch {
        log("A3b corrupt model -> \(error)")
    }
    log("A3b folder still present: \(fm.fileExists(atPath: repoDir.appendingPathComponent(vadFile).path))")
}

let speechA = try readMono16k("a.wav")
let speechB = try readMono16k("b.wav")
let speechC = try readMono16k("c.wav")
let gap = [Float](repeating: 0, count: 16_000)
let dialogue = speechA + gap + speechB + gap + speechC

// A4 VAD: own MLModel -> VadManager(config:vadModel:).
log("VAD")
do {
    let model = try await timed("load") {
        try MLModel(contentsOf: modelsRoot.appendingPathComponent("vad/\(vadFile)"), configuration: config(.cpuAndNeuralEngine))
    }
    let vad = VadManager(config: .default, vadModel: model)
    let results = try await timed("process \(dialogue.count / 16_000) s") { try await vad.process(dialogue) }
    let active = results.filter(\.isVoiceActive).count
    log("  chunks=\(results.count) voiced=\(active)")
} catch { log("  VAD FAIL: \(error)") }

// A4 diarizer: own MLModels + ported PLDA psi -> initialize(models:).
func loadPLDAPsi(_ url: URL) throws -> [Double] {
    let root = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    let psi = (root["tensors"] as! [String: Any])["psi"] as! [String: Any]
    let data = Data(base64Encoded: psi["data_base64"] as! String, options: [.ignoreUnknownCharacters])!
    var floats = [Float](repeating: 0, count: data.count / MemoryLayout<Float>.size)
    _ = floats.withUnsafeMutableBytes { data.copyBytes(to: $0) }
    return floats.map(Double.init)
}

log("Diarizer")
do {
    let d = modelsRoot.appendingPathComponent("diarizer")
    let t0 = Date()
    let c = config(.all)
    let models = OfflineDiarizerModels(
        segmentationModel: try MLModel(contentsOf: d.appendingPathComponent("Segmentation.mlmodelc"), configuration: c),
        fbankModel: try MLModel(contentsOf: d.appendingPathComponent("FBank.mlmodelc"), configuration: c),
        embeddingModel: try MLModel(contentsOf: d.appendingPathComponent("Embedding.mlmodelc"), configuration: c),
        pldaRhoModel: try MLModel(contentsOf: d.appendingPathComponent("PldaRho.mlmodelc"), configuration: c),
        pldaPsi: try loadPLDAPsi(d.appendingPathComponent("plda-parameters.json")),
        compilationDuration: Date().timeIntervalSince(t0))
    log(String(format: "  load: %.3f s (psi=%d)", models.compilationDuration, models.pldaPsi.count))
    let diarizer = OfflineDiarizerManager(config: .default)
    diarizer.initialize(models: models)
    let result = try await timed("process") { try await diarizer.process(audio: dialogue) }
    let speakers = Set(result.segments.map(\.speakerId))
    log("  speakers=\(speakers.sorted()) segments=\(result.segments.count)")
    for s in result.segments {
        log(String(format: "    %@ %.2f-%.2f", s.speakerId, s.startTimeSeconds, s.endTimeSeconds))
    }
} catch { log("  DIARIZER FAIL: \(error)") }

// A4 Kokoro ANE: phonemes only, never G2P or initialize().
log("Kokoro ANE")
do {
    let kokoro = KokoroAneManager(variant: .english, directory: modelsRoot.appendingPathComponent("kokoro"))
    let phonemes = "h\u{259}l\u{2c8}O w\u{2c8}\u{25c}\u{279}ld. \u{f0}\u{26a}s \u{26a}z \u{259} t\u{2c8}\u{25b}st \u{28c}v \u{f0}\u{259} n\u{2c8}u\u{279}\u{259}l \u{2c8}\u{25b}n\u{2a4}\u{259}n."
    let r = try await timed("first synth (load + run)") { try await kokoro.synthesizeFromPhonemesDetailed(phonemes) }
    let r2 = try await timed("second synth") { try await kokoro.synthesizeFromPhonemesDetailed(phonemes) }
    let finite = r.samples.allSatisfy(\.isFinite)
    let rms = sqrt(r.samples.reduce(0) { $0 + $1 * $1 } / Float(max(1, r.samples.count)))
    log("  samples=\(r.samples.count) rate=\(r.sampleRate) finite=\(finite) rms=\(rms) repeatSame=\(r.samples == r2.samples)")
    let fmt = AVAudioFormat(standardFormatWithSampleRate: Double(r.sampleRate), channels: 1)!
    let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(r.samples.count))!
    buf.frameLength = buf.frameCapacity
    r.samples.withUnsafeBufferPointer { buf.floatChannelData![0].update(from: $0.baseAddress!, count: r.samples.count) }
    let wav = try AVAudioFile(forWriting: outDir.appendingPathComponent("kokoro.wav"), settings: fmt.settings)
    try wav.write(from: buf)
    // A3c: an unpinned voice reaches AssetDownloader, which ignores offlineMode.
    let t0 = Date()
    do {
        _ = try await kokoro.synthesizeFromPhonemesDetailed(phonemes, voice: "bf_emma")
        log("  A3c FAIL: unpinned voice synthesized")
    } catch {
        log(String(format: "  A3c unpinned voice -> %@ (%.3f s)", "\(error)", Date().timeIntervalSince(t0)))
    }
} catch { log("  KOKORO FAIL: \(error)") }
}

// Zero writes into the installs, no FluidAudio cache directories.
for (dir, old) in zip(installs, before) {
    let now = snapshot(dir)
    let changed = Set(old.keys).symmetricDifference(now.keys).union(old.keys.filter { now[$0] != nil && now[$0] != old[$0] })
    log("install \(dir.lastPathComponent): files=\(now.count) changed=\(changed.sorted())")
}
for c in cacheDirs { log("cache \(c.path) exists=\(fm.fileExists(atPath: c.path))") }
