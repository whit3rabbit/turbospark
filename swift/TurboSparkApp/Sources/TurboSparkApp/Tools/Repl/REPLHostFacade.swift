import Foundation
import JavaScriptCore

typealias REPLConsoleSink = @Sendable (REPLTextOutputEvent) -> Void

/// Installs the worker's host-visible surface onto a fresh context: the
/// console methods, the rooted file facade, and the image emission
/// interface. Every member is defined non-writable and non-configurable,
/// and both the host objects and their method functions are frozen, so
/// tampering during one call cannot weaken the surface for later calls.
/// The app never receives JavaScriptCore values or objects through this
/// seam.
///
/// Capability pieces install in two steps so a context that installs
/// several pieces seals exactly once. Each `install...Bridge` method
/// exposes its Swift seam under a temporary global, and
/// `sealReplSurface` assembles the frozen `repl` object from every bridge
/// present at seal time and removes the temporary globals. A sealed
/// `repl` object can never be extended, so the self-sealing convenience
/// installers (`installFileSystem`, `installImageEmitter`) exist only for
/// contexts that use a single capability piece; a context combining
/// pieces must call the bridge steps and then exactly one seal before any
/// script runs.
final class REPLHostFacade: @unchecked Sendable {
    private let output: REPLConsoleSink

    /// File responses are delivered on this serial queue rather than the
    /// evaluation thread. JavaScriptCore serializes context access
    /// internally, and delivering off the evaluation thread is what lets
    /// the worker keep servicing requests while a top-level await is
    /// pending on a file promise.
    private let fileResponseQueue = DispatchQueue(
        label: "com.turbospark.repl.file-response")

    init(output: @escaping REPLConsoleSink) {
        self.output = output
    }

    func installConsole(into context: JSContext) {
        let output = self.output
        let consoleSink: @convention(block) (String, String) -> Void = { level, text in
            guard let level = REPLTextOutputEvent.Level(rawValue: level) else { return }
            output(REPLTextOutputEvent(level: level, text: text))
        }
        context.setObject(consoleSink, forKeyedSubscript: "__turbosparkConsoleSink" as NSString)
        context.evaluateScript(Self.consoleInstallationScript)
    }

    /// Installs the rooted file facade onto a context that uses only the
    /// file capability: the request bridge followed by the seal. The
    /// facade has no filesystem API of its own: `repl.fs.readFile(path)`,
    /// `repl.fs.writeFile(path, contents)`, and `repl.fs.list(path)` each
    /// send one bounded request over the channel and settle the returned
    /// promise with the typed result, so the worker never learns the
    /// granted roots and never touches FileManager. Must be called on the
    /// worker queue, once per fresh context, alongside `installConsole`.
    /// Contexts that also emit images must use `installFileRequestBridge`
    /// and one `sealReplSurface` after every bridge instead, so `repl` is
    /// assembled with both pieces before it freezes.
    func installFileSystem(into context: JSContext, channel: any REPLFileRequestChannel) {
        installFileRequestBridge(into: context, channel: channel)
        sealReplSurface(into: context)
    }

    /// Installs `repl.emitImage(base64Data, label)` onto a context that
    /// uses only the image capability: the emission bridge followed by the
    /// seal. The bridge validates the decoded payload by PNG or JPEG magic
    /// bytes (never the claimed label), rejects payloads over
    /// `limits.maximumImageBytes` with an explanatory error, writes the
    /// file into `config.artifactDirectory`, reports the emitted image to
    /// `emitter`, and returns the file reference to the script. Rejected
    /// emissions surface to the script as ordinary errors while everything
    /// already captured for the call survives (6.2, 6.4). Must be called
    /// on the worker queue, once per fresh context; contexts that also
    /// expose `repl.fs` must use `installImageEmitterBridge` and one
    /// `sealReplSurface` after every bridge.
    func installImageEmitter(
        into context: JSContext,
        config: REPLSessionConfiguration,
        limits: REPLLimits,
        emitter: @escaping (REPLEmittedImage) -> Void
    ) {
        installImageEmitterBridge(
            into: context, config: config, limits: limits, emitter: emitter)
        sealReplSurface(into: context)
    }

    /// Exposes the file request bridge under a temporary global without
    /// sealing; see the class doc for the bridge-and-seal contract.
    func installFileRequestBridge(into context: JSContext, channel: any REPLFileRequestChannel) {
        let responseQueue = fileResponseQueue
        // `contents` arrives as JSValue rather than String? because the
        // automatic block bridging renders a JS null as the four-character
        // string "null"; the explicit isNull/isUndefined check below keeps
        // a payload-less read request payload-less.
        let requestBridge: @convention(block) (String, String, JSValue, JSValue) -> Void =
            { operationName, path, contents, callback in
                guard let operation = REPLFileAccessRequest.Operation.named(operationName) else {
                    responseQueue.async {
                        callback.call(withArguments: [
                            "repl.fs." + operationName + " is not a known file operation.",
                            NSNull()
                        ])
                    }
                    return
                }
                let payloadText: String?
                if contents.isNull || contents.isUndefined {
                    payloadText = nil
                } else {
                    payloadText = contents.toString()
                }
                let request = REPLFileAccessRequest(
                    id: UUID(),
                    operation: operation,
                    path: path,
                    payload: payloadText.map { Data($0.utf8) })
                Task {
                    let result = await channel.send(request)
                    responseQueue.async {
                        Self.deliver(result, to: callback)
                    }
                }
            }
        context.setObject(requestBridge, forKeyedSubscript: "__turbosparkFileRequest" as NSString)
    }

    /// Exposes the image emission bridge under a temporary global without
    /// sealing; see the class doc for the bridge-and-seal contract. The
    /// bridge runs synchronously on the JavaScript thread, so the emitter
    /// sees images in emission order on the worker queue.
    func installImageEmitterBridge(
        into context: JSContext,
        config: REPLSessionConfiguration,
        limits: REPLLimits,
        emitter: @escaping (REPLEmittedImage) -> Void
    ) {
        let emitImage: @convention(block) (JSValue, JSValue) -> NSArray = { data, label in
            switch Self.emitImagePayload(
                data: data,
                label: label,
                limits: limits,
                artifactDirectory: config.artifactDirectory)
            {
            case let .emitted(image):
                emitter(image)
                return NSArray(objects: NSNull(), image.fileURL.path as NSString)
            case let .rejected(message):
                return NSArray(objects: message as NSString, NSNull())
            }
        }
        context.setObject(emitImage, forKeyedSubscript: "__turbosparkEmitImage" as NSString)
    }

    /// Assembles the frozen `repl` object from every bridge present at
    /// seal time, defines the non-writable non-configurable global
    /// binding, and removes the temporary bridge globals. A no-op when no
    /// bridge is present or the surface was already sealed, so a stray
    /// second seal cannot weaken or duplicate the surface. Must be called
    /// on the worker queue after every intended bridge step and before
    /// any script runs.
    func sealReplSurface(into context: JSContext) {
        context.evaluateScript(Self.replSurfaceInstallationScript)
    }

    /// Delivers one broker result to the facade's resolve/reject callback.
    /// The script side sees only text, arrays, null, or an ordinary error;
    /// roots, identifiers, and transport detail never cross.
    private static func deliver(_ result: REPLFileAccessResult, to callback: JSValue) {
        switch result {
        case let .data(data):
            if let text = String(data: data, encoding: .utf8) {
                callback.call(withArguments: [NSNull(), text])
            } else {
                callback.call(withArguments: [
                    "REPL file read failed: the file is not valid UTF-8 text.",
                    NSNull()
                ])
            }
        case let .entries(entries):
            callback.call(withArguments: [NSNull(), entries])
        case .written:
            callback.call(withArguments: [NSNull(), NSNull()])
        case let .denied(message):
            callback.call(withArguments: [message, NSNull()])
        }
    }

    // MARK: Image emission (6.2, 6.4)

    /// The image formats the facade accepts, detected only from magic
    /// bytes. The claimed label never selects the format or the extension.
    private enum REPLImageFormat {
        case png
        case jpeg

        var fileExtension: String {
            switch self {
            case .png: return "png"
            case .jpeg: return "jpg"
            }
        }
    }

    private enum REPLImageEmissionOutcome {
        case emitted(REPLEmittedImage)
        case rejected(String)
    }

    private static let pngMagicBytes: [UInt8] = [
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A
    ]
    private static let jpegMagicBytes: [UInt8] = [0xFF, 0xD8, 0xFF]

    /// Validates one emission, writes the file, and reports the emitted
    /// image. Every rejection carries an explanatory message that
    /// surfaces to the script as an ordinary error while the rest of the
    /// call result survives.
    private static func emitImagePayload(
        data: JSValue,
        label: JSValue,
        limits: REPLLimits,
        artifactDirectory: URL
    ) -> REPLImageEmissionOutcome {
        guard data.isString, let base64 = data.toString() else {
            return .rejected(
                "repl.emitImage requires base64-encoded image data as a string.")
        }
        // Reject clearly oversized payloads before decoding so a payload
        // over the cap never materializes its decoded bytes. N image
        // bytes need at most ceil(N / 3) * 4 base64 characters plus
        // padding, so anything longer cannot fit under the cap.
        let maximumBase64Characters = ((limits.maximumImageBytes + 2) / 3) * 4 + 4
        if base64.count > maximumBase64Characters {
            return .rejected(
                Self.overCapMessage(byteCount: base64.count / 4 * 3, limits: limits))
        }
        guard let bytes = Data(base64Encoded: base64) else {
            return .rejected(
                "repl.emitImage rejected the image: the data is not valid base64.")
        }
        if bytes.count > limits.maximumImageBytes {
            return .rejected(Self.overCapMessage(byteCount: bytes.count, limits: limits))
        }
        guard let format = Self.imageFormat(of: bytes) else {
            return .rejected(
                "repl.emitImage rejected the image: the data does not start with "
                    + "PNG or JPEG magic bytes.")
        }
        var labelValue: String?
        if !(label.isNull || label.isUndefined) {
            guard label.isString, let text = label.toString() else {
                return .rejected("repl.emitImage requires the label to be a string.")
            }
            labelValue = text
        }
        // The file name never derives from the label: the label is model
        // input, so it rides along verbatim in the result and cannot steer
        // a path. The detected magic bytes pick the extension, and the
        // UUID keeps every emission its own file.
        let fileURL = artifactDirectory.appendingPathComponent(
            "repl-image-\(UUID().uuidString).\(format.fileExtension)")
        do {
            try FileManager.default.createDirectory(
                at: artifactDirectory, withIntermediateDirectories: true)
            try bytes.write(to: fileURL, options: .atomic)
        } catch {
            return .rejected(
                "repl.emitImage could not write the image file: "
                    + error.localizedDescription)
        }
        return .emitted(REPLEmittedImage(fileURL: fileURL, label: labelValue))
    }

    private static func imageFormat(of data: Data) -> REPLImageFormat? {
        if data.starts(with: pngMagicBytes) { return .png }
        if data.starts(with: jpegMagicBytes) { return .jpeg }
        return nil
    }

    private static func overCapMessage(byteCount: Int, limits: REPLLimits) -> String {
        "repl.emitImage rejected the image: \(byteCount) bytes exceeds the "
            + "\(limits.maximumImageBytes) byte image limit."
    }

    private static let replSurfaceInstallationScript = """
    (() => {
      const fileRequest = globalThis.__turbosparkFileRequest;
      const imageEmission = globalThis.__turbosparkEmitImage;
      if (fileRequest === undefined && imageEmission === undefined) return;
      const repl = {};
      if (fileRequest !== undefined) {
        const fs = {};
        for (const name of ["readFile", "writeFile", "list"]) {
          const method = (path, contents) => new Promise((resolve, reject) => {
            if (typeof path !== "string") {
              reject(new Error("repl.fs." + name + " requires a string path"));
              return;
            }
            fileRequest(name, path, contents, (error, value) => {
              if (error !== null) reject(new Error(error));
              else resolve(value);
            });
          });
          Object.freeze(method);
          Object.defineProperty(fs, name, {
            value: method,
            enumerable: true,
            writable: false,
            configurable: false
          });
        }
        Object.freeze(fs);
        Object.defineProperty(repl, "fs", {
          value: fs,
          enumerable: true,
          writable: false,
          configurable: false
        });
      }
      if (imageEmission !== undefined) {
        const emitImage = (data, label) => {
          const outcome = imageEmission(
            data === undefined ? null : data,
            label === undefined ? null : label);
          if (outcome[0] !== null) throw new Error(outcome[0]);
          return outcome[1];
        };
        Object.freeze(emitImage);
        Object.defineProperty(repl, "emitImage", {
          value: emitImage,
          enumerable: true,
          writable: false,
          configurable: false
        });
      }
      Object.freeze(repl);
      Object.defineProperty(globalThis, "repl", {
        value: repl,
        enumerable: true,
        writable: false,
        configurable: false
      });
      delete globalThis.__turbosparkFileRequest;
      delete globalThis.__turbosparkEmitImage;
    })();
    """

    private static let consoleInstallationScript = """
    (() => {
      const sink = globalThis.__turbosparkConsoleSink;
      const stringify = JSON.stringify;
      const stringValue = String;
      const format = values => {
        let formatted = "";
        for (let index = 0; index < values.length; index++) {
          const value = values[index];
          let part;
          if (typeof value === "string") {
            part = value;
          } else {
            try {
              const encoded = stringify(value);
              part = encoded === undefined ? stringValue(value) : encoded;
            } catch (_) {
              part = stringValue(value);
            }
          }
          if (index > 0) formatted += " ";
          formatted += part;
        }
        return formatted;
      };
      const console = {};
      for (const level of ["log", "info", "debug", "warn", "error"]) {
        const method = (...values) => sink(level, format(values));
        Object.freeze(method);
        Object.defineProperty(console, level, {
          value: method,
          enumerable: true,
          writable: false,
          configurable: false
        });
      }
      Object.freeze(console);
      Object.defineProperty(globalThis, "console", {
        value: console,
        enumerable: true,
        writable: false,
        configurable: false
      });
      delete globalThis.__turbosparkConsoleSink;
    })();
    """
}
