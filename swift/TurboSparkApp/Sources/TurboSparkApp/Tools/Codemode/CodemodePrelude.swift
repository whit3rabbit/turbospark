import Foundation

/// JavaScript evaluated inside the worker's fresh JavaScriptCore context
/// before the script runs. A port of pi's codemode prelude, reduced to the
/// surface this app exposes: `tools.<name>(args)`, `ALL_TOOLS`, `console`,
/// `text()`, `exit()`, `store()`, and `load()`.
///
/// Discipline carried over from pi: the script's only way to reach the host
/// is the bridges installed by `CodemodeWorkerContext`, every payload
/// crossing them is a JSON string, and the intrinsics the host relies on
/// (`JSON.stringify`/`parse`, `String`) are captured at install time so a
/// script cannot tamper with what the host receives. Settlement is
/// idempotent and owned by the prelude: an output-cap breach reports the
/// failure through the done bridge BEFORE throwing, so catching the throw
/// cannot resume output or overwrite the result.
///
/// Deliberate deviations from pi, noted in the README: no global-lockdown
/// walk (the context is fresh per execution and discarded with its process),
/// no `image()` (this app's tool results are text), and no host globals.
enum CodemodePrelude {
    /// Interpolates the numeric bounds into the prelude source. Only Int
    /// values reach the string, never script-controlled data; the tool
    /// list and store snapshot arrive through bridges instead.
    static func source(limits: CodemodeLimits) -> String {
        """
        (() => {
          const toolsInit = globalThis.__codemodeToolsInit;
          const storeInit = globalThis.__codemodeStoreInit;
          const callBridge = globalThis.__codemodeCall;
          const outputBridge = globalThis.__codemodeOutput;
          const doneBridge = globalThis.__codemodeDone;
          const stringify = JSON.stringify;
          const parse = JSON.parse;
          const stringValue = String;
          const MAX_OUTPUT_CHARS = \(limits.maximumOutputCharacters);
          const MAX_OUTPUT_ITEMS = \(limits.maximumOutputItems);
          const MAX_STORE_VALUE_CHARS = \(limits.maximumStoreValueCharacters);
          const MAX_STORE_TOTAL_CHARS = \(limits.maximumStoreTotalCharacters);

          const entries = parse(toolsInit());
          const snapshot = parse(storeInit());

          delete globalThis.__codemodeToolsInit;
          delete globalThis.__codemodeStoreInit;
          // The runtime bridges stay reachable to the prelude's closures
          // but not to the script: deleting the globals after capture
          // leaves the only reference inside this IIFE, the same way pi's
          // prelude closes over its bridge.
          delete globalThis.__codemodeCall;
          delete globalThis.__codemodeOutput;
          delete globalThis.__codemodeDone;

          let finished = false;
          let outputCharacters = 0;
          let outputItemCount = 0;
          const pendingWrites = new Map();

          const describeError = (error) => {
            let name = "Error";
            let message = "the script threw an undescribable value";
            let stack;
            try {
              if (error !== null && error !== undefined) {
                if (error.name !== undefined) name = stringValue(error.name);
                if (error.message !== undefined) message = stringValue(error.message);
                else message = stringValue(error);
                if (error.stack !== undefined && error.stack !== null) stack = stringValue(error.stack);
              } else {
                message = stringValue(error);
              }
            } catch (_) {}
            const described = { name: name, message: message };
            if (stack !== undefined) described.stack = stack;
            return described;
          };

          const serializeWrites = () => {
            const writes = [];
            for (const pair of pendingWrites) {
              if (pair[1] === undefined) writes.push([pair[0]]);
              else writes.push([pair[0], pair[1]]);
            }
            pendingWrites.clear();
            return stringify(writes);
          };

          const succeed = (valueJSON) => {
            if (finished) return;
            finished = true;
            doneBridge(true, valueJSON === undefined ? null : valueJSON, null, serializeWrites());
          };

          const fail = (described) => {
            if (finished) return;
            finished = true;
            doneBridge(false, null, stringify(described), serializeWrites());
          };

          const output = (kind, text, level) => {
            if (finished) return;
            outputItemCount += 1;
            outputCharacters += text.length + 1;
            if (outputCharacters > MAX_OUTPUT_CHARS || outputItemCount > MAX_OUTPUT_ITEMS) {
              const message = "script output exceeded the limit of " + MAX_OUTPUT_CHARS
                + " characters or " + MAX_OUTPUT_ITEMS + " text() and console calls. Print a summary "
                + "instead of the full data, or return the useful part.";
              fail({ name: "RangeError", message: message });
              throw new RangeError(message);
            }
            outputBridge(kind, text, level === undefined ? null : level);
          };

          const formatValue = (value) => {
            if (typeof value === "string") return value;
            try {
              const encoded = stringify(value);
              return encoded === undefined ? stringValue(value) : encoded;
            } catch (_) {
              return stringValue(value);
            }
          };

          // --- store / load: synchronous against the snapshot, JSON only ---
          const storeMap = new Map();
          let storeTotalCharacters = 0;
          for (const key of Object.keys(snapshot)) {
            const value = snapshot[key];
            if (typeof value === "string") {
              storeMap.set(key, value);
              storeTotalCharacters += key.length + value.length;
            }
          }

          const store = (key, value) => {
            if (typeof key !== "string" || key.length === 0) {
              throw new TypeError("store() requires a non-empty string key");
            }
            if (value === undefined) {
              const existing = storeMap.get(key);
              if (existing !== undefined) {
                storeTotalCharacters -= key.length + existing.length;
                storeMap.delete(key);
                pendingWrites.set(key, undefined);
              }
              return;
            }
            let json;
            try { json = stringify(value); } catch (_) { json = undefined; }
            if (json === undefined) {
              throw new TypeError("store('" + key + "') requires a JSON-serializable value");
            }
            if (json.length > MAX_STORE_VALUE_CHARS) {
              throw new RangeError("store() value of " + json.length + " characters exceeds the "
                + MAX_STORE_VALUE_CHARS + " character per-value limit. store() is for small state "
                + "such as ids or summaries; use text() to report larger data.");
            }
            const existing = storeMap.get(key);
            const existingCost = existing === undefined ? 0 : key.length + existing.length;
            const nextTotal = storeTotalCharacters - existingCost + key.length + json.length;
            if (nextTotal > MAX_STORE_TOTAL_CHARS) {
              throw new RangeError("store is full: storing that value would exceed the "
                + MAX_STORE_TOTAL_CHARS + " character total. Delete keys with store(key, undefined).");
            }
            storeTotalCharacters = nextTotal;
            storeMap.set(key, json);
            pendingWrites.set(key, json);
          };

          const load = (key) => {
            if (typeof key !== "string") {
              throw new TypeError("load() requires a string key");
            }
            const json = storeMap.get(key);
            if (json === undefined) return undefined;
            try { return parse(json); } catch (_) { return undefined; }
          };

          // --- tools: one promise-returning function per entry ---
          const makeCaller = (toolName) => {
            return (...args) => new Promise((resolve, reject) => {
              let json = null;
              if (args.length === 1) {
                try { json = stringify(args[0]); } catch (error) {
                  reject(new TypeError("tools." + toolName + " arguments must be JSON-serializable"));
                  return;
                }
                if (json === undefined) {
                  reject(new TypeError("tools." + toolName + " arguments must be JSON-serializable"));
                  return;
                }
              } else if (args.length > 1) {
                reject(new TypeError("tools." + toolName + " takes one arguments object"));
                return;
              }
              callBridge(toolName, json, (error, payload) => {
                if (error !== null && error !== undefined) {
                  reject(new Error(error));
                  return;
                }
                let value = null;
                if (payload !== null && payload !== undefined) {
                  try { value = parse(payload); } catch (_) {
                    reject(new Error("tool " + toolName + " returned a result that was not valid JSON"));
                    return;
                  }
                }
                resolve(value);
              });
            });
          };

          const normalize = (value) => stringValue(value).toLowerCase().replace(/[^a-z0-9]/g, "");
          const target = {};
          const names = [];
          for (const entry of entries) {
            const caller = makeCaller(entry.name);
            Object.freeze(caller);
            const define = (label) => {
              if (target[label] !== undefined) return;
              Object.defineProperty(target, label, {
                value: caller, enumerable: true, writable: false, configurable: false
              });
            };
            define(entry.jsName);
            define(entry.name);
            names.push(entry.jsName);
          }
          Object.freeze(target);

          const suggestionMessage = (missing) => {
            const wanted = normalize(missing);
            const close = names.filter((name) => {
              const candidate = normalize(name);
              return candidate.includes(wanted) || wanted.includes(candidate);
            }).slice(0, 5);
            let listing;
            if (close.length > 0) listing = close.map((name) => "tools." + name).join(", ");
            else if (names.length <= 20) listing = names.map((name) => "tools." + name).join(", ");
            else listing = names.length + " tools are available";
            return "tools." + missing + " is not an available tool. Available: " + listing
              + ". ALL_TOOLS lists every tool; use the direct MCP tool instead when the name is unknown.";
          };

          const tools = new Proxy(target, {
            get(targetObject, property) {
              if (typeof property !== "string") return targetObject[property];
              if (property in targetObject) return targetObject[property];
              if (property === "then" || property === "toJSON") return undefined;
              throw new TypeError(suggestionMessage(property));
            }
          });

          const allTools = Object.freeze(entries.map((entry) => Object.freeze({
            name: entry.jsName,
            description: entry.description === undefined ? "" : stringValue(entry.description)
          })));

          const text = (value) => output("text", formatValue(value), undefined);

          const console = {};
          for (const level of ["log", "info", "debug", "warn", "error"]) {
            const method = (...values) => {
              let formatted = "";
              for (let index = 0; index < values.length; index++) {
                if (index > 0) formatted += " ";
                formatted += formatValue(values[index]);
              }
              output("console", formatted, level);
            };
            Object.freeze(method);
            Object.defineProperty(console, level, {
              value: method, enumerable: true, writable: false, configurable: false
            });
          }
          Object.freeze(console);

          const exitSentinel = new Error("__codemode_exit");
          Object.freeze(exitSentinel);

          const exit = () => {
            succeed(undefined);
            throw exitSentinel;
          };

          const install = (label, value, enumerable) => {
            Object.defineProperty(globalThis, label, {
              value: value, enumerable: enumerable, writable: false, configurable: false
            });
          };
          install("tools", tools, true);
          install("ALL_TOOLS", allTools, true);
          install("console", console, true);
          install("text", text, true);
          install("exit", exit, true);
          install("store", store, true);
          install("load", load, true);
          install("__codemodeComplete", (value) => {
            if (finished) return;
            let json;
            if (value !== undefined) {
              try {
                const encoded = stringify(value);
                json = encoded === undefined ? stringValue(value) : encoded;
              } catch (_) { json = stringValue(value); }
            }
            succeed(json);
          }, false);
          install("__codemodeFail", (error) => {
            if (finished) return;
            fail(describeError(error));
          }, false);
          install("__codemodeReportStalled", () => {
            fail({
              name: "Error",
              message: "The script is waiting on a promise that can never settle: no tool call "
                + "is pending, and timers do not exist here."
            });
          }, false);
        })()
        """
    }

    /// Wraps the script body as one async function body so `return` and
    /// top-level `await` work, then reports settlement through the hidden
    /// prelude hooks. The wrapper's opener shares line 1 with the script so
    /// stack-trace line numbers match the original source (columns on
    /// line 1 shift), the same trick pi uses.
    static func wrapper(code: String) -> String {
        "(async () => {\(code)\n"
            + "})().then(\n"
            + "  (value) => { globalThis.__codemodeComplete(value); },\n"
            + "  (error) => { globalThis.__codemodeFail(error); }\n"
            + ");"
    }
}
