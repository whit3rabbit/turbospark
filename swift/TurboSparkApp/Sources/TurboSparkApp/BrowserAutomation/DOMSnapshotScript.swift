import Foundation
import WebKit

extension DOMSnapshotService {
    static let walkerScript = #"""
    (() => {
      const referencesByNode = new WeakMap();
      const nodesByReference = new Map();
      const maximumLocatorNodes = 5000;
      const actionableSelector = [
        "button", "a[href]", "input:not([type='hidden'])", "textarea", "select",
        "[role='button']", "[role='link']", "[role='checkbox']", "[role='radio']",
        "[role='switch']", "[role='tab']", "[tabindex]:not([tabindex='-1'])"
      ].join(",");
      const excludedTags = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "TEMPLATE"]);

      const normalize = value => (value || "").replace(/\s+/g, " ").trim().slice(0, 500);
      const roleFor = element => {
        const explicit = element.getAttribute("role");
        if (explicit) return explicit.trim().split(/\s+/)[0].toLowerCase();
        const tag = element.tagName.toLowerCase();
        if (tag === "a" && element.hasAttribute("href")) return "link";
        if (tag === "button" || tag === "summary") return "button";
        if (tag === "textarea") return "textbox";
        if (tag === "select") return element.multiple ? "listbox" : "combobox";
        if (tag === "input") {
          const type = (element.getAttribute("type") || "text").toLowerCase();
          if (type === "checkbox") return "checkbox";
          if (type === "radio") return "radio";
          if (type === "range") return "slider";
          if (["button", "submit", "reset", "image"].includes(type)) return "button";
          return "textbox";
        }
        if (/^h[1-6]$/.test(tag)) return "heading";
        if (tag === "main") return "main";
        if (tag === "nav") return "navigation";
        if (tag === "form") return "form";
        if (tag === "dialog") return "dialog";
        if (tag === "p") return "paragraph";
        return "generic";
      };
      const accessibleName = element => {
        const type = element.tagName === "INPUT" ? (element.getAttribute("type") || "text").toLowerCase() : "";
        const labelledBy = normalize((element.getAttribute("aria-labelledby") || "")
          .split(/\s+/).map(id => {
            const label = document.getElementById(id);
            if (!label || ["INPUT", "TEXTAREA", "SELECT"].includes(label.tagName)) return "";
            return label.textContent || "";
          }).join(" "));
        if (labelledBy) return labelledBy;
        const ariaLabel = normalize(element.getAttribute("aria-label"));
        if (ariaLabel) return ariaLabel;
        if (type === "password") return "Password field";
        if (element.labels && element.labels.length) {
          const label = normalize(Array.from(element.labels).map(node => node.textContent || "").join(" "));
          if (label) return label;
        }
        if (["INPUT", "TEXTAREA", "SELECT"].includes(element.tagName)) return "";
        const alt = normalize(element.getAttribute("alt"));
        if (alt) return alt;
        const title = normalize(element.getAttribute("title"));
        if (title) return title;
        const tag = element.tagName;
        if (["BUTTON", "A", "SUMMARY", "OPTION", "DIALOG"].includes(tag)) {
          return normalize(element.textContent);
        }
        return normalize(Array.from(element.childNodes)
          .filter(node => node.nodeType === Node.TEXT_NODE)
          .map(node => node.nodeValue || "").join(" "));
      };
      const visible = element => {
        const style = getComputedStyle(element);
        if (style.display === "none" || style.visibility === "hidden" || Number(style.opacity) === 0) return false;
        const rect = element.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
      };
      const referenceFor = (element, generation) => {
        let reference = referencesByNode.get(element);
        if (!reference) {
          const bytes = crypto.getRandomValues(new Uint8Array(16));
          reference = Array.from(bytes, byte => byte.toString(16).padStart(2, "0")).join("");
          referencesByNode.set(element, reference);
        }
        nodesByReference.set(reference, { node: new WeakRef(element), generation });
        return reference;
      };
      const boundsFor = element => {
        const rect = element.getBoundingClientRect();
        // Integers: 17-digit doubles inflate when Swift re-encodes the JSON and
        // would push a near-budget snapshot over the byte cap.
        return { x: Math.round(rect.x), y: Math.round(rect.y), width: Math.round(rect.width), height: Math.round(rect.height) };
      };
      const isActionable = element => element.matches(actionableSelector);
      const hasExactKeys = (value, expected) => {
        if (!value || typeof value !== "object" || Array.isArray(value)) return false;
        const actual = Object.keys(value).sort();
        return actual.length === expected.length && actual.every((key, index) => key === expected.slice().sort()[index]);
      };
      const validReference = value => typeof value === "string" && /^[0-9a-f]{32}$/.test(value);
      const validBoundedText = (value, maximum) =>
        typeof value === "string" && value.length > 0 && new TextEncoder().encode(value).length <= maximum;
      const validKey = value => {
        const namedKeys = new Set([
          "Enter", "Escape", "Tab", "Backspace", "Delete", "ArrowUp", "ArrowDown",
          "ArrowLeft", "ArrowRight", "Home", "End", "PageUp", "PageDown", " "
        ]);
        return typeof value === "string" && new TextEncoder().encode(value).length <= 32
          && (namedKeys.has(value) || [...value].length === 1);
      };
      const validLocator = locator => {
        if (!locator || typeof locator !== "object" || Array.isArray(locator)) return false;
        if (locator.kind === "reference") {
          return hasExactKeys(locator, ["kind", "reference"]) && validReference(locator.reference);
        }
        if (locator.kind === "role") {
          return hasExactKeys(locator, ["kind", "role", "name"])
            && typeof locator.role === "string" && /^[a-z][a-z0-9-]{0,63}$/.test(locator.role)
            && validBoundedText(locator.name, 500);
        }
        if (locator.kind === "text") {
          return hasExactKeys(locator, ["kind", "text"]) && validBoundedText(locator.text, 500);
        }
        if (locator.kind === "css") {
          return hasExactKeys(locator, ["kind", "selector"]) && validBoundedText(locator.selector, 512);
        }
        return false;
      };
      const validAction = action => {
        if (!action || typeof action !== "object" || Array.isArray(action)) return false;
        if (action.kind === "click") return hasExactKeys(action, ["kind"]);
        if (action.kind === "type") {
          return hasExactKeys(action, ["kind", "text", "submit"])
            && typeof action.text === "string" && new TextEncoder().encode(action.text).length <= 4096
            && typeof action.submit === "boolean";
        }
        if (action.kind === "pressKey") {
          return hasExactKeys(action, ["kind", "key"]) && validKey(action.key);
        }
        if (action.kind === "scroll") {
          return hasExactKeys(action, ["kind", "direction", "amount"])
            && ["up", "down", "left", "right"].includes(action.direction)
            && Number.isInteger(action.amount) && action.amount >= 1 && action.amount <= 5000;
        }
        return false;
      };
      const findTarget = request => {
        if (!request || typeof request.generation !== "string" || !validLocator(request.locator)) {
          return { error: "invalid_locator" };
        }
        const locator = request.locator;
        if (locator.kind === "reference") {
          const record = nodesByReference.get(locator.reference);
          const node = record && record.node.deref();
          if (!node || record.generation !== request.generation || !node.isConnected
              || referencesByNode.get(node) !== locator.reference) {
            return { error: "stale_reference" };
          }
          return { element: node, ambiguous: false };
        }

        let matches;
        if (locator.kind === "role") {
          const role = locator.role.toLowerCase();
          const name = normalize(locator.name);
          matches = element => roleFor(element) === role && accessibleName(element) === name;
        } else if (locator.kind === "text") {
          const text = normalize(locator.text);
          matches = element => normalize(element.innerText || "") === text;
        } else {
          try {
            document.documentElement.matches(locator.selector);
          } catch (_) {
            return { error: "invalid_locator" };
          }
          matches = element => element.matches(locator.selector);
        }

        const walker = document.createTreeWalker(document, NodeFilter.SHOW_ELEMENT);
        let first = null;
        let ambiguous = false;
        let visited = 0;
        let element = walker.nextNode();
        while (element) {
          if (visited >= maximumLocatorNodes) {
            // Past the walk budget a first match is still a usable target;
            // report it as ambiguous instead of failing the whole action
            // (a reference-less scroll on a large page matches body early).
            return first ? { element: first, ambiguous: true } : { error: "locator_limit" };
          }
          visited += 1;
          // matches() first: visible() forces layout and most nodes do not match.
          if (matches(element) && visible(element)) {
            if (first) {
              ambiguous = true;
              break;
            }
            first = element;
          }
          element = walker.nextNode();
        }
        return first ? { element: first, ambiguous } : { error: "not_found" };
      };
      const targetResult = (request, target) => ({
        ok: true,
        reference: referenceFor(target.element, request.generation),
        ambiguous: target.ambiguous === true
      });
      const dispatchKey = (element, key) => {
        const init = { key, bubbles: true, cancelable: true, view: window };
        element.dispatchEvent(new KeyboardEvent("keydown", init));
        if (key.length === 1) element.dispatchEvent(new KeyboardEvent("keypress", init));
        element.dispatchEvent(new KeyboardEvent("keyup", init));
      };
      const performAction = (element, action) => {
        if (!validAction(action)) return { error: "invalid_action" };
        if (action.kind === "click") {
          if (element instanceof HTMLElement) element.click();
          else element.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, view: window }));
          return { didSubmit: false };
        }
        if (action.kind === "type") {
          if (!(element instanceof HTMLElement)) return { error: "action_rejected" };
          const form = action.submit ? (element.form || element.closest("form")) : null;
          if (action.submit && !form) return { error: "action_rejected" };
          const tag = element.tagName;
          if (tag === "INPUT") {
            const inputType = (element.getAttribute("type") || "text").toLowerCase();
            if (!["text", "search", "email", "url", "tel", "password", "number"].includes(inputType)) {
              return { error: "action_rejected" };
            }
            element.focus();
            const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
            setter.call(element, action.text);
          } else if (tag === "TEXTAREA") {
            element.focus();
            const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value").set;
            setter.call(element, action.text);
          } else if (element.isContentEditable) {
            element.focus();
            element.textContent = action.text;
          } else {
            return { error: "action_rejected" };
          }
          element.dispatchEvent(new InputEvent("input", {
            bubbles: true,
            cancelable: false,
            inputType: "insertText",
            data: action.text
          }));
          element.dispatchEvent(new Event("change", { bubbles: true }));
          if (!action.submit) return { didSubmit: false };
          dispatchKey(element, "Enter");
          form.requestSubmit();
          return { didSubmit: true };
        }
        if (action.kind === "pressKey") {
          if (element instanceof HTMLElement) element.focus();
          dispatchKey(element, action.key);
          return { didSubmit: false };
        }
        const delta = action.amount * (action.direction === "up" || action.direction === "left" ? -1 : 1);
        const deltaX = action.direction === "left" || action.direction === "right" ? delta : 0;
        const deltaY = action.direction === "up" || action.direction === "down" ? delta : 0;
        element.dispatchEvent(new WheelEvent("wheel", {
          deltaX, deltaY, bubbles: true, cancelable: true, view: window
        }));
        if (element === document.scrollingElement || element === document.documentElement || element === document.body) {
          window.scrollBy(deltaX, deltaY);
        } else if (element instanceof HTMLElement) {
          element.scrollBy(deltaX, deltaY);
        }
        return { didSubmit: false };
      };
      const version = "ts_snapshot_v1";

      globalThis.__tsSnapshotV1 = {
        async snapshot(request) {
          const nodes = [];
          let truncated = false;
          let visitedElements = 0;
          const envelopeFor = (candidateNodes, isTruncated) => ({
            version,
            generation: request.generation,
            nodes: candidateNodes,
            truncated: isTruncated
          });
          const fits = candidateNodes => new TextEncoder().encode(
            JSON.stringify(envelopeFor(candidateNodes, true))
          ).length <= request.maximumBytes;
          const visit = (element, parentIndex, depth) => {
            if (truncated) return;
            visitedElements += 1;
            if (visitedElements > request.maximumNodes * 4) {
              truncated = true;
              return;
            }
            if (nodes.length >= request.maximumNodes || depth > 32) {
              truncated = true;
              return;
            }
            if (excludedTags.has(element.tagName)) return;
            if (!visible(element)) {
              // A zero-size box (display: contents, a collapsed float wrapper)
              // can still hold visible children, so descend without emitting a
              // node. Only display:none, visibility:hidden and opacity 0
              // genuinely hide the subtree.
              const style = getComputedStyle(element);
              if (style.display === "none" || style.visibility === "hidden" || Number(style.opacity) === 0) return;
              for (const child of element.children) visit(child, parentIndex, depth + 1);
              return;
            }

            const actionable = isActionable(element);
            const candidate = {
              role: roleFor(element),
              name: accessibleName(element),
              parentIndex,
              bounds: boundsFor(element),
              reference: actionable ? referenceFor(element, request.generation) : null
            };
            if (!fits(nodes.concat(candidate))) {
              truncated = true;
              return;
            }
            const index = nodes.length;
            nodes.push(candidate);
            for (const child of element.children) visit(child, index, depth + 1);
          };

          if (document.documentElement) visit(document.documentElement, null, 0);
          const result = envelopeFor(nodes, truncated);
          const acknowledgement = await webkit.messageHandlers.turboSparkDOMSnapshot.postMessage({
            type: "snapshot",
            generation: request.generation,
            snapshot: result
          });
          if (!acknowledgement || acknowledgement.accepted !== true) {
            throw new Error("Snapshot bridge rejected the page message");
          }
          return result;
        },
        resolve(request) {
          const record = nodesByReference.get(request.reference);
          const node = record && record.node.deref();
          return Boolean(node && record.generation === request.generation && node.isConnected
            && referencesByNode.get(node) === request.reference);
        },
        waitCondition(request) {
          if (!request || !validReference(request.reference)
              || !["exists", "visible", "hidden", "enabled"].includes(request.condition)) {
            return false;
          }
          const record = nodesByReference.get(request.reference);
          const node = record && record.node.deref();
          if (!node || record.generation !== request.generation || !node.isConnected
              || referencesByNode.get(node) !== request.reference) {
            return request.condition === "hidden";
          }
          if (request.condition === "exists") return true;
          if (request.condition === "visible") return visible(node);
          if (request.condition === "hidden") return !visible(node);
          return !("disabled" in node && node.disabled)
            && node.getAttribute("aria-disabled") !== "true";
        },
        async resolveLocator(request) {
          const acknowledgement = await webkit.messageHandlers.turboSparkDOMSnapshot.postMessage({
            type: "resolve",
            generation: request.generation,
            locator: request.locator
          });
          if (!acknowledgement || acknowledgement.accepted !== true) return { error: "rejected" };
          const target = findTarget(request);
          return target.error ? { error: target.error } : targetResult(request, target);
        },
        async perform(request) {
          const acknowledgement = await webkit.messageHandlers.turboSparkDOMSnapshot.postMessage({
            type: "action",
            generation: request.generation,
            locator: request.locator,
            action: request.action
          });
          if (!acknowledgement || acknowledgement.accepted !== true) return { error: "rejected" };
          const target = findTarget(request);
          if (target.error) return { error: target.error };
          const reference = referenceFor(target.element, request.generation);
          const actionResult = performAction(target.element, request.action);
          if (actionResult.error) return { error: actionResult.error };
          return {
            ok: true,
            reference,
            ambiguous: target.ambiguous === true,
            didSubmit: actionResult.didSubmit === true
          };
        },
        async pickAt(request) {
          if (!Number.isFinite(request.x) || !Number.isFinite(request.y)
              || request.x < 0 || request.y < 0) return { error: "invalid" };
          const element = document.elementFromPoint(request.x, request.y);
          if (!element || excludedTags.has(element.tagName) || !visible(element)) return null;
          const candidate = {
            version: "ts_element_pick_v1",
            generation: request.generation,
            reference: referenceFor(element, request.generation),
            role: roleFor(element),
            name: accessibleName(element),
            bounds: boundsFor(element)
          };
          const acknowledgement = await webkit.messageHandlers.turboSparkDOMSnapshot.postMessage({
            type: "pick",
            generation: request.generation,
            candidate
          });
          if (!acknowledgement || acknowledgement.accepted !== true) return { error: "rejected" };
          return candidate;
        }
      };
    })();
    """#

}
