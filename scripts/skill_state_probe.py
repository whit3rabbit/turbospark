#!/usr/bin/env python3
"""SKILL.state go/no-go probe: can a LOCAL install drive a bounded-state agent loop?

Measures the two numbers docs/SKILL_STATE.md's "The measurement that decides"
section names: the valid-patch rate (first try, and after one retry) and the
state accuracy at T steps, against the append-only baseline on the same events.

Two arms over one deterministic warehouse task:

  state    [fixed spec P][current state Sigma][one event]  -> JSON Merge Patch
  history  [fixed spec P][every event and reply so far]    -> complete state

Both arms emit a state at every step, so scoring is identical and the ONLY
variable is what the prompt carries. Needs a running `turbospark-server`.

Stdlib only, no dependency, nothing installed. Offline self-check first:

  python3 scripts/skill_state_probe.py --selfcheck

That runs the oracle, noop and clobber agents with no server at all and
asserts the scorer awards 1.00 to a correct agent and less to both wrong
ones. A harness that cannot fail cannot measure; run it before believing any
number below it.

  python3 scripts/skill_state_probe.py --arm state --steps 50 --port 8080 \
      --label gemma4 --out /tmp/skill-state/gemma4-state.json
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

SHELVES = [f"shelf-{i}" for i in range(1, 9)]
ITEMS = [f"sku-{i:02d}" for i in range(1, 15)]

SCHEMA_TEXT = (
    '{"shelves": {"<shelf-id>": ["<item-id>", ...]}, "shipped": ["<item-id>", ...]}'
)

RULES = f"""The state schema is exactly:
  {SCHEMA_TEXT}
Shelf ids are shelf-1 through shelf-8. Item ids look like sku-07."""

SPEC_STATE = f"""You are a warehouse state tracker. Each step you are given the CURRENT STATE as
JSON and one EVENT. Reply with ONE JSON object and nothing else, with keys:
  "reasoning": a short string, at most 15 words
  "patch": a JSON Merge Patch (RFC 7386) applied to the current state

{RULES}

Patch rules:
- Include ONLY the keys that change. An event that changes nothing takes {{}}.
- To change a shelf, give that shelf's COMPLETE new item list. Arrays are
  replaced wholesale, never merged.
- To empty a shelf, map it to [] . To remove the shelf key, map it to null.
- Never invent an item or a shelf the event did not mention."""

SPEC_HISTORY = f"""You are a warehouse state tracker. You are given a sequence of EVENTS, one per
step. Reply with ONE JSON object and nothing else, with keys:
  "reasoning": a short string, at most 15 words
  "state": the COMPLETE current state after applying every event so far

{RULES}

Never invent an item or a shelf no event mentioned."""

NOISE = [
    "NOTICE: the loading dock closes early on Friday.",
    "NOTICE: the break room coffee machine is out of order.",
    "NOTICE: a safety drill is scheduled for next Tuesday.",
    "NOTICE: new high-visibility vests arrived for staff.",
    "NOTICE: the forklift battery charger was serviced today.",
]


# ---------------------------------------------------------------- task


def empty_state():
    return {"shelves": {}, "shipped": []}


def locate(state, item):
    """Where the state believes `item` is: a shelf id, 'shipped', or None."""
    for shelf, items in sorted(state.get("shelves", {}).items()):
        if isinstance(items, list) and item in items:
            return shelf
    if item in state.get("shipped", []):
        return "shipped"
    return None


def build_task(seed, steps, noise_every):
    """Deterministic event stream plus the ground-truth state after each event.

    Conservation invariant, asserted here rather than trusted: after every
    event each item is in exactly one place (a shelf, shipped, or unplaced).
    """
    rng = random.Random(seed)
    state = empty_state()
    events, truths = [], []
    for step in range(steps):
        if noise_every and step % noise_every == noise_every - 1:
            events.append(rng.choice(NOISE))
            truths.append(json.loads(json.dumps(state)))
            continue
        on_shelf = [i for i in ITEMS if locate(state, i) not in (None, "shipped")]
        unplaced = [i for i in ITEMS if locate(state, i) is None]
        choices = ["store"] if not on_shelf else ["ship", "move", "audit"]
        if unplaced and on_shelf:
            choices = ["store", "store", "ship", "move", "audit"]
        action = rng.choice(choices)
        if action == "store":
            item, shelf = rng.choice(unplaced), rng.choice(SHELVES)
            state["shelves"].setdefault(shelf, []).append(item)
            events.append(f"STORE {item} ON {shelf}")
        elif action == "ship":
            item = rng.choice(on_shelf)
            state["shelves"][locate(state, item)].remove(item)
            state["shipped"].append(item)
            events.append(f"SHIP {item}")
        elif action == "move":
            item = rng.choice(on_shelf)
            src = locate(state, item)
            dst = rng.choice([s for s in SHELVES if s != src])
            state["shelves"][src].remove(item)
            state["shelves"].setdefault(dst, []).append(item)
            events.append(f"MOVE {item} TO {dst}")
        else:
            shelf = rng.choice(sorted(state["shelves"]))
            listed = ", ".join(state["shelves"][shelf]) or "nothing"
            events.append(f"AUDIT {shelf} reports: {listed}")
        seen = [i for i in ITEMS if locate(state, i) is not None]
        assert len(seen) == len(set(seen)), "ground truth lost an item"
        truths.append(json.loads(json.dumps(state)))
    return events, truths


# ------------------------------------------------------- patch and score


def merge_patch(target, patch):
    """RFC 7386 JSON Merge Patch. Arrays replace; null deletes."""
    if not isinstance(patch, dict):
        return json.loads(json.dumps(patch))
    out = json.loads(json.dumps(target)) if isinstance(target, dict) else {}
    for key, value in patch.items():
        if value is None:
            out.pop(key, None)
        else:
            out[key] = merge_patch(out.get(key), value)
    return out


def validate_patch(patch):
    """Schema errors in a proposed patch. Empty list means valid."""
    errors = []
    if not isinstance(patch, dict):
        return ["patch is not a JSON object"]
    for key, value in patch.items():
        if key not in ("shelves", "shipped"):
            errors.append(f"unknown key {key!r}")
        elif key == "shipped":
            if not isinstance(value, list) or not all(
                isinstance(x, str) for x in value
            ):
                errors.append("'shipped' must be an array of strings")
        elif not isinstance(value, dict):
            errors.append("'shelves' must be an object")
        else:
            for shelf, items in value.items():
                if items is None:
                    continue
                if not isinstance(items, list) or not all(
                    isinstance(x, str) for x in items
                ):
                    errors.append(f"shelf {shelf!r} must be an array of strings")
    return errors


def item_accuracy(state, truth):
    """Fraction of the 14 items whose location matches ground truth."""
    hits = sum(1 for i in ITEMS if locate(state, i) == locate(truth, i))
    return hits / len(ITEMS)


def extract_json(text):
    """First balanced JSON object in `text`, mirroring the server's rescue.

    Returns (object, was_whole_reply) or (None, False). `was_whole_reply`
    distinguishes a model that emitted clean JSON from one that needed
    rescuing out of prose, which is the difference between the strict and
    rescued validity rates this probe reports separately.
    """
    stripped = text.strip()
    try:
        return json.loads(stripped), True
    except ValueError:
        pass
    depth, start, in_str, esc = 0, None, False, False
    for idx, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : idx + 1]), False
                except ValueError:
                    start = None
    return None, False


# ------------------------------------------------------------- agents


class LocalAgent:
    """Server-free controls. These are what make the harness falsifiable."""

    def __init__(self, kind, truths):
        self.kind, self.truths, self.usage = kind, truths, (0, 0)

    def reply(self, step, _messages, arm):
        truth = self.truths[step]
        if self.kind == "oracle":
            body = {"state": truth} if arm == "history" else {"patch": truth}
        elif self.kind == "noop":
            body = {"state": empty_state()} if arm == "history" else {"patch": {}}
        else:  # clobber: valid shape, destroys state the event did not mention
            wipe = {"shelves": {s: [] for s in SHELVES}, "shipped": []}
            body = {"state": wipe} if arm == "history" else {"patch": wipe}
        return json.dumps(body), (0, 0)


class ServerAgent:
    def __init__(self, port, max_tokens, timeout, seed):
        self.url = f"http://127.0.0.1:{port}/v1/chat/completions"
        self.max_tokens, self.timeout, self.seed = max_tokens, timeout, seed

    def reply(self, _step, messages, _arm):
        payload = {
            "model": "local",
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "stream": False,
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode())
        usage = body.get("usage") or {}
        return (
            body["choices"][0]["message"].get("content") or "",
            (usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)),
        )


# --------------------------------------------------------------- run


def run(agent, events, truths, arm, retries=1):
    state, history, steps = empty_state(), [], []
    p_tokens = c_tokens = 0
    for idx, event in enumerate(events):
        if arm == "state":
            messages = [
                {"role": "system", "content": SPEC_STATE},
                {
                    "role": "user",
                    "content": "CURRENT STATE:\n"
                    + json.dumps(state, sort_keys=True)
                    + f"\n\nEVENT:\n{event}",
                },
            ]
        else:
            messages = [{"role": "system", "content": SPEC_HISTORY}] + history + [
                {"role": "user", "content": f"EVENT:\n{event}"}
            ]

        record = {"step": idx, "event": event, "attempts": []}
        applied, strict_ok = False, False
        for attempt in range(retries + 1):
            try:
                text, (pt, ct) = agent.reply(idx, messages, arm)
            except urllib.error.HTTPError as exc:
                # The server's own message: this is where a prompt that
                # outgrew the context window reports itself.
                try:
                    detail = exc.read().decode()[:400]
                except OSError:
                    detail = ""
                record["attempts"].append(
                    {"class": "transport", "error": f"HTTP {exc.code}: {detail}"}
                )
                break
            except (urllib.error.URLError, OSError, KeyError) as exc:
                record["attempts"].append(
                    {"class": "transport", "error": f"{type(exc).__name__}: {exc}"}
                )
                break
            p_tokens, c_tokens = p_tokens + pt, c_tokens + ct
            # Per step, so the O(T) vs O(T^2) prompt growth is recoverable
            # from the artifact rather than only its totals.
            record.setdefault("tokens", []).append([pt, ct])
            record.setdefault("raw", []).append(text[:600])
            obj, whole = extract_json(text)
            if obj is None:
                record["attempts"].append({"class": "syntax"})
                nudge = "Your reply was not JSON. Reply with ONE JSON object only."
            else:
                key = "state" if arm == "history" else "patch"
                body = obj.get(key)
                if body is None:
                    record["attempts"].append({"class": "schema",
                                               "errors": [f"missing {key!r}"]})
                    nudge = f'Your reply had no "{key}" key. Include it.'
                else:
                    errors = validate_patch(body)
                    if errors:
                        record["attempts"].append({"class": "schema",
                                                   "errors": errors})
                        nudge = "Your patch was invalid: " + "; ".join(errors)
                    else:
                        record["attempts"].append({"class": "ok",
                                                   "whole_reply": whole})
                        state = (
                            merge_patch(empty_state(), body)
                            if arm == "history"
                            else merge_patch(state, body)
                        )
                        applied, strict_ok = True, whole
                        break
            if attempt < retries:
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": nudge},
                ]

        if arm == "history":
            history += [
                {"role": "user", "content": f"EVENT:\n{event}"},
                {"role": "assistant",
                 "content": json.dumps({"state": state}, sort_keys=True)},
            ]
        record.update(
            applied=applied,
            strict=strict_ok,
            accuracy=item_accuracy(state, truths[idx]),
            state=json.loads(json.dumps(state)),
        )
        steps.append(record)
    return steps, (p_tokens, c_tokens)


def summarize(steps, truths, tokens, label, arm):
    n = len(steps)
    first = [
        s for s in steps if s["attempts"] and s["attempts"][0].get("class") == "ok"
    ]
    classes = {}
    for s in steps:
        for a in s["attempts"]:
            cls = a.get("class", "unknown")
            classes[cls] = classes.get(cls, 0) + 1
    # The step a transport error first appeared, which for the history arm is
    # where the growing prompt outran the server's context window.
    stalled = next(
        (s["step"] for s in steps
         if any(a.get("class") == "transport" for a in s["attempts"])),
        None,
    )
    diverged = next((s["step"] for s in steps if s["accuracy"] < 1.0), None)
    return {
        "label": label,
        "arm": arm,
        "steps": n,
        "valid_first": len(first) / n,
        "valid_strict_first": sum(1 for s in first if s["strict"]) / n,
        "valid_after_retry": sum(1 for s in steps if s["applied"]) / n,
        "final_item_accuracy": steps[-1]["accuracy"] if steps else 0.0,
        "mean_item_accuracy": sum(s["accuracy"] for s in steps) / n,
        "exact_final_state": steps[-1]["state"] == truths[-1] if steps else False,
        "first_divergence_step": diverged,
        "first_transport_error_step": stalled,
        "attempt_classes": classes,
        "prompt_tokens": tokens[0],
        "completion_tokens": tokens[1],
        "total_tokens": tokens[0] + tokens[1],
    }


VALIDATOR_CASES = [
    # (patch, expect_valid, why)
    ({}, True, "empty patch is the correct answer to a no-op event"),
    ({"shelves": {"shelf-1": ["sku-01"]}}, True, "shelf replaced with a list"),
    ({"shelves": {"shelf-1": None}}, True, "null deletes a shelf"),
    ({"shipped": []}, True, "empty shipped list"),
    ({"inventory": {}}, False, "unknown top-level key"),
    ({"shipped": "sku-01"}, False, "shipped must be an array"),
    ({"shipped": [1, 2]}, False, "shipped must hold strings"),
    ({"shelves": ["shelf-1"]}, False, "shelves must be an object"),
    ({"shelves": {"shelf-1": "sku-01"}}, False, "a shelf must be an array"),
    ({"shelves": {"shelf-1": [3]}}, False, "a shelf must hold strings"),
    ([], False, "a patch must be an object"),
]

# No control agent emits a null-deletion, so merge_patch's delete branch is
# reachable only from real model output. Found by mutation: deleting that
# branch left the whole selfcheck green.
MERGE_CASES = [
    ({"shelves": {"a": ["x"]}}, {"shelves": {"a": None}},
     {"shelves": {}}, "null deletes a shelf"),
    ({"shelves": {"a": ["x"], "b": ["y"]}}, {"shelves": {"a": None}},
     {"shelves": {"b": ["y"]}}, "null deletes only its own key"),
    ({"shelves": {"a": ["x", "y"]}}, {"shelves": {"a": ["z"]}},
     {"shelves": {"a": ["z"]}}, "arrays replace, never merge"),
    ({"shelves": {"a": ["x"]}, "shipped": []}, {},
     {"shelves": {"a": ["x"]}, "shipped": []}, "empty patch changes nothing"),
    ({"shelves": {"a": ["x"]}, "shipped": []}, {"shipped": ["x"]},
     {"shelves": {"a": ["x"]}, "shipped": ["x"]}, "absent keys untouched"),
]

EXTRACTOR_CASES = [
    ('{"patch": {}}', True, "clean JSON is whole-reply"),
    ('```json\n{"patch": {}}\n```', False, "fenced JSON needs rescuing"),
    ('Sure! {"patch": {}} hope that helps', False, "JSON buried in prose"),
    ('{"patch": {"shelves": {"a": ["}{"]}}}', True, "braces inside a string"),
    ("no json here at all", None, "unparseable"),
]


def check_units():
    """Direct cases for the two functions the control agents cannot reach."""
    ok = True
    for patch, expect, why in VALIDATOR_CASES:
        got = not validate_patch(patch)
        if got != expect:
            print(f"  FAIL validator: {why} -> valid={got}, want {expect}")
            ok = False
    for target, patch, want, why in MERGE_CASES:
        got = merge_patch(target, patch)
        if got != want:
            print(f"  FAIL merge: {why} -> {got!r}, want {want!r}")
            ok = False
    for text, expect_whole, why in EXTRACTOR_CASES:
        obj, whole = extract_json(text)
        if expect_whole is None:
            if obj is not None:
                print(f"  FAIL extractor: {why} -> parsed {obj!r}")
                ok = False
        elif obj is None or whole != expect_whole:
            print(f"  FAIL extractor: {why} -> obj={obj!r} whole={whole}")
            ok = False
    print(f"  units    validator={len(VALIDATOR_CASES)} "
          f"merge={len(MERGE_CASES)} extractor={len(EXTRACTOR_CASES)} "
          f"{'ok' if ok else 'FAILED'}")
    return ok


def selfcheck(steps_n, seed):
    """The harness must award 1.00 to a correct agent and less to wrong ones."""
    events, truths = build_task(seed, steps_n, 7)
    print(f"selfcheck: {len(events)} events, seed {seed}")
    units_ok = check_units()
    results = {}
    for arm in ("state", "history"):
        for kind in ("oracle", "noop", "clobber"):
            recs, tok = run(LocalAgent(kind, truths), events, truths, arm)
            s = summarize(recs, truths, tok, kind, arm)
            results[(arm, kind)] = s
            print(f"  {arm:8s} {kind:8s} valid={s['valid_after_retry']:.2f} "
                  f"final_acc={s['final_item_accuracy']:.3f} "
                  f"exact={s['exact_final_state']}")
    ok = units_ok
    for arm in ("state", "history"):
        o, no, cl = (results[(arm, k)] for k in ("oracle", "noop", "clobber"))
        for name, cond in (
            ("oracle scores 1.00", o["final_item_accuracy"] == 1.0),
            ("oracle is exact", o["exact_final_state"]),
            ("oracle is always valid", o["valid_after_retry"] == 1.0),
            ("noop scores below oracle", no["final_item_accuracy"] < 1.0),
            ("clobber scores below oracle", cl["final_item_accuracy"] < 1.0),
            ("clobber is schema-VALID", cl["valid_after_retry"] == 1.0),
        ):
            if not cond:
                print(f"  FAIL [{arm}] {name}")
                ok = False
    print("selfcheck:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("state", "history"), default="state")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--noise-every", type=int, default=7,
                    help="every Nth event is an irrelevant notice (0 disables)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--label", default="server")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck(args.steps, args.seed)

    events, truths = build_task(args.seed, args.steps, args.noise_every)
    agent = ServerAgent(args.port, args.max_tokens, args.timeout, args.seed)
    started = time.time()
    recs, tok = run(agent, events, truths, args.arm, args.retries)
    summary = summarize(recs, truths, tok, args.label, args.arm)
    summary["wall_seconds"] = round(time.time() - started, 1)
    summary["seed"] = args.seed
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"summary": summary, "steps": recs}, fh, indent=1)
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
