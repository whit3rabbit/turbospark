#!/usr/bin/env python3
"""Report when upstream ggml's quantizers drift from what crates/repack ported.

The GGUF quantizer in crates/repack/src/gguf_quantize is a port of ggml's
encoders, golden-tested byte for byte against ggml at one commit
(GGML_PARITY_COMMIT in provenance.rs). If upstream changes a function we
ported, the port can silently stop matching `llama-quantize`. This compares the
ported functions between the pinned commit and another ref and says which
encoders are affected.

  scripts/check_ggml_drift.py                  pinned commit vs master
  scripts/check_ggml_drift.py --ref b9000      pinned commit vs a tag or commit
  scripts/check_ggml_drift.py --self-test      no network; checks the extractor

Exit status: 0 nothing relevant changed, 1 a ported function changed,
2 the comparison could not be made.

Only public GitHub API reads are made (a few file fetches, unauthenticated).
After reviewing a reported change, re-run the golden tests
(`cargo test -p turbospark-repack --lib gguf_quantize`), regenerate fixtures if
ggml's output changed, and bump GGML_PARITY_COMMIT.
"""
import argparse
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "ggml-org/llama.cpp"
ROOT = Path(__file__).resolve().parent.parent
PRIMARY = "ggml/src/ggml-quants.c"
PROVENANCE = ROOT / "crates/repack/src/gguf_quantize/provenance.rs"

# file -> {C function name: encoders in this crate that depend on it}
PORTED = {
    "ggml/src/ggml-quants.c": {
        "nearest_int": "all K-quants and IQ types",
        "make_qx_quants": "Q3_K, Q6_K, Q4_0, Q5_0 (imatrix)",
        "make_q3_quants": "Q3_K",
        "make_qkx2_quants": "Q2_K, Q4_K, Q5_K",
        "make_qkx3_quants": "Q2_K, Q4_K, Q5_K (imatrix)",
        "make_qp_quants": "Q2_K, Q4_K, Q5_K (imatrix), IQ2_XXS",
        "quantize_row_q2_K_ref": "Q2_K",
        "quantize_row_q2_K_impl": "Q2_K (imatrix)",
        "quantize_row_q3_K_ref": "Q3_K",
        "quantize_row_q3_K_impl": "Q3_K (imatrix)",
        "quantize_row_q4_K_ref": "Q4_K",
        "quantize_row_q4_K_impl": "Q4_K (imatrix)",
        "quantize_row_q5_K_ref": "Q5_K",
        "quantize_row_q5_K_impl": "Q5_K (imatrix)",
        "quantize_row_q6_K_ref": "Q6_K",
        "quantize_row_q6_K_impl": "Q6_K (imatrix)",
        "quantize_row_q4_0_ref": "Q4_0",
        "quantize_row_q4_0_impl": "Q4_0 (imatrix)",
        "quantize_row_q5_0_ref": "Q5_0",
        "quantize_row_q5_0_impl": "Q5_0 (imatrix)",
        "quantize_row_q8_0_ref": "Q8_0",
        "iq2xs_init_impl": "IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, IQ1_M (grids and neighbour tables)",
        "iq3xs_init_impl": "IQ3_XXS, IQ3_S (grids and neighbour tables)",
        "iq2_find_best_neighbour": "IQ2_XXS, IQ2_XS, IQ2_S",
        "iq3_find_best_neighbour": "IQ3_XXS, IQ3_S",
        "iq1_find_best_neighbour2": "IQ1_S, IQ1_M",
        "quantize_row_iq2_xxs_impl": "IQ2_XXS",
        "quantize_row_iq2_xs_impl": "IQ2_XS",
        "quantize_row_iq2_s_impl": "IQ2_S",
        "quantize_row_iq3_xxs_impl": "IQ3_XXS",
        "quantize_row_iq3_s_impl": "IQ3_S",
        "quantize_row_iq1_s_impl": "IQ1_S",
        "quantize_row_iq1_m_impl": "IQ1_M",
        "quantize_row_iq4_nl_impl": "IQ4_NL, IQ4_XS",
        "best_index_iq4nl": "IQ4_NL, IQ4_XS",
        "quantize_iq4_nl": "IQ4_NL (ntry)",
        "quantize_iq4_xs": "IQ4_XS (ntry)",
    },
    "ggml/src/ggml.c": {
        "ggml_quantize_requires_imatrix": "which types need an imatrix",
    },
    "src/llama-quant.cpp": {
        "tensor_requires_imatrix": "the llama.cpp imatrix policy for low-bit types",
    },
    "common/imatrix-loader.cpp": {
        "common_imatrix_load_legacy": "legacy imatrix reader",
        "common_imatrix_load": "GGUF imatrix reader",
    },
}


def fetch(path: str, ref: str) -> str:
    url = f"https://api.github.com/repos/{REPO}/contents/{path}?ref={ref}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.raw", "User-Agent": "turbospark-ggml-drift"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return ""  # the file does not exist at this ref (or the ref is unknown)
        raise


def strip_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    return re.sub(r"//[^\n]*", "", s)


def preprocess(s: str) -> str:
    """Drop comments and preprocessor lines. For `#if/#else/#endif`, keep the
    `#else` branch: some ggml functions open a brace in each branch (for
    example the Apple-linker `volatile` workarounds), which would otherwise
    unbalance brace matching."""
    s = strip_comments(s)
    pat = re.compile(r"^[ \t]*#[ \t]*if[^\n]*\n(?:(?!^[ \t]*#[ \t]*if).)*?^[ \t]*#[ \t]*else[^\n]*\n(.*?)^[ \t]*#[ \t]*endif[^\n]*\n", re.M | re.S)
    while True:
        s, n = pat.subn(lambda m: m.group(1), s)
        if not n:
            break
    return re.sub(r"^[ \t]*#[^\n]*\n", "", s, flags=re.M)


def extract(src: str, name: str):
    """Text of the C/C++ function `name` (signature through its closing brace)."""
    src = preprocess(src)
    for m in re.finditer(r"^[A-Za-z_][^\n;{}]*\b%s\s*\(" % re.escape(name), src, re.M):
        i = src.find("{", m.end())
        semi = src.find(";", m.end())
        if i < 0 or (0 <= semi < i):
            continue  # a declaration, not the definition
        depth, j = 0, i
        while j < len(src):
            c = src[j]
            depth += (c == "{") - (c == "}")
            j += 1
            if depth == 0:
                return re.sub(r"\s+", " ", src[m.start():j]).strip()
    return None


def pinned_commit() -> str:
    m = re.search(r'GGML_PARITY_COMMIT: &str = "([0-9a-f]+)"', PROVENANCE.read_text())
    if not m:
        sys.exit("could not read GGML_PARITY_COMMIT from " + str(PROVENANCE))
    return m.group(1)


def compare(old: str, new: str, names: dict):
    changed, missing = [], []
    for name, users in names.items():
        a, b = extract(old, name), extract(new, name)
        if a is None:
            missing.append((name, "not found at the pinned commit"))
        elif b is None:
            missing.append((name, "missing upstream (renamed or removed)"))
        elif a != b:
            changed.append((name, users))
    return changed, missing


def self_test():
    old = ("static int f(int x) { /* c */ return x + 1; }\nint g(void);\nstatic int g(void) { return 2; }\n"
           "static int h(int n) {\n#ifdef A\n for (volatile int i = 0; i < n; ++i) {\n#else\n for (int i = 0; i < n; ++i) {\n#endif\n  n++; }\n return n; }\n")
    new = "static int f(int x) { return x + 2; }\nstatic int g(void) { return 2; }\n" + old.split("static int h")[1].join(["static int h", ""]) if False else (
           "static int f(int x) { return x + 2; }\nstatic int g(void) { return 2; }\n"
           "static int h(int n) {\n#ifdef A\n for (volatile int i = 0; i < n; ++i) {\n#else\n for (int i = 0; i < n; ++i) {\n#endif\n  n++; }\n return n; }\n")
    assert extract(old, "f") == "static int f(int x) { return x + 1; }"
    assert extract(old, "g") == "static int g(void) { return 2; }"
    assert extract(old, "h").endswith("return n; }"), extract(old, "h")
    ch, mi = compare(old, new, {"f": "A", "g": "B", "h": "C", "nope": "D"})
    assert ch == [("f", "A")] and [n for n, _ in mi] == ["nope"], (ch, mi)
    print("self-test ok")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", default="master", help="upstream ref to compare against (default master)")
    ap.add_argument("--pinned", help="override the pinned commit read from provenance.rs")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    pin = a.pinned or pinned_commit()
    print(f"comparing ported ggml functions: pinned {pin} vs {a.ref}")
    all_changed, all_missing = [], []
    try:
        for path, names in PORTED.items():
            old, new = fetch(path, pin), fetch(path, a.ref)
            if path == PRIMARY:
                for label, text in (("pinned", old), ("compared", new)):
                    if not text:
                        print(f"{PRIMARY} not found at the {label} ref "
                              f"({pin if label == 'pinned' else a.ref}); is it a valid tag or commit?", file=sys.stderr)
                        return 2
            ch, mi = compare(old, new, names)
            all_changed += [(path, *c) for c in ch]
            all_missing += [(path, *m) for m in mi]
    except Exception as e:  # network, rate limit, unknown ref
        print(f"could not compare: {e}", file=sys.stderr)
        return 2
    for path, name, why in all_missing:
        print(f"  ?  {path}: {name}: {why}")
    for path, name, users in all_changed:
        print(f"  CHANGED {path}: {name}  (affects: {users})")
    if all_changed or all_missing:
        print("\nReview the upstream diff, rerun the gguf_quantize golden tests, regenerate "
              "fixtures if ggml's output changed, then bump GGML_PARITY_COMMIT.")
        return 1
    print("no ported function changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
