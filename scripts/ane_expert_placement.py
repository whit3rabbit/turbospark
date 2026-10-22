#!/usr/bin/env python3
"""Run the shared-plane hardware probe on isolated expert programs.

Record actual compute plans and exact sync/async output agreement. Timings
in raw probe output do not include expert preparation or qualify a speedup.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("programs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist")
    root = args.programs.resolve()
    manifest = json.loads((root / "experiment.json").read_text())
    if manifest.get("format") != "expert-probe-v1":
        parser.error("expert-probe-v1 manifest required")
    names = [row["name"] for row in manifest["programs"]]
    if not names or len(set(names)) != len(names):
        parser.error("nonempty unique program names required")
    for name in names:
        if (root / name).resolve().parent != root or not name.endswith(".mlpackage"):
            parser.error("program must be a package directly inside the program directory")
    args.output.mkdir(parents=True)
    rows = []
    for row in manifest["programs"]:
        name = row["name"]
        print(f"running {name}", flush=True)
        with (args.output / (name + ".stdout")).open("w") as out, \
                (args.output / (name + ".stderr")).open("w") as err:
            subprocess.run([str(args.probe.resolve()), str(root / name), "2", "paired"],
                           stdout=out, stderr=err, check=True, timeout=180,
                           env=dict(os.environ, ANE_PROBE_UNITS="ane"))
        samples = [json.loads(line) for line in
                   (args.output / (name + ".stdout")).read_text().splitlines()]
        if len(samples) != 4 or {(s["round"], s["executor"]) for s in samples} != {
                (r, executor) for r in (1, 2) for executor in ("sync", "async")}:
            raise RuntimeError(f"missing paired execution evidence for {name}")
        stderr = (args.output / (name + ".stderr")).read_text()
        ops = []
        for record in re.findall(r"OpPlacement \{([^}]+)\}", stderr):
            operator = re.search(r'operator: "([^"]+)"', record)
            preferred = re.search(r"preferred: Some\((\w+)\)", record)
            if operator and preferred:
                ops.append(dict(operator=operator[1], preferred=preferred[1]))
        if not ops or not any("matmul" in op["operator"] for op in ops):
            raise RuntimeError(f"missing matmul placement evidence for {name}")
        rows.append(dict(row, preferred_ops=ops, sync_async_equal=True))
        # Preserve partial results if a later child fails or times out.
        (args.output / "summary.json").write_text(json.dumps(dict(
            format="expert-placement-v1", programs=rows,
            complete=len(rows) == len(names),
            compute_units="cpu_and_ane",
            includes_preparation=False, throughput_qualified=False), indent=2) + "\n")


if __name__ == "__main__":
    main()
