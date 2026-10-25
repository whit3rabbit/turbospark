"""List or fetch a Hugging Face repo subset at one immutable commit.

usage:
  hf_pin.py list  <repo> [prefix ...]
  hf_pin.py fetch <repo> <dest_dir> [prefix ...]

Prefixes select files by path prefix. The commit is resolved once from
`main` and printed; every URL then uses that 40-hex revision. LFS files are
checked against their sha256 after download, git blobs by size only (their
sha256 is computed and printed so a catalog row can record it).
"""
import hashlib
import json
import os
import sys
import urllib.request

API = "https://huggingface.co/api/models"


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def resolve(repo):
    sha = get_json(f"{API}/{repo}")["sha"]
    info = get_json(f"{API}/{repo}/revision/{sha}?blobs=true")
    return sha, info["siblings"]


def select(siblings, prefixes):
    if not prefixes:
        return siblings
    return [s for s in siblings if any(s["rfilename"].startswith(p) for p in prefixes)]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(repo, sha, files, dest):
    for s in files:
        rel = s["rfilename"]
        out = os.path.join(dest, rel)
        size = s.get("size")
        lfs = s.get("lfs") or {}
        if os.path.exists(out) and os.path.getsize(out) == size:
            pass
        else:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            url = f"https://huggingface.co/{repo}/resolve/{sha}/{urllib.request.quote(rel)}"
            tmp = out + ".part"
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, out)
        got = os.path.getsize(out)
        if got != size:
            sys.exit(f"size mismatch {rel}: {got} != {size}")
        digest = sha256_file(out)
        if lfs.get("sha256") and digest != lfs["sha256"]:
            sys.exit(f"sha256 mismatch {rel}")
        print(f"{digest}  {size:>12}  {rel}")


def main():
    cmd, repo = sys.argv[1], sys.argv[2]
    if cmd == "list":
        sha, sib = resolve(repo)
        files = select(sib, sys.argv[3:])
        print(f"# {repo}@{sha}  files={len(files)}  bytes={sum(s.get('size') or 0 for s in files)}")
        for s in files:
            lfs = (s.get("lfs") or {}).get("sha256", "-")
            print(f"{s.get('size'):>12}  {lfs[:12]}  {s['rfilename']}")
    elif cmd == "fetch":
        dest = sys.argv[3]
        sha, sib = resolve(repo)
        files = select(sib, sys.argv[4:])
        print(f"# {repo}@{sha}  files={len(files)}  bytes={sum(s.get('size') or 0 for s in files)}")
        fetch(repo, sha, files, dest)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
