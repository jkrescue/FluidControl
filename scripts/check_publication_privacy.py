"""Read-only publication audit. Prints categories and object IDs, never matched values."""
import argparse
import collections
import re
import subprocess
from pathlib import Path

PATTERNS = {
    "user_path": re.compile(rb'/(?:home|Users)/(?!USER(?:/|\b))[^/\s"\x27<>:;,]+'),
    "private_ip": re.compile(rb'(?<![\d.])(?:10\.(?:\d{1,3}\.){2}\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(?![\d.])'),
    "private_key": re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----'),
    "ssh_key": re.compile(rb'\b(?:ssh-rsa|ssh-ed25519|ecdsa-sha2-[\w-]+)\s+[A-Za-z0-9+/]{40,}'),
    "token": re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|glpat-[A-Za-z0-9_-]{15,}|sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b'),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-history", action="store_true", help="Audit every local Git object in a dedicated publication clone")
    parser.add_argument("--terms-file", type=Path, help="Optional untracked local file of private terms; values are never printed")
    args = parser.parse_args()
    if args.terms_file:
        for index, term in enumerate(args.terms_file.read_bytes().splitlines()):
            if term.strip():
                PATTERNS["private_term_" + str(index)] = re.compile(re.escape(term.strip()), re.I)
    counts = collections.Counter()
    checked = 0
    if args.all_history:
        process = subprocess.Popen(["git", "cat-file", "--batch-all-objects", "--batch"], stdout=subprocess.PIPE)
    else:
        objects = subprocess.check_output(["git", "ls-tree", "-r", "HEAD"]).splitlines()
        ids = sorted({line.split()[2] for line in objects})
        process = subprocess.Popen(["git", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        # communicate drains output while sending IDs, avoiding pipe deadlock.
        raw, _ = process.communicate(b"\n".join(ids) + b"\n")
        import io
        process.stdout = io.BytesIO(raw)
    for header in iter(process.stdout.readline, b""):
        digest, kind, size = header.split()
        data = process.stdout.read(int(size))
        if process.stdout.read(1) != b"\n":
            raise ValueError("invalid Git batch output")
        if kind not in {b"blob", b"commit", b"tag"}:
            continue
        checked += 1
        for category, pattern in PATTERNS.items():
            if pattern.search(data):
                counts[category] += 1
                print(category, digest.decode())
        if kind == b"commit":
            for line in data.split(b"\n\n", 1)[0].splitlines():
                if line.startswith((b"author ", b"committer ")):
                    if b"FluidControl Contributors <contributors@example.invalid>" not in line:
                        counts["commit_identity"] += 1
    if process.wait() != 0:
        raise RuntimeError("Git audit failed")
    print("PASS" if not counts else "FAIL", "objects=" + str(checked), dict(counts))
    return bool(counts)

if __name__ == "__main__":
    raise SystemExit(main())
