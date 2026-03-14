#!/usr/bin/env python3
"""
sync_logs.py — Pull results.tsv + claude_loop.log from RunPod and display them.

Usage:
    python sync_logs.py              # one-shot sync + print
    python sync_logs.py --watch      # poll every 60s
    python sync_logs.py --interval 30  # poll every 30s
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

POD_HOST = "qjgwqvg5jlnvb8-64410b0c@ssh.runpod.io"
SSH_KEY = Path.home() / ".ssh" / "id_ed25519"
REMOTE_RESULTS = "/autoresearch/results.tsv"
REMOTE_LOG = "/autoresearch/claude_loop.log"
LOCAL_RESULTS = Path(__file__).parent / "results.tsv"
LOG_TAIL_LINES = 20


def scp_pull(remote_path: str, local_path: Path) -> bool:
    """Pull a single file from the pod via scp. Returns True on success."""
    cmd = [
        "scp",
        "-i", str(SSH_KEY),
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=15",
        f"{POD_HOST}:{remote_path}",
        str(local_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [scp error] {remote_path}: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def ssh_tail(remote_path: str, n: int = LOG_TAIL_LINES) -> str:
    """Fetch last n lines of a remote file via ssh."""
    cmd = [
        "ssh",
        "-i", str(SSH_KEY),
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=15",
        POD_HOST,
        f"tail -n {n} {remote_path} 2>/dev/null || echo '[file not found]'",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def parse_results(path: Path) -> list[dict]:
    """Parse results.tsv into a list of row dicts."""
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    if not lines:
        return []
    header = lines[0].split("\t")
    for line in lines[1:]:
        parts = line.split("\t")
        row = dict(zip(header, parts))
        rows.append(row)
    return rows


def print_table(rows: list[dict]) -> None:
    """Print results as a formatted table."""
    if not rows:
        print("  (no results yet)")
        return

    # Determine columns present
    all_keys = list(rows[0].keys()) if rows else []

    # Column display order (show what's available)
    preferred = ["commit", "val_bpb", "mem_gb", "status", "description"]
    cols = [k for k in preferred if k in all_keys]
    extra = [k for k in all_keys if k not in cols]
    cols = cols + extra

    # Compute widths
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0)) for c in cols}

    # Header
    idx_w = len(str(len(rows)))
    header_parts = [f"{'#':>{idx_w}}"] + [f"{c:<{widths[c]}}" for c in cols]
    header = "  ".join(header_parts)
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    # Find best val_bpb row
    best_idx = None
    try:
        bpb_vals = [float(r.get("val_bpb", "inf")) for r in rows]
        best_idx = bpb_vals.index(min(bpb_vals))
    except (ValueError, TypeError):
        pass

    for i, row in enumerate(rows):
        marker = " ← BEST" if i == best_idx else ""
        parts = [f"{i+1:>{idx_w}}"] + [f"{str(row.get(c, '')):  <{widths[c]}}" for c in cols]
        print("  ".join(parts) + marker)

    print(sep)


def sync_and_print() -> None:
    print(f"\n[{time.strftime('%H:%M:%S')}] Syncing from pod...")

    # Pull results.tsv
    ok = scp_pull(REMOTE_RESULTS, LOCAL_RESULTS)
    if ok:
        print(f"  results.tsv → {LOCAL_RESULTS}")
    else:
        print("  results.tsv: could not pull (pod may be offline or file not yet created)")

    rows = parse_results(LOCAL_RESULTS)
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT RESULTS  ({len(rows)} runs)")
    print(f"{'='*60}")
    print_table(rows)

    # Tail the log
    print(f"\n--- claude_loop.log (last {LOG_TAIL_LINES} lines) ---")
    log_tail = ssh_tail(REMOTE_LOG, LOG_TAIL_LINES)
    print(log_tail or "  (empty or unreachable)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync and display autoresearch logs from RunPod.")
    parser.add_argument("--watch", action="store_true", help="Poll continuously")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds (default: 60)")
    args = parser.parse_args()

    if args.watch:
        print(f"Watching pod logs (interval={args.interval}s). Ctrl+C to stop.")
        try:
            while True:
                sync_and_print()
                print(f"\nNext sync in {args.interval}s...")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        sync_and_print()


if __name__ == "__main__":
    main()
