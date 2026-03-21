"""Automated dataset collection manager for hypTcn.

Wraps the hyptcn binary to collect labeled memory frames into ml/data/raw/<label>/.

Usage:
    from ml.pipeline.collect import CollectionManager
    mgr = CollectionManager()
    frames = mgr.start("normal", duration_sec=60)

CLI:
    python ml/pipeline/collect.py --label normal --duration 60
    python ml/pipeline/collect.py --label normal --duration 60 --vm hyptcn-guest
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_OUTPUT = os.path.join(_REPO_ROOT, "ml", "data", "raw")
_BINARY = os.path.join(_REPO_ROOT, "bin", "hyptcn")

VALID_LABELS = ("normal", "malware", "shellcode", "rootkit", "cryptominer", "ransomware")


class CollectionManager:
    """Manages labeled dataset collection via the hyptcn binary.

    Args:
        output_dir: Root directory; frames go into <output_dir>/<label>/
        binary:     Path to the hyptcn binary.
        interval_ms: Sampling interval in milliseconds.
    """

    def __init__(
        self,
        output_dir: str = _DEFAULT_OUTPUT,
        binary: str = _BINARY,
        interval_ms: int = 100,
    ) -> None:
        self.output_dir = output_dir
        self.binary = binary
        self.interval_ms = interval_ms

    def start(
        self,
        label: str,
        duration_sec: int,
        vm_name: str = "",
        sysmap: str = "",
    ) -> int:
        """Collect frames for one label.

        Args:
            label:        One of VALID_LABELS.
            duration_sec: How long to collect (0 = run until Ctrl-C).
            vm_name:      KVM domain name; empty = mock mode.
            sysmap:       Path to System.map; empty = skip OS layer.

        Returns:
            Number of .bin files written to <output_dir>/<label>/.
        """
        if label not in VALID_LABELS:
            raise ValueError(f"Unknown label {label!r}. Valid: {VALID_LABELS}")

        label_dir = os.path.join(self.output_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        cmd = [
            self.binary,
            "--collect",
            "--collect-label", label,
            "--collect-dir", self.output_dir,
            "--interval", str(self.interval_ms),
            "--quiet",
        ]

        if vm_name:
            cmd += ["--vm", vm_name]
        else:
            cmd.append("--mock")

        if sysmap:
            cmd += ["--sysmap", sysmap]

        if duration_sec > 0:
            cmd += ["--collect-duration", str(duration_sec)]

        if not os.path.isfile(self.binary):
            raise FileNotFoundError(
                f"hyptcn binary not found at {self.binary!r}. Run 'make' first."
            )

        before = _count_bins(label_dir)
        print(
            f"[collect] {label}: running for {duration_sec}s "
            f"({'mock' if not vm_name else vm_name}) → {label_dir}",
            flush=True,
        )
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=False)
        elapsed = time.time() - t0
        after = _count_bins(label_dir)
        frames = after - before
        print(
            f"[collect] {label}: {frames} frames in {elapsed:.1f}s "
            f"(exit={proc.returncode})",
            flush=True,
        )
        return frames


def _count_bins(directory: str) -> int:
    if not os.path.isdir(directory):
        return 0
    return sum(1 for f in os.listdir(directory) if f.endswith(".bin"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect labeled frames via hyptcn binary")
    parser.add_argument("--label", required=True, choices=VALID_LABELS)
    parser.add_argument("--duration", type=int, default=60, help="seconds (0=unlimited)")
    parser.add_argument("--vm", default="", help="KVM domain name; empty=mock")
    parser.add_argument("--sysmap", default="", help="Path to System.map")
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT)
    parser.add_argument("--interval", type=int, default=100, help="ms between samples")
    args = parser.parse_args()

    mgr = CollectionManager(
        output_dir=args.output_dir,
        interval_ms=args.interval,
    )
    mgr.start(args.label, args.duration, vm_name=args.vm, sysmap=args.sysmap)


if __name__ == "__main__":
    main()
