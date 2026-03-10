#!/usr/bin/env python
"""Run filter on all detect run directories."""

import subprocess
import sys
from pathlib import Path

def main():
    runs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/runs")

    if not runs_dir.exists():
        print(f"Error: {runs_dir} does not exist")
        sys.exit(1)

    run_dirs = sorted(runs_dir.iterdir())
    print(f"Found {len(run_dirs)} run directories in {runs_dir}\n")

    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue

        results_dir = run_dir / "results"
        if not results_dir.exists():
            print(f"[{run_dir.name}] Skipping - no results/ directory")
            continue

        # Check for events results file
        events_files = list(results_dir.glob("*events_results*.csv")) + \
                       list(results_dir.glob("*events_results*.parquet"))
        if not events_files:
            print(f"[{run_dir.name}] Skipping - no events results file")
            continue

        print(f"\n[{run_dir.name}] Running filter...")
        cmd = [
            sys.executable, "-m", "malca.filter",
            "--detect-run", str(run_dir),
            "-v"
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[{run_dir.name}] Failed!")
        else:
            print(f"[{run_dir.name}] Done")

    print("\n" + "=" * 40)
    print("All runs processed")

if __name__ == "__main__":
    main()
