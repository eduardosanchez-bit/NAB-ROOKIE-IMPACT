"""Download raw NBA data and rebuild the processed rookie dataset.

Run from the project root:
    python scripts/update_data.py                       # 2015-16 ... 2025-26
    python scripts/update_data.py --start 2022 --end 2024 --skip-onoff
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rookies.fetch import fetch_all  # noqa: E402
from rookies.metrics import build_rookie_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2015)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--skip-onoff", action="store_true")
    parser.add_argument("--min-minutes", type=float, default=50.0)
    args = parser.parse_args()

    raw_dir = ROOT / "data" / "raw"
    fetch_all(args.start, args.end, raw_dir, args.pause, not args.skip_onoff)

    table = build_rookie_table(raw_dir, args.min_minutes)
    out = ROOT / "data" / "processed" / "rookies.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"Wrote {len(table)} rookie seasons to {out}")


if __name__ == "__main__":
    main()
