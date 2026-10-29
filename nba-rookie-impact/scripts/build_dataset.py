"""Build data/processed/rookies.csv from the raw downloads in data/raw/.

Needs pandas (run it in Codespaces or any machine where pandas works):
    python scripts/build_dataset.py
    python scripts/build_dataset.py --min-minutes 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rookies.metrics import build_rookie_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "rookies.csv")
    parser.add_argument("--min-minutes", type=float, default=50.0, help="drop rookies with fewer total minutes")
    args = parser.parse_args()

    table = build_rookie_table(args.raw_dir, args.min_minutes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    seasons = table["season"].nunique()
    print(f"Wrote {len(table)} rookie seasons ({seasons} seasons) to {args.out}")


if __name__ == "__main__":
    main()
