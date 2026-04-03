#!/usr/bin/env python3
"""
Propagate occlusion tags from original game CSVs to their augmented variants.

Since augmented images (bright/color/dark/noisy) share the same frame numbers
and board positions, occlusion tags apply identically.

Usage:
    python propagate_occlusions.py
    python propagate_occlusions.py --games 8,9,10,11,12
"""

import argparse
from pathlib import Path
import pandas as pd


DATA_DIR = Path("data")
VARIANTS = ["bright", "color", "dark", "noisy"]


def propagate_game(game_num: int):
    """Copy the 'occluded' column from the original game CSV to all augmented variants."""
    base_dir = DATA_DIR / f"game{game_num}_per_frame"
    base_csvs = list(base_dir.glob("*.csv"))
    if not base_csvs:
        print(f"  Skip game{game_num}: no CSV in {base_dir}")
        return

    base_df = pd.read_csv(base_csvs[0])
    if "occluded" not in base_df.columns:
        print(f"  Skip game{game_num}: no 'occluded' column in {base_csvs[0].name}")
        return

    frame_col = "from_frame" if "from_frame" in base_df.columns else "frame"
    occlusion_map = dict(zip(base_df[frame_col], base_df["occluded"].fillna("")))

    tagged_count = sum(1 for v in occlusion_map.values() if v)
    print(f"  game{game_num}: {tagged_count} frames with occlusions -> propagating to {len(VARIANTS)} variants")

    for variant in VARIANTS:
        variant_dir = DATA_DIR / f"game{game_num}_per_frame_{variant}"
        if not variant_dir.exists():
            continue

        variant_csvs = list(variant_dir.glob("*.csv"))
        if not variant_csvs:
            continue

        vdf = pd.read_csv(variant_csvs[0])
        v_frame_col = "from_frame" if "from_frame" in vdf.columns else "frame"

        if "occluded" not in vdf.columns:
            vdf["occluded"] = ""

        updated = 0
        for i, row in vdf.iterrows():
            frame_num = row[v_frame_col]
            if frame_num in occlusion_map and occlusion_map[frame_num]:
                vdf.at[i, "occluded"] = occlusion_map[frame_num]
                updated += 1

        vdf.to_csv(variant_csvs[0], index=False)
        print(f"    {variant}: {updated} frames updated in {variant_csvs[0].name}")


def main():
    parser = argparse.ArgumentParser(description="Propagate occlusion tags to augmented game variants")
    parser.add_argument("--games", type=str, default=None,
                        help="Comma-separated game numbers (default: all found)")
    args = parser.parse_args()

    if args.games:
        game_nums = [int(g.strip()) for g in args.games.split(",")]
    else:
        game_nums = sorted(set(
            int("".join(filter(str.isdigit, d.name.split("_")[0])))
            for d in DATA_DIR.iterdir()
            if d.is_dir() and "game" in d.name and "per_frame" in d.name and "_" not in d.name.split("per_frame")[-1]
        ))

    print(f"Propagating occlusions for games: {game_nums}\n")
    for num in game_nums:
        propagate_game(num)

    print("\nDone.")


if __name__ == "__main__":
    main()
