#!/usr/bin/env python3
"""
Interactive annotation tool for tagging occluded squares on chessboard frames.

Occlusions are stored in a separate 'occluded' column in the CSV (e.g. "f8,g8,h8"),
leaving the FEN string untouched.

Usage:
    python tag_occlusions.py data/game2_per_frame
    python tag_occlusions.py data/game2_per_frame --board_size 512

Controls:
    Left-click   : Toggle a square as occluded (red overlay)
    N / Right     : Next frame (auto-saves if changes)
    P / Left      : Previous frame (auto-saves if changes)
    S             : Save current frame's occlusions to CSV
    U             : Undo all occlusion marks on current frame
    Q / Escape    : Quit
"""

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

WINDOW = "Tag Occlusions"
BOARD_DISPLAY = 640


def rc_to_square_name(r: int, c: int) -> str:
    """Convert (row, col) to chess notation like 'a8', 'h1'."""
    return f"{chr(ord('a') + c)}{8 - r}"


def square_name_to_rc(name: str) -> tuple:
    """Convert chess notation like 'a8' to (row, col)."""
    c = ord(name[0]) - ord('a')
    r = 8 - int(name[1])
    return r, c


def load_game(game_dir: str):
    """Load CSV and discover frame images for a game folder."""
    game_path = Path(game_dir)
    csv_files = list(game_path.glob("*.csv"))
    if not csv_files:
        sys.exit(f"No CSV found in {game_path}")
    csv_path = csv_files[0]
    df = pd.read_csv(csv_path)

    # Ensure the 'occluded' column exists
    if "occluded" not in df.columns:
        df["occluded"] = ""

    images_dir = game_path / "tagged_images"
    if not images_dir.exists():
        images_dir = game_path / "images"
    if not images_dir.exists():
        sys.exit(f"No images directory in {game_path}")

    frame_col = "from_frame" if "from_frame" in df.columns else "frame"
    frames = []
    for i, row in df.iterrows():
        frame_num = int(row[frame_col])
        img_path = images_dir / f"frame_{frame_num:06d}.jpg"
        if not img_path.exists():
            img_path = images_dir / f"frame_{frame_num:06d}.png"
        if not img_path.exists():
            continue
        frames.append({
            "frame_num": frame_num,
            "fen": row["fen"],
            "img_path": str(img_path),
            "df_index": i,
        })

    return csv_path, df, frame_col, frames


def parse_occluded_column(value) -> set:
    """Parse the 'occluded' column value into a set of (row, col) tuples."""
    if pd.isna(value) or str(value).strip() == "":
        return set()
    names = str(value).strip().split(",")
    result = set()
    for name in names:
        name = name.strip()
        if len(name) == 2 and name[0].isalpha() and name[1].isdigit():
            result.add(square_name_to_rc(name))
    return result


def occluded_set_to_str(occluded: set) -> str:
    """Convert a set of (row, col) to a sorted comma-separated string of square names."""
    if not occluded:
        return ""
    names = sorted([rc_to_square_name(r, c) for r, c in occluded])
    return ",".join(names)


class OcclusionTagger:
    def __init__(self, game_dir: str, board_size: int = 512):
        self.board_size = board_size
        self.csv_path, self.df, self.frame_col, self.frames = load_game(game_dir)

        if not self.frames:
            sys.exit("No frames with matching images found.")

        self.idx = 0
        self.occluded: set = set()
        self.board_img = None
        self.dirty = False
        self._display_w = BOARD_DISPLAY
        self._display_h = BOARD_DISPLAY

        print(f"Loaded {len(self.frames)} frames from {game_dir}")
        print(f"CSV: {self.csv_path}")
        print(f"\nControls: click=toggle | N/Right=next | P/Left=prev | S=save | U=undo | Q=quit\n")

    def _load_frame(self):
        """Load the current frame as-is and read existing occlusion tags."""
        frame = self.frames[self.idx]
        self.board_img = cv2.imread(frame["img_path"])
        occluded_val = self.df.at[frame["df_index"], "occluded"]
        self.occluded = parse_occluded_column(occluded_val)
        self.dirty = False

    def _render(self) -> np.ndarray:
        """Render the original frame with grid overlay and occlusion markers."""
        h, w = self.board_img.shape[:2]
        scale = BOARD_DISPLAY / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        display = cv2.resize(self.board_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        self._display_h, self._display_w = new_h, new_w

        sq_w = new_w // 8
        sq_h = new_h // 8

        # Draw grid
        for i in range(1, 8):
            cv2.line(display, (i * sq_w, 0), (i * sq_w, new_h), (0, 255, 0), 1)
            cv2.line(display, (0, i * sq_h), (new_w, i * sq_h), (0, 255, 0), 1)

        # Draw occlusion overlays
        overlay = display.copy()
        for r, c in self.occluded:
            x1, y1 = c * sq_w, r * sq_h
            x2, y2 = x1 + sq_w, y1 + sq_h
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.putText(overlay, "X", (x1 + sq_w // 3, y1 + 2 * sq_h // 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)

        # Draw square labels
        for r in range(8):
            for c in range(8):
                label = rc_to_square_name(r, c)
                x = c * sq_w + 2
                y = r * sq_h + 14
                cv2.putText(display, label, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # HUD
        frame = self.frames[self.idx]
        status = "*UNSAVED*" if self.dirty else "saved"
        occ_str = occluded_set_to_str(self.occluded) if self.occluded else "none"
        hud = f"Frame {self.idx + 1}/{len(self.frames)}  |  #{frame['frame_num']}  |  Occluded: {occ_str}  |  {status}"
        cv2.putText(display, hud, (10, new_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        return display

    def _pixel_to_square(self, x: int, y: int):
        """Convert pixel coords to (row, col) board square."""
        sq_w = self._display_w // 8
        sq_h = self._display_h // 8
        col = min(x // sq_w, 7)
        row = min(y // sq_h, 7)
        return row, col

    def _save(self):
        """Save occlusion tags to the 'occluded' column in the CSV."""
        frame = self.frames[self.idx]
        occ_str = occluded_set_to_str(self.occluded)

        self.df.at[frame["df_index"], "occluded"] = occ_str
        self.df.to_csv(self.csv_path, index=False)
        self.dirty = False

        display_str = occ_str if occ_str else "none"
        print(f"  Saved frame #{frame['frame_num']}: occluded=[{display_str}] -> {self.csv_path.name}")

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            r, c = self._pixel_to_square(x, y)
            if (r, c) in self.occluded:
                self.occluded.discard((r, c))
            else:
                self.occluded.add((r, c))
            self.dirty = True

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self._on_mouse)

        self._load_frame()

        while True:
            display = self._render()
            cv2.imshow(WINDOW, display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord("q") or key == 27:
                if self.dirty:
                    print("  Warning: unsaved changes. Press S to save or Q again to quit.")
                    cv2.imshow(WINDOW, display)
                    key2 = cv2.waitKey(0) & 0xFF
                    if key2 == ord("s"):
                        self._save()
                    elif key2 != ord("q") and key2 != 27:
                        continue
                break

            elif key == ord("n") or key == 83 or key == 3:
                if self.dirty:
                    self._save()
                self.idx = min(self.idx + 1, len(self.frames) - 1)
                self._load_frame()

            elif key == ord("p") or key == 81 or key == 2:
                if self.dirty:
                    self._save()
                self.idx = max(self.idx - 1, 0)
                self._load_frame()

            elif key == ord("s"):
                self._save()

            elif key == ord("u"):
                self.occluded.clear()
                self.dirty = True

        cv2.destroyAllWindows()
        print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Tag occluded squares on chessboard frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("game_dir", type=str,
                        help="Path to game folder (e.g. data/game2_per_frame)")
    parser.add_argument("--board_size", type=int, default=512,
                        help="Internal board size for perspective transform (default: 512)")
    args = parser.parse_args()

    tagger = OcclusionTagger(args.game_dir, board_size=args.board_size)
    tagger.run()


if __name__ == "__main__":
    main()
