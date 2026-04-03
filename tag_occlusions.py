#!/usr/bin/env python3
"""
Interactive annotation tool for tagging occluded squares on chessboard frames.

Occlusions are stored in a separate 'occluded' column in the CSV (e.g. "f8,g8,h8"),
leaving the FEN string untouched.

Usage:
    python tag_occlusions.py data/game8_per_frame

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
    return f"{chr(ord('a') + c)}{8 - r}"


def square_name_to_rc(name: str) -> tuple:
    c = ord(name[0]) - ord('a')
    r = 8 - int(name[1])
    return r, c


def load_game(game_dir: str):
    game_path = Path(game_dir)
    csv_files = list(game_path.glob("*.csv"))
    if not csv_files:
        sys.exit(f"No CSV found in {game_path}")
    csv_path = csv_files[0]
    df = pd.read_csv(csv_path)

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
    if not occluded:
        return ""
    names = sorted([rc_to_square_name(r, c) for r, c in occluded])
    return ",".join(names)


class OcclusionTagger:
    def __init__(self, game_dir: str):
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
        frame = self.frames[self.idx]
        self.board_img = cv2.imread(frame["img_path"])
        occluded_val = self.df.at[frame["df_index"], "occluded"]
        self.occluded = parse_occluded_column(occluded_val)
        self.dirty = False

    def _render(self) -> np.ndarray:
        h, w = self.board_img.shape[:2]
        scale = BOARD_DISPLAY / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        display = cv2.resize(self.board_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        self._display_h, self._display_w = new_h, new_w

        sq_w = new_w // 8
        sq_h = new_h // 8

        for i in range(1, 8):
            cv2.line(display, (i * sq_w, 0), (i * sq_w, new_h), (0, 255, 0), 1)
            cv2.line(display, (0, i * sq_h), (new_w, i * sq_h), (0, 255, 0), 1)

        overlay = display.copy()
        for r, c in self.occluded:
            x1, y1 = c * sq_w, r * sq_h
            x2, y2 = x1 + sq_w, y1 + sq_h
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.putText(overlay, "X", (x1 + sq_w // 3, y1 + 2 * sq_h // 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)

        for r in range(8):
            for c in range(8):
                label = rc_to_square_name(r, c)
                x = c * sq_w + 2
                y = r * sq_h + 14
                cv2.putText(display, label, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        frame = self.frames[self.idx]
        status = "*UNSAVED*" if self.dirty else "saved"
        occ_str = occluded_set_to_str(self.occluded) if self.occluded else "none"
        hud = f"Frame {self.idx + 1}/{len(self.frames)}  |  #{frame['frame_num']}  |  Occluded: {occ_str}  |  {status}"
        cv2.putText(display, hud, (10, new_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        return display

    def _pixel_to_square(self, x: int, y: int):
        sq_w = self._display_w // 8
        sq_h = self._display_h // 8
        col = min(x // sq_w, 7)
        row = min(y // sq_h, 7)
        return row, col

    def _save(self):
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
    parser = argparse.ArgumentParser(description="Tag occluded squares on chessboard frames")
    parser.add_argument("game_dir", type=str, help="Path to game folder (e.g. data/game8_per_frame)")
    args = parser.parse_args()
    tagger = OcclusionTagger(args.game_dir)
    tagger.run()


if __name__ == "__main__":
    main()
