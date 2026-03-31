#!/usr/bin/env python3
"""
Extract FEN strings from PGN files and map them to frame numbers.

For each game, we:
1. Parse the PGN file using chess.pgn
2. Replay the game move by move, collecting board positions
3. Extract board placement only (no castling/en-passant metadata)
4. Distribute positions evenly across available frames
5. Output as CSV with columns: from_frame,to_frame,fen,occluded
"""

import csv
import os
import sys
from pathlib import Path
import chess
import chess.pgn


def extract_board_only_fen(board):
    """Extract board placement only from FEN (without castling/en-passant/halfmove/fullmove)."""
    return board.board_fen()


def get_existing_frames(images_dir):
    """Get sorted list of existing frame numbers in the images directory."""
    frames = []
    if os.path.exists(images_dir):
        for filename in os.listdir(images_dir):
            if filename.startswith('frame_') and filename.endswith('.jpg'):
                try:
                    frame_num = int(filename.split('_')[1].split('.')[0])
                    frames.append(frame_num)
                except (ValueError, IndexError):
                    pass
    return sorted(frames)


def find_closest_frame(target_frame, available_frames):
    """Find the closest available frame number to target frame."""
    if not available_frames:
        return None
    if target_frame in available_frames:
        return target_frame
    # Find closest frame
    closest = min(available_frames, key=lambda x: abs(x - target_frame))
    return closest


def extract_fen_from_pgn(pgn_path, images_dir, output_csv_path):
    """
    Extract FEN positions from a PGN file and map to frame numbers.

    Args:
        pgn_path: Path to the .pgn file
        images_dir: Path to the images directory
        output_csv_path: Path where to write the CSV output
    """
    # Get available frames
    available_frames = get_existing_frames(images_dir)
    total_frames = len(available_frames)

    if total_frames == 0:
        print(f"ERROR: No frames found in {images_dir}")
        return False

    print(f"Processing {pgn_path}")
    print(f"  Available frames: {total_frames}")

    # Parse PGN file
    with open(pgn_path, 'r') as f:
        game = chess.pgn.read_game(f)

    if game is None:
        print(f"ERROR: Could not parse PGN file {pgn_path}")
        return False

    # Collect all positions
    positions = []  # List of (fen_string, move_number)
    board = game.board()
    positions.append((extract_board_only_fen(board), 0))  # Starting position

    move_count = 0
    for move in game.mainline_moves():
        board.push(move)
        move_count += 1
        positions.append((extract_board_only_fen(board), move_count))

    print(f"  Total moves: {move_count}")
    print(f"  Total positions (including starting): {len(positions)}")

    # Map positions to frames
    # Formula: position i gets frame = round((i+1) * F / (M+1))
    # where F = total frames, M = total moves
    M = move_count
    F = total_frames

    fen_to_frames = {}  # Map FEN to list of frame numbers

    for i, (fen, move_num) in enumerate(positions):
        # Calculate target frame
        target_frame_num = round((i + 1) * F / (M + 1))
        closest_frame = find_closest_frame(target_frame_num, available_frames)

        if closest_frame is not None:
            if fen not in fen_to_frames:
                fen_to_frames[fen] = []
            fen_to_frames[fen].append(closest_frame)

    # Write CSV: each unique FEN gets one or more rows with from_frame=to_frame=frame_number
    # Group consecutive frames with the same FEN
    csv_rows = []
    sorted_frames = sorted(set(f for frames in fen_to_frames.values() for f in frames))

    for frame_num in sorted_frames:
        # Find which FEN this frame belongs to
        assigned_fen = None
        for fen, frames in fen_to_frames.items():
            if frame_num in frames:
                assigned_fen = fen
                break

        if assigned_fen:
            csv_rows.append((frame_num, frame_num, assigned_fen, ''))

    # Write CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['from_frame', 'to_frame', 'fen', 'occluded'])
        writer.writerows(csv_rows)

    print(f"  Wrote {len(csv_rows)} rows to {output_csv_path}")

    return True


def main():
    base_downloads = Path('/sessions/blissful-wonderful-mccarthy/mnt/Downloads/c06')
    base_data = Path('/sessions/blissful-wonderful-mccarthy/mnt/DL_project/data')

    games = ['game8', 'game9', 'game10']

    for game_name in games:
        print(f"\n{'='*60}")
        print(f"Processing {game_name}")
        print('='*60)

        pgn_path = base_downloads / game_name / f'{game_name}.pgn'
        images_dir = base_downloads / game_name / 'images'
        output_csv = base_data / f'{game_name}_per_frame' / f'{game_name}.csv'

        if not pgn_path.exists():
            print(f"ERROR: PGN file not found: {pgn_path}")
            continue

        if not images_dir.exists():
            print(f"ERROR: Images directory not found: {images_dir}")
            continue

        success = extract_fen_from_pgn(str(pgn_path), str(images_dir), str(output_csv))

        if success:
            # Print first 5 and last 5 rows
            print(f"\nFirst 5 rows of {game_name}.csv:")
            with open(output_csv, 'r') as f:
                rows = f.readlines()
                for row in rows[:6]:  # Header + 5 data rows
                    print(f"  {row.rstrip()}")

            if len(rows) > 7:
                print(f"\nLast 5 rows of {game_name}.csv:")
                for row in rows[-5:]:
                    print(f"  {row.rstrip()}")

    print(f"\n{'='*60}")
    print("Done!")


if __name__ == '__main__':
    main()
