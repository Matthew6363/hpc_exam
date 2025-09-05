#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import os
import re
from glob import glob
from collections import defaultdict


def load_dump(filename, x_size, y_size):
    """Load binary dump file into a 2D numpy array."""
    data = np.fromfile(filename, dtype=np.float32)
    expected = x_size * y_size
    if data.size != expected:
        raise ValueError(f"{filename}: Expected {expected} values, got {data.size}")
    return data.reshape((y_size, x_size))


def get_sorted_bin_files(folder, limit=None):
    """(legacy) Retrieve plane_*.bin files sorted by step number (from filename)."""
    files = glob(os.path.join(folder, 'plane_*.bin'))
    step_re = re.compile(r'plane_(\d+)\.bin')
    files = sorted(files, key=lambda f: int(step_re.search(f).group(1)))
    return files[:limit] if limit else files


def find_rank_iteration_files(folder):
    """
    Find files matching pattern '<rank>_plane_<iteration>.bin'.
    Returns a dict: iteration(int) -> dict(rank(int) -> filepath).
    """
    files = glob(os.path.join(folder, '*_plane_*.bin'))
    rx = re.compile(r'(?P<rank>\d+)_plane_(?P<iter>\d+)\.bin$')
    it_map = defaultdict(dict)
    for f in files:
        m = rx.search(os.path.basename(f))
        if not m:
            continue
        rank = int(m.group('rank'))
        it = int(m.group('iter'))
        it_map[it][rank] = f
    return dict(it_map)


def assemble_full_matrix(rank_files_map, x_size, y_size, SX, SY):
    """
    Given a mapping rank -> filepath for a single iteration, assemble the full matrix.
    - ranks are arranged row-major across SX columns and SY rows (SX * SY ranks).
    - requires x_size % SX == 0 and y_size % SY == 0 (equal block sizes).
    """
    expected_ranks = SX * SY
    ranks_present = sorted(rank_files_map.keys())
    if len(ranks_present) != expected_ranks:
        raise ValueError(f"Expected {expected_ranks} ranks for this iteration, found {len(ranks_present)}: {ranks_present}")

    if x_size % SX != 0 or y_size % SY != 0:
        raise ValueError(f"x_size ({x_size}) must be divisible by SX ({SX}) and y_size ({y_size}) by SY ({SY}) "
                         "for equal block sizes. If your blocks differ, modify the code to provide block sizes per rank.")

    bx = x_size // SX
    by = y_size // SY

    full = np.empty((y_size, x_size), dtype=np.float32)

    for rank, filepath in rank_files_map.items():
        block = load_dump(filepath, bx, by)  # each rank file is a bx-by block but reshape uses (by,bx)
        row = rank // SX   # row index in [0..SY-1]
        col = rank % SX    # col index in [0..SX-1]
        y0 = row * by
        y1 = y0 + by
        x0 = col * bx
        x1 = x0 + bx
        full[y0:y1, x0:x1] = block

    return full


def animate_frames(frames, interval=100, save=None):
    """Create and show or save a matplotlib animation."""
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], cmap='hot', interpolation='nearest', origin='lower')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Heat")

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)

    if save:
        print(f"Saving animation to {save}...")
        if save.endswith('.gif'):
            ani.save(save, writer='Pillow', fps=10)
        else:
            ani.save(save, writer='ffmpeg', fps=10)
    else:
        plt.show()


# —————————————————————————————————————————————————————————————
# MAIN
# —————————————————————————————————————————————————————————————
def main():
    parser = argparse.ArgumentParser(description="Animate heatmap .bin files (single or parallel rank outputs).")
    parser.add_argument("folder", help="Folder containing bin files")
    parser.add_argument("-x", "--xsize", type=int, required=True, help="Global grid size in x-direction")
    parser.add_argument("-y", "--ysize", type=int, required=True, help="Global grid size in y-direction")
    parser.add_argument("--sx", type=int, help="Number of rank partitions in x (columns) for parallel output")
    parser.add_argument("--sy", type=int, help="Number of rank partitions in y (rows) for parallel output")
    parser.add_argument("-n", "--num", type=int, help="Number of frames to use (optional)")
    parser.add_argument("--save", help="Save animation as .mp4 or .gif instead of displaying")
    parser.add_argument("--interval", type=int, default=100, help="Frame interval in ms (default 100)")
    args = parser.parse_args()

    # Try detection: if files match '<rank>_plane_<iter>.bin' -> parallel mode
    it_map = find_rank_iteration_files(args.folder)
    if it_map:
        if args.sx is None or args.sy is None:
            raise SystemExit("Parallel output detected: you must supply --sx and --sy (number of partitions in x and y).")

        iterations = sorted(it_map.keys())
        if args.num:
            iterations = iterations[:args.num]

        print(f"Found parallel files for {len(iterations)} iterations in '{args.folder}'. Using SX={args.sx}, SY={args.sy}.")
        frames = []
        for it in iterations:
            rank_map = it_map[it]
            try:
                full = assemble_full_matrix(rank_map, args.xsize, args.ysize, args.sx, args.sy)
            except Exception as e:
                raise SystemExit(f"Error assembling iteration {it}: {e}")
            frames.append(full)
    else:
        # Legacy single-file-per-iteration mode: plane_XXXXX.bin
        bin_files = get_sorted_bin_files(args.folder, args.num)
        if not bin_files:
            raise SystemExit("No .bin files found in the folder.")
        print(f"Loading {len(bin_files)} frames from '{args.folder}' (single-file-per-iteration mode)...")
        frames = [load_dump(f, args.xsize, args.ysize) for f in bin_files]

    animate_frames(frames, interval=args.interval, save=args.save)


if __name__ == "__main__":
    main()
# python plot_parallel.py data_parallel -x 256 -y 256 --sx 2 --sy 2 -n 100 --save parallel.mp4
