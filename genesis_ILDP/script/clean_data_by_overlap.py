#!/usr/bin/env python3
"""
Data Cleaning Script for Genesis Zarr Datasets
Filters out episodes with final intersection ratio below a threshold.
"""

import zarr
import numpy as np
import time
from pathlib import Path
import sys


def clean_zarr_by_overlap(input_path, output_path, overlap_threshold=0.55):
    """
    Clean zarr dataset by removing episodes with low final overlap ratio.

    Args:
        input_path: Path to input zarr dataset
        output_path: Path to output cleaned zarr dataset
        overlap_threshold: Minimum final intersection ratio to keep (default: 0.5)

    Returns:
        Tuple of (kept_episodes, removed_episodes)
    """

    print(f"Loading dataset: {input_path}")
    input_store = zarr.DirectoryStore(input_path)
    input_root = zarr.group(store=input_store)

    # Load final_intersection_ratio metadata
    if 'final_intersection_ratio' not in input_root:
        raise ValueError(
            "Dataset does not contain 'final_intersection_ratio' metadata. "
            "This script requires data collected with collect_data_v1.py"
        )

    final_ratios = input_root['final_intersection_ratio'][:]
    episode_ends = input_root.meta['episode_ends'][:]

    print(f"Total episodes: {len(final_ratios)}")
    print(f"Overlap threshold: {overlap_threshold}")
    print(f"Final intersection ratios - min: {final_ratios.min():.3f}, "
          f"max: {final_ratios.max():.3f}, mean: {final_ratios.mean():.3f}")

    # Find episodes to keep (overlap >= threshold)
    keep_mask = final_ratios >= overlap_threshold
    episodes_to_keep = np.where(keep_mask)[0]
    episodes_to_remove = np.where(~keep_mask)[0]

    print(f"\nEpisodes to keep: {len(episodes_to_keep)}")
    print(f"Episodes to remove: {len(episodes_to_remove)}")

    if len(episodes_to_keep) == 0:
        raise ValueError("No episodes meet the overlap threshold! Try lowering the threshold.")

    # Show which episodes are being removed
    if len(episodes_to_remove) > 0:
        print(f"\nRemoving episodes with overlap ratios:")
        for ep_idx in episodes_to_remove:
            print(f"  Episode {ep_idx}: overlap={final_ratios[ep_idx]:.3f}")

    # Calculate episode boundaries
    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    # Collect timestep indices to keep
    timesteps_to_keep = []
    new_episode_ends = []
    new_final_ratios = []
    current_timestep = 0

    for ep_idx in episodes_to_keep:
        start_idx = episode_starts[ep_idx]
        end_idx = episode_ends[ep_idx]
        episode_length = end_idx - start_idx

        # Record timesteps for this episode
        timesteps_to_keep.extend(range(start_idx, end_idx))

        # Record new episode end
        current_timestep += episode_length
        new_episode_ends.append(current_timestep)

        # Record final ratio
        new_final_ratios.append(final_ratios[ep_idx])

    timesteps_to_keep = np.array(timesteps_to_keep, dtype=np.int64)
    new_episode_ends = np.array(new_episode_ends, dtype=np.int64)
    new_final_ratios = np.array(new_final_ratios, dtype=np.float32)

    print(f"\nOriginal timesteps: {episode_ends[-1]}")
    print(f"Cleaned timesteps: {len(timesteps_to_keep)}")

    # Create output directory
    output_path = Path(output_path)
    if output_path.exists():
        import shutil
        print(f"Removing existing output directory: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating cleaned dataset: {output_path}")
    output_store = zarr.DirectoryStore(str(output_path))
    output_root = zarr.group(store=output_store)

    # Create data and meta groups
    output_data = output_root.create_group('data')
    output_meta = output_root.create_group('meta')

    # Copy filtered data arrays
    input_data = input_root.data
    data_keys = list(input_data.keys())

    print("\nCopying data arrays:")
    for key in data_keys:
        input_arr = input_data[key]

        # Determine optimal chunking based on data type (same as collect_data_v1.py)
        if key == 'img':
            chunks = (32, 96, 96, 3)
            compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE)
        elif key in ['cur_keypoints', 'target_keypoints']:
            chunks = (128, 8, 2)
            compressor = zarr.Blosc(cname='lz4', clevel=5, shuffle=zarr.Blosc.SHUFFLE)
        else:  # action, state
            chunks = (512, input_arr.shape[1]) if len(input_arr.shape) > 1 else (512,)
            compressor = zarr.Blosc(cname='lz4', clevel=5, shuffle=zarr.Blosc.SHUFFLE)

        # Filter data to keep only selected timesteps
        filtered_data = input_arr[timesteps_to_keep]

        print(f"  {key}: {input_arr.shape} -> {filtered_data.shape}")

        # Create output array
        output_data.create_dataset(
            key,
            data=filtered_data,
            chunks=chunks,
            compressor=compressor,
            dtype=input_arr.dtype
        )

    # Save metadata
    output_meta.create_dataset('episode_ends', data=new_episode_ends)

    # Save filtered final_intersection_ratio
    output_root.create_dataset(
        'final_intersection_ratio',
        data=new_final_ratios,
        dtype=np.float32,
        compressor=zarr.Blosc(cname='lz4', clevel=5)
    )

    print(f"\n✓ Cleaned dataset created:")
    print(f"  Input episodes: {len(final_ratios)} -> Output episodes: {len(new_final_ratios)}")
    print(f"  Input timesteps: {episode_ends[-1]} -> Output timesteps: {len(timesteps_to_keep)}")
    print(f"  Removed {len(episodes_to_remove)} episodes ({len(episodes_to_remove)/len(final_ratios)*100:.1f}%)")
    print(f"  New overlap stats - min: {new_final_ratios.min():.3f}, "
          f"max: {new_final_ratios.max():.3f}, mean: {new_final_ratios.mean():.3f}")
    print(f"  Saved to: {output_path}")

    return len(episodes_to_keep), len(episodes_to_remove)


def main():
    """Main function to clean dataset."""

    # Example usage - modify these paths
    base_path = Path("../data/train_data/pusht")

    # Input dataset path - change this to your dataset
    input_dataset = base_path / "genesis_data_20251027_004448.zarr"

    # Check if dataset exists
    if not input_dataset.exists():
        print(f"Error: Dataset not found: {input_dataset}")
        print("\nUsage: Modify the input_dataset path in main() to point to your zarr file")
        sys.exit(1)

    # Generate output path with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dataset = base_path / f"genesis_data_cleaned_{timestamp}.zarr"

    # Overlap threshold (default 0.5 = 50%)
    overlap_threshold = 0.55

    print("=" * 60)
    print("Genesis Dataset Cleaner - Overlap-based Filtering")
    print("=" * 60)

    # Clean dataset
    try:
        kept, removed = clean_zarr_by_overlap(
            str(input_dataset),
            str(output_dataset),
            overlap_threshold=overlap_threshold
        )
        print("\n" + "=" * 60)
        print("✓ Dataset cleaning completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Error during cleaning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
