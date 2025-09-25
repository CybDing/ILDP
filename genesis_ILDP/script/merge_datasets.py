#!/usr/bin/env python3
"""
Dataset Merging Script for Genesis Zarr Datasets
Merges two zarr datasets into a single larger dataset.
"""

import zarr
import numpy as np
import time
from pathlib import Path
import sys

def merge_zarr_datasets(dataset1_path, dataset2_path, output_path):
    """Merge two zarr datasets into a new combined dataset."""

    print(f"Loading dataset 1: {dataset1_path}")
    store1 = zarr.DirectoryStore(dataset1_path)
    root1 = zarr.group(store=store1)

    print(f"Loading dataset 2: {dataset2_path}")
    store2 = zarr.DirectoryStore(dataset2_path)
    root2 = zarr.group(store=store2)

    # Get data arrays
    data1, data2 = root1.data, root2.data
    meta1, meta2 = root1.meta, root2.meta

    # Verify same keys
    assert set(data1.keys()) == set(data2.keys()), "Datasets have different data keys"
    data_keys = list(data1.keys())

    print(f"Dataset 1: {data1[data_keys[0]].shape[0]} timesteps, {len(meta1['episode_ends'])} episodes")
    print(f"Dataset 2: {data2[data_keys[0]].shape[0]} timesteps, {len(meta2['episode_ends'])} episodes")

    # Create output directory and clean if exists
    output_path = Path(output_path)
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Creating merged dataset: {output_path}")
    output_store = zarr.DirectoryStore(str(output_path))

    # Create root group first
    output_root = zarr.group(store=output_store)

    # Create data and meta groups
    output_data = output_root.create_group('data')
    output_meta = output_root.create_group('meta')

    # Merge data arrays
    dataset1_size = data1[data_keys[0]].shape[0]

    for key in data_keys:
        arr1, arr2 = data1[key], data2[key]

        # Determine optimal chunking based on data type
        if key == 'img':
            chunks = (32, 96, 96, 3)
            compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE)
        elif key in ['cur_keypoints', 'target_keypoints']:
            chunks = (128, 8, 2)
            compressor = zarr.Blosc(cname='lz4', clevel=5, shuffle=zarr.Blosc.SHUFFLE)
        else:  # action, state
            chunks = (512, arr1.shape[1]) if len(arr1.shape) > 1 else (512,)
            compressor = zarr.Blosc(cname='lz4', clevel=5, shuffle=zarr.Blosc.SHUFFLE)

        print(f"Merging {key}: {arr1.shape} + {arr2.shape}")

        # Create merged array
        merged_shape = (arr1.shape[0] + arr2.shape[0],) + arr1.shape[1:]
        merged_array = output_data.create_dataset(
            key,
            shape=merged_shape,
            dtype=arr1.dtype,
            chunks=chunks,
            compressor=compressor
        )

        # Copy data
        merged_array[:arr1.shape[0]] = arr1[:]
        merged_array[arr1.shape[0]:] = arr2[:]

    # Merge episode boundaries
    episode_ends1 = meta1['episode_ends'][:]
    episode_ends2 = meta2['episode_ends'][:] + dataset1_size  # Offset by first dataset size

    merged_episode_ends = np.concatenate([episode_ends1, episode_ends2])
    output_meta.create_dataset('episode_ends', data=merged_episode_ends)

    total_timesteps = merged_shape[0]
    total_episodes = len(merged_episode_ends)

    print(f"✓ Merged dataset created:")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Saved to: {output_path}")

    return output_path

def main():
    """Main function with dataset paths."""

    # Dataset paths
    base_path = Path("../data/train_data/pusht")
    dataset1 = base_path / "genesis_data_20250921_004830.zarr"
    dataset2 = base_path / "genesis_data_20250920_003037.zarr"

    # Generate output path with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output = base_path / f"genesis_data_merged_{timestamp}.zarr"

    # Check datasets exist
    if not dataset1.exists():
        print(f"Error: Dataset 1 not found: {dataset1}")
        sys.exit(1)

    if not dataset2.exists():
        print(f"Error: Dataset 2 not found: {dataset2}")
        sys.exit(1)

    # Merge datasets
    try:
        merge_zarr_datasets(str(dataset1), str(dataset2), str(output))
        print("\n✓ Dataset merging completed successfully!")
    except Exception as e:
        print(f"✗ Error during merging: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()