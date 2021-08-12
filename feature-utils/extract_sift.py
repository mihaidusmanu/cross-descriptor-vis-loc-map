import argparse

import os

import subprocess

import types


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--colmap_path', type=str, required=True,
        help='path to the COLMAP executable folder'
    )
    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='path to the dataset'
    )
    parser.add_argument(
        '--image_path', type=str, default=None,
        help='path to the images'
    )

    args = parser.parse_args()

    if args.image_path is None:
        args.image_path = args.dataset_path

    # Dataset paths.
    paths = types.SimpleNamespace()
    paths.database_path = os.path.join(args.dataset_path, 'sift-features.db')

    if os.path.exists(paths.database_path):
        raise FileExistsError('Database already exists at %s.' % paths.database_path)

    # Extract SIFT features.
    subprocess.call([
        os.path.join(args.colmap_path, 'colmap'), 'feature_extractor',
        '--database_path', paths.database_path,
        '--image_path', args.image_path,
        '--SiftExtraction.first_octave', str(0),
        '--SiftExtraction.num_threads', str(1)
    ])
