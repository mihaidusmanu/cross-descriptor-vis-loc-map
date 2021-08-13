import argparse

import json

import os

import shutil

import types

import torch

from utils import build_hybrid_database, match_features_subset, geometric_verification, reconstruct, compute_extra_stats


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='path to the dataset'
    )

    parser.add_argument(
        '--colmap_path', type=str, required=True,
        help='path to the COLMAP executable folder'
    )

    parser.add_argument(
        '--features', nargs='+', type=str, required=True,
        help='list of descriptors to consider'
    )

    parser.add_argument(
        '--feature', type=str, required=True,
        help='descriptors to map'
    )

    parser.add_argument(
        '--exp_name', type=str, required=True,
        help='name of the experiment'
    )

    args = parser.parse_args()
    return args


def main():
    # Set CUDA.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.set_grad_enabled(False)

    # Parse arguments.
    args = parse_args()

    paths = types.SimpleNamespace()
    paths.database_path = os.path.join(
        args.dataset_path, '%s.db' % args.exp_name
    )
    paths.image_path = os.path.join(
        args.dataset_path, 'images'
    )
    paths.match_list_path = os.path.join(
        args.dataset_path, 'match-list-exh.txt'
    )
    paths.sparse_path = os.path.join(
        args.dataset_path, 'sparse-%s' % args.exp_name
    )
    paths.output_path = os.path.join(
        args.dataset_path, 'stats-%s.txt' % args.exp_name
    )

    # Copy reference database.
    if os.path.exists(paths.database_path):
        raise FileExistsError('Database file already exists.')
    shutil.copy(
        os.path.join(args.dataset_path, 'database.db'),
        paths.database_path
    )

    # Build and translate database.
    image_features = build_hybrid_database(
        args.features,
        args.dataset_path,
        paths.database_path
    )

    # Matching + GV + reconstruction.
    match_features_subset(
        args.feature,
        image_features,
        args.colmap_path,
        paths.database_path, paths.image_path, paths.match_list_path
    )
    torch.cuda.empty_cache()
    matching_stats = geometric_verification(
        args.colmap_path,
        paths.database_path, paths.match_list_path + '.aux'
    )
    os.remove(paths.match_list_path + '.aux')
    largest_model_path, reconstruction_stats = reconstruct(
        args.colmap_path,
        paths.database_path, paths.image_path, paths.sparse_path
    )
    extra_stats = compute_extra_stats(image_features, largest_model_path)

    with open(paths.output_path, 'w') as f:
        f.write(json.dumps(matching_stats))
        f.write('\n')
        f.write(json.dumps(reconstruction_stats))
        f.write('\n')
        f.write(json.dumps(extra_stats))
        f.write('\n')


if __name__ == '__main__':
    main()
