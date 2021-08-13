import argparse

import json

import numpy as np

import os

import shutil

import sqlite3

import sys

import types

import torch

from utils import translate_descriptors, build_hybrid_database, match_features, geometric_verification, reconstruct, blob_to_array, compute_extra_stats

sys.path.append(os.getcwd())
from lib.utils import create_network_for_feature


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
        '--exp_name', type=str, required=True,
        help='name of the experiment'
    )

    parser.add_argument(
        '--checkpoint', type=str, default='checkpoints-pretrained/model.pth',
        help='path to the checkpoint'
    )

    parser.add_argument(
        '--batch_size', type=int, default=4096,
        help='batch size'
    )

    args = parser.parse_args()
    return args


def translate_database(image_features, database_path, encoders, batch_size, device):
    shutil.copy(database_path, database_path + '.aux')
    
    connection = sqlite3.connect(database_path + '.aux')
    cursor = connection.cursor()

    output_connection = sqlite3.connect(database_path)
    output_cursor = output_connection.cursor()
    output_cursor.execute(
        'DELETE FROM descriptors'
    )
    output_connection.commit()

    for image_id, feature in image_features.items():
        image_id = int(image_id)
        result = (cursor.execute("SELECT data, rows, cols FROM descriptors WHERE image_id=?", (image_id,))).fetchall()
        data, rows, cols = result[0]
        try:
            descriptors = blob_to_array(data, np.float32, (rows, cols))
        except ValueError:
            descriptors = blob_to_array(data, bool, (rows, cols)).astype(np.float32)
        
        descriptors = translate_descriptors(descriptors, feature, None, encoders, {}, batch_size, device)

        output_cursor.execute(
            'INSERT INTO descriptors VALUES (?, ?, ?, ?)',
            (image_id,) + descriptors.shape + (descriptors.tobytes(),)
        )
    output_connection.commit()

    cursor.close()
    connection.close()
    os.remove(database_path + '.aux')

    output_cursor.close()
    output_connection.close()


def main():
    # Set CUDA.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.set_grad_enabled(False)

    #  Load config json.
    with open('checkpoints-pretrained/config.json', 'r') as f:
        config = json.load(f)

    # Parse arguments.
    args = parse_args()
    assert(len(args.features) > 1)

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

    # Create networks.
    checkpoint = torch.load(args.checkpoint)
    encoders = {}
    for feature in args.features:
        encoder, _ = create_network_for_feature(feature, config, use_cuda)
        state_dict = list(filter(lambda x: x[0] == feature, checkpoint['encoders']))[0]
        encoder.load_state_dict(state_dict[1])
        encoder.eval()
        encoders[feature] = encoder

    # Build and translate database.
    image_features = build_hybrid_database(
        args.features,
        args.dataset_path,
        paths.database_path
    )
    np.save(os.path.join(args.dataset_path, 'features.npy'), image_features)
    translate_database(
        image_features,
        paths.database_path,
        encoders, args.batch_size, device
    )

    # Matching + GV + reconstruction.
    match_features(
        args.colmap_path,
        paths.database_path, paths.image_path, paths.match_list_path
    )
    torch.cuda.empty_cache()
    matching_stats = geometric_verification(
        args.colmap_path,
        paths.database_path, paths.match_list_path
    )
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
