import argparse

import numpy as np

import os

import sqlite3

from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='path to the dataset'
    )
    parser.add_argument(
        '--feature', type=str, required=True,
        help='descriptor'
    )

    args = parser.parse_args()

    database_path = os.path.join(
        args.dataset_path, '%s-features.db' % args.feature
    )
    output_path = os.path.join(
        args.dataset_path, '%s-features.npy' % args.feature
    )

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    all_descriptors = []
    for (image_id,) in tqdm(cursor.execute('SELECT image_id FROM images').fetchall()):
        r, c, blob = cursor.execute('SELECT rows, cols, data FROM descriptors WHERE image_id=?', (image_id,)).fetchall()[0]
        try:
            descriptors = np.frombuffer(blob, dtype=np.float32).reshape(r, c)
        except ValueError:
            descriptors = np.frombuffer(blob, dtype=bool).reshape(r, c)
        all_descriptors.append(descriptors)
    all_descriptors = np.concatenate(all_descriptors, axis=0)    

    # Random shuffle - not required.
    random = np.random.RandomState(seed=1)
    random.shuffle(all_descriptors)

    np.save(output_path, all_descriptors)

    cursor.close()
    connection.close()