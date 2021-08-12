# Adapted from https://github.com/vcg-uvic/image-matching-benchmark-baselines/blob/master/extract_descriptors_hardnet.py.

import argparse

import numpy as np

import os

import cv2

import kornia

import shutil

import sqlite3

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms

import tqdm

from extract_patches.core import extract_patches


def get_transforms():
    transform = transforms.Compose([
        transforms.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)),
        transforms.Lambda(lambda x: np.reshape(x, (32, 32, 1))),
        transforms.ToTensor(),
    ])
    return transform


class BRIEFDescriptor(nn.Module):
    # Adapted from https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.BRIEF.
    def __init__(self, desc_size=512, patch_size=32, seed=1):
        super(BRIEFDescriptor, self).__init__()

        # Sampling pattern.
        random = np.random.RandomState()
        random.seed(seed)
        samples = (patch_size / 5.0) * random.randn(desc_size * 8)
        samples = np.array(samples, dtype=np.int32)
        samples = samples[
            (samples <= (patch_size // 2)) & (samples >= - (patch_size - 2) // 2)
        ]
        samples += (patch_size // 2 - 1)
        pos1 = samples[: desc_size * 2].reshape(desc_size, 2)
        pos2 = samples[desc_size * 2 : desc_size * 4].reshape(desc_size, 2)

        # Create tensors.
        self.pos1 = torch.from_numpy(pos1).long()
        self.pos2 = torch.from_numpy(pos2).long()

    def forward(self, patches):
        pixel_values1 = patches[:, 0, self.pos1[:, 0], self.pos1[:, 1]]
        pixel_values2 = patches[:, 0, self.pos2[:, 0], self.pos2[:, 1]]
        descriptors = (pixel_values1 < pixel_values2)
        return descriptors


def recover_database_images_and_ids(database_path):
    # Connect to the database.
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    # Recover database images and ids.
    images = {}
    cursor.execute('SELECT name, image_id FROM images;')
    for row in cursor:
        images[row[1]] = row[0]

    # Close the connection to the database.
    cursor.close()
    connection.close()

    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='path to the dataset'
    )
    parser.add_argument(
        '--image_path', type=str, default=None,
        help='path to the images'
    )
    parser.add_argument(
        '--feature', type=str, required=True,
        choices=['brief', 'sift-kornia', 'hardnet', 'sosnet'],
        help='descriptors to be extracted'
    )
    parser.add_argument(
        '--mr_size', type=float, default=12.0,
        help='patch size in image is mr_size * pt.size'
    )
    parser.add_argument(
        '--batch_size', type=int, default=512,
        help='path to the model weights'
    )

    args = parser.parse_args()

    if args.image_path is None:
        args.image_path = args.dataset_path

    # Dataset paths.
    paths = types.SimpleNamespace()
    paths.sift_database_path = os.path.join(args.dataset_path, 'sift-features.db')
    paths.database_path = os.path.join(args.dataset_path, '%s-features.db' % args.feature)

    # Copy SIFT database.
    if os.path.exists(paths.database_path):
        raise FileExistsError('Database already exists at %s.' % paths.database_path)
    shutil.copy(paths.sift_database_path, paths.database_path)

    # PyTorch settings.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.set_grad_enabled(False)

    # Network and input transforms.
    dim = 128
    dtype = np.float32
    if args.feature == 'brief':
        model = BRIEFDescriptor()
        model = model.to(device)
        dim = 512
        dtype = bool
    elif args.feature == 'sift-kornia':
        model = kornia.feature.SIFTDescriptor(patch_size=32, rootsift=False)
        model = model.to(device)
    elif args.feature == 'hardnet':
        model = kornia.feature.HardNet(pretrained=True)
        model = model.to(device)
        model.eval()
    elif args.feature == 'sosnet':
        model = kornia.feature.SOSNet(pretrained=True)
        model = model.to(device)
        model.eval()
    transform = get_transforms()

    # Recover list of images.
    images = recover_database_images_and_ids(paths.database_path)

    # Connect to database.
    connection = sqlite3.connect(paths.database_path)
    cursor = connection.cursor()
    
    cursor.execute('DELETE FROM descriptors;')
    connection.commit()

    cursor.execute('SELECT image_id, rows, cols, data FROM keypoints;')
    raw_keypoints = cursor.fetchall()
    for row in tqdm.tqdm(raw_keypoints):
        assert(row[2] == 6)
        image_id = row[0]
        image_relative_path = images[image_id]
        if row[1] == 0:
            keypoints = np.zeros([0, 6])
        else:
            keypoints = np.frombuffer(row[-1], dtype=np.float32).reshape(row[1], row[2])

        keypoints = np.copy(keypoints)
        # In COLMAP, the upper left pixel has the coordinate (0.5, 0.5).
        keypoints[:, 0] = keypoints[:, 0] - .5
        keypoints[:, 1] = keypoints[:, 1] - .5

        # Extract patches.
        image = cv2.cvtColor(
            cv2.imread(os.path.join(args.image_path, image_relative_path)),
            cv2.COLOR_BGR2RGB
        )

        patches = extract_patches(
            keypoints, image, 32, args.mr_size, 'xyA'
        )

        # Extract descriptors.
        descriptors = np.zeros((len(patches), dim), dtype=dtype)
        for i in range(0, len(patches), args.batch_size):
            data_a = patches[i : i + args.batch_size]
            data_a = torch.stack(
                [transform(patch) for patch in data_a]
            ).to(device)
            # Predict
            out_a = model(data_a)
            descriptors[i : i + args.batch_size] = out_a.cpu().detach().numpy()

        # Insert into database.
        cursor.execute(
            'INSERT INTO descriptors(image_id, rows, cols, data) VALUES(?, ?, ?, ?);',
            (image_id, descriptors.shape[0], descriptors.shape[1], descriptors.tobytes())
        )
    connection.commit()

    # Close connection to database.
    cursor.close()
    connection.close()
