import numpy as np

import os

import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(
            self,
            base_path=None,
            features=['brief', 'sift-kornia', 'hardnet', 'sosnet'],
            subsampling_ratio=1.0
    ):
        self.features = features

        self.arrays = {}
        for feature in self.features:
            npy_path = os.path.join(
                base_path, '%s-features.npy' % feature
            )
            descriptors = np.load(npy_path)
            # Deterministic subsampling.
            if subsampling_ratio < 1.0:
                num_descriptors = int(np.ceil(subsampling_ratio * descriptors.shape[0]))
                random = np.random.RandomState(seed=1)
                selected_ids = random.choice(descriptors.shape[0], num_descriptors, replace=False)
            else:
                selected_ids = np.arange(descriptors.shape[0])
            self.arrays[feature] = descriptors[selected_ids, :]
        
        self.len = self.arrays[features[0]].shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = {}
        for feature in self.features:
            sample[feature] = torch.from_numpy(
                self.arrays[feature][idx, :]
            ).float()
        return sample
