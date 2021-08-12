import argparse

import json

import numpy as np

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from tqdm import tqdm

from lib.datasets import TranslationDataset as Dataset
from lib.losses import exhaustive_loss
from lib.utils import create_network_for_feature


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training script')

    parser.add_argument(
        '--random_seed', type=int, default=1,
        help='random seed for numpy and PyTorch'
    )

    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='path to the dataset'
    )

    parser.add_argument(
        '--features', nargs='+', type=str, required=True,
        help='list of descriptors to consider'
    )

    parser.add_argument(
        '--initial_checkpoint', type=str, default=None,
        help='path to the initial checkpoint'
    )

    parser.add_argument(
        '--num_epochs', type=int, default=5,
        help='number of training epochs'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='learning rate'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1024,
        help='batch size'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='number of workers for data loading'
    )

    parser.add_argument(
        '--log_interval', type=int, default=1000,
        help='loss logging interval'
    )

    parser.add_argument(
        '--checkpoint_directory', type=str, default='checkpoints',
        help='directory for training checkpoints'
    )
    parser.add_argument(
        '--checkpoint_prefix', type=str, default='multi',
        help='prefix for training checkpoints'
    )

    parser.add_argument(
        '--alpha', type=float, default=0.1,
        help='consistency loss weight'
    )
    parser.add_argument(
        '--margin', type=float, default=1.0,
        help='margin for the negative margin loss'
    )

    args = parser.parse_args()

    print(args)

    return args


# Updating mean class for loss aggregation.
class UpdatingMean():
    def __init__(self):
        self.sum = 0
        self.n = 0

    def mean(self):
        return self.sum / self.n

    def add(self, loss):
        self.sum += loss
        self.n += 1


# Epoch training / validation loop.
def run_epoch(
        encoders,
        decoders,
        loss_function,
        optimizer,
        dataloader,
        device,
        log_file, train=True
):
    epoch_loss = UpdatingMean()
    epoch_t_loss = UpdatingMean()
    epoch_e_loss = UpdatingMean()

    torch.set_grad_enabled(train)

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch_idx, batch in progress_bar:
        # Move batch to device.
        for key in batch.keys():
            batch[key] = batch[key].to(device)

        # Reset gradient if needed.
        if train:
            optimizer.zero_grad()

        # Compute loss.
        loss, (t_loss, e_loss) = loss_function(encoders, decoders, batch, device)

        # Add loss to history.
        epoch_loss.add(loss.data.cpu().numpy())
        epoch_t_loss.add(t_loss)
        epoch_e_loss.add(e_loss)

        # Update progress bar.
        progress_bar.set_postfix(
            loss=('%.4f' % epoch_loss.mean()),
            t_loss=('%.4f' % epoch_t_loss.mean()),
            e_loss=('%.4f' % epoch_e_loss.mean())
        )

        # Update logs.
        if batch_idx % args.log_interval == 0:
            log_file.write('[%s] epoch %02d - batch %04d / %04d - avg_loss: %f, avg_t_loss: %f, avg_e_loss: %f\n' % (
                'train' if train else 'valid',
                epoch_idx, batch_idx, len(dataloader),
                epoch_loss.mean(), epoch_t_loss.mean(), epoch_e_loss.mean()
            ))

        # Backprop.
        if train:
            loss.backward()
            optimizer.step()
    
    # Update logs.
    log_file.write('[%s] epoch %02d - avg_loss: %f, avg_t_loss: %f, avg_e_loss: %f\n' % (
        'train' if train else 'valid',
        epoch_idx,
        epoch_loss.mean(), epoch_t_loss.mean(), epoch_e_loss.mean()
    ))
    log_file.flush()

    return epoch_loss.mean()


if __name__ == '__main__':
    # Set CUDA.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    #  Load config json.
    with open('checkpoints-pretrained/config.json', 'r') as f:
        config = json.load(f)

    # Command line arguments.
    args = parse_arguments()

    # Fix random seed.
    torch.manual_seed(args.random_seed)
    if use_cuda:
        torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Networks.
    encoders = {}
    decoders = {}
    for feature in args.features:
        encoder, decoder = create_network_for_feature(feature, config, use_cuda)
        
        encoders[feature] = encoder
        decoders[feature] = decoder

    # Load initial checkpoint if needed.
    if args.initial_checkpoint is not None:
        checkpoint = torch.load(args.initial_checkpoint)
        for feature, state_dict in checkpoint['encoders']:
            encoders[feature].load_state_dict(state_dict)
        for feature, state_dict in checkpoint['decoders']:
            decoders[feature].load_state_dict(state_dict)

    # Dataset.
    training_dataset = Dataset(
        base_path=args.dataset_path,
        features=args.features
    )
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )

    # Optimizer and loss.
    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            [param for _, enc in encoders.items() for param in enc.parameters()] +
            [param for _, dec in decoders.items() for param in dec.parameters()]
        ),
        lr=args.lr
    )
    loss_function = lambda encoders, decoders, batch, device: exhaustive_loss(
        encoders, decoders, batch, device, 
        alpha=args.alpha, margin=args.margin
    )

    # Create the checkpoint directory.
    if os.path.isdir(args.checkpoint_directory):
        print('[Warning] Checkpoint directory already exists.')
    else:
        os.mkdir(args.checkpoint_directory)
    
    # Open the log file for writing
    if os.path.exists(os.path.join(args.checkpoint_directory, 'log.txt')):
        print('[Warning] Log file already exists.')
    log_file = open(os.path.join(args.checkpoint_directory, 'log.txt'), 'a+')

    # Training loop.
    train_loss_history = []
    for epoch_idx in range(1, args.num_epochs + 1):
        # Run training epoch.
        train_loss_history.append(
            run_epoch(
                encoders, decoders,
                loss_function,
                optimizer,
                training_dataloader,
                device,
                log_file
            )
        )

        # Save the current checkpoint
        checkpoint_path = os.path.join(
            args.checkpoint_directory,
            '%s.%02d.pth' % (args.checkpoint_prefix, epoch_idx)
        )
        checkpoint = {
            'args': args,
            'epoch_idx': epoch_idx,
            'encoders': [(feature, enc.state_dict()) for feature, enc in encoders.items()],
            'decoders': [(feature, dec.state_dict()) for feature, dec in decoders.items()],
            'optimizer': optimizer.state_dict(),
            'train_loss_history': train_loss_history
        }
        torch.save(checkpoint, checkpoint_path)

    # Close the log file
    log_file.close()
