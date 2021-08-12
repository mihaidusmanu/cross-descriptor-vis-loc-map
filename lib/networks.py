import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
            self,
            num_channels,
            use_cuda=True,
            use_bn=False,
            last_activation=None,
            l2_norm=False
    ):
        super(MLP, self).__init__()

        # Retrieve from num_channels.
        source_num_channels = num_channels[0]
        target_num_channels = num_channels[-1]
        hidden_num_channels = num_channels[1 : -1]

        # Prepare layers.
        layers = []
        previous_num_channels = source_num_channels
        for current_num_channels in hidden_num_channels:
            layers.append(
                nn.Linear(previous_num_channels, current_num_channels)
            )
            if use_bn:
                layers.append(
                    nn.BatchNorm1d(current_num_channels)
                )
            layers.append(
                nn.ReLU(inplace=True)
            )
            previous_num_channels = current_num_channels
        layers.append(nn.Linear(previous_num_channels, target_num_channels))

        # Make a sequential model.
        self.network = nn.Sequential(*layers)

        # Last activation.
        if last_activation is None:
            self.last_activation = None
        elif last_activation.lower() == 'relu':
            self.last_activation = nn.ReLU()
        elif last_activation.lower() == 'sigmoid':
            self.last_activation = nn.Sigmoid()
        else:
            raise NotImplementedError('Unknown activation "%s".' % last_activation)

        # L2-normalize output.
        self.l2_norm = l2_norm

        # Move to GPU if needed.
        if use_cuda:
            self.cuda()

    def forward(self, batch):
        output = self.network(batch)
        if self.last_activation is not None:
            output = self.last_activation(output)
        if self.l2_norm:
            output = F.normalize(output, dim=1)
        return output
