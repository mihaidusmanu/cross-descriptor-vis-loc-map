import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def exhaustive_loss(encoders, decoders, batch, device, alpha=0.1, margin=1.0):
    # Translation loss.
    embeddings = {}
    for source_feature in batch.keys():
        source_descriptors = batch[source_feature]
        embeddings[source_feature] = encoders[source_feature](source_descriptors)
    
    all_embeddings = torch.cat(list(embeddings.values()), dim=0)

    t_loss = torch.tensor(0.).float().to(device)
    for target_feature in batch.keys():
        target_descriptors = batch[target_feature]
        output_descriptors = decoders[target_feature](all_embeddings)
        if target_feature == 'brief':
            current_loss = F.binary_cross_entropy(
                output_descriptors,
                torch.cat([target_descriptors] * len(batch), dim=0)
            )
        else:
            current_loss = torch.mean(
                torch.norm(output_descriptors - torch.cat([target_descriptors] * len(batch), dim=0), dim=1)
            )
        t_loss += current_loss
    t_loss /= len(batch)

    # Triplet loss in embedding space.
    e_loss = torch.tensor(0.).float().to(device)
    if alpha > 0:
        for source_feature in embeddings.keys():
            for target_feature in embeddings.keys():
                # TODO: Implement symmetric negative mining.
                sqdist_matrix = 2 - 2 * embeddings[source_feature] @ embeddings[target_feature].T
                pos_dist = torch.norm(torch.diag(sqdist_matrix).unsqueeze(-1), dim=-1)
                sqdist_matrix = sqdist_matrix + torch.diag(torch.full((sqdist_matrix.shape[0],), np.inf)).to(device)
                # neg_sqdist = torch.min(torch.min(sqdist_matrix, dim=-1)[0], torch.min(sqdist_matrix, dim=0)[0])
                neg_sqdist = torch.min(sqdist_matrix, dim=-1)[0]
                neg_dist = torch.norm(neg_sqdist.unsqueeze(-1), dim=-1)
                e_loss = e_loss + torch.mean(
                    F.relu(margin + pos_dist - neg_dist)
                )
        e_loss /= (len(batch) ** 2)

    # Final loss.
    if alpha > 0:
        loss = t_loss + alpha * e_loss
    else:
        loss = t_loss
    
    return loss, (t_loss.detach(), e_loss.detach())


def autoencoder_loss(encoders, decoders, batch, device, alpha=0.1, margin=1.0):
    # AE loss.
    embeddings = {}
    t_loss = torch.tensor(0.).float().to(device)
    for source_feature in batch.keys():
        source_descriptors = batch[source_feature]
        current_embeddings = encoders[source_feature](source_descriptors)
        embeddings[source_feature] = current_embeddings
        output_descriptors = decoders[source_feature](current_embeddings)
        if source_feature == 'brief':
            current_loss = F.binary_cross_entropy(
                output_descriptors, source_descriptors
            )
        else:
            current_loss = torch.mean(
                torch.norm(output_descriptors - source_descriptors, dim=1)
            )
        t_loss += current_loss
    t_loss /= len(batch)

    # Triplet loss in embedding space.
    e_loss = torch.tensor(0.).float().to(device)
    if alpha > 0:
        for source_feature in embeddings.keys():
            for target_feature in embeddings.keys():
                # TODO: Implement symmetric negative mining.
                sqdist_matrix = 2 - 2 * embeddings[source_feature] @ embeddings[target_feature].T
                pos_dist = torch.norm(torch.diag(sqdist_matrix).unsqueeze(-1), dim=-1)
                sqdist_matrix = sqdist_matrix + torch.diag(torch.full((sqdist_matrix.shape[0],), np.inf)).to(device)
                # neg_sqdist = torch.min(torch.min(sqdist_matrix, dim=-1)[0], torch.min(sqdist_matrix, dim=0)[0])
                neg_sqdist = torch.min(sqdist_matrix, dim=-1)[0]
                neg_dist = torch.norm(neg_sqdist.unsqueeze(-1), dim=-1)
                e_loss = e_loss + torch.mean(
                    F.relu(margin + pos_dist - neg_dist)
                )
        e_loss /= (len(batch) ** 2)

    # Final loss.
    if alpha > 0:
        loss = t_loss + alpha * e_loss
    else:
        loss = t_loss
    
    return loss, (t_loss.detach(), e_loss.detach())


def subset_loss(encoders, decoders, batch, device, alpha=0.1, margin=1.0):
    # Select random pairs of descriptors.
    # Make sure that all encoders and all encoders are used.
    source_features = np.array(list(batch.keys()))
    target_features = np.array(source_features)
    np.random.shuffle(target_features)

    # Translation loss.
    embeddings = {}
    t_loss = torch.tensor(0.).float().to(device)
    for source_feature, target_feature in zip(source_features, target_features):
        source_descriptors = batch[source_feature]
        target_descriptors = batch[target_feature]
        current_embeddings = encoders[source_feature](source_descriptors)
        embeddings[source_feature] = current_embeddings
        output_descriptors = decoders[target_feature](current_embeddings)
        if target_feature == 'brief':
            current_loss = F.binary_cross_entropy(
                output_descriptors, target_descriptors
            )
        else:
            current_loss = torch.mean(
                torch.norm(output_descriptors - target_descriptors, dim=1)
            )
        t_loss += current_loss
    t_loss /= len(batch)

    # Triplet loss in embedding space.
    e_loss = torch.tensor(0.).float().to(device)
    if alpha > 0:
        for source_feature, target_feature in zip(source_features, target_features):
            # TODO: Implement symmetric negative mining.
            sqdist_matrix = 2 - 2 * embeddings[source_feature] @ embeddings[target_feature].T
            pos_dist = torch.norm(torch.diag(sqdist_matrix).unsqueeze(-1), dim=-1)
            sqdist_matrix = sqdist_matrix + torch.diag(torch.full((sqdist_matrix.shape[0],), np.inf)).to(device)
            # neg_sqdist = torch.min(torch.min(sqdist_matrix, dim=-1)[0], torch.min(sqdist_matrix, dim=0)[0])
            neg_sqdist = torch.min(sqdist_matrix, dim=-1)[0]
            neg_dist = torch.norm(neg_sqdist.unsqueeze(-1), dim=-1)
            e_loss = e_loss + torch.mean(
                F.relu(margin + pos_dist - neg_dist)
            )
        e_loss /= len(batch)

    # Final loss.
    if alpha > 0:
        loss = t_loss + alpha * e_loss
    else:
        loss = t_loss
    
    return loss, (t_loss.detach(), e_loss.detach())
