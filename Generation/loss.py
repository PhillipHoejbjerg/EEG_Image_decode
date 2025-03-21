"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import torch
import torch.distributed.nn
from torch import distributed as dist, nn as nn
from torch.nn import functional as F

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
    image_features,
    text_features,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    use_horovod=False,
):
    if use_horovod:
        assert hvd is not None, "Please install horovod"
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(
                    all_image_features.chunk(world_size, dim=0)
                )
                gathered_text_features = list(
                    all_text_features.chunk(world_size, dim=0)
                )
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(
                torch.distributed.nn.all_gather(image_features), dim=0
            )
            all_text_features = torch.cat(
                torch.distributed.nn.all_gather(text_features), dim=0
            )
        else:
            gathered_image_features = [
                torch.zeros_like(image_features) for _ in range(world_size)
            ]
            gathered_text_features = [
                torch.zeros_like(text_features) for _ in range(world_size)
            ]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
        uniformity_loss_weight=0,  # Flag to add uniformity loss
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

        # Additional loss flags
        self.uniformity_loss_weight = uniformity_loss_weight

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = (
                    logit_scale * all_image_features @ all_text_features.T
                )
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2

        # Uniformity loss
        if self.uniformity_loss_weight:
            total_loss += self.uniformity_loss_weight * (1/2 * (uniformity_loss(image_features) + uniformity_loss(text_features)))

        # Cross-modal uniformity loss
        if self.uniformity_loss_weight:
            total_loss += self.uniformity_loss_weight * (cross_modal_uniformity_loss(image_features, text_features))

        # Alignment loss
        if self.uniformity_loss_weight:
            total_loss += self.uniformity_loss_weight * (alignment_loss(image_features, text_features))

        return total_loss
    
def normalize_embeddings(features):
    """L2 normalizes the embeddings along the last dimension."""
    return features / (features.norm(dim=-1, keepdim=True) + 1e-6)  # Prevent div by zero

def uniformity_loss(features):
    """
    Computes the uniformity loss for the given embeddings.
    """
    features = normalize_embeddings(features)  # Normalize embeddings
    N = features.shape[0]

    dist_matrix = torch.cdist(features, features, p=2) ** 2
    
    # Apply exponential function with -2 factor
    exp_matrix = torch.exp(-2 * dist_matrix)
    
    loss = torch.log(exp_matrix.sum(dim=1).mean() + 1e-6)  # Stability in log

    return loss

def cross_modal_uniformity_loss(image_features, text_features):
    """
    Computes the XUniform loss between image and text embeddings.
    """
    image_features = normalize_embeddings(image_features)  # Normalize
    text_features = normalize_embeddings(text_features)    # Normalize
    
    N = image_features.shape[0]

    # Compute pairwise squared Euclidean distances
    dist_matrix = torch.cdist(image_features, text_features, p=2) ** 2  # Shape: (N, N)

    # Apply exponential function with -2 factor
    exp_matrix = torch.exp(-2 * dist_matrix)

    # Remove diagonal elements without modifying exp_matrix in-place
    mask = torch.eye(N, device=image_features.device, dtype=torch.bool)
    exp_matrix = exp_matrix * (~mask)  # Element-wise multiplication instead of masked_fill_()

    # Compute mean and log
    loss = torch.log(exp_matrix.sum(dim=1).mean() + 1e-6)  # Stability in log
    return loss

def alignment_loss(image_features, text_features):
    """
    Computes the alignment loss between normalized image and text embeddings.
    """
    image_features = normalize_embeddings(image_features)  # Normalize
    text_features = normalize_embeddings(text_features)    # Normalize
    
    pairwise_diff = image_features - text_features
    pairwise_dist_sq = torch.sum(pairwise_diff ** 2, dim=-1)
    loss = torch.mean(pairwise_dist_sq)

    return loss



class VICReg_loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.embedding = 1024 
        self.projector = Projector(args, self.embedding)

    def forward(self, x, y):
        x = self.projector(x)
        y = self.projector(y)

        repr_loss = F.mse_loss(x, y)

        # For multi-processing 
        #x = torch.cat(FullGatherLayer.apply(x), dim=0)
        #y = torch.cat(FullGatherLayer.apply(y), dim=0)
        # without it x = x, and y=y


        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )

        return loss


def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]
    

import torch.nn.functional as F

class SoftContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        """
        Soft Contrastive Loss with dynamically computed soft negatives.
        
        :param temperature: Temperature scaling parameter for softmax.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, projections, targets):
        """
        Compute the Soft Contrastive Loss.

        :param projections: (batch_size, embedding_dim) - New modality embeddings.
        :param targets: (batch_size, embedding_dim) - Corresponding embeddings in the existing space.
        :return: Scalar loss value.
        """
        batch_size = projections.shape[0]

        # Normalize both embeddings
        projections = F.normalize(projections, dim=-1)
        targets = F.normalize(targets, dim=-1)

        # Compute cosine similarity matrix (between new modality and existing space)
        sim_matrix = torch.matmul(projections, targets.T)  # (batch_size, batch_size)

        # Compute cosine similarity within the existing space (to identify soft negatives)
        existing_similarity = torch.matmul(targets, targets.T)  # (batch_size, batch_size)

        # Compute soft negative weights: w_ij = 1 - sim(existing_i, existing_j)
        soft_negative_weights = 1 - existing_similarity  # (batch_size, batch_size)

        # Apply temperature scaling and softmax for contrastive loss
        exp_logits = torch.exp(sim_matrix / self.temperature)

        # Compute weighted sum of negatives
        weighted_negatives = (exp_logits * soft_negative_weights).sum(dim=1)

        # Compute final Soft Contrastive Loss
        loss = -torch.log(exp_logits.diag() / (weighted_negatives + exp_logits.diag()))

        return loss.mean()
    


class HybridSoftContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, lambda_cosine=0.5, lambda_mse=0.5):
        """
        Soft Contrastive Loss using both Cosine Similarity and MSE for hybrid alignment.

        :param temperature: Temperature scaling for softmax.
        :param lambda_cosine: Weight for cosine similarity component.
        :param lambda_mse: Weight for MSE-based similarity component.
        """
        super().__init__()
        self.temperature = temperature
        self.lambda_cosine = lambda_cosine
        self.lambda_mse = lambda_mse

    def forward(self, projections, targets):
        """
        Compute the Hybrid Soft Contrastive Loss.

        :param projections: Tensor of shape (batch_size, embedding_dim) - New modality embeddings.
        :param targets: Tensor of shape (batch_size, embedding_dim) - Corresponding embeddings in the existing space.
        :return: Scalar loss value.
        """
        batch_size, embedding_dim = projections.shape  # (batch_size, embedding_dim)

        # Normalize embeddings to ensure cosine similarity is valid
        projections_norm = F.normalize(projections, dim=-1)  # Shape: (batch_size, embedding_dim)
        targets_norm = F.normalize(targets, dim=-1)  # Shape: (batch_size, embedding_dim)

        # Compute Cosine Similarity matrix (between new modality and existing space)
        cosine_sim_matrix = torch.matmul(projections_norm, targets_norm.T)  
        # Shape: (batch_size, batch_size) - Cosine similarity between each projection and each target

        # Compute MSE Similarity: Convert MSE distance into a similarity score
        mse_distance_matrix = torch.cdist(projections, targets, p=2)  
        # Shape: (batch_size, batch_size) - Pairwise L2 distance between new modality and existing embeddings

        mse_sim_matrix = 1 - mse_distance_matrix / mse_distance_matrix.max()  
        # Shape: (batch_size, batch_size) - Normalize distances into [0,1] similarity scores

        # Compute Hybrid Similarity matrix (weighted combination of Cosine and MSE similarity)
        hybrid_sim_matrix = self.lambda_cosine * cosine_sim_matrix + self.lambda_mse * mse_sim_matrix  
        # Shape: (batch_size, batch_size) - Hybrid similarity score

        ### Compute soft negative weights based on existing space similarity ###

        # Don't weigh the score based on existing space if batch size is 1
        if hybrid_sim_matrix.numel() == 1:  # test case with batch size == 1
            soft_negative_weights = torch.tensor([[1.0]]).to(hybrid_sim_matrix.device)

        else:
            # Cosine similarity within the existing space
            existing_cosine_similarity = torch.matmul(targets_norm, targets_norm.T)  
            # Shape: (batch_size, batch_size) - Cosine similarity between existing embeddings

            # Compute pairwise MSE distance in the existing space
            existing_mse_distance = torch.cdist(targets, targets, p=2)  
            # Shape: (batch_size, batch_size) - L2 distance between existing embeddings

            # Convert MSE distances into similarity scores
            if existing_mse_distance.numel() == 1:  # If only one element
                existing_mse_sim = torch.tensor([[1.0]]).to(existing_mse_distance.device)
            else:
                existing_mse_sim = 1 - existing_mse_distance / existing_mse_distance.max()  
            # Shape: (batch_size, batch_size) - MSE-based similarity in existing space

            # Compute hybrid similarity within the existing space
            existing_hybrid_similarity = self.lambda_cosine * existing_cosine_similarity + self.lambda_mse * existing_mse_sim  
            # Shape: (batch_size, batch_size) - Hybrid similarity score in existing space

            # Compute soft negative weights: w_ij = 1 - HybridSim(existing_i, existing_j)
            soft_negative_weights = 1 - existing_hybrid_similarity  
            # Shape: (batch_size, batch_size) - Determines how much each negative sample contributes

        ### Compute the Contrastive Loss ###

        # Apply temperature scaling
        exp_logits = torch.exp(hybrid_sim_matrix / self.temperature)  
        # Shape: (batch_size, batch_size) - Exponentiated similarity scores

        # Compute weighted sum of negatives
        weighted_negatives = (exp_logits * soft_negative_weights).sum(dim=1)  
        # Shape: (batch_size,) - Summed negative scores for each sample

        # Compute final Soft Contrastive Loss
        loss = -torch.log(exp_logits.diag() / (weighted_negatives + exp_logits.diag()))  
        # Shape: (batch_size,) - Contrastive loss for each sample

        return loss.mean()  # Scalar loss value