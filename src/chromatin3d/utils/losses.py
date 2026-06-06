import logging
import torch
import torchsort
from torchmetrics.image import StructuralSimilarityIndexMeasure
logger = logging.getLogger("chromatin3d")


def data_loss_soft_rank(true_hic_matrices, pred_hic_matrices):
    """ batch version """

    # soft rank, so loss is differentiable
    r_x = torchsort.soft_rank(true_hic_matrices.flatten(start_dim=1), regularization_strength=0.1)
    r_y = torchsort.soft_rank(pred_hic_matrices.flatten(start_dim=1), regularization_strength=0.1)

    # calculating pearson correlation on ranks
    r_x -= r_x.mean(dim=1, keepdim=True)
    r_y -= r_y.mean(dim=1, keepdim=True)

    n = (r_x * r_y).sum(dim=1, keepdim=True)
    d = r_x.norm(dim=1, keepdim=True) * r_y.norm(dim=1, keepdim=True)

    return 1 - n / (d + 1e-8)


def data_loss_ssim(true_hic_matrices, pred_hic_matrices):
    """ batch version """

    ssim = StructuralSimilarityIndexMeasure(data_range=None, reduction=None)

    r = 1 - ssim(pred_hic_matrices.unsqueeze(1), true_hic_matrices.unsqueeze(1))

    logger.debug(f"SSIM: {r}")

    return r


# insulation score (take diagonal and average across perpendicular diagonals)
def data_loss_insulation_score(true_hic_matrices, pred_hic_matrices):
    """ batch version """

    return torch.Tensor([0])


# decay along from diagonal
def data_loss_decay(true_hic_matrices, pred_positions):
    """ batch version """

    return torch.Tensor([0])


# SVD ranking?
def data_loss_svd(true_hic_matrices, pred_positions):
    """ batch version """

    return torch.Tensor([0])


# implement (convolution)
def loops_loss(true_hic_matrices, pred_positions):
    """ batch version """

    return torch.Tensor([0])


# compartments loss
def compartments_loss(true_hic_matrices, pred_positions):
    """ batch version """

    return torch.Tensor([0])


def rouse_loss(positions):
    """ batch version """

    diffs = positions[:, 1:, :] - positions[:, :-1, :]

    l = diffs.norm(dim=2)

    return ((l - 1) ** 2).mean(dim=1)


def smoothness_loss(positions):
    """ batch version """

    u = positions[:, 1:-1:, :] - positions[:, :-2, :]
    v = positions[:, 1:-1, :] - positions[:, 2:, :]

    n = (u * v).sum(dim=2)
    d = u.norm(dim=2) * v.norm(dim=2)

    cos_theta = (n / d).clip(min=-1, max=1)

    theta = torch.arccos(cos_theta)

    return ((theta - torch.pi) ** 2).mean(dim=1)
