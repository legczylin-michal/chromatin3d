import logging

import torch

logger = logging.getLogger("chromatin3d")


def pos_2_matrix_spacial(positions):
    """ batch version """

    a = positions.unsqueeze(1)
    b = positions.unsqueeze(2)

    return torch.sqrt(torch.sum((a - b) ** 2, dim=3) + 1e-8)


def pos_2_hic(positions):
    """ batch version """

    distance_matrices = pos_2_matrix_spacial(positions)

    return ((distance_matrices + 1e-8) ** -1.5).clamp(max=1)


def normalise(hic):
    hic = torch.log1p(hic)
    hic = hic / hic.amax(dim=(1, 2), keepdim=True)

    return hic
