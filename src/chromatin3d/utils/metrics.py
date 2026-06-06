import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanrho


def linker_length_metric(positions):
    """ batch version """

    diffs = positions[:, 1:, :] - positions[:, :-1, :]

    l = diffs.norm(dim=2)

    return l.mean(dim=1), l.std(dim=1)


def tangents(points):
    v = points[1:] - points[:-1]
    t = F.normalize(v, dim=-1)
    return t


def tangent_correlations(t):
    """
    t: [M, 3]
    returns:
        corr[k] = average dot product at lag k
    """
    M = t.shape[0]

    corrs = []

    for k in range(M):
        dots = (t[:-k or None] * t[k:]).sum(dim=-1)
        corrs.append(dots.mean())

    return torch.stack(corrs)


def persistence_length(corrs, ds=1.0):
    k = torch.arange(len(corrs), device=corrs.device)

    # avoid log(0)
    valid = corrs > 0

    x = k[valid] * ds
    y = torch.log(corrs[valid])

    # linear fit
    slope = np.polyfit(x, y, deg=1)[0]

    Lp = -1.0 / slope
    return Lp


# revise
def persistence_length_metric(positions):
    """ batch version """

    result = []
    for points in positions:
        t = tangents(points)
        corrs = tangent_correlations(t)
        result.append(persistence_length(corrs))

    return torch.Tensor(result)


def heatmap_correlation_metric(true_hic_matrices, pred_hic_matrices):
    """ batch version """

    with torch.no_grad():
        af = true_hic_matrices.flatten(start_dim=1)
        bf = pred_hic_matrices.flatten(start_dim=1)

        return torch.Tensor(spearmanrho(af, bf, axis=1)[0])
