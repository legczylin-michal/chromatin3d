import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def insulation_score(hic, w):
    """batch version"""

    kernel = torch.ones(1, 1, w, w)

    cross = F.conv2d(hic.unsqueeze(1), kernel, padding=0).squeeze()
    means = hic.flatten(start_dim=1).mean(dim=1, keepdim=True)

    return ((cross.diagonal(dim1=-2, dim2=-1) + 1e-8) / (means + 1e-8)).log2()


def visualise_hic(hic, max_value=None):
    hic_clipped = hic.clip(min=0, max=max_value)

    plt.figure(figsize=(6, 6))
    plt.imshow(hic_clipped, cmap=mcolors.LinearSegmentedColormap.from_list("white_to_red", [(1, 1, 1), (1, 0, 0)]))
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    plt.show()



def prettify_duration(duration):
    s = duration % 60
    duration -= s
    duration //= 60
    s = round(s)

    m = duration % 60
    duration -= m
    duration //= 60
    m = round(m)

    h = duration % 24
    duration -= h
    duration //= 24
    h = round(h)
    d = round(duration)

    return f"{d}d {h}h {m}m {s}s"
