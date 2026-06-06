import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def visualise_hic(hic, max_value=None):
    hic_clipped = hic.clip(max=max_value)

    plt.figure(figsize=(6, 6))
    plt.imshow(hic_clipped, cmap=mcolors.LinearSegmentedColormap.from_list("white_to_red", [(1, 1, 1), (1, 0, 0)]))
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    plt.show()

    return


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
