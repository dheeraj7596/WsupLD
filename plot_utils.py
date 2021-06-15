import numpy as np
import matplotlib.pyplot as plt


def get_custom_bar_width(hist_x):
    # custom heights - difference
    heights = np.diff(hist_x) / 2
    return np.hstack([heights[0], heights])


def side_corr_plot(plot_x, plot_y, xlim=[0, 1], ylim=[0, 1], xlog=False, ylog=False,
                   xlabel='x', ylabel='y', save=None):
    fig, axes = plt.subplots(1, 3, figsize=(5, 5))
    fontsize = 15
    width = 0.7
    side_width = 0.12
    gap = 0.03
    bins = 40
    bar_width = 0.5
    ax = axes[0]
    ax.set_position([0, 0, width, width])
    ax.scatter(plot_x, plot_y, s=1, rasterized=True)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    ax = axes[1]
    ax.set_xlim(xlim)
    ax.set_position([0, width + gap, width, side_width])
    hist_x, hist_y = get_hist(plot_x, frequency=True, bins=bins, logbin=xlog)
    widths = get_custom_bar_width(hist_x)
    ax.bar(hist_x, hist_y, width=widths)
    ax.set_yscale('log')
    if xlog:
        ax.set_xscale('log')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([])
    # ax.set_yticks([])
    ax = axes[2]
    ax.set_ylim(ylim)
    ax.set_position([width + gap, 0, side_width, width])
    hist_x, hist_y = get_hist(plot_y, frequency=True, bins=bins, logbin=ylog)
    heights = get_custom_bar_width(hist_x)
    ax.barh(hist_x, hist_y, height=heights)
    ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.set_xticks([])
    ax.set_yticks([])
    if save is not None:
        fig.savefig(save, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()


def get_hist(data, bins=20, rang=None, density=False, logbin=False, frequency=False):
    if isinstance(data, np.ndarray):
        data = data.ravel()
    if logbin:
        if rang:
            bins = np.logspace(np.log10(rang[0]), np.log10(rang[1]), bins)
        else:
            bins = np.logspace(np.log10(np.min(data[data > 0])), np.log10(np.max(data[data > 0])), bins)
    hist, bin_edges = np.histogram(data, bins=bins, range=rang, density=density)
    if frequency:
        hist = hist / len(data)
    bin_c = (bin_edges[1:] + bin_edges[:-1]) / 2
    return bin_c, hist


if __name__ == "__main__":
    x = [0, 0.2, 0.4, 0.5, 0.6]
    y = [1, 2, 3, 4, 5]
    side_corr_plot(x, y,
                   xlim=[0, 0.7], ylim=[0, 10],
                   xlog=False, ylog=False,
                   xlabel=r'xlabel',
                   ylabel=r'ylabel',
                   save=None)
