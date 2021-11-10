import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import Counter


def get_hist(bins=10, rang=None, density=False, logbin=False, frequency=False):
    dataset = "nyt-coarse"
    # base_path = "/data/dheeraj/WsupLD/data/"
    base_path = "./data/"
    data_path = base_path + dataset + "/"
    data = pickle.load(open(data_path + "probs_prob_filter.pkl", "rb"))

    if isinstance(data, np.ndarray):
        data = data.ravel()
    if logbin:
        if rang:
            bins = np.logspace(np.log10(max(rang[0], np.min(data))),
                               np.log10(min(rang[1], np.max(data))), bins)
        else:
            bins = np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), bins)
    hist, bin_edges = np.histogram(data, bins=bins, range=rang, density=density)
    bin_c = (bin_edges[1:] + bin_edges[:-1]) / 2
    if frequency:
        hist = np.array(hist) / len(data)
    return bin_c, hist


if __name__ == "__main__":
    plt.bar(*get_hist(frequency=True), width=0.1)
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # plt.show()
    plt.xlabel("Probability")
    plt.ylabel("Proportion of Samples")
    plt.title("Prediction Probability Distribution")
    # plt.savefig("./data/nyt-coarse/prob_dist_nyt.png")
    plt.show()