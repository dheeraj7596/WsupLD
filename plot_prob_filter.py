import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import Counter

# def get_hist(bins=10, rang=None, density=False, logbin=False, frequency=False):
#     dataset = "nyt-coarse"
#     # base_path = "/data/dheeraj/WsupLD/data/"
#     base_path = "./data/"
#     data_path = base_path + dataset + "/"
#     data = pickle.load(open(data_path + "probs_prob_filter.pkl", "rb"))
#
#     if isinstance(data, np.ndarray):
#         data = data.ravel()
#     if logbin:
#         if rang:
#             bins = np.logspace(np.log10(max(rang[0], np.min(data))),
#                                np.log10(min(rang[1], np.max(data))), bins)
#         else:
#             bins = np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), bins)
#     hist, bin_edges = np.histogram(data, bins=bins, range=rang, density=density)
#     bin_c = (bin_edges[1:] + bin_edges[:-1]) / 2
#     if frequency:
#         hist = np.array(hist) / len(data)
#     return bin_c, hist
#
#
# if __name__ == "__main__":
#     plt.bar(*get_hist(frequency=True), width=0.1)
#     plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
#     plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
#     # plt.show()
#     plt.xlabel("Probability")
#     plt.ylabel("Proportion of Samples")
#     plt.title("Prediction Probability Distribution")
#     # plt.savefig("./data/nyt-coarse/prob_dist_nyt.png")
#     plt.show()

import pickle
from collections import Counter


def get_x_y():
    dataset = "nyt-coarse"
    # base_path = "/data/dheeraj/WsupLD/data/"
    base_path = "./data/"
    data_path = base_path + dataset + "/"

    correct_list = pickle.load(open(data_path + "correct_probs_1it.pkl", "rb"))
    wrong_list = pickle.load(open(data_path + "wrong_probs_1it.pkl", "rb"))

    cor_y = np.array(correct_list)
    wro_y = np.array(wrong_list)
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    pos_hist, bin_edges = np.histogram(cor_y, bins=bins, range=None, density=False)
    pos_hist = np.array(pos_hist) / len(correct_list)

    neg_hist, bin_edges = np.histogram(wro_y, bins=bins, range=None, density=False)
    neg_hist = np.array(neg_hist) / len(wrong_list)

    return pos_hist, neg_hist


if __name__ == "__main__":
    correct, wrong = get_x_y()
    N = 10
    # men_means = (20, 35, 30, 35, 27)
    # men_std = (2, 3, 4, 1, 2)

    ind = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # the x locations for the groups
    width = 0.05  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, correct, width=0.05, color='g')

    # women_means = (25, 32, 34, 20, 25)
    # women_std = (3, 5, 2, 3, 3)
    rects2 = ax.bar(ind + width, wrong, width=0.05, color='r')

    # add some text for labels, title and axes ticks
    ax.set_xlabel('Probability')
    ax.set_ylabel('Proportion of Samples')
    ax.set_title('Prediction Probability Distribution')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'))

    ax.legend((rects1[0], rects2[0]), ('Correct', 'Wrong'))
    plt.savefig("./data/nyt-coarse/prob_dist_nyt.png")
    # plt.show()
