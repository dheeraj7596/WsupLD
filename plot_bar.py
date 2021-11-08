import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import Counter


def get_x_y():
    dataset = "nyt-coarse"
    # base_path = "/data/dheeraj/WsupLD/data/"
    base_path = "./data/"
    data_path = base_path + dataset + "/"

    correct_list = pickle.load(open(data_path + "correct_it_first_ep_0.pkl", "rb"))
    wrong_list = pickle.load(open(data_path + "wrong_it_first_ep_0.pkl", "rb"))

    cor = Counter(correct_list)
    wro = Counter(wrong_list)

    count_cor = 0
    for c in cor:
        count_cor += cor[c]

    count_wro = 0
    for w in wro:
        count_wro += wro[w]

    for c in cor:
        cor[c] = cor[c] / count_cor

    for w in wro:
        wro[w] = wro[w] / count_wro

    cor_y = np.array([cor[1], cor[2], cor[3], cor[4], cor[0]])
    wro_y = np.array([wro[1], wro[2], wro[3], wro[4], wro[0]])
    return cor_y, wro_y


if __name__ == "__main__":
    correct, wrong = get_x_y()
    N = 5
    # men_means = (20, 35, 30, 35, 27)
    # men_std = (2, 3, 4, 1, 2)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, correct, width, color='g')

    # women_means = (25, 32, 34, 20, 25)
    # women_std = (3, 5, 2, 3, 3)
    rects2 = ax.bar(ind + width, wrong, width, color='r')

    # add some text for labels, title and axes ticks
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Proportion of Samples learnt')
    ax.set_title('Proportion of Samples vs Learnt Epoch')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('1', '2', '3', '4', 'Never'))

    ax.legend((rects1[0], rects2[0]), ('Correct', 'Wrong'))
    plt.savefig('./data/nyt-coarse/prelim_exp.png')
    # plt.show()
