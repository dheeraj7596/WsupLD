import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt


def adjustFigAspect(fig, aspect=1.0):
    '''
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    '''
    xsize, ysize = fig.get_size_inches()
    minsize = min(xsize, ysize)
    xlim = .4 * minsize / xsize
    ylim = .4 * minsize / ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5 - xlim,
                        right=.5 + xlim,
                        bottom=.5 - ylim,
                        top=.5 + ylim)


if __name__ == "__main__":
    dataset = "20news-fine-nomisc"
    # base_path = "/data/dheeraj/WsupLD/data/"
    base_path = "./data/"
    cls = "bert"
    data_path = base_path + dataset + "/"

    correct_list = pickle.load(open(data_path + "correct_list_batch_epoch_filter" + "_" + cls + ".pkl", "rb"))
    wrong_list = pickle.load(open(data_path + "wrong_list_batch_epoch_filter" + "_" + cls + ".pkl", "rb"))
    coverage_list = pickle.load(open(data_path + "coverage_list_batch_epoch_filter" + "_" + cls + ".pkl", "rb"))

    probs = pickle.load(open(data_path + "probs_prob_filter" + "_" + cls + ".pkl", "rb"))
    y_pseudo_orig = pickle.load(open(data_path + "y_pseudo_prob_filter" + "_" + cls + ".pkl", "rb"))
    y_true = pickle.load(open(data_path + "y_true_prob_filter" + "_" + cls + ".pkl", "rb"))

    noise_epoch = []
    noise_prob = []

    for i in range(len(correct_list)):
        noise_epoch.append(wrong_list[i] / coverage_list[i])

    risk_epoch = correct_list
    if dataset == "nyt-coarse":
        risk_epoch.append(8375)
        noise_epoch.append(1086 / 9460)
    elif dataset == "20news-fine-nomisc":
        risk_epoch.append(7771)
        noise_epoch.append(2684 / 10455)
    else:
        risk_epoch.append(4892)
        noise_epoch.append(2913 / 7805)

    risk_prob = []
    coverage = []

    inds = list(np.argsort(probs)[::-1])
    for c in coverage_list:
        train_inds = inds[:c]

        correct = 0
        wrong = 0
        for loop_ind in train_inds:
            if y_pseudo_orig[loop_ind] == y_true[loop_ind]:
                correct += 1
            else:
                wrong += 1

        risk_prob.append(correct)
        noise_prob.append(wrong / c)

    for c in coverage_list:
        if dataset == "nyt-coarse":
            coverage.append(c / 9460)
            # coverage.append(c)
        elif dataset == "20news-fine-nomisc":
            coverage.append(c / 10455)
        else:
            coverage.append(c / 7805)

    if dataset == "nyt-coarse":
        risk_prob.append(8375)
        noise_prob.append(1086 / 9460)
    elif dataset == "20news-fine-nomisc":
        risk_prob.append(7771)
        noise_prob.append(2684 / 10455)
    else:
        risk_prob.append(4892)
        noise_prob.append(2913 / 7805)
    coverage.append(1)
    # coverage.append(9460)

    coverage, risk_epoch = zip(*sorted(zip(coverage, risk_epoch)))
    coverage, risk_prob = zip(*sorted(zip(coverage, risk_prob)))

    fig = plt.figure()
    adjustFigAspect(fig, aspect=.5)
    ax = fig.add_subplot(111)
    ax.plot(coverage, risk_epoch, marker='o', markersize=4, label="LOPS")
    ax.plot(coverage, risk_prob, marker='o', markersize=4, label="Probability-based selection")

    # if dataset == "nyt-coarse":
    #     plt.plot(coverage, [8375] * len(coverage), '--', label="No filter")
    # else:
    #     plt.plot(coverage, [7771] * len(coverage), '--', label="No filter")

    plt.xlabel("Coverage")
    plt.ylabel("#Correctly labeled samples")
    # fig.legend(prop={'size': 6})
    fig.legend(prop={'size': 5}, bbox_to_anchor=(0.15, -0.33, 0.5, 0.5))
    # if dataset == "nyt-coarse":
    #     plt.title("NYT-Coarse")
    # elif dataset == "20news-fine-nomisc":
    #     plt.title("20News-Fine")
    # else:
    #     plt.title("Books")
    # plt.savefig(data_path + "plots/" + dataset + "_" + cls + "_rc.png")
    fig.show()

    fig = plt.figure()
    adjustFigAspect(fig, aspect=.5)
    ax = fig.add_subplot(111)
    ax.plot(coverage, noise_epoch, marker='o', markersize=4, label="LOPS")
    ax.plot(coverage, noise_prob, marker='o', markersize=4, label="Probability-based selection")

    # if dataset == "nyt-coarse":
    #     plt.plot(coverage, [8375] * len(coverage), '--', label="No filter")
    # else:
    #     plt.plot(coverage, [7771] * len(coverage), '--', label="No filter")

    plt.xlabel("Coverage")
    plt.ylabel("#Noise Ratio")
    if dataset == "20news-fine-nomisc":
        fig.legend(prop={'size': 5}, bbox_to_anchor=(0.15, 0.4, 0.5, 0.5))
    else:
        fig.legend(prop={'size': 5}, bbox_to_anchor=(0.15, -0.33, 0.5, 0.5))
    # if dataset == "nyt-coarse":
    #     plt.title("NYT-Coarse")
    # elif dataset == "20news-fine-nomisc":
    #     plt.title("20News-Fine")
    # else:
    #     plt.title("Books")
    # plt.savefig(data_path + "plots/" + dataset + "_" + cls + "_noise.png")
    plt.show()
