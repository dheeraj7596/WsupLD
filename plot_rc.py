import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset = "20news-fine-nomisc"
    # base_path = "/data/dheeraj/WsupLD/data/"
    base_path = "./data/"
    data_path = base_path + dataset + "/"

    correct_list = pickle.load(open(data_path + "correct_list_batch_epoch_filter.pkl", "rb"))
    wrong_list = pickle.load(open(data_path + "wrong_list_batch_epoch_filter.pkl", "rb"))
    coverage_list = pickle.load(open(data_path + "coverage_list_batch_epoch_filter.pkl", "rb"))

    probs = pickle.load(open(data_path + "probs_prob_filter.pkl", "rb"))
    pred_labels = pickle.load(open(data_path + "pred_labels_prob_filter.pkl", "rb"))
    y_pseudo_orig = pickle.load(open(data_path + "y_pseudo_prob_filter.pkl", "rb"))
    y_true = pickle.load(open(data_path + "y_true_prob_filter.pkl", "rb"))

    risk_epoch = correct_list
    if dataset == "nyt-coarse":
        risk_epoch.append(8375)
    else:
        risk_epoch.append(7771)

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

    for c in coverage_list:
        if dataset == "nyt-coarse":
            coverage.append(c / 9460)
            # coverage.append(c)
        else:
            coverage.append(c / 10455)

    if dataset == "nyt-coarse":
        risk_prob.append(8375)
    else:
        risk_prob.append(7771)
    coverage.append(1)
    # coverage.append(9460)

    coverage, risk_epoch = zip(*sorted(zip(coverage, risk_epoch)))
    coverage, risk_prob = zip(*sorted(zip(coverage, risk_prob)))

    plt.plot(coverage, risk_epoch, label="Epoch-based filter")
    plt.plot(coverage, risk_prob, label="Probability-based filter")

    if dataset == "nyt-coarse":
        plt.plot(coverage, [8375] * len(coverage), '--', label="No filter")
    else:
        plt.plot(coverage, [7771] * len(coverage), '--', label="No filter")

    plt.xlabel("Coverage")
    plt.ylabel("Risk=#Correctly labeled samples")
    plt.legend(loc='lower right')
    plt.title(dataset)
    plt.savefig(data_path + "plots/" + dataset + "_rc.png")
    # plt.show()
