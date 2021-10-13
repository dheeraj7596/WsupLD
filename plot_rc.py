import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset = sys.argv[1]
    base_path = "/data/dheeraj/WsupLD/data/"
    data_path = base_path + dataset + "/"

    correct_list = pickle.load(open(data_path + "correct_list_batch_epoch_filter.pkl", "rb"))
    wrong_list = pickle.load(open(data_path + "wrong_list_batch_epoch_filter.pkl", "rb"))
    coverage_list = pickle.load(open(data_path + "coverage_list_batch_epoch_filter.pkl", "rb"))

    probs = pickle.load(open(data_path + "probs_prob_filter.pkl", "rb"))
    pred_labels = pickle.load(open(data_path + "pred_labels_prob_filter.pkl", "rb"))
    y_pseudo_orig = pickle.load(open(data_path + "y_pseudo_prob_filter.pkl", "rb"))
    y_true = pickle.load(open(data_path + "y_true_prob_filter.pkl", "rb"))

    risk_epoch = correct_list
    risk_prob = []
    coverage = []

    inds = list(np.argsort(probs)[::-1])
    for c in coverage_list:
        train_inds = inds[:c]

        correct = 0
        wrong = 0
        for loop_ind in train_inds:
            if pred_labels[loop_ind] == y_true[loop_ind]:
                correct += 1
            else:
                wrong += 1

        risk_prob.append(correct)

    for c in coverage_list:
        if dataset == "nyt-coarse":
            coverage.append(c / 9460)
        else:
            coverage.append(c / 10455)

    plt.plot(coverage, risk_epoch, label="Epoch-based filter")
    plt.plot(coverage, risk_prob, label="Probability-based filter")
    plt.xlabel("Coverage")
    plt.ylabel("Risk=#Correctly labeled samples")
    plt.legend()
    plt.title(dataset)
    plt.savefig(data_path + "plots/" + dataset + "_rc.png")
