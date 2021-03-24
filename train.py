from bert_train import *
import pickle
import json, sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from util import *
import matplotlib.pyplot as plt


def create_label_index_maps(labels):
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(labels):
        label_to_index[label] = i
        index_to_label[i] = label
    return label_to_index, index_to_label


def generate_pseudo_labels(df, labels, label_term_dict, tokenizer):
    def argmax_label(count_dict):
        maxi = 0
        max_label = None
        for l in count_dict:
            count = 0
            for t in count_dict[l]:
                count += count_dict[l][t]
            if count > maxi:
                maxi = count
                max_label = l
        return max_label

    y = []
    X = []
    y_true = []
    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w
    for index, row in df.iterrows():
        line = row["sentence"]
        label = row["label"]
        tokens = tokenizer.texts_to_sequences([line])[0]
        words = []
        for tok in tokens:
            words.append(index_word[tok])
        count_dict = {}
        flag = 0
        for l in labels:
            seed_words = set()
            for w in label_term_dict[l]:
                seed_words.add(w)
            int_labels = list(set(words).intersection(seed_words))
            if len(int_labels) == 0:
                continue
            for word in words:
                if word in int_labels:
                    flag = 1
                    try:
                        temp = count_dict[l]
                    except:
                        count_dict[l] = {}
                    try:
                        count_dict[l][word] += 1
                    except:
                        count_dict[l][word] = 1
        if flag:
            lbl = argmax_label(count_dict)
            if not lbl:
                continue
            y.append(lbl)
            X.append(line)
            y_true.append(label)
    return X, y, y_true


if __name__ == "__main__":
    # base_path = "./data/"
    base_path = "/data/dheeraj/WsupLD/data/"
    dataset = "nyt"
    data_path = base_path + dataset + "/"
    plot_dump_dir = data_path + "plots/"
    os.makedirs(plot_dump_dir, exist_ok=True)
    thresh = 0.6
    use_gpu = int(sys.argv[1])
    gpu_id = int(sys.argv[2])
    bins = [0, 0.25, 0.5, 0.75, 1]
    # use_gpu = 0

    if use_gpu:
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device("cpu")

    df = pickle.load(open(data_path + "df_coarse.pkl", "rb"))
    # with open(data_path + "seedwords.json") as fp:
    #     label_term_dict = json.load(fp)

    labels = list(set(df["label"]))
    label_to_index, index_to_label = create_label_index_maps(labels)

    X_all = list(df["text"])
    y_all = list(df["label"])
    y_all_inds = [label_to_index[l] for l in y_all]

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all_inds, test_size=0.9, random_state=42,
                                                        stratify=y_all)

    for it in range(5):
        print("Iteration:", it)

        if it == 0:
            model, _, _ = train_bert(X_train, y_train, device, None, None, label_dyn=False)
        else:
            print("Correct Samples:", len(correct_bootstrap["text"]))
            print("Wrong Samples:", len(wrong_bootstrap["text"]))
            model, correct_bootstrap, wrong_bootstrap = train_bert(X_train, y_train, device, correct_bootstrap,
                                                                   wrong_bootstrap, label_dyn=True)
            plt.figure()
            plt.hist(correct_bootstrap["match"], color='blue', edgecolor='black', bins=bins)
            plt.xticks(bins)
            plt.savefig(plot_dump_dir + "correct_it_" + str(it) + ".png")

            plt.figure()
            plt.hist(wrong_bootstrap["match"], color='blue', edgecolor='black', bins=bins)
            plt.xticks(bins)
            plt.savefig(plot_dump_dir + "wrong_it_" + str(it) + ".png")

            plt.figure()
            plt.hist(correct_bootstrap["first_ep"], color='blue', edgecolor='black', bins=bins)
            plt.xticks(bins)
            plt.savefig(plot_dump_dir + "correct_it_first_ep_" + str(it) + ".png")

            plt.figure()
            plt.hist(wrong_bootstrap["first_ep"], color='blue', edgecolor='black', bins=bins)
            plt.xticks(bins)
            plt.savefig(plot_dump_dir + "wrong_it_first_ep_" + str(it) + ".png")

        correct_bootstrap = {"text": [], "true": [], "pred": [], "match": [], "first_ep": []}
        wrong_bootstrap = {"text": [], "true": [], "pred": [], "match": [], "first_ep": []}

        predictions = test(model, X_test, y_test, device)
        for i, p in enumerate(predictions):
            if i == 0:
                pred = p
            else:
                pred = np.concatenate((pred, p))

        pred_labels = []
        removed_inds = []
        for i, p in enumerate(pred):
            sample = X_test[i]
            true_lbl = y_test[i]
            max_prob = p.max(axis=-1)
            lbl = p.argmax(axis=-1)
            pred_labels.append(index_to_label[lbl])
            if max_prob >= thresh:
                X_train.append(sample)
                y_train.append(lbl)
                removed_inds.append(i)
                if true_lbl == lbl:
                    correct_bootstrap["text"].append(sample)
                    correct_bootstrap["true"].append(true_lbl)
                    correct_bootstrap["pred"].append(lbl)
                    correct_bootstrap["match"].append(0)
                    correct_bootstrap["first_ep"].append(0)
                else:
                    wrong_bootstrap["text"].append(sample)
                    wrong_bootstrap["true"].append(true_lbl)
                    wrong_bootstrap["pred"].append(lbl)
                    wrong_bootstrap["match"].append(0)
                    wrong_bootstrap["first_ep"].append(0)

        removed_inds.sort(reverse=True)
        for i in removed_inds:
            del X_test[i]
            del y_test[i]

        print("****************** CLASSIFICATION REPORT FOR All DOCUMENTS ********************")
        predictions = test(model, X_all, y_all_inds, device)
        pred_inds = get_labelinds_from_probs(predictions)
        pred_labels = []
        for p in pred_inds:
            pred_labels.append(index_to_label[p])
        print(classification_report(y_all, pred_labels))
        print("*" * 80)
