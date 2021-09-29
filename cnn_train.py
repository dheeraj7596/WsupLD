import pickle
import json, sys
import os
from sklearn.metrics import classification_report
import torch
from util import *
import matplotlib.pyplot as plt
from cnn_model.train_cnn import filter, test, train_cnn
import copy
import torchtext.legacy.data as data
import random

if __name__ == "__main__":
    base_path = "/data/dheeraj/WsupLD/data/"
    dataset = sys.argv[3]
    data_path = base_path + dataset + "/"
    plot_dump_dir = data_path + "plots/no_filter/"
    os.makedirs(plot_dump_dir, exist_ok=True)
    thresh = 0.6
    use_gpu = int(sys.argv[1])
    gpu_id = int(sys.argv[2])
    dump_flag = False
    plt_flag = int(sys.argv[5])
    filter_flag = int(sys.argv[4])
    percent_thresh = float(sys.argv[6])
    bins = [0, 0.25, 0.5, 0.75, 1]
    bins_five = [0, 1, 2, 3, 4, 5]
    num_its = 5

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if use_gpu:
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device("cpu")

    df = pickle.load(open(data_path + "df.pkl", "rb"))
    with open(data_path + "seedwords.json") as fp:
        label_term_dict = json.load(fp)

    labels = list(set(df["label"]))
    label_to_index, index_to_label = create_label_index_maps(labels)

    df_copy = copy.deepcopy(df)
    df_copy = preprocess(df_copy)
    tokenizer = fit_get_tokenizer(df_copy.text, max_words=150000)

    X_all = list(df["text"])
    y_all = list(df["label"])
    y_all_inds = [label_to_index[l] for l in y_all]

    print("Generating pseudo labels..", flush=True)
    X_train_inds, y_train, y_true = generate_pseudo_labels(df_copy, labels, label_term_dict, tokenizer)
    X_test_inds = list(set(range(len(df))) - set(X_train_inds))

    X_train = list(df.iloc[X_train_inds]["text"])
    y_train = [label_to_index[l] for l in y_train]
    y_true = [label_to_index[l] for l in y_true]

    X_test = list(df.iloc[X_test_inds]["text"])
    y_test = list(df.iloc[X_test_inds]["label"])
    y_test = [label_to_index[l] for l in y_test]

    correct_bootstrap = {"text": [], "true": [], "pred": [], "match": [], "first_ep": []}
    wrong_bootstrap = {"text": [], "true": [], "pred": [], "match": [], "first_ep": []}

    for i, sent in enumerate(X_train):
        if y_train[i] == y_true[i]:
            correct_bootstrap["text"].append(sent)
            correct_bootstrap["true"].append(y_true[i])
            correct_bootstrap["pred"].append(y_train[i])
            correct_bootstrap["match"].append(0)
            correct_bootstrap["first_ep"].append(0)
        else:
            wrong_bootstrap["text"].append(sent)
            wrong_bootstrap["true"].append(y_true[i])
            wrong_bootstrap["pred"].append(y_train[i])
            wrong_bootstrap["match"].append(0)
            wrong_bootstrap["first_ep"].append(0)

    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False,
                             use_vocab=False,
                             pad_token=None,
                             unk_token=None
                             )

    for it in range(num_its):
        temp_label_to_index = {}
        temp_index_to_label = {}

        print("Iteration:", it, flush=True)

        print("Correct Samples:", len(correct_bootstrap["text"]), flush=True)
        print("Wrong Samples:", len(wrong_bootstrap["text"]), flush=True)

        if dump_flag:
            pickle.dump(X_train, open(data_path + "X_train_" + str(it) + ".pkl", "wb"))
            pickle.dump(y_train, open(data_path + "y_train_" + str(it) + ".pkl", "wb"))
            pickle.dump(y_true, open(data_path + "y_true_" + str(it) + ".pkl", "wb"))

        non_train_data = []
        non_train_labels = []
        true_non_train_labels = []

        print("****************BEFORE FILTERING: classification report of pseudo-labels******************", flush=True)
        print(classification_report(y_true, y_train), flush=True)

        if filter_flag:
            for i, y in enumerate(sorted(list(set(y_train)))):
                temp_label_to_index[y] = i
                temp_index_to_label[i] = y
            y_train = [temp_label_to_index[y] for y in y_train]

        text_field.build_vocab(X_train)
        label_field.build_vocab(y_train)

        if filter_flag == 1:
            print("Filtering started..", flush=True)
            X_train, y_train, y_true, non_train_data, non_train_labels, true_non_train_labels = filter(X_train,
                                                                                                       y_train,
                                                                                                       y_true,
                                                                                                       percent_thresh,
                                                                                                       device,
                                                                                                       text_field,
                                                                                                       label_field,
                                                                                                       it)
            y_train = [temp_index_to_label[y] for y in y_train]
            non_train_labels = [temp_index_to_label[y] for y in non_train_labels]
        # elif filter_flag == 2:
        #     print("Filtering started..", flush=True)
        #     X_train, y_train, y_true, non_train_data, non_train_labels, true_non_train_labels, probs, cutoff_prob = prob_filter(
        #         X_train,
        #         y_train,
        #         y_true,
        #         device,
        #         it)
        #     y_train = [temp_index_to_label[y] for y in y_train]
        #     non_train_labels = [temp_index_to_label[y] for y in non_train_labels]

        # probs = np.sort(probs)[::-1]
        # plt.figure()
        # plt.plot(probs)
        # plt.axhline(cutoff_prob, color='r')
        # plt.savefig(plot_dump_dir + "prob_filter_cutoff_prob_" + str(it) + ".png")
        # pickle.dump(X_train, open(data_path + "X_train_prob_" + str(it) + ".pkl", "wb"))
        print("******************AFTER FILTERING: classification report of pseudo-labels******************", flush=True)
        print(classification_report(y_true, y_train), flush=True)

        if len(set(y_train)) < len(label_to_index):
            print("Number of labels in training set after filtering:", len(set(y_train)))
            # raise Exception(
            #     "Number of labels expected " + str(len(label_to_index)) + " but found " + str(len(set(y_train))))

        if dump_flag:
            pickle.dump(X_train, open(data_path + "X_train_filtered_" + str(it) + ".pkl", "wb"))
            pickle.dump(y_train, open(data_path + "y_train_filtered_" + str(it) + ".pkl", "wb"))
            pickle.dump(y_true, open(data_path + "y_true_filtered_" + str(it) + ".pkl", "wb"))

            pickle.dump(non_train_data, open(data_path + "non_train_data_" + str(it) + ".pkl", "wb"))
            pickle.dump(non_train_labels, open(data_path + "non_train_labels_" + str(it) + ".pkl", "wb"))
            pickle.dump(true_non_train_labels, open(data_path + "true_non_train_labels_" + str(it) + ".pkl", "wb"))

        for i in range(len(non_train_data)):
            X_test.append(non_train_data[i])
            y_test.append(true_non_train_labels[i])

        correct_bootstrap = {"text": [], "true": [], "pred": [], "match": [], "first_ep": []}
        wrong_bootstrap = {"text": [], "true": [], "pred": [], "match": [], "first_ep": []}

        for i, sent in enumerate(X_train):
            if y_train[i] == y_true[i]:
                correct_bootstrap["text"].append(sent)
                correct_bootstrap["true"].append(y_true[i])
                correct_bootstrap["pred"].append(y_train[i])
                correct_bootstrap["match"].append(0)
                correct_bootstrap["first_ep"].append(0)
            else:
                wrong_bootstrap["text"].append(sent)
                wrong_bootstrap["true"].append(y_true[i])
                wrong_bootstrap["pred"].append(y_train[i])
                wrong_bootstrap["match"].append(0)
                wrong_bootstrap["first_ep"].append(0)

        print("Filtering completed..", flush=True)
        print("Correct Samples in New training data:", len(correct_bootstrap["text"]), flush=True)
        print("Wrong Samples in New training data:", len(wrong_bootstrap["text"]), flush=True)

        for i, y in enumerate(sorted(list(set(y_train)))):
            temp_label_to_index[y] = i
            temp_index_to_label[i] = y
        y_train = [temp_label_to_index[y] for y in y_train]

        print("Training model..", flush=True)
        model, correct_bootstrap, wrong_bootstrap = train_cnn(X_train, y_train, device, text_field, label_field,
                                                              correct_bootstrap, wrong_bootstrap, label_dyn=filter_flag)
        if plt_flag:
            plt.figure()
            plt.hist(correct_bootstrap["match"], color='blue', edgecolor='black', bins=bins)
            plt.xticks(bins)
            plt.savefig(plot_dump_dir + "correct_it_" + str(it) + ".png")

            plt.figure()
            plt.hist(wrong_bootstrap["match"], color='blue', edgecolor='black', bins=bins)
            plt.xticks(bins)
            plt.savefig(plot_dump_dir + "wrong_it_" + str(it) + ".png")

            plt.figure()
            plt.hist(correct_bootstrap["first_ep"], color='blue', edgecolor='black', bins=bins_five)
            plt.xticks(bins_five)
            plt.savefig(plot_dump_dir + "correct_it_first_ep_" + str(it) + ".png")

            plt.figure()
            plt.hist(wrong_bootstrap["first_ep"], color='blue', edgecolor='black', bins=bins_five)
            plt.xticks(bins_five)
            plt.savefig(plot_dump_dir + "wrong_it_first_ep_" + str(it) + ".png")

        print("****************** CLASSIFICATION REPORT FOR All DOCUMENTS ********************", flush=True)
        pred_inds, _ = test(model, X_all, y_all_inds, text_field, label_field, device)
        pred_labels = []
        for p in pred_inds:
            pred_labels.append(index_to_label[temp_index_to_label[p]])
        print(classification_report(y_all, pred_labels), flush=True)
        print("*" * 80, flush=True)

        if filter_flag:
            print(
                "****************** CLASSIFICATION REPORT FOR FIRST EP CORRECT DOCUMENTS WRT PSEUDO ********************",
                flush=True)
            pred_inds, _ = test(model, X_train, y_train, text_field, label_field, device)
            pred_labels = []
            for p in pred_inds:
                pred_labels.append(index_to_label[temp_index_to_label[p]])
            y_train_strs = [index_to_label[temp_index_to_label[lbl]] for lbl in y_train]
            print(classification_report(y_train_strs, pred_labels), flush=True)
            print("*" * 80, flush=True)

            print("****************** CLASSIFICATION REPORT FOR FIRST EP CORRECT DOCUMENTS WRT GT ********************",
                  flush=True)
            pred_inds, _ = test(model, X_train, y_true, text_field, label_field, device)
            pred_labels = []
            for p in pred_inds:
                pred_labels.append(index_to_label[temp_index_to_label[p]])
            y_true_train_strs = [index_to_label[lbl] for lbl in y_true]
            print(classification_report(y_true_train_strs, pred_labels), flush=True)
            print("*" * 80, flush=True)

            print(
                "****************** CLASSIFICATION REPORT FOR FIRST EP WRONG DOCUMENTS WRT PSEUDO ********************",
                flush=True)
            pred_inds, _ = test(model, non_train_data, non_train_labels, text_field, label_field, device)
            pred_labels = []
            for p in pred_inds:
                pred_labels.append(index_to_label[temp_index_to_label[p]])
            non_train_labels_strs = [index_to_label[lbl] for lbl in non_train_labels]
            print(classification_report(non_train_labels_strs, pred_labels), flush=True)
            print("*" * 80, flush=True)

            print("****************** CLASSIFICATION REPORT FOR FIRST EP WRONG DOCUMENTS WRT GT ********************",
                  flush=True)
            pred_inds, _ = test(model, non_train_data, true_non_train_labels, text_field, label_field, device)
            pred_labels = []
            for p in pred_inds:
                pred_labels.append(index_to_label[temp_index_to_label[p]])
            true_non_train_labels_strs = [index_to_label[lbl] for lbl in true_non_train_labels]
            print(classification_report(true_non_train_labels_strs, pred_labels), flush=True)
            print("*" * 80, flush=True)

        print("****************** CLASSIFICATION REPORT FOR REST DOCUMENTS WRT GT ********************", flush=True)
        pred_inds, _ = test(model, X_test, y_test, text_field, label_field, device)
        pred_labels = []
        for p in pred_inds:
            pred_labels.append(index_to_label[temp_index_to_label[p]])
        y_test_strs = [index_to_label[lbl] for lbl in y_test]
        print(classification_report(y_test_strs, pred_labels), flush=True)
        print("*" * 80, flush=True)

        _, predictions = test(model, X_test, y_test, text_field, label_field, device)
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
            lbl = temp_index_to_label[p.argmax(axis=-1)]
            pred_labels.append(index_to_label[lbl])
            if max_prob >= thresh:
                X_train.append(sample)
                y_train.append(lbl)
                y_true.append(true_lbl)
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

        # Resetting match and first_ep for all samples
        assert len(correct_bootstrap["match"]) == len(correct_bootstrap["first_ep"])
        assert len(wrong_bootstrap["match"]) == len(wrong_bootstrap["first_ep"])

        for i in range(len(correct_bootstrap["match"])):
            correct_bootstrap["match"][i] = 0
            correct_bootstrap["first_ep"][i] = 0

        for i in range(len(wrong_bootstrap["match"])):
            wrong_bootstrap["match"][i] = 0
            wrong_bootstrap["first_ep"][i] = 0
