import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
import string


def preprocess(df):
    print("Preprocessing data..", flush=True)
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    for index, row in df.iterrows():
        if index % 100 == 0:
            print("Finished rows: " + str(index) + " out of " + str(len(df)), flush=True)
        line = row["text"]
        words = line.strip().split()
        new_words = []
        for word in words:
            word_clean = word.translate(str.maketrans('', '', string.punctuation))
            if len(word_clean) == 0 or word_clean in stop_words:
                continue
            new_words.append(word_clean)
        df["text"][index] = " ".join(new_words)
    return df


def generate_pseudo_labels(df, labels, label_term_dict, tokenizer):
    def argmax_label(count_dict):
        maxi = 0
        max_label = None
        keys = sorted(count_dict.keys())
        for l in keys:
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
        line = row["text"]
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
            X.append(index)
            y_true.append(label)
    return X, y, y_true


def get_labelinds_from_probs(predictions):
    for i, p in enumerate(predictions):
        if i == 0:
            pred = p
        else:
            pred = np.concatenate((pred, p))
    pred_inds = []
    for p in pred:
        pred_inds.append(p.argmax(axis=-1))
    return pred_inds


def compute_train_non_train_inds(pred_inds, true_inds, inds_map, lbl):
    train_inds = []
    non_train_inds = []
    for ind in inds_map[lbl]:
        if pred_inds[ind] == true_inds[ind]:
            train_inds.append(ind)
        else:
            non_train_inds.append(ind)
    return train_inds, non_train_inds


def fit_get_tokenizer(data, max_words):
    tokenizer = Tokenizer(num_words=max_words, filters='!"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(data)
    return tokenizer


def create_label_index_maps(labels):
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(labels):
        label_to_index[label] = i
        index_to_label[i] = label
    return label_to_index, index_to_label


def bert_data_it_dict():
    # dic = {"nyt-coarse": {0: 8817, 1: 12821, 2: 12875, 3: 12966, 4: 12999},
    #        "nyt-fine": {0: 6738, 1: 10770, 2: 11199, 3: 11420, 4: 11527},
    #        "20news-coarse-nomisc": {0: 7377, 1: 17047, 2: 17409, 3: 17470, 4: 17524},
    #        "20news-fine-nomisc": {0: 9751, 1: 14727, 2: 15668, 3: 15841, 4: 15862},
    #        "agnews": {0: 32264, 1: 114160, 2: 118153, 3: 118060, 4: 118198}}

    dic = {"nyt-coarse": {0: 9128, 1: 12821, 2: 12875, 3: 12966, 4: 12999},
           "nyt-fine": {0: 6738, 1: 10770, 2: 11199, 3: 11420, 4: 11527},
           "20news-coarse-nomisc": {0: 7377, 1: 17047, 2: 17409, 3: 17470, 4: 17524},
           "20news-fine-nomisc": {0: 10136, 1: 14727, 2: 15668, 3: 15841, 4: 15862},
           "agnews": {0: 32264, 1: 114160, 2: 118153, 3: 118060, 4: 118198}}
    return dic


def roberta_data_it_dict():
    dic = {"nyt-coarse": {0: 8743, 1: 12747, 2: 12902, 3: 12862, 4: 12896},
           "nyt-fine": {0: 6657, 1: 10694, 2: 11349, 3: 11396, 4: 11430},
           "20news-coarse-nomisc": {0: 7390, 1: 16639, 2: 17149, 3: 17386, 4: 17434},
           "20news-fine-nomisc": {0: 9377, 1: 14547, 2: 15475, 3: 15449, 4: 15831},
           "agnews": {0: 32257, 1: 113995, 2: 116278, 3: 117804, 4: 117518}}
    return dic


def xlnet_data_it_dict():
    dic = {
        "nyt-coarse": {0: 8735, 1: 12850, 2: 12892, 3: 12909, 4: 12937},
        "nyt-fine": {0: 6674, 1: 10601, 2: 11324, 3: 11420, 4: 11366},
        "20news-coarse-nomisc": {0: 7340, 1: 16650, 2: 17138, 3: 17374, 4: 17344},
        "20news-fine-nomisc": {0: 9393, 1: 15018, 2: 15459, 3: 15573, 4: 15639},
    }
    return dic


def compute_stability_scores(stability_list):
    stability_scores = []
    for lst in stability_list:
        den = 0
        num = 0
        for i, ent in enumerate(lst):
            if ent == 1:
                den += 1
                num += 1
            elif ent == 0 and den != 0:
                den += 1

        if den == 0:
            temp = 0
        else:
            temp = num / den
        stability_scores.append(temp)
    return stability_scores


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
