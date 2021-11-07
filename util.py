import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
import string
import random


def generate_name(id):
    return "fnust" + str(id)


def modify_phrases(label_term_dict, phrase_id_map, random_k=0):
    for l in label_term_dict:
        temp_list = []
        for term in label_term_dict[l]:
            try:
                temp_list.append(generate_name(phrase_id_map[term]))
            except:
                temp_list.append(term)
        if random_k:
            random.shuffle(temp_list)
            label_term_dict[l] = temp_list[:random_k]
        else:
            label_term_dict[l] = temp_list
    return label_term_dict


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

    dic = {"nyt-coarse": {0: 8794, 1: 12846, 2: 12961, 3: 12961, 4: 12960},
           "nyt-fine": {0: 6728, 1: 10474, 2: 11240, 3: 11326, 4: 11503},
           "20news-coarse-nomisc": {0: 7427, 1: 16782, 2: 17365, 3: 17349, 4: 17560},
           "20news-fine-nomisc": {0: 9612, 1: 14795, 2: 15617, 3: 15638, 4: 15893},
           "books": {0: 7540, 1: 29934, 2: 32287, 3: 32447, 4: 32317},
           "dblp": {0: 9251, 1: 33927, 2: 36447, 3: 36599, 4: 36872},
           # "agnews": {0: 32264, 1: 114160, 2: 118153, 3: 118060, 4: 118198}
           }
    return dic


def roberta_data_it_dict():
    dic = {"nyt-coarse": {0: 8711, 1: 12741, 2: 12952, 3: 12953, 4: 12924},
           "nyt-fine": {0: 6686, 1: 10581, 2: 11247, 3: 11409, 4: 11353},
           "20news-coarse-nomisc": {0: 7326, 1: 16851, 2: 17052, 3: 17367, 4: 17322},
           "20news-fine-nomisc": {0: 9334, 1: 14886, 2: 15326, 3: 15729, 4: 15810},
           "agnews": {0: 32257, 1: 113995, 2: 116278, 3: 117804, 4: 117518},
           "books": {0: 7539, 1: 29487, 2: 31668, 3: 32168, 4: 32059},
           "dblp": {0: 9216, 1: 33720, 2: 35843, 3: 35834, 4: 35782},
           }
    return dic


def xlnet_data_it_dict():
    dic = {
        "nyt-coarse": {0: 8658, 1: 12743, 2: 12728, 3: 12768, 4: 12839},
        "nyt-fine": {0: 6555, 1: 11056, 2: 11421, 3: 11398, 4: 11462},
        "20news-coarse-nomisc": {0: 7383, 1: 16751, 2: 17130, 3: 17391, 4: 17232},
        "20news-fine-nomisc": {0: 9466, 1: 15080, 2: 15605, 3: 15759, 4: 15650},
        "agnews": {0: 32266, 1: 112158, 2: 116953, 3: 116676, 4: 115854},
        "dblp": {0: 9200, 1: 34122, 2: 36141, 3: 36059, 4: 36480},
        "books": {0: 7534, 1: 29366, 2: 30930, 3: 31746, 4: 32090},
    }
    return dic


def cnn_data_it_dict():
    dic = {
        "nyt-coarse": {0: 8735, 1: 12850, 2: 12892, 3: 12909, 4: 12937},
        "nyt-fine": {0: 6674, 1: 10601, 2: 11324, 3: 11420, 4: 11366},
        "20news-coarse-nomisc": {0: 7340, 1: 16650, 2: 17138, 3: 17374, 4: 17344},
        "20news-fine-nomisc": {0: 9393, 1: 15018, 2: 15459, 3: 15573, 4: 15639},
    }
    return dic


def gpt2_data_it_dict():
    dic = {
        "nyt-coarse": {0: 8800, 1: 12655, 2: 12807, 3: 12877, 4: 12961},
        "nyt-fine": {0: 6447, 1: 10885, 2: 11158, 3: 11218, 4: 11310},
        "20news-coarse-nomisc": {0: 7452, 1: 16407, 2: 17131, 3: 17294, 4: 17404},
        "20news-fine-nomisc": {0: 9506, 1: 14958, 2: 15458, 3: 15623, 4: 15679},
        "agnews": {0: 32167, 1: 109142, 2: 115443, 3: 116411, 4: 115758},
        "books": {0: 7568, 1: 29542, 2: 31528, 3: 31687, 4: 31701},
        # "dblp": {0: 9251, 1: 33927, 2: 36447, 3: 36599, 4: 36872},
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
