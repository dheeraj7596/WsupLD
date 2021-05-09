import numpy as np
from keras.preprocessing.text import Tokenizer


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
