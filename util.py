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


def fit_get_tokenizer(data, max_words):
    tokenizer = Tokenizer(num_words=max_words, filters='!"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(data)
    return tokenizer
