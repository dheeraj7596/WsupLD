from keras_han.model import HAN
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
from nltk import tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow
import pickle
import sys
from util import create_label_index_maps, compute_train_non_train_inds
from sklearn.metrics import classification_report
from word2vec import train_word2vec, fit_get_tokenizer
import os
from collections import Counter


class EpochFilterCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, X_train, y_train, stop_ep):
        super().__init__()
        self.train_inds_map = {}
        self.non_train_inds_map = {}
        self.X_train = X_train
        self.true_inds = y_train
        self.stop_ep = stop_ep
        self.union_train_inds = set([])

        for i in list(set(y_train)):
            self.train_inds_map[i] = set([])
            self.non_train_inds_map[i] = set([])

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.X_train)
        pred_inds = get_labelinds_from_probs(predictions)

        for loop_ind in range(len(pred_inds)):
            if pred_inds[loop_ind] == self.true_inds[loop_ind]:
                self.train_inds_map[self.true_inds[loop_ind]].add(loop_ind)
                self.union_train_inds.add(loop_ind)
            else:
                self.non_train_inds_map[self.true_inds[loop_ind]].add(loop_ind)

        if epoch == self.stop_ep:
            self.model.stop_training = True
            for i in list(set(self.true_inds)):
                self.non_train_inds_map[i] = self.non_train_inds_map[i] - self.union_train_inds


class Top50UnionFilterCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, thresh_map, inds_map, X_train, y_train):
        super().__init__()
        self.thresh_map = thresh_map
        self.inds_map = inds_map
        self.filter_flag_map = {}
        self.train_inds_map = {}
        self.non_train_inds_map = {}
        self.X_train = X_train
        self.true_inds = y_train

        for i in self.thresh_map:
            self.filter_flag_map[i] = False
            self.train_inds_map[i] = set([])
            self.non_train_inds_map[i] = set([])

    def on_epoch_end(self, epoch, logs=None):
        # predict on all training data
        # check if the threshold is hit for all labels
        # then stop
        predictions = self.model.predict(self.X_train)
        pred_inds = get_labelinds_from_probs(predictions)

        count = 0
        for lbl in self.filter_flag_map:
            if not self.filter_flag_map[lbl]:
                train_inds, non_train_inds = compute_train_non_train_inds(pred_inds, self.true_inds, self.inds_map, lbl)
                self.train_inds_map[lbl].update(set(train_inds))
                self.non_train_inds_map[lbl].update(set(non_train_inds))

                if len(self.train_inds_map[lbl]) >= self.thresh_map[lbl]:
                    self.filter_flag_map[lbl] = True
                    count += 1
            else:
                count += 1

        print("Number of labels reached 50 percent threshold", count)
        for i in self.filter_flag_map:
            if not self.filter_flag_map[i]:
                print("For label ", i, " Number expected ", self.thresh_map[i], " Found ", len(self.train_inds_map[i]))

        temp_flg = True
        for i in self.filter_flag_map:
            temp_flg = temp_flg and self.filter_flag_map[i]

        if temp_flg:
            self.model.stop_training = temp_flg

    def on_train_end(self, logs=None):
        print("Executing on_train_end in filter", flush=True)
        for i in self.filter_flag_map:
            if not self.filter_flag_map[i]:
                self.train_inds_map[i] = self.inds_map[i]
                self.non_train_inds_map[i] = []

        union_train_inds = set([])
        for i in self.train_inds_map:
            union_train_inds.update(self.train_inds_map[i])

        for i in self.non_train_inds_map:
            self.non_train_inds_map[i] = self.non_train_inds_map[i] - union_train_inds


class Top50FilterCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, thresh_map, inds_map, X_train, y_train):
        super().__init__()
        self.thresh_map = thresh_map
        self.inds_map = inds_map
        self.filter_flag_map = {}
        self.train_inds_map = {}
        self.non_train_inds_map = {}
        self.X_train = X_train
        self.true_inds = y_train

        for i in self.thresh_map:
            self.filter_flag_map[i] = False
            self.train_inds_map[i] = []
            self.non_train_inds_map[i] = []

    def on_epoch_end(self, epoch, logs=None):
        # predict on all training data
        # check if the threshold is hit for all labels
        # then stop
        predictions = self.model.predict(self.X_train)
        pred_inds = get_labelinds_from_probs(predictions)

        count = 0
        for lbl in self.filter_flag_map:
            if not self.filter_flag_map[lbl]:
                train_inds, non_train_inds = compute_train_non_train_inds(pred_inds, self.true_inds, self.inds_map, lbl)
                self.train_inds_map[lbl] = train_inds
                self.non_train_inds_map[lbl] = non_train_inds
                if len(train_inds) >= self.thresh_map[lbl]:
                    self.filter_flag_map[lbl] = True
                    count += 1
            else:
                count += 1

        print("Number of labels reached 50 percent threshold", count)
        for i in self.filter_flag_map:
            if not self.filter_flag_map[i]:
                print("For label ", i, " Number expected ", self.thresh_map[i], " Found ", len(self.train_inds_map[i]))
                self.train_inds_map[i] = self.inds_map[i]
                self.non_train_inds_map[i] = []

        temp_flg = True
        for i in self.filter_flag_map:
            temp_flg = temp_flg and self.filter_flag_map[i]

        if temp_flg:
            self.model.stop_training = temp_flg


class ProblematicScoreCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, thresh_map, inds_map, X_train, y_train):
        super().__init__()
        self.match = []
        for i in X_train:
            self.match.append(0)
        self.X_train = X_train
        self.y_train = y_train
        self.thresh_map = thresh_map
        self.inds_map = inds_map
        self.train_inds_map = {}
        self.non_train_inds_map = {}

        for i in self.thresh_map:
            self.train_inds_map[i] = []
            self.non_train_inds_map[i] = []

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.X_train)
        label_inds = get_labelinds_from_probs(predictions)
        for index, pred_ind in enumerate(label_inds):
            if pred_ind == self.y_train[index]:
                self.match[index] += 1

    def on_train_end(self, logs=None):
        print("Executing on_train_end in filter", flush=True)
        self.match = np.array(self.match)
        for lbl in self.thresh_map:
            inds = np.array(self.inds_map[lbl])
            scores = self.match[inds]
            high_scoring_inds = inds[np.argsort(scores)[::-1][:self.thresh_map[lbl]]]
            self.train_inds_map[lbl] = list(high_scoring_inds)
            self.non_train_inds_map[lbl] = list(set(inds) - set(high_scoring_inds))


class PlotCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, correct, wrong):
        super().__init__()
        self.correct = correct
        self.wrong = wrong

    def on_epoch_end(self, epoch, logs=None):
        if len(self.correct["text"]) > 0:
            corr_bs_predictions = self.model.predict(self.correct["text_np"])
            cor_bs_label_inds = get_labelinds_from_probs(corr_bs_predictions)
            for index, pred_ind in enumerate(cor_bs_label_inds):
                if pred_ind == self.correct["pred"][index]:
                    self.correct["stability"][index].append(1)
                    self.correct["match"][index] += 1
                    if self.correct["first_ep"][index] == 0:
                        self.correct["first_ep"][index] = epoch + 1
                else:
                    self.correct["stability"][index].append(0)

        if len(self.wrong["text"]) > 0:
            wrong_bs_predictions = self.model.predict(self.wrong["text_np"])
            wrong_bs_label_inds = get_labelinds_from_probs(wrong_bs_predictions)
            for index, pred_ind in enumerate(wrong_bs_label_inds):
                if pred_ind == self.wrong["pred"][index]:
                    self.wrong["stability"][index].append(1)
                    self.wrong["match"][index] += 1
                    if self.wrong["first_ep"][index] == 0:
                        self.wrong["first_ep"][index] = epoch + 1
                else:
                    self.wrong["stability"][index].append(0)


def filter(X, y_pseudo, y_true, tokenizer, embedding_matrix, percent_thresh, iteration=None, dataset=None):
    inds_map = {}
    for i, j in enumerate(y_pseudo):
        try:
            inds_map[j].append(i)
        except:
            inds_map[j] = [i]

    if iteration == 0:
        if "nyt" in dataset:
            percent_thresh = 1
        if "20news" in dataset:
            percent_thresh = 0.9

    thresh_map = dict(Counter(y_pseudo))
    print("Counts of pseudo-labels ", thresh_map, flush=True)
    for i in thresh_map:
        thresh_map[i] = int(thresh_map[i] * percent_thresh)

    print("Threshold map ", thresh_map, flush=True)

    print("Going to train classifier..")
    max_sentence_length = 100
    max_sentences = 15
    max_words = 20000

    y_one_hot = make_one_hot(y_pseudo)
    print("Fitting tokenizer...")
    print("Splitting into train, dev...")
    X_train, y_train, X_val, y_val = create_train_dev(X, labels=y_one_hot, tokenizer=tokenizer,
                                                      max_sentences=max_sentences,
                                                      max_sentence_length=max_sentence_length,
                                                      max_words=max_words)
    print("Initializing model...")
    model = HAN(max_words=max_sentence_length, max_sentences=max_sentences, output_size=len(y_train[0]),
                embedding_matrix=embedding_matrix)
    print("Compiling model...")
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    print("model fitting - Hierachical attention network...")
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    X_all = prep_data(texts=X, max_sentences=max_sentences, max_sentence_length=max_sentence_length,
                      tokenizer=tokenizer)
    # fc = Top50UnionFilterCallback(thresh_map=thresh_map, inds_map=inds_map, X_train=X_all, y_train=y_pseudo)
    # fc = EpochFilterCallback(X_train=X_all, y_train=y_pseudo, stop_ep=9)
    fc = ProblematicScoreCallback(thresh_map=thresh_map, inds_map=inds_map, X_train=X_all, y_train=y_pseudo)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=256, callbacks=[es, fc])

    train_data = []
    train_labels = []
    true_train_labels = []
    non_train_data = []
    non_train_labels = []
    true_non_train_labels = []

    for lbl in fc.train_inds_map:
        for loop_ind in fc.train_inds_map[lbl]:
            train_data.append(X[loop_ind])
            train_labels.append(y_pseudo[loop_ind])
            true_train_labels.append(y_true[loop_ind])

    for lbl in fc.non_train_inds_map:
        for loop_ind in fc.non_train_inds_map[lbl]:
            non_train_data.append(X[loop_ind])
            non_train_labels.append(y_pseudo[loop_ind])
            true_non_train_labels.append(y_true[loop_ind])

    return train_data, train_labels, true_train_labels, non_train_data, non_train_labels, true_non_train_labels


def get_labelinds_from_probs(predictions):
    pred_inds = []
    for p in predictions:
        pred_inds.append(p.argmax(axis=-1))
    return pred_inds


def get_from_one_hot(pred, index_to_label):
    pred_labels = np.argmax(pred, axis=-1)
    ans = []
    for l in pred_labels:
        ans.append(index_to_label[l])
    return ans


def create_train_dev(texts, labels, tokenizer, max_sentences=15, max_sentence_length=100, max_words=20000):
    data = prep_data(max_sentence_length, max_sentences, texts, tokenizer)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)
    return X_train, y_train, X_test, y_test


def make_one_hot(y):
    n_classes = len(set(y))
    y_new = []
    for label in y:
        current = np.zeros(n_classes)
        current[label] = 1.0
        y_new.append(current)
    y_new = np.asarray(y_new)
    return y_new


def prep_data(max_sentence_length, max_sentences, texts, tokenizer):
    data = np.zeros((len(texts), max_sentences, max_sentence_length), dtype='int32')
    documents = []
    for text in texts:
        sents = tokenize.sent_tokenize(text)
        documents.append(sents)
    for i, sentences in enumerate(documents):
        tokenized_sentences = tokenizer.texts_to_sequences(
            sentences
        )
        tokenized_sentences = pad_sequences(
            tokenized_sentences, maxlen=max_sentence_length
        )

        pad_size = max_sentences - tokenized_sentences.shape[0]

        if pad_size < 0:
            tokenized_sentences = tokenized_sentences[0:max_sentences]
        else:
            tokenized_sentences = np.pad(
                tokenized_sentences, ((0, pad_size), (0, 0)),
                mode='constant', constant_values=0
            )

        data[i] = tokenized_sentences[None, ...]
    return data


def test(model, tokenizer, X_test):
    max_sentence_length = 100
    max_sentences = 15
    X_all = prep_data(texts=X_test, max_sentences=max_sentences, max_sentence_length=max_sentence_length,
                      tokenizer=tokenizer)
    pred = model.predict(X_all)
    return pred


def train_han(X, y, tokenizer, embedding_matrix, correct, wrong, label_dyn=False):
    print("Going to train classifier..")
    max_sentence_length = 100
    max_sentences = 15
    max_words = 20000

    y_one_hot = make_one_hot(y)
    print("Fitting tokenizer...")
    print("Splitting into train, dev...")
    X_train, y_train, X_val, y_val = create_train_dev(X, labels=y_one_hot, tokenizer=tokenizer,
                                                      max_sentences=max_sentences,
                                                      max_sentence_length=max_sentence_length,
                                                      max_words=max_words)
    print("Initializing model...")
    model = HAN(max_words=max_sentence_length, max_sentences=max_sentences, output_size=len(y_train[0]),
                embedding_matrix=embedding_matrix)
    print("Compiling model...")
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    print("model fitting - Hierachical attention network...")
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    if label_dyn:
        correct["text_np"] = prep_data(texts=correct["text"], max_sentences=max_sentences,
                                       max_sentence_length=max_sentence_length, tokenizer=tokenizer)
        wrong["text_np"] = prep_data(texts=wrong["text"], max_sentences=max_sentences,
                                     max_sentence_length=max_sentence_length, tokenizer=tokenizer)
        plt_cb = PlotCallback(correct=correct, wrong=wrong)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=256, callbacks=[plt_cb])
        correct = plt_cb.correct
        wrong = plt_cb.wrong
        correct["match"] = list(np.array(correct["match"]) / 50)
        wrong["match"] = list(np.array(wrong["match"]) / 50)
    else:
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=256, callbacks=[es])
    return model, correct, wrong


if __name__ == "__main__":
    # base_path = "./data/"
    base_path = "/data/dheeraj/WsupLD/data/"
    dataset = sys.argv[3]
    data_path = base_path + dataset + "/"
    use_gpu = int(sys.argv[1])
    gpu_id = int(sys.argv[2])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    df = pickle.load(open(data_path + "df.pkl", "rb"))

    df = df[:1500]

    labels = list(set(df["label"]))
    label_to_index, index_to_label = create_label_index_maps(labels)

    X_all = list(df["text"])
    y_all = list(df["label"])
    y_all_inds = [label_to_index[l] for l in y_all]

    X_train, X_test, y_train, y_test, y_train_inds, y_test_inds = train_test_split(X_all, y_all, y_all_inds,
                                                                                   stratify=y_all, test_size=0.1)

    embedding_matrix = train_word2vec(X_all, data_path)
    tokenizer = fit_get_tokenizer(X_all, data_path, max_words=20000)

    print("Before filtering ", len(X_train))
    X_train, y_train_inds, y_true_inds, non_train_data, non_train_labels, true_non_train_labels = filter(X_train,
                                                                                                         y_train_inds,
                                                                                                         y_train_inds,
                                                                                                         tokenizer,
                                                                                                         embedding_matrix)
    print("After filtering ", len(X_train))

    model = train_han(X_train, y_train_inds, tokenizer, embedding_matrix, None, None)

    predictions = test(model, tokenizer, X_test)
    pred_inds = get_labelinds_from_probs(predictions)
    pred_labels = []
    for p in pred_inds:
        pred_labels.append(index_to_label[p])
    print(classification_report(y_test, pred_labels), flush=True)
    print("*" * 80, flush=True)
