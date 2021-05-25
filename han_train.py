from keras_han.model import HAN
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
from nltk import tokenize
from keras.preprocessing.sequence import pad_sequences
import pickle
import sys
from util import create_label_index_maps, get_labelinds_from_probs
from sklearn.metrics import classification_report
from word2vec import train_word2vec, fit_get_tokenizer


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


def make_one_hot(y, label_to_index):
    labels = list(label_to_index.keys())
    n_classes = len(labels)
    y_new = []
    for label in y:
        current = np.zeros(n_classes)
        i = label_to_index[label]
        current[i] = 1.0
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


def train_han(X, y, label_to_index, tokenizer, embedding_matrix):
    print("Going to train classifier..")
    max_sentence_length = 100
    max_sentences = 15
    max_words = 20000

    y_one_hot = make_one_hot(y, label_to_index)
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
    model.fit(X_train, y_train, validation_data=(X_val, y_val), nb_epoch=100, batch_size=256, callbacks=[es])
    return model


# def filter_han(X, y_pseudo, y_true, device):


if __name__ == "__main__":
    # base_path = "./data/"
    base_path = "/data/dheeraj/WsupLD/data/"
    dataset = sys.argv[3]
    data_path = base_path + dataset + "/"
    use_gpu = int(sys.argv[1])
    gpu_id = int(sys.argv[2])

    df = pickle.load(open(data_path + "df.pkl", "rb"))

    labels = list(set(df["label"]))
    label_to_index, index_to_label = create_label_index_maps(labels)

    X_all = list(df["text"])
    y_all = list(df["label"])
    y_all_inds = [label_to_index[l] for l in y_all]

    X_train, X_test, y_train, y_test, y_train_inds, y_test_inds = train_test_split(X_all, y_all, y_all_inds,
                                                                                   stratify=y_all, test_size=0.1)

    embedding_matrix = train_word2vec(X_all, data_path)
    tokenizer = fit_get_tokenizer(X_all, data_path, max_words=20000)
    model = train_han(X_train, y_train_inds, label_to_index, tokenizer, embedding_matrix)

    predictions = test(model, tokenizer, X_test)
    pred_inds = get_labelinds_from_probs(predictions)
    pred_labels = []
    for p in pred_inds:
        pred_labels.append(index_to_label[p])
    print(classification_report(y_test, pred_labels), flush=True)
    print("*" * 80, flush=True)
