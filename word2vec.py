from gensim.models import word2vec
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle


def fit_get_tokenizer(data, dataset_path, max_words):
    try:
        tokenizer = pickle.load(open(dataset_path + "tokenizer.pkl", "rb"))
    except:
        tokenizer = Tokenizer(num_words=max_words, filters='!"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(data)
        pickle.dump(tokenizer, open(dataset_path + "tokenizer.pkl", "wb"))
    return tokenizer


def train_word2vec(texts, dataset_path):
    def get_embeddings(inp_data, vocabulary_inv, size_features=100,
                       mode='skipgram',
                       min_word_count=2,
                       context=5):
        num_workers = 15  # Number of threads to run in parallel
        downsampling = 1e-3  # Downsample setting for frequent words
        print('Training Word2Vec model...')
        sentences = [[vocabulary_inv[w] for w in s] for s in inp_data]
        if mode == 'skipgram':
            sg = 1
            print('Model: skip-gram')
        elif mode == 'cbow':
            sg = 0
            print('Model: CBOW')
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers,
                                            sg=sg,
                                            vector_size=size_features,
                                            min_count=min_word_count,
                                            window=context,
                                            sample=downsampling)
        embedding_model.init_sims(replace=True)
        embedding_weights = np.zeros((len(vocabulary_inv) + 1, size_features))
        embedding_weights[0] = 0
        for i, word in vocabulary_inv.items():
            if word in embedding_model:
                embedding_weights[i] = embedding_model[word]
            else:
                embedding_weights[i] = np.random.uniform(-0.25, 0.25, embedding_model.vector_size)

        return embedding_weights

    try:
        embedding_mat = pickle.load(open(dataset_path + "embedding_matrix.pkl", "rb"))
    except:
        tokenizer = fit_get_tokenizer(texts, dataset_path, max_words=150000)
        print("Total number of words: ", len(tokenizer.word_index))
        tagged_data = tokenizer.texts_to_sequences(texts)
        vocabulary_inv = {}
        for word in tokenizer.word_index:
            vocabulary_inv[tokenizer.word_index[word]] = word
        embedding_mat = get_embeddings(tagged_data, vocabulary_inv)
        pickle.dump(embedding_mat, open(dataset_path + "embedding_matrix.pkl", "wb"))
    return embedding_mat
