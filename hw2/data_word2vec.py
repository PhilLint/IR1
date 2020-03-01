import collections
import os
import pickle
import random
from io import open
import numpy as np
import tqdm as tqdm
import torch


def get_dataset(docs, min_freq):
    """ Returns word frequencies, id2word and word2id dicts
    Args:
        docs (dict): Dictionary 'docs' returned from read_ap.get_processed_docs()
        min_freq (int): Minimum frequency of words. Assignment sheet says 50
    Returns:
        data (list): All indices
        counter (dict): (word, freq) as (key, value) pairs
        word2id (dict): (word, id) as (key, value) pairs
        id2word (dict): (id, word) as (key, value) pairs
    """

    doc_ids = list(docs.keys())
    word_freq = collections.Counter()
    print("Building Corpus For word2vec")
    # count all words over all docs
    for doc_id in doc_ids:
        doc = docs[doc_id]
        word_freq = word_freq + collections.Counter(doc)
    word_freq = word_freq.most_common()
    word_freq = dict(word_freq)
    # remove words with less than min_freq mentions
    word_freq = dict((word, v) for word, v in word_freq.items() if v > min_freq)
    word2id = {}
    # build dictionaries
    for word, _ in word_freq.items():
        word2id[word] = len(word2id)
    index = []
    indices = []

    for i, doc_id in enumerate(doc_ids):
        doc = docs[doc_id]
        for word in doc:
            try:
                index.append(word2id[word])

            except:
                continue
        indices.append(index)

    id2word = dict(zip(word2id.values(), word2id.keys()))
    return [indices, word_freq, word2id, id2word]


def load_dataset(docs, min_freq, doc_set_name="indices"):
    path = f"./datasets/{doc_set_name}.list"

    if not os.path.exists(path):
        print("Creating word2vec dataset")
        dataset = get_dataset(docs, min_freq)
        print("Saving as list file")
        save_dataset(dataset)
        indices, word_freq, word2id, id2word = dataset[0], dataset[1], dataset[2], dataset[3]
    else:
        print("Dataset already saved. Loading from disk")
        indices, word_freq, word2id, id2word = read_dataset()

    return indices, word_freq, word2id, id2word


def save_dataset(dataset):
    """ Save returns of get_dataset to save time for full dataset
    Args:
        data:
        count:
        dictionary:
        reversed_dictionary:
    """
    dataset_names = ["indices", "word_freq", "word2id", "id2word"]

    for i, element in enumerate(dataset):
        pickle.dump(dataset[i], open("datasets/" + dataset_names[i] + ".list", "wb"))


def read_dataset():
    """ Reads previously saved dataset files
    Returns:
        data:
        count:
        dictionary:
        reversed_dictionary:
    """
    dataset_names = ["indices", "word_freq", "word2id", "id2word"]
    indices, word_freq, word2id, id2word = [], {}, {}, {}

    for i, file_name in enumerate(dataset_names):
        file = pickle.load(open("datasets/" + file_name + ".list", "rb"))
        if file_name == "indices":
            indices = file
        elif file_name == "word_freq":
            word_freq = file
        elif file_name == "word2id":
            word2id = file
        elif file_name == "id2word":
            id2word = file

    return indices, word_freq, word2id, id2word


class Dataloader:
    def __init__(self, data_indices, word_count, word2id, device="cpu"):

        self.data_indices = data_indices
        self.data_index = 0
        self.document_index = 0
        self.word_count = word_count
        self.word2id = word2id
        self.device = device
        self.noise_table = self.get_noise_distribution()

    def get_noise_distribution(self):
        """ Returns the noise distribution needed for NCE loss
        As in 2.2 Negative Sampling of 'Distributed Representations of Words and Phrases and their Compositionality'
        Args:
            word_count (dict):
            word2id (dict):

        Returns:
            unigram_table (list)
        """
        Z = 0.001
        sample_table = []
        num_total_words = sum([c for w, c in self.word_count.items()])
        for k, v in self.word2id.items():
            unigram_dist = (self.word_count[k] / num_total_words) ** (3 / 4)
            sample_table += [v] * int(unigram_dist / Z)
        return sample_table

    def generate_batch(self, window_size, batch_size, num_neg_samples):

        context_size = 2 * window_size + 1
        context = np.ndarray(shape=(batch_size, 2 * window_size), dtype=np.int64)
        labels = np.ndarray(shape=(batch_size), dtype=np.int64)
        print("doc: {}".format(self.data_indices[self.document_index]))
        if self.data_index + context_size > len(self.data_indices[self.document_index]):
            self.data_index = 0
            self.document_index += 1

        pos_u = []
        pos_v = []

        doc = self.data_indices[self.document_index]
        sliding_window = doc[self.data_index : self.data_index + context_size]

        for i in range(batch_size):
            self.data_index += 1
            # extract all context words
            context[i, :] = sliding_window[:window_size] + sliding_window[window_size + 1:]
            labels[i] = sliding_window[window_size]

            if self.data_index + context_size > len(doc):
                sliding_window[:] = doc[:context_size]
                self.data_index = 0
                self.document_index += 1
                doc = self.data_indices[self.document_index]
                sliding_window = doc[self.data_index:self.data_index + context_size]

            else:
                sliding_window = doc[self.data_index:self.data_index + context_size]

            for j in range(context_size - 1):
                pos_u.append(labels[i])
                pos_v.append(context[i, j])

        neg_v = np.random.choice(self.noise_table, size=(batch_size * 2 * window_size, num_neg_samples))

        pos_u = torch.LongTensor(pos_u).to(self.device)
        pos_v = torch.LongTensor(pos_v).to(self.device)
        neg_v = torch.LongTensor(neg_v).to(self.device)

        return pos_u, pos_v, neg_v
