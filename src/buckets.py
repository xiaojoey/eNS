import numpy as np
from tensorflow.keras import utils
import math
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SequenceBuckets(utils.Sequence):
    """
    Sequence bucket padding
    Args:
        sequences - (list) A list of sequences of tokens
        choose_length - (function) A function for choosing length from numpy arrays
        other_feature - (list) A list of tensors with other features
        labels - (np.array) labels
        indices - (np.array) Numpy array of indices
        shuffle - (bool) To shuffle or not
        batch_size - (int) batch size of the samples
    """
    def __init__(self, sequences, choose_length, other_features=None,
    labels = None, indices=None, shuffle=False, batch_size=512):
        super(SequenceBuckets, self).__init__()
        self.sequences = np.array(sequences, dtype=object)
        self.lengths = np.array([len(x) for x in sequences])
        self.n_samples = len(sequences)
        self.choose_length = choose_length
        self.other_features = other_features
        self.labels = labels

        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(len(sequences))

        self.batch_size = batch_size
        self.shuffle = shuffle

        if self.shuffle:
            self._shuffle()

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def _shuffle(self):
        self.indices = np.random.permutation(self.indices)

    def __getitem__(self, i):
        idx = self.indices[(self.batch_size * i):(self.batch_size * (i + 1))]

        if self.shuffle and i == len(self) - 1:
            self._shuffle()

        pad_length = math.ceil(self.choose_length(self.lengths[idx]))
        padded_sequences = pad_sequences(self.sequences[idx], maxlen=pad_length, padding='post', truncating='post')

        x_batch = padded_sequences

        if self.other_features is not None:
            x_batch += [x[idx] for x in self.other_features]

        if self.labels is not None:
            out = x_batch, self.labels[idx]
        else:
            out = x_batch
        return out
