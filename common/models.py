import numpy as np

import keras.layers as kl
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from common.common import *
from common.generate import DataGenerator
from common.cells import build_word_cnn
from common.vocabulary import Vocabulary

class EmbeddingsLearner:

    NEPOCHS, BATCH_SIZE, PATIENCE = 50, 16, 5

    def __init__(self, vocabulary=None, min_length=-1, max_length=30,
                 validation_split=0.2, char_embeddings_size=32, dropout=0.0,
                 use_lstm=False, lstm_units=128,
                 layers=1, conv_windows=5, conv_filters=128,
                 char_filter_multiple=25, highway_layers=1,
                 dense_layers=0, dense_units=128,
                 train_params=None, random_state=189):
        self.min_length = min_length
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.validation_split = validation_split
        self.char_embeddings_size = char_embeddings_size
        self.use_lstm = use_lstm
        self.lstm_units = lstm_units
        self.layers = layers
        self.conv_windows = conv_windows
        self.conv_filters = conv_filters
        self.char_filter_multiple = char_filter_multiple
        self.highway_layers = highway_layers
        self.dense_layers = dense_layers
        self.dense_units = dense_units
        self.dropout = dropout
        self.train_params = train_params or dict()
        for key in ["nepochs", "batch_size", "patience"]:
            if key not in self.train_params:
                self.train_params[key] = getattr(self, key.upper())
        self.random_state = random_state

    def _recode(self, word):
        return [BEGIN] + [self.vocabulary.toidx(x) for x in word] + [END]

    def train(self, embedder):
        self.dim = embedder.dim
        words = [x for x in embedder if len(x) >= self.min_length and len(x) <= self.max_length]
        embeddings = np.array([embedder._get_word_vector(x) for x in words])
        if self.vocabulary is None:
            self.vocabulary = Vocabulary(min_count=3).train(words, from_sentences=False)
        data = [self._recode(x) for x in words]
        # preparing data
        indexes = np.random.permutation(len(words))
        L = int((1.0 - self.validation_split) * len(words))
        train_indexes, dev_indexes = indexes[:L], indexes[L:]
        train_data, train_targets = [data[i] for i in train_indexes], embeddings[train_indexes]
        dev_data, dev_targets = [data[i] for i in dev_indexes], embeddings[dev_indexes]
        # training the model
        self.model = self.build()
        train_gen = DataGenerator(train_data, train_targets, shuffle=True,
                                  batch_size=self.train_params["batch_size"])
        dev_gen = DataGenerator(dev_data, dev_targets, batch_size=self.train_params["batch_size"])
        if self.train_params["patience"] >= 0:
            callbacks = [EarlyStopping(patience=self.train_params["patience"], restore_best_weights=True)]
        else:
            callbacks = []
        self.model.fit_generator(train_gen, steps_per_epoch=train_gen.steps_per_epoch,
                                 validation_data=dev_gen, validation_steps=dev_gen.steps_per_epoch,
                                 epochs=self.train_params["nepochs"], callbacks=callbacks)
        return self

    def build(self):
        inputs = kl.Input(shape=(None,), dtype="int32")
        if self.use_lstm:
            embeddings = kl.Embedding(self.vocabulary.symbols_number_, self.char_embeddings_size)(inputs)
            raise NotImplementedError("")
        else:
            outputs = build_word_cnn(inputs, symbols_number=self.vocabulary.symbols_number_,
                                     char_embeddings_size=self.char_embeddings_size,
                                     char_window_size=self.conv_windows, char_filters=self.conv_filters,
                                     char_filter_multiple=self.char_filter_multiple,
                                     highway_layers=self.highway_layers, dropout=self.dropout,
                                     from_one_hot=False)
        for _ in range(self.dense_layers):
            outputs = kl.Dense(self.dense_units, activation="tanh")(outputs)
        output_embeddings = kl.Dense(self.dim, name="output_embeddings")(outputs)
        model = Model(inputs, output_embeddings)
        model.compile(optimizer=Adam(clipnorm=5.0), loss="mean_squared_error")
        print(model.summary())
        return model
