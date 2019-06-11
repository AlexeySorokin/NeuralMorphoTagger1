import ujson as json
import numpy as np

import tensorflow as tf
import keras.backend.tensorflow_backend as kbt

import keras.backend as kb
import keras.layers as kl
import keras.optimizers as kopt
from keras.layers import Layer
from keras import Model
from keras.engine.topology import InputSpec
from keras.callbacks import EarlyStopping

from common.read import read_syntax_infile
from common.vocabulary import Vocabulary, vocabulary_from_json
from common.generate import DataGenerator
from common.common import gather_indexes
from common.cells import BiaffineAttention
from syntax.common import pad_data, load_elmo, make_indexes_for_syntax
from dependency_decoding import chu_liu_edmonds

from deeppavlov import build_model, configs
from deeppavlov.core.common.params import from_params
from deeppavlov.core.commands.utils import parse_config


class PositionEmbedding(Layer):

    def __init__(self, max_length, dim, **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.max_length = max_length
        self.dim = dim
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(self.max_length+1, self.dim),
                                      initializer='glorot_uniform', name='kernel')
        self.built = True

    def call(self, inputs):
        while kb.ndim(inputs) > 2:
            inputs = inputs[...,0]
        positions = kb.cumsum(kb.ones_like(inputs, dtype="int32"), axis=-1) - 1
        positions = kb.maximum(positions, self.max_length)
        answer = kb.gather(self.kernel, positions)
        return answer

    def compute_output_shape(self, input_shape):
        output_shape = tuple(input_shape[:2]) + (self.dim,)
        return output_shape


def load_syntactic_parser(infile):
    with open(infile, "r", encoding="utf8") as fin:
        config = json.load(fin)
    info = {key: config.get(key) for key in ["head_model_params", "dep_model_params",
                                             "head_train_params", "dep_train_params"]}
    embedder = load_elmo()
    parser = SyntacticParser(embedder=embedder, **info)
    parser.dep_vocab = vocabulary_from_json(config["dep_vocab"])
    parser.head_model = parser.build_head_network(**parser.head_model_params)
    parser.dep_model = parser.build_dep_network(**parser.dep_model_params)
    if "head_model_save_file" in config:
        parser.head_model.load_weights(config["head_model_save_file"])
    if "dep_model_save_file" in config:
        parser.dep_model.load_weights(config["dep_model_save_file"])
    return parser


class SyntacticParser:

    def __init__(self, embedder, head_model_params=None, dep_model_params=None,
                 head_train_params=None, dep_train_params=None):
        self.embedder = embedder
        self.head_model_params = head_model_params or dict()
        self.dep_model_params = dep_model_params or dict()
        self.head_train_params = head_train_params or dict()
        self.dep_train_params = dep_train_params or dict()

    def build_head_network(self, use_lstm=True, lstm_size=128, state_size=384, activation="relu"):
        word_inputs = kl.Input(shape=(None, self.embedder.dim), dtype="float32")
        if use_lstm:
            projected_inputs = kl.Bidirectional(kl.LSTM(units=lstm_size, return_sequences=True))(word_inputs)
        else:
            projected_inputs = kl.Dense(256, activation="tanh")(word_inputs)
        position_embeddings = PositionEmbedding(max_length=128, dim=128)(projected_inputs)
        embeddings = kl.Concatenate()([projected_inputs, position_embeddings])
        head_states = kl.Dense(state_size, activation=activation)(embeddings)
        dep_states = kl.Dense(state_size, activation=activation)(embeddings)
        attention = BiaffineAttention(state_size)([head_states, dep_states])
        attention_probs = kl.Softmax()(attention)
        model = Model(word_inputs, attention_probs)
        model.compile(optimizer=kopt.Adam(clipnorm=5.0), loss="categorical_crossentropy", metrics=["accuracy"])
        print(model.summary())
        return model

    def build_dep_network(self, lstm_units=128, state_units=256, dense_units=None):
        dense_units = dense_units or []
        word_inputs = kl.Input(shape=(None, self.embedder.dim), dtype="float32")
        dep_inputs = kl.Input(shape=(None,), dtype="int32")
        head_inputs = kl.Input(shape=(None,), dtype="int32")
        inputs = [word_inputs, dep_inputs, head_inputs]
        if lstm_units > 0:
            word_inputs = kl.Bidirectional(kl.LSTM(lstm_units, return_sequences=True))(word_inputs)
        # dep_inputs = kl.Lambda(kb.expand_dims, output_shape=(lambda x: x + (1,)))(dep_inputs)
        # head_inputs = kl.Lambda(kb.expand_dims, output_shape=(lambda x: x + (1,)))(head_inputs)
        dep_embeddings = kl.Lambda(gather_indexes, arguments={"B": dep_inputs})(word_inputs)
        head_embeddings = kl.Lambda(gather_indexes, arguments={"B": head_inputs})(word_inputs)
        dep_states = kl.Dense(state_units, activation=None)(dep_embeddings)
        dep_states = kl.ReLU()(kl.BatchNormalization()(dep_states))
        head_states = kl.Dense(state_units, activation=None)(head_embeddings)
        head_states = kl.ReLU()(kl.BatchNormalization()(head_states))
        state = kl.Concatenate()([dep_states, head_states])
        for units in dense_units:
            state = kl.Dense(units, activation="relu")(state)
        output = kl.Dense(self.dep_vocab.symbols_number_, activation="softmax")(state)
        model = Model(inputs, output)
        model.compile(optimizer=kopt.Adam(clipnorm=5.0), loss="categorical_crossentropy", metrics=["accuracy"])
        print(model.summary())
        return model

    def train(self, sents, heads, deps, dev_sents=None, dev_heads=None, dev_deps=None):
        sents, heads, deps = pad_data(sents, heads, deps)
        if dev_sents is not None:
            dev_sents, dev_heads, dev_deps = pad_data(dev_sents, dev_heads, dev_deps)
        self.train_head_model(sents, heads, dev_sents, dev_heads, **self.head_train_params)
        self.dep_vocab = Vocabulary(min_count=3).train(deps)
        self.train_dep_model(sents, heads, deps, dev_sents, dev_heads, dev_deps, **self.dep_train_params)
        return self

    def train_head_model(self, sents, heads, dev_sents, dev_heads,
                         nepochs=5, batch_size=16, patience=1):
        self.head_model = self.build_head_network(**self.head_model_params)
        head_gen_params = {"embedder": self.embedder, "classes_number": DataGenerator.POSITIONS_AS_CLASSES}
        train_gen = DataGenerator(sents, heads, **head_gen_params)
        if dev_sents is not None:
            dev_gen = DataGenerator(dev_sents, dev_heads, shuffle=False, **head_gen_params)
            validation_steps = dev_gen.steps_per_epoch
        else:
            dev_gen, validation_steps = None, None
        callbacks = []
        if patience >= 0:
            callbacks.append(EarlyStopping(monitor="val_acc", restore_best_weights=True, patience=patience))
        self.head_model.fit_generator(train_gen, train_gen.steps_per_epoch,
                                      validation_data=dev_gen, validation_steps=validation_steps,
                                      callbacks=callbacks, epochs=nepochs, batch_size=batch_size)
        return self

    def train_dep_model(self, sents, heads, deps, dev_sents, dev_heads, dev_deps,
                        nepochs=2, batch_size=16, patience=1):
        raise NotImplementedError("")

    def predict(self, data):
        data = pad_data(data)
        head_probs, chl_pred_heads = self.predict_heads(data)
        deps = self.predict_deps(data, chl_pred_heads)
        return chl_pred_heads, deps

    def predict_heads(self, data):
        probs, heads = [None] * len(data), [None] * len(data)
        test_gen = DataGenerator(data, embedder=self.embedder,
                                 yield_targets=False, yield_indexes=True, nepochs=1)
        for batch_index, (batch, indexes) in enumerate(test_gen):
            batch_probs = self.head_model.predict(batch)
            for i, index in enumerate(indexes):
                L = len(data[index])
                curr_probs = batch_probs[i][:L - 1, :L - 1]
                curr_probs /= np.sum(curr_probs, axis=-1)
                probs[index] = curr_probs
                heads[index] = np.argmax(curr_probs[1:], axis=-1)
        chl_pred_heads = [chu_liu_edmonds(elem.astype("float64"))[0][1:] for elem in probs]
        return probs, chl_pred_heads

    def predict_deps(self, data, heads):
        dep_indexes, head_indexes = make_indexes_for_syntax(heads)
        generator_params = {"embedder": self.embedder,
                            "additional_padding": [DataGenerator.POSITION_AS_PADDING] * 2}
        test_gen = DataGenerator(data, additional_data=[dep_indexes, head_indexes],
                                 yield_indexes=True, yield_targets=False, shuffle=False,
                                 nepochs=1, **generator_params)
        answer = [None] * len(data)
        for batch, indexes in test_gen:
            batch_probs = self.dep_model.predict(batch)
            batch_labels = np.argmax(batch_probs, axis=-1)
            for i, index in enumerate(indexes):
                L = len(sents[index])
                curr_labels = batch_labels[i][1:L - 1]
                answer[index] = [self.dep_vocab.symbols_[elem] for elem in curr_labels]
        return answer




def evaluate_heads(corr_heads, pred_heads):
    corr, total, corr_sents = 0, 0, 0
    for corr_sent, pred_sent in zip(corr_heads, pred_heads):
        if len(corr_sent) == len(pred_sent) + 2:
            corr_sent = corr_sent[1:-1]
        if len(corr_sent) != len(pred_sent):
            raise ValueError("Different sentence lengths.")
        has_nonequal = False
        for x, y in zip(corr_sent, pred_sent):
            corr += int(x == y)
            has_nonequal |= (x != y)
        corr_sents += 1 - int(has_nonequal)
        total += len(corr_sent)
    return corr, total, corr / total, corr_sents, len(corr_heads), corr_sents / len(corr_heads)


if __name__ == "__main__":
    parser = load_syntactic_parser("syntax/config/config_load_basic.json")
    test_infile = "/home/alexeysorokin/data/Data/UD2.3/UD_Russian-SynTagRus/ru_syntagrus-ud-test.conllu"
    sents, heads, deps = read_syntax_infile(test_infile, to_process_word=False)
    pred_heads, pred_deps = parser.predict(sents)
    print(evaluate_heads(heads, pred_heads))



