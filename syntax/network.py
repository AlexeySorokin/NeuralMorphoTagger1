import os
import ujson as json
import numpy as np
import inspect

import tensorflow as tf
import keras.backend.tensorflow_backend as kbt
import keras.backend as kb
import keras.layers as kl
import keras.optimizers as kopt
from keras.layers import Layer
from keras import Model
from keras.engine.topology import InputSpec
from keras.callbacks import EarlyStopping

from common.read import read_syntax_infile, process_word, make_UD_pos_and_tag
from common.vocabulary import Vocabulary, FeatureVocabulary, vocabulary_from_json
from common.generate import DataGenerator
from common.common import BEGIN, END, PAD
from common.common import gather_indexes
from common.cells import BiaffineAttention, BiaffineLayer, build_word_cnn
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
    parser = StrangeSyntacticParser(embedder=embedder, **info)
    parser.dep_vocabulary_ = vocabulary_from_json(config["dep_vocab"])
    parser.head_model = parser.build_head_network(**parser.head_model_params)
    parser.dep_model = parser.build_dep_network(**parser.dep_model_params)
    if "head_model_save_file" in config:
        parser.head_model.load_weights(config["head_model_save_file"])
    if "dep_model_save_file" in config:
        parser.dep_model.load_weights(config["dep_model_save_file"])
    return parser


def load_parser(infile):
    with open(infile, "r", encoding="utf8") as fin:
        config = json.load(fin)
    embedder = config.pop("embedder", None)
    if embedder is not None:
        embedder = load_elmo()
    config["embedder"] = embedder
    vocabulary_keys = [key for key in config if key.endswith("vocabulary_")]
    vocab_config = {key: vocabulary_from_json(
                        config.pop(key), use_features=(key=="tag_vocabulary_"))
                    for key in vocabulary_keys}
    model_config = {key: config.pop(key) for key in ["model_file", "head_model_file", "dep_model_file"]
                    if key in config}
    parser = StrangeSyntacticParser(**config)
    for key, value in vocab_config.items():
        setattr(parser, key, value)
    if parser.use_joint_model:
        parser.model_, parser.head_model_, parser.dep_model_ = parser.build_Dozat_network(**parser.model_params)
    else:
        parser.head_model_ = parser.build_head_network(**parser.head_model_params)
        parser.dep_model_ = parser.build_dep_network(**parser.dep_model_params)
    for key, weights_file in model_config.items():
        model_key = key[:-4]
        model = getattr(parser, model_key)
        weights_file = os.path.join(os.path.dirname(infile), weights_file)
        model.load_weights(weights_file)
    return parser


class StrangeSyntacticParser:

    MAX_WORD_LENGTH = 30

    def __init__(self, use_joint_model=True, embedder=None, use_tags=False,
                 use_char_model=False, max_word_length=MAX_WORD_LENGTH,
                 head_model_params=None, dep_model_params=None,
                 head_train_params=None, dep_train_params=None,
                 model_params=None, train_params=None,
                 char_layer_params=None):
        self.use_joint_model = use_joint_model
        self.embedder = embedder
        self.use_tags = use_tags
        self.use_char_model = use_char_model
        self.max_word_length = max_word_length
        self.head_model_params = head_model_params or dict()
        self.dep_model_params = dep_model_params or dict()
        self.head_train_params = head_train_params or dict()
        self.dep_train_params = dep_train_params or dict()
        self.model_params = model_params or dict()
        self.train_params = train_params or dict()
        self.char_layer_params = char_layer_params or dict()
        self.head_model_params["char_layer_params"] = self.char_layer_params
        self.dep_model_params["char_layer_params"] = self.char_layer_params
        if self.embedder is None and not self.use_char_model and not self.use_tags:
            raise ValueError("")

    def to_json(self, outfile, model_file=None, head_model_file=None, dep_model_file=None):
        info = dict()
        outdir = os.path.dirname(outfile)
        for (attr, val) in inspect.getmembers(self):
            print(attr)
            if not (attr.startswith("__") or inspect.ismethod(val) or
                    isinstance(getattr(StrangeSyntacticParser, attr, None), property) or
                    isinstance(val, np.ndarray) or
                    isinstance(val, Vocabulary) or attr.isupper() or
                    attr.endswith("model_") or attr == "embedder"):
                info[attr] = val
            elif isinstance(val, Vocabulary):
                info[attr] = val.jsonize()
            elif isinstance(val, np.ndarray):
                val = val.tolist()
                info[attr] = val
            elif attr == "embedder":
                if val is not None:
                    info[attr] = "elmo"
        for key in ["model_", "head_model_", "dep_model_"]:
            keyfile = locals()[key + "file"]
            if keyfile is not None:
                info[key + "file"] = os.path.relpath(os.path.abspath(keyfile), outdir)
                model = getattr(self, key)
                model.save_weights(keyfile)
        with open(outfile, "w", encoding="utf8") as fout:
            json.dump(info, fout)
        return

    @property
    def first_input_index(self):
        return 0 if self.embedder is not None else 1 if self.use_char_model else 2

    @property
    def active_input_indexes(self):
        inputs = [self.embedder is not None, self.use_char_model, self.use_tags]
        answer = [i for i, x in enumerate(inputs) if x]
        return answer

    def _initialize_position_embeddings(self):
        new_weight = np.eye(128, dtype="float") / np.sqrt(128)
        new_weight = np.concatenate([new_weight, [[0] * 128]], axis=0)
        layer: kl.Layer = self.head_model_.get_layer(name="pos_embeddings")
        layer.set_weights([new_weight])
        return

    def build_head_network(self, use_lstm=True, lstm_size=128, state_size=384,
                           activation="relu", char_layer_params=None,
                           tag_embeddings_size=64):
        if self.embedder is not None:
            word_inputs = kl.Input(shape=(None, self.embedder.dim), dtype="float32")
            word_embeddings = word_inputs
        else:
            word_inputs = kl.Input(shape=(None, self.max_word_length + 2), dtype="float32")
            char_layer_params = char_layer_params or dict()
            word_embeddings = build_word_cnn(word_inputs, from_one_hot=False,
                                             symbols_number=self.symbol_vocabulary_.symbols_number_,
                                             char_embeddings_size=32, **char_layer_params)
        inputs = [word_inputs]
        if self.use_tags:
            tag_inputs = kl.Input(shape=(None, self.tag_vocabulary_.symbol_vector_size_), dtype="float32")
            tag_embeddings = kl.Dense(tag_embeddings_size, activation="relu")(tag_inputs)
            inputs.append(tag_inputs)
            word_embeddings = kl.Concatenate()([word_embeddings, tag_embeddings])
        if use_lstm:
            projected_inputs = kl.Bidirectional(kl.LSTM(units=lstm_size, return_sequences=True))(word_embeddings)
        else:
            projected_inputs = kl.Dense(256, activation="tanh")(word_embeddings)
        position_embeddings = PositionEmbedding(max_length=128, dim=128, name="pos_embeddings")(projected_inputs)
        embeddings = kl.Concatenate()([projected_inputs, position_embeddings])
        head_states = kl.Dense(state_size, activation=activation)(embeddings)
        dep_states = kl.Dense(state_size, activation=activation)(embeddings)
        attention = BiaffineAttention(state_size)([head_states, dep_states])
        attention_probs = kl.Softmax()(attention)
        model = Model(inputs, attention_probs)
        model.compile(optimizer=kopt.Adam(clipnorm=5.0), loss="categorical_crossentropy", metrics=["accuracy"])
        print(model.summary())
        return model

    def build_dep_network(self, lstm_units=128, state_units=256, dense_units=None,
                          tag_embeddings_size=64, char_layer_params=None):
        dense_units = dense_units or []
        char_layer_params = char_layer_params or dict()
        if self.embedder is not None:
            word_inputs = kl.Input(shape=(None, self.embedder.dim), dtype="float32")
            word_embeddings = word_inputs
        else:
            word_inputs = kl.Input(shape=(None, self.max_word_length + 2), dtype="float32")
            char_layer_params = char_layer_params or dict()
            word_embeddings = build_word_cnn(word_inputs, from_one_hot=False,
                                             symbols_number=self.symbol_vocabulary_.symbols_number_,
                                             char_embeddings_size=32, **char_layer_params)
        dep_inputs = kl.Input(shape=(None,), dtype="int32")
        head_inputs = kl.Input(shape=(None,), dtype="int32")
        inputs = [word_inputs, dep_inputs, head_inputs]
        if self.use_tags:
            tag_inputs = kl.Input(shape=(None, self.tag_vocabulary_.symbol_vector_size_), dtype="float32")
            tag_embeddings = kl.Dense(tag_embeddings_size, activation="relu")(tag_inputs)
            inputs.append(tag_inputs)
            word_embeddings = kl.Concatenate()([word_embeddings, tag_embeddings])
        if lstm_units > 0:
            word_embeddings = kl.Bidirectional(kl.LSTM(lstm_units, return_sequences=True))(word_embeddings)
        dep_embeddings = kl.Lambda(gather_indexes, arguments={"B": dep_inputs})(word_embeddings)
        head_embeddings = kl.Lambda(gather_indexes, arguments={"B": head_inputs})(word_embeddings)
        dep_states = kl.Dense(state_units, activation=None)(dep_embeddings)
        dep_states = kl.ReLU()(kl.BatchNormalization()(dep_states))
        head_states = kl.Dense(state_units, activation=None)(head_embeddings)
        head_states = kl.ReLU()(kl.BatchNormalization()(head_states))
        state = kl.Concatenate()([dep_states, head_states])
        for units in dense_units:
            state = kl.Dense(units, activation="relu")(state)
        output = kl.Dense(self.dep_vocabulary_.symbols_number_, activation="softmax")(state)
        model = Model(inputs, output)
        model.compile(optimizer=kopt.Adam(clipnorm=5.0), loss="categorical_crossentropy", metrics=["accuracy"])
        print(model.summary())
        return model

    def build_Dozat_network(self, state_units=256, dropout=0.2,
                            lstm_layers=1, lstm_size=128, lstm_dropout=0.2,
                            char_layer_params=None, tag_embeddings_size=64):
        inputs, embeddings = [], []
        if self.embedder is not None:
            word_inputs = kl.Input(shape=(None, self.embedder.dim), dtype="float32")
            inputs.append(word_inputs)
            embeddings.append(word_inputs)
        if self.use_char_model:
            char_inputs = kl.Input(shape=(None, self.max_word_length + 2), dtype="float32")
            char_layer_params = char_layer_params or dict()
            word_embeddings = build_word_cnn(char_inputs, from_one_hot=False,
                                             symbols_number=self.symbol_vocabulary_.symbols_number_,
                                             char_embeddings_size=32, **char_layer_params)
            embeddings.append(word_embeddings)
        if self.use_tags:
            tag_inputs = kl.Input(shape=(None, self.tag_vocabulary_.symbol_vector_size_), dtype="float32")
            tag_embeddings = kl.Dense(tag_embeddings_size, activation="relu")(tag_inputs)
            inputs.append(tag_inputs)
            embeddings.append(tag_embeddings)
        embeddings = kl.Concatenate()(embeddings) if len(embeddings) > 1 else embeddings[0]
        lstm_input = embeddings
        for i in range(lstm_layers-1):
            lstm_layer = kl.Bidirectional(kl.LSTM(lstm_size, dropout=lstm_dropout, return_sequences=True))
            lstm_input = lstm_layer(lstm_input)
        lstm_layer = kl.Bidirectional(kl.LSTM(lstm_size, dropout=lstm_dropout, return_sequences=True))
        lstm_output = lstm_layer(embeddings)
        # selecting each word head
        head_encodings = kl.Dropout(dropout)(kl.Dense(state_units, activation="relu")(lstm_output))
        dep_encodings = kl.Dropout(dropout)(kl.Dense(state_units, activation="relu")(lstm_output))
        head_similarities = BiaffineAttention(state_units, use_first_bias=True)([dep_encodings, head_encodings])
        head_probs = kl.Softmax(name="heads", axis=-1)(head_similarities)
        # selecting each word dependency type (with gold heads)
        dep_inputs = kl.Input(shape=(None,), dtype="int32")
        head_inputs = kl.Input(shape=(None,), dtype="int32")
        inputs.extend([dep_inputs, head_inputs])
        dep_embeddings = kl.Lambda(gather_indexes, arguments={"B": dep_inputs})(lstm_output)
        head_embeddings = kl.Lambda(gather_indexes, arguments={"B": head_inputs})(lstm_output)
        dep_encodings = kl.Dropout(dropout)(kl.Dense(state_units, activation="relu")(dep_embeddings))
        head_encodings = kl.Dropout(dropout)(kl.Dense(state_units, activation="relu")(head_embeddings))
        dep_probs = BiaffineLayer(self.dep_vocabulary_.symbols_number_, state_units,
                                  name="deps", use_first_bias=True, use_second_bias=True,
                                  use_label_bias=True, activation="softmax")([dep_encodings, head_encodings])
        outputs = [head_probs, dep_probs]
        model = Model(inputs, outputs)
        model.compile(optimizer=kopt.Adam(clipnorm=5.0), loss=["categorical_crossentropy"] * 2,
                      loss_weights=[1.0, 1.0], metrics=["accuracy"])
        head_model = kb.Function(inputs[:-2] + [kb.learning_phase()], [head_probs])
        dep_model = kb.Function(inputs + [kb.learning_phase()], [dep_probs])
        print(model.summary())
        return model, head_model, dep_model

    def _recode(self, sent):
        if isinstance(sent[0], str):
            sent, from_word = [sent], True
        else:
            from_word = False
        answer = np.full(shape=(len(sent), self.max_word_length+2), fill_value=PAD, dtype="int32")
        for i, word in enumerate(sent):
            word = word[-self.max_word_length:]
            answer[i, 0], answer[i, len(word) + 1] = BEGIN, END
            answer[i, 1:len(word) + 1] = self.symbol_vocabulary_.toidx(word)
        return answer[0] if from_word else answer

    def _transform_data(self, sents, to_train=False):
        sents = [[process_word(word, to_lower=True, append_case="first",
                               special_tokens=["<s>", "</s>"]) for word in sent] for sent in sents]
        if to_train:
            self.symbol_vocabulary_ = Vocabulary(character=True, min_count=3).train(sents)
        sents = [self._recode(sent) for sent in sents]
        return sents

    def _transform_tags(self, sents, to_train=False):
        sents = [['BEGIN'] + sent + ['END'] for sent in sents]
        if to_train:
            self.tag_vocabulary_ = FeatureVocabulary(min_count=3).train(sents)
        answer = [[self.tag_vocabulary_.to_vector(x, return_vector=True) for x in sent] for sent in sents]
        return answer

    def train(self, sents, heads, deps, dev_sents=None, dev_heads=None, dev_deps=None,
              tags=None, dev_tags=None, save_file=None, model_file=None,
              head_model_file=None, dep_model_file=None):
        sents, heads, deps = pad_data(sents, heads, deps)
        if self.use_char_model:
            sent_data = self._transform_data(sents, to_train=True)
        else:
            sent_data = sents
        if tags is not None:
            tag_data = self._transform_tags(tags, to_train=True)
        else:
            tag_data = None
        if dev_sents is not None:
            dev_sents, dev_heads, dev_deps = pad_data(dev_sents, dev_heads, dev_deps)
            dev_sent_data = self._transform_data(dev_sents) if self.use_char_model else dev_sents
            dev_tag_data = self._transform_tags(dev_tags) if self.use_tags else None
        else:
            dev_sent_data, dev_heads, dev_deps, dev_tag_data = None, None, None, None
        self.dep_vocabulary_ = Vocabulary(min_count=3).train(deps)
        if self.use_joint_model:
            self.train_joint_model(sents, sent_data, heads, deps,
                                   dev_sents, dev_sent_data, dev_heads, dev_deps,
                                   tags=tag_data, dev_tags=dev_tag_data, **self.train_params)
        else:
            self.train_head_model(sent_data, heads, dev_sent_data, dev_heads,
                                  tags=tag_data, dev_tags=dev_tag_data, **self.head_train_params)
            self.train_dep_model(sent_data, heads, deps, dev_sent_data, dev_heads, dev_deps,
                                 tags=tag_data, dev_tags=dev_tag_data, **self.dep_train_params)
        if save_file is not None:
            self.to_json(save_file, model_file, head_model_file, dep_model_file)
        return self

    def train_joint_model(self, sent_data, sents, heads, deps,
                          dev_sent_data, dev_sents, dev_heads, dev_deps,
                          tags=None, dev_tags=None,
                          nepochs=3, batch_size=16, patience=1):
        self.model_, self.head_model_, self.dep_model_ = self.build_Dozat_network(**self.model_params)
        gen_params = {"embedder": self.embedder, "batch_size": batch_size,
                      "classes_number": DataGenerator.POSITIONS_AS_CLASSES,
                      "additional_classes_number": [self.dep_vocabulary_.symbols_number_],
                      "additional_target_paddings": [PAD]}
        dep_indexes, head_indexes, dep_codes = \
            make_indexes_for_syntax(heads, deps, dep_vocab=self.dep_vocabulary_, to_pad=False)
        # все входные данные
        data = [sent_data, sents, tags, dep_indexes, head_indexes]
        paddings = [PAD] * 3 + [DataGenerator.POSITION_AS_PADDING] * 2
        additional_input_indexes = self.active_input_indexes[1:] + [3, 4]
        # используемые входные данные
        additional_data = [data[i] for i in additional_input_indexes]
        gen_params["additional_padding"] = [paddings[i] for i in additional_input_indexes]
        train_gen = DataGenerator(data[self.first_input_index], targets=heads,
                                  additional_data=additional_data,  additional_targets=[dep_codes],
                                  **gen_params)
        if dev_sent_data is not None:
            dev_dep_indexes, dev_head_indexes, dev_dep_codes = \
                make_indexes_for_syntax(dev_heads, dev_deps, dep_vocab=self.dep_vocabulary_, to_pad=False)
            dev_data = [dev_sent_data, dev_sents, dev_tags, dev_dep_indexes, dev_head_indexes]
            additional_dev_data = [dev_data[i] for i in additional_input_indexes]
            dev_gen = DataGenerator(dev_data[self.first_input_index], targets=dev_heads,
                                    additional_data=additional_dev_data,
                                    additional_targets=[dev_dep_codes], shuffle=False, **gen_params)
            validation_steps = dev_gen.steps_per_epoch
        else:
            dev_gen, validation_steps = None, None
        callbacks = []
        if patience >= 0:
            callbacks.append(EarlyStopping(monitor="val_heads_acc", restore_best_weights=True, patience=patience))
        self.model_.fit_generator(train_gen, train_gen.steps_per_epoch,
                                  validation_data=dev_gen,
                                  validation_steps=validation_steps,
                                  callbacks=callbacks, epochs=nepochs)
        return self

    def train_head_model(self, sents, heads, dev_sents, dev_heads, tags=None, dev_tags=None,
                         nepochs=5, batch_size=16, patience=1):
        self.head_model_ = self.build_head_network(**self.head_model_params)
        # self._initialize_position_embeddings()
        head_gen_params = {"embedder": self.embedder, "batch_size": batch_size,
                           "classes_number": DataGenerator.POSITIONS_AS_CLASSES}
        additional_data = [tags] if self.use_tags else None
        train_gen = DataGenerator(sents, heads, additional_data=additional_data, **head_gen_params)
        if dev_sents is not None:
            additional_data = [dev_tags] if self.use_tags else None
            dev_gen = DataGenerator(dev_sents, dev_heads, additional_data=additional_data,
                                    shuffle=False, **head_gen_params)
            validation_steps = dev_gen.steps_per_epoch
        else:
            dev_gen, validation_steps = None, None
        callbacks = []
        if patience >= 0:
            callbacks.append(EarlyStopping(monitor="val_acc", restore_best_weights=True, patience=patience))
        self.head_model_.fit_generator(train_gen, train_gen.steps_per_epoch,
                                       validation_data=dev_gen, validation_steps=validation_steps,
                                       callbacks=callbacks, epochs=nepochs)
        return self

    def train_dep_model(self, sents, heads, deps, dev_sents, dev_heads, dev_deps,
                        tags=None, dev_tags=None, nepochs=2, batch_size=16, patience=1):
        self.dep_model_ = self.build_dep_network(**self.dep_model_params)
        dep_indexes, head_indexes, dep_codes =\
            make_indexes_for_syntax(heads, deps, dep_vocab=self.dep_vocabulary_, to_pad=False)
        dep_gen_params = {"embedder": self.embedder, "classes_number": self.dep_vocabulary_.symbols_number_,
                          "batch_size": batch_size, "target_padding": PAD,
                          "additional_padding": [DataGenerator.POSITION_AS_PADDING] * 2}
        additional_data = [dep_indexes, head_indexes]
        if self.use_tags:
            additional_data.append(tags)
            dep_gen_params["additional_padding"].append(0)
        train_gen = DataGenerator(sents, targets=dep_codes, additional_data=additional_data, **dep_gen_params)
        if dev_sents is not None:
            dev_dep_indexes, dev_head_indexes, dev_dep_codes = \
                make_indexes_for_syntax(dev_heads, dev_deps, dep_vocab=self.dep_vocabulary_, to_pad=False)
            additional_data = [dev_dep_indexes, dev_head_indexes]
            if self.use_tags:
                additional_data.append(dev_tags)
            dev_gen = DataGenerator(data=dev_sents, targets=dev_dep_codes,
                                    additional_data=additional_data,
                                    shuffle=False, **dep_gen_params)
            validation_steps = dev_gen.steps_per_epoch
        else:
            dev_gen, validation_steps = None, None
        callbacks = []
        if patience >= 0:
            callbacks.append(EarlyStopping(monitor="val_acc", restore_best_weights=True, patience=patience))
        self.dep_model_.fit_generator(train_gen, train_gen.steps_per_epoch,
                                      validation_data=dev_gen,
                                      validation_steps=validation_steps,
                                      callbacks=callbacks, epochs=nepochs)
        return self

    def predict(self, sents, tags=None):
        sents = pad_data(sents)
        if self.use_char_model:
            sent_data = self._transform_data(sents, to_train=False)
        else:
            sent_data = sents
        if tags is not None:
            tag_data = self._transform_tags(tags, to_train=False)
        else:
            tag_data = None
        head_probs, chl_pred_heads = self.predict_heads(sents, sent_data, tag_data)
        deps = self.predict_deps(sents, sent_data, chl_pred_heads, tag_data)
        return chl_pred_heads, deps

    def predict_heads(self, sents, sent_data, tags=None):
        probs, heads = [None] * len(sents), [None] * len(sents)
        data = [sents, sent_data, tags]
        additional_data = [data[i] for i in self.active_input_indexes[1:]]
        test_gen = DataGenerator(data[self.first_input_index], additional_data=additional_data,
                                 embedder=self.embedder, yield_targets=False, yield_indexes=True,
                                 shuffle=False, nepochs=1)
        for batch_index, (batch, indexes) in enumerate(test_gen):
            if self.use_joint_model:
                batch_probs = self.head_model_(batch + [0])[0]
            else:
                batch_probs = self.head_model_.predict(batch)
            for i, index in enumerate(indexes):
                L = len(sents[index])
                curr_probs = batch_probs[i][:L - 1, :L - 1]
                curr_probs /= np.sum(curr_probs, axis=-1)
                probs[index] = curr_probs
                heads[index] = np.argmax(curr_probs[1:], axis=-1)
        chl_pred_heads = [chu_liu_edmonds(elem.astype("float64"))[0][1:] for elem in probs]
        return probs, chl_pred_heads

    def predict_deps(self, sents, sent_data, heads, tags=None):
        dep_indexes, head_indexes = make_indexes_for_syntax(heads)
        data = [sents, sent_data, tags]
        additional_data = [data[i] for i in self.active_input_indexes[1:]] + [dep_indexes, head_indexes]
        additional_padding = [0] * (len(additional_data)-2) + [DataGenerator.POSITION_AS_PADDING] * 2
        generator_params = {"embedder": self.embedder, "additional_data": additional_data,
                            "additional_padding": additional_padding}
        test_gen = DataGenerator(data[self.first_input_index],
                                 yield_indexes=True, yield_targets=False,
                                 shuffle=False, nepochs=1, **generator_params)
        answer = [None] * len(sents)
        for batch, indexes in test_gen:
            if self.use_joint_model:
                batch_probs = self.dep_model_(batch + [0])[0]
            else:
                batch_probs = self.dep_model_.predict(batch)
            batch_labels = np.argmax(batch_probs, axis=-1)
            for i, index in enumerate(indexes):
                L = len(sents[index])
                curr_labels = batch_labels[i][1:L - 1]
                answer[index] = [self.dep_vocabulary_.symbols_[elem] for elem in curr_labels]
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



