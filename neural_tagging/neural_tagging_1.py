import sys
from collections import defaultdict
import inspect
import json
import os
import copy
from typing import List
# import statprof

import keras.layers as kl
import keras.optimizers as ko
import keras.regularizers as kreg
from keras import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.activations import softmax

from neural_LM.UD_preparation.read_tags import is_subsumed, descr_to_feats
from neural_LM.UD_preparation.extract_tags_from_UD import decode_word
from neural_LM.vocabulary import Vocabulary, FeatureVocabulary, vocabulary_from_json
from neural_LM.neural_lm import make_bucket_indexes
from neural_LM.common import *
from neural_LM.common_new import DataGenerator, MultirunEarlyStopping
from neural_LM import load_lm
from neural_LM.cells import SelfAttentionEncoder, SelfAttentionDecoder, LayerNorm1D, WeightedSum
from neural_tagging.cells import Highway, WeightedCombinationLayer, TemporalDropout, leader_loss, positions_func
from neural_tagging.dictionary import read_dictionary
from neural_tagging.vectorizers import UnimorphVectorizer, MatchingVectorizer, load_vectorizer

BUCKET_SIZE = 32
MAX_WORD_LENGTH = 30

CASHED_INDEXES = dict()

def load_tagger(infile):
    with open(infile, "r", encoding="utf8") as fin:
        json_data = json.load(fin)
    args = {key: value for key, value in json_data.items()
            if not (key.endswith("_") or key.endswith("callback") or
                    key in ["dump_file", "lm_file", "vectorizer_save_data", "tag_vectorizer_save_data"])}
    callbacks = []
    early_stopping_callback_data = json_data.get("early_stopping_callback")
    if early_stopping_callback_data is not None:
        callbacks.append(EarlyStopping(**early_stopping_callback_data))
    lr_callback_data = json_data.get("LR_callback")
    if lr_callback_data is not None:
        callbacks.append(ReduceLROnPlateau(**lr_callback_data))
    model_checkpoint_callback_data = json_data.get("model_checkpoint_callback")
    if model_checkpoint_callback_data is not None:
        model_checkpoint_callback_data["save_weights_only"] = True
        model_checkpoint_callback_data["save_best_only"] = True
        callbacks.append(ModelCheckpoint(**model_checkpoint_callback_data))
    args['callbacks'] = callbacks
    # создаём языковую модель
    tagger = CharacterTagger(**args)
    # обучаемые параметры
    args = {key: value for key, value in json_data.items() if key[-1] == "_"}
    for key, value in args.items():
        if key == "symbols_":
            value = vocabulary_from_json(value)
        elif key == "tags_":
            value = vocabulary_from_json(value, use_features=True)
        elif key == "tag_embeddings_":
            value = np.asarray(value)
        elif key == "morpho_dict_":
            use_features = tagger.morpho_dict_params.get("type") in ["features" "native"]
            value = vocabulary_from_json(value, use_features=use_features)
        setattr(tagger, key, value)
    # loading language model
    if tagger.use_lm and "lm_file" in json_data:
        tagger.lm_ = load_lm(json_data["lm_file"])
    # loading word vectorizers

    for cls, infile, dim in json_data.get("vectorizer_save_data", []):
        vectorizer = load_vectorizer(cls, infile)
        tagger.word_vectorizers.append([vectorizer, dim])
    # compiling morpho dictionary (if any)
    if tagger.morpho_dict is not None:
        tagger._make_morpho_dict_indexes_func(tagger.morpho_dict_params["type"])
    # модель
    tagger.build()  # не работает сохранение модели, приходится сохранять только веса
    tagger.model_.load_weights(os.path.abspath(json_data['dump_file']))
    return tagger


def extract_matching_tags(tag, tag_dictionary):
    """
    Returns a list of dictionary tags for a given word
    """
    # if not hasattr(extract_matching_tags, "CALLS"):
    #     extract_matching_tags.CALLS = 0
    if tag not in CASHED_INDEXES:
        CASHED_INDEXES[tag] = []
        for i, dict_tag in enumerate(tag_dictionary):
            if is_subsumed(tag, dict_tag):
                CASHED_INDEXES[tag].append(i)
        # extract_matching_tags.CALLS += 1
    return CASHED_INDEXES[tag]


def extract_feature_indexes(tag, tag_dictionary):
    if tag not in CASHED_INDEXES:
        symbol, feats = descr_to_feats(tag)
        curr_labels = {symbol} | {"{}_{}_{}".format(symbol, *x) for x in feats}
        CASHED_INDEXES[tag] = [tag_dictionary.symbol_labels_codes_[label]
                               for label in curr_labels
                               if label in tag_dictionary.symbol_labels_codes_]
    return CASHED_INDEXES[tag]


class CharacterTagger:
    """
    A class for character-based neural morphological tagger
    """
    def __init__(self, reverse=False, use_lm_loss=False, use_lm=False,
                 morpho_dict=None, morpho_dict_params=None,
                 morpho_dict_embeddings_size=256, word_vectorizers=None,
                 word_tag_vectorizers=None,
                 additional_inputs_number=0, additional_inputs_weight=0,
                 use_additional_symbol_features=False, transfer_warmup_epochs=0,
                 freeze_after_transfer=False,
                 normalize_lm_embeddings=False, base_model_weight=0.25,
                 word_rnn = "cnn", min_char_count=1, char_embeddings_size=16,
                 char_conv_layers = 1, char_window_size = 5, char_filters = None,
                 total_char_filters=None, char_filter_multiple=25, max_window_filters=200,
                 char_highway_layers = 1, conv_dropout = 0.0, highway_dropout = 0.0,
                 intermediate_dropout = 0.0, word_dropout=0.0, lm_dropout=0.0,
                 word_lstm_layers=1, word_lstm_units=128, lstm_dropout = 0.0,
                 use_rnn_for_weight_state=False, weight_state_rnn_units=64,
                 use_fusion=False, fusion_state_units=256, use_dimension_bias=False,
                 use_intermediate_activation_for_weights=False,
                 intermediate_units_for_weights=64,
                 use_leader_loss=False, leader_loss_weight=0.2,
                 regularizer=None, fusion_regularizer=None,
                 probs_threshold=None, lm_probs_threshold=None,
                 batch_size=16, validation_split=0.2, nepochs=25,
                 min_prob=0.01, max_diff=2.0, to_weigh_loss=True,
                 callbacks=None, verbose=1, random_state=189):
        self.reverse = reverse
        self.use_lm_loss = use_lm_loss
        self.use_lm = use_lm
        self.morpho_dict = morpho_dict
        self.morpho_dict_params = morpho_dict_params
        self.morpho_dict_embeddings_size = morpho_dict_embeddings_size
        self.word_vectorizers = word_vectorizers
        self.word_tag_vectorizers = word_tag_vectorizers
        self.additional_inputs_number = additional_inputs_number
        self.additional_inputs_weight = additional_inputs_weight
        self.use_additional_symbol_features = use_additional_symbol_features
        self.transfer_warmup_epochs = transfer_warmup_epochs
        self.freeze_after_transfer = freeze_after_transfer
        self.normalize_lm_embeddings = normalize_lm_embeddings
        self.base_model_weight = base_model_weight
        self.word_rnn = word_rnn
        self.min_char_count = min_char_count
        self.char_embeddings_size = char_embeddings_size
        self.char_conv_layers = char_conv_layers
        self.char_window_size = char_window_size
        self.char_filters = char_filters
        self.total_char_filters = total_char_filters
        self.char_filter_multiple = char_filter_multiple
        self.max_window_filters = max_window_filters
        self.char_highway_layers = char_highway_layers
        self.conv_dropout = conv_dropout
        self.highway_dropout = highway_dropout
        self.intermediate_dropout = intermediate_dropout
        self.word_dropout = word_dropout
        self.word_lstm_layers = word_lstm_layers
        self.word_lstm_units = word_lstm_units
        self.lstm_dropout = lstm_dropout
        self.lm_dropout = lm_dropout
        self.use_rnn_for_weight_state = use_rnn_for_weight_state
        self.weight_state_rnn_units = weight_state_rnn_units
        self.use_fusion = use_fusion
        self.fusion_state_units = fusion_state_units
        self.use_dimension_bias = use_dimension_bias
        self.use_intermediate_activation_for_weights = use_intermediate_activation_for_weights
        self.intermediate_units_for_weights = intermediate_units_for_weights
        self.use_leader_loss = use_leader_loss
        self.leader_loss_weight = leader_loss_weight
        self.regularizer = regularizer
        self.fusion_regularizer = fusion_regularizer
        self.probs_threshold = probs_threshold
        self.lm_probs_threshold = lm_probs_threshold
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.nepochs=nepochs
        self.min_prob = min_prob
        self.max_diff = max_diff
        self.to_weigh_loss = to_weigh_loss
        self.callbacks = callbacks
        self.verbose = verbose
        self.random_state = random_state
        self._initialize()

    def _initialize(self):
        if isinstance(self.char_window_size, int):
            self.char_window_size = [self.char_window_size]
        self.char_window_size = sorted(self.char_window_size)
        if self.char_filters is None:
            self._make_char_filters()
        if isinstance(self.char_filters, int):
            self.char_filters = [self.char_filters] * len(self.char_window_size)
        if len(self.char_window_size) != len(self.char_filters):
            raise ValueError("There should be the same number of window sizes and filter sizes")
        if isinstance(self.word_lstm_units, int):
            self.word_lstm_units = [self.word_lstm_units] * self.word_lstm_layers
        if len(self.word_lstm_units) != self.word_lstm_layers:
            raise ValueError("There should be the same number of lstm layer units and lstm layers")
        if self.word_vectorizers is None:
            self.word_vectorizers = []
        self.vectorizer_save_data = []
        for i, (data, dim) in enumerate(self.word_vectorizers):
            cls, params = eval(data["cls"]), data.get("params", dict())
            train_params = data.get("train_params", dict())
            save_file = data["save_file"]
            vectorizer = cls(**params).train(save_file=save_file, **train_params)
            self.word_vectorizers[i] = (vectorizer, dim)
            self.vectorizer_save_data.append((data["cls"], save_file, dim))
        if self.word_tag_vectorizers is None:
            self.word_tag_vectorizers = []
        self.tag_vectorizer_save_data = []
        for i, data in enumerate(self.word_tag_vectorizers):
            cls, params = eval(data["cls"]), data.get("params", dict())
            train_params = data.get("train_params", dict())
            # save_file = data["save_file"]
            self.word_tag_vectorizers[i] = cls(**params).train(**train_params)
            self.tag_vectorizer_save_data.append((data["cls"],))
        if self.regularizer is not None:
            self.regularizer = kreg.l2(self.regularizer)
        if self.fusion_regularizer is not None:
            self.fusion_regularizer = kreg.l2(self.fusion_regularizer)
        np.random.seed(self.random_state)

    def _make_char_filters(self):
        """
        Initializes window filters in case they are not given explicitly
        """
        if self.total_char_filters is None:
            self.char_filters = [min(self.char_filter_multiple * window_size,
                                     self.max_window_filters)
                                 for window_size in self.char_window_size]
        else:
            if self.total_char_filters >= self.windows_number * self.max_window_filters:
                raise ValueError(
                    "Cannot have {} filters with {} windows and {} filters per window".format(
                        self.total_char_filters, self.windows_number, self.max_window_filters))
            # setting the number of layer filters proportionally to window size
            window_sizes_sum = sum(self.char_window_size)
            char_filter_multiple = self.total_char_filters // window_sizes_sum
            filters = [None] * len(self.char_window_size)
            for i, window_size in enumerate(self.char_window_size[::-1], 1):
                filters[-i] = min(char_filter_multiple * window_size, self.max_window_filters)
            total_filters = sum(filters)
            for i, curr_filters in enumerate(filters[::-1], 1):
                if total_filters == self.total_char_filters:
                    break
                filters[-i] = min(curr_filters + (self.total_char_filters - total_filters),
                                  self.max_window_filters)
                total_filters += (filters[-i] - curr_filters)
            self.char_filters = filters
        return

    def to_json(self, outfile, model_file, lm_file=None, dictionary_file=None):
        info = dict()
        if lm_file is not None:
            info["lm_file"] = lm_file
        # model_file = os.path.abspath(model_file)
        for (attr, val) in inspect.getmembers(self):
            if not (attr.startswith("__") or inspect.ismethod(val) or
                    isinstance(getattr(CharacterTagger, attr, None), property) or
                    isinstance(val, np.ndarray) or isinstance(val, Vocabulary) or
                    attr.isupper() or isinstance(val, kb.Function) or
                    attr in ["callbacks", "_compile_args",
                             "model_", "_basic_model_", "warmup_model_",
                             "_decoder_", "lm_",
                             "morpho_dict_indexes_func_", "word_tag_mapper_", "morpho_dict_",
                             "regularizer", "fusion_regularizer",
                             "word_vectorizers", "word_tag_vectorizers"]):
                info[attr] = val
            elif isinstance(val, Vocabulary):
                info[attr] = val.jsonize()
            elif isinstance(val, np.ndarray):
                val = val.tolist()
                info[attr] = val
            elif attr == "model_":
                info["dump_file"] = model_file
                self.model_.save_weights(model_file)
            elif attr == "callbacks":
                for callback in val:
                    if isinstance(callback, EarlyStopping):
                        info["early_stopping_callback"] = {"patience": callback.patience,
                                                           "monitor": callback.monitor}
                    elif isinstance(callback, ModelCheckpoint):
                        info["model_checkpoint_callback"] =\
                            {key: getattr(callback, key) for key in ["monitor", "filepath"]}
                    elif isinstance(callback, ReduceLROnPlateau):
                        info["LR_callback"] =\
                            {key: getattr(callback, key) for key in
                             ["monitor", "factor", "patience", "cooldown", "min_delta"]}
            elif attr.endswith("regularizer"):
                if val is not None:
                    info[attr] = float(val.l2)
        with open(outfile, "w", encoding="utf8") as fout:
            json.dump(info, fout)

    @property
    def symbols_number_(self):
        return self.symbols_.symbols_number_ + self.additional_inputs_number + self.has_additional_inputs

    @property
    def tags_number_(self):
        return self.tags_.symbols_number_

    @property
    def windows_number(self):
        return len(self.char_window_size)

    def _additional_symbol_index(self, i):
        return self.symbols_.symbols_number_ + i

    @property
    def has_additional_inputs(self):
        return int(self.additional_inputs_number > 0)

    def _make_word_dictionary(self, data=None):
        self._make_word_tag_mapper(data)
        if self.morpho_dict_params is None:
            self.morpho_dict_params = {"type": "subsume"}
        morpho_dict_params = copy.copy(self.morpho_dict_params)
        morpho_dict_type = morpho_dict_params.pop("type")
        # need to train dictionaries
        if morpho_dict_type == "tags":
            self.morpho_dict_ = Vocabulary(**morpho_dict_params)
        elif morpho_dict_type == "features":
            self.morpho_dict_ = FeatureVocabulary(**morpho_dict_params)
        elif morpho_dict_type == "native":
            self.morpho_dict_ = self.tags_
        if morpho_dict_type in ["tags", "features"]:
            self.morpho_dict_.train(self.word_tag_mapper_.values())
        self._make_morpho_dict_indexes_func(morpho_dict_type)
        return self

    def _make_word_tag_mapper(self, data=None):
        if self.morpho_dict == "pymorphy":
            raise NotImplementedError
        elif isinstance(self.morpho_dict, str) and os.path.exists(self.morpho_dict):
            self.word_tag_mapper_ = read_dictionary(self.morpho_dict)
        else:
            raise ValueError("Dictionary should be 'pymorphy' or path to a file")

    def _make_morpho_dict_indexes_func(self, morpho_dict_type):
        """
        Creates a function that transforms morpho dict tags to indexes
        """
        if morpho_dict_type == "subsume":
            func = lambda x: extract_matching_tags(x, self.tags_.symbols_)
        elif morpho_dict_type == "tags":
            func = lambda x: [self.morpho_dict_.toidx(x)]
        elif morpho_dict_type in ["features", "native"]:
            func = lambda x: extract_feature_indexes(x, self.morpho_dict_)
        else:
            raise ValueError("Unknown morpho_dict_type: {}".format(morpho_dict_type))
        self.morpho_dict_indexes_func_ = func

    def transform(self, data, labels=None, pad=True, return_indexes=True,
                  buckets_number=None, bucket_size=None, join_buckets=True,
                  dataset_codes=None, join_datasets=True):
        if dataset_codes is None:
            dataset_codes = [0] * len(data)
        if join_datasets:
            indexes_by_datasets = [list(range(len(data)))]
        else:
            indexes_by_datasets = [[] for _ in range(self.additional_inputs_number + 1)]
            for i, code in enumerate(dataset_codes):
                indexes_by_datasets[code].append(i)
        lengths = [[len(data[i])+2 for i in elem] for elem in indexes_by_datasets]
        bucket_data = []
        for dataset_index, (curr_lengths, curr_indexes) in enumerate(zip(lengths, indexes_by_datasets)):
            if len(curr_indexes) == 0:
                continue
            if pad:
                if len(curr_indexes) > 0:
                    curr_indexes_by_buckets, curr_level_lengths = make_bucket_indexes(
                        curr_lengths, buckets_number=buckets_number,
                        bucket_size=bucket_size, join_buckets=join_buckets)
                    curr_indexes_by_buckets = [[curr_indexes[index] for index in elem]
                                               for elem in curr_indexes_by_buckets]
            else:
                curr_indexes_by_buckets = [[i] for i in curr_indexes]
                curr_level_lengths = curr_lengths
            for bucket_indexes, length in zip(curr_indexes_by_buckets, curr_level_lengths):
                bucket_data.append((bucket_indexes, length, dataset_index))
        X = [None] * len(data)
        for r, (bucket_indexes, bucket_length, dataset_code) in enumerate(bucket_data):
            bucket_length += int(self.additional_inputs_number > 0)
            for i in bucket_indexes:
                sent = data[i] if not self.reverse else data[i][::-1]
                length_func = lambda x: min(len(x), MAX_WORD_LENGTH)+2
                X[i] = [self._make_sent_vector(sent, dataset_codes[i],
                                               bucket_length=bucket_length,
                                               classes_number=self.symbols_number_)]
                # X[i] = [self._make_sent_vector(sent, bucket_length=bucket_length),
                #         self._make_tags_vector(sent, bucket_length=bucket_length,
                #                                func=length_func)]
                if labels is not None:
                    tags = labels[i] if not self.reverse else labels[i][::-1]
                    X[i].append(self._make_tags_vector(tags, bucket_length=bucket_length))
            if labels is not None and hasattr(self, "lm_"):
                curr_bucket = np.array([X[i][-1] for i in bucket_indexes])
                padding = np.full((curr_bucket.shape[0], 1), BEGIN, dtype=int)
                curr_bucket = np.hstack((padding, curr_bucket[:,:-1]))
                # transforming indexes to features
                curr_bucket = self.lm_.vocabulary_.symbol_matrix_[curr_bucket]
                lm_probs = self.lm_.model_.predict(curr_bucket)
                lm_states = self.lm_.hidden_state_func_([curr_bucket, 0])[0]
                self.lm_state_dim_ = lm_states.shape[-1]
                for i, index in enumerate(bucket_indexes):
                    X[index].insert(1, lm_states[i])
                    if not self.use_fusion:
                        X[index].insert(1, lm_probs[i])
            if hasattr(self, "lm_") and labels is not None:
                insert_pos = 3 - int(self.use_fusion)
            else:
                insert_pos = 1
            for vectorizer, _ in self.word_vectorizers[::-1]:
                for i, index in enumerate(bucket_indexes):
                    curr_sent_tags = np.zeros(shape=(bucket_length, vectorizer.dim), dtype=np.float)
                    sent = data[index] if not self.reverse else data[i][::-1]
                    for j, word in enumerate(sent):
                        word = decode_word(word)
                        if word is not None:
                            word_indexes = vectorizer[word]
                            if word_indexes is not None and len(word_indexes) > 0:
                                for elem in word_indexes:
                                    curr_sent_tags[j, elem] += 1.0
                                curr_sent_tags /= len(word_indexes)
                    X[index].insert(insert_pos, curr_sent_tags)
            insert_pos += len(self.word_vectorizers)
            for vectorizer in self.word_tag_vectorizers[::-1]:
                for i, index in enumerate(bucket_indexes):
                    curr_sent_tags = np.zeros(shape=(bucket_length, self.tags_number_), dtype=np.float)
                    sent = data[index] if not self.reverse else data[i][::-1]
                    for j, word in enumerate(sent):
                        word = decode_word(word)
                        if word is not None:
                            word_tags = vectorizer[word]
                            word_indexes = [self.tags_.toidx(tag) for tag in word_tags]
                            word_indexes = [x for x in word_indexes if x != UNKNOWN]
                            if len(word_indexes) > 0:
                                # curr_sent_tags[j, word_indexes] += 1 / np.log2(len(word_indexes) + 1)
                                curr_sent_tags[j, word_indexes] += 1 / len(word_indexes)
                    X[index].insert(insert_pos, curr_sent_tags)
            # if self.morpho_dict is not None:
            #     if hasattr(self, "lm_") and labels is not None:
            #         insert_pos = 3 - int(self.use_fusion)
            #     else:
            #         insert_pos = 1
            #     for i, index in enumerate(bucket_indexes):
            #         curr_sent_tags = np.zeros(
            #             shape=(bucket_length, self.tags_number_), dtype=np.int32)
            #         sent = data[index] if not self.reverse else data[i][::-1]
            #         for j, word in enumerate(sent):
            #             word = decode_word(word)
            #             if word is not None and word in self.word_tag_mapper_:
            #                 word_tags = self.word_tag_mapper_[word]
            #                 for tag in word_tags:
            #                     tag_indexes = self.morpho_dict_indexes_func_(tag)
            #                     curr_sent_tags[j][tag_indexes] = 1
            #         X[index].insert(insert_pos, curr_sent_tags)
        indexes, datasets = [elem[0] for elem in bucket_data], [elem[2] for elem in bucket_data]
        if return_indexes:
            return X, indexes, datasets
        else:
            return X

    def _make_sent_vector(self, sent, dataset_code=0, bucket_length=None,
                          classes_number=None):
        if bucket_length is None:
            bucket_length = len(sent)
        if dataset_code is not None:
            dataset_code = self._additional_symbol_index(dataset_code)
        output_shape = (bucket_length, MAX_WORD_LENGTH+2)
        # if classes_number is not None:
        #     output_shape += (classes_number,)
        answer = np.full(shape=output_shape, fill_value=PAD, dtype="int32")
        answer[:len(sent),0] = BEGIN
        if self.additional_inputs_number > 0:
            answer[:len(sent),1] = dataset_code
        lengths = [min(len(word), MAX_WORD_LENGTH - self.has_additional_inputs) for word in sent]
        for i, word in enumerate(sent):
            m = lengths[i]
            for j, x in enumerate(word[-m:], self.has_additional_inputs):
                answer[i, j+1] = self.symbols_.toidx(x)
            answer[i, self.has_additional_inputs+m+1] = END
        answer = np.expand_dims(answer, -1).tolist()
        if classes_number is not None:
            if self.use_additional_symbol_features:
                for i, L in enumerate(lengths):
                    for j in range(2, L+2):
                        answer[i][j].append(dataset_code)
        return answer

    def _make_tags_vector(self, tags, bucket_length=None, func=None):
        m = len(tags)
        if bucket_length is None:
            bucket_length = m
        answer = np.zeros(shape=(bucket_length,), dtype=np.int32)
        for i, tag in enumerate(tags):
            answer[i] = self.tags_.toidx(tag) if func is None else func(tag)
        return answer

    def train(self, data, labels, dev_data=None, dev_labels=None,
              additional_data=None, additional_labels=None,
              symbol_vocabulary_file=None, tags_vocabulary_file=None,
              lm_file=None, model_file=None, save_file=None):
        """
        Trains the tagger on data :data: with labels :labels:

        data: list of lists of sequences, a list of sentences
        labels: list of lists of strs,
            a list of sequences of tags, each tag is a feature-value structure
        :return:
        """
        # vocabularies for symbols and tags
        data_for_vocab, labels_for_vocab = data[:], labels[:]
        if additional_data is not None:
            for (elem, elem_labels) in zip(additional_data, additional_labels):
                data_for_vocab += elem
                labels_for_vocab += elem_labels
        if symbol_vocabulary_file is None:
            self.symbols_ = Vocabulary(character=True, min_count=self.min_char_count).train(data)
        else:
            self.symbols_ = vocabulary_from_json(symbol_vocabulary_file, use_features=False)
        if tags_vocabulary_file is None:
            self.tags_ = FeatureVocabulary(character=False).train(labels)
        else:
            with open(tags_vocabulary_file, "r", encoding="utf8") as fin:
                tags_info = json.load(fin)
            self.tags_ = vocabulary_from_json(tags_info, use_features=True)
        if self.verbose > 0:
            print("{} characters, {} tags".format(self.symbols_number_, self.tags_number_))
        # language model
        if lm_file is not None and (self.use_lm or self.use_lm_loss):
            lm = load_lm(lm_file)
        if lm_file is not None and self.use_lm_loss:
            self._make_tag_embeddings(lm)
        if lm_file is not None and self.use_lm:
            self.lm_ = lm
        # external word-tag dictionaries
        if self.morpho_dict is not None:
            data_for_dict = data + dev_data if dev_data is not None else data
            self._make_word_dictionary(data_for_dict)
        if dev_data is None:
            indexes = np.arange(len(data))
            np.random.shuffle(indexes)
            train_data_length = int(len(data) * (1.0 - self.validation_split))
            train_indexes, dev_indexes = indexes[:train_data_length], indexes[train_data_length:]
            data, dev_data = [data[i] for i in train_indexes], [data[i] for i in dev_indexes]
            labels, dev_labels = [labels[i] for i in train_indexes], [labels[i] for i in dev_indexes]
        dataset_codes = [0] * len(data)
        if self.additional_inputs_number > 0:
            for code, (add_dataset, add_labels) in enumerate(zip(additional_data, additional_labels), 1):
                data += add_dataset
                dataset_codes += [code] * len(add_dataset)
                labels += add_labels
        X_train, indexes_by_buckets, dataset_codes_by_buckets = self.transform(
            data, labels, buckets_number=10, dataset_codes=dataset_codes,
            join_datasets=(self.transfer_warmup_epochs==0))
        if dev_data is not None:
            X_dev, dev_indexes_by_buckets, _ =\
                self.transform(dev_data, dev_labels, bucket_size=BUCKET_SIZE,
                               join_datasets=(self.transfer_warmup_epochs==0))
        else:
            X_dev, dev_indexes_by_buckets = [None] * 2
        self.build()
        if save_file is not None and model_file is not None:
            self.to_json(save_file, model_file, lm_file)
        self._train_on_data(X_train, indexes_by_buckets, X_dev,
                            dev_indexes_by_buckets, dataset_codes=dataset_codes,
                            model_file=model_file,
                            train_dataset_codes_by_buckets=dataset_codes_by_buckets)
        return self

    def _train_on_data(self, X, indexes_by_buckets, X_dev=None,
                       dev_indexes_by_buckets=None, model_file=None,
                       dataset_codes=None, train_dataset_codes_by_buckets=None,
                       dev_dataset_codes_by_buckets=None):
        if dataset_codes is None:
            weights = np.ones(shape=(len(X)))
        else:
            if X_dev is None:
                raise ValueError("You should implicitly pass X_dev when using additional data")
            dataset_codes = np.array(dataset_codes)
            weights = np.where(dataset_codes == 0.0, 1.0, self.additional_inputs_weight)
        if X_dev is None:
            X_dev, dev_indexes_by_buckets = X, []
            validation_split = self.validation_split
        else:
            validation_split = 0.0
        train_indexes_by_buckets = []
        for curr_indexes in indexes_by_buckets:
            np.random.shuffle(curr_indexes)
            if validation_split != 0.0:
                train_bucket_size = int((1.0 - self.validation_split) * len(curr_indexes))
                train_indexes_by_buckets.append(curr_indexes[:train_bucket_size])
                dev_indexes_by_buckets.append(curr_indexes[train_bucket_size:])
            else:
                train_indexes_by_buckets.append(curr_indexes)
        if train_dataset_codes_by_buckets is None:
            train_dataset_codes_by_buckets = [0] * len(train_indexes_by_buckets)
        if dev_dataset_codes_by_buckets is None:
            dev_dataset_codes_by_buckets = [0] * len(dev_indexes_by_buckets)
        # if hasattr(self, "lm_") and self.n_warmup_epochs > 0:
        #     fields_number = 1 + int(self.morpho_dict is not None)
        #     train_gen = generate_data(X, train_indexes_by_buckets, self.tags_number_,
        #                               self.batch_size, use_last=False,
        #                               duplicate_answer=False, fields_number=fields_number,
        #                               yield_weights=self.to_weigh_loss)
        #     dev_gen = generate_data(X_dev, dev_indexes_by_buckets, self.tags_number_,
        #                             use_last=False, shuffle=False, duplicate_answer=False,
        #                             fields_number=fields_number,
        #                             yield_weights=self.to_weigh_loss)
        #     self.warmup_model_.fit_generator(
        #         train_gen, steps_per_epoch=train_steps, epochs=self.n_warmup_epochs,
        #         callbacks=self.callbacks, validation_data=dev_gen,
        #         validation_steps=dev_steps, verbose=1)
        if model_file is not None:
            monitor = "val_p_output_acc" if self.use_lm else "val_acc"
            callback = ModelCheckpoint(model_file, monitor=monitor,
                                       save_weights_only=True, save_best_only=True)
            if self.callbacks is not None:
                self.callbacks.append(callback)
            else:
                self.callbacks = [callback]
        are_train_buckets_active = self._make_active_buckets(train_dataset_codes_by_buckets)
        active_train_epochs = are_train_buckets_active.max(axis=1).astype(bool)
        are_dev_buckets_active = self._make_active_buckets(dev_dataset_codes_by_buckets)
        active_dev_epochs = are_dev_buckets_active.max(axis=1).astype(bool)
        for t in range(self.nepochs):
            if t == self.transfer_warmup_epochs and self.freeze_after_transfer:
                self._freeze_output_network()
            train_steps, dev_steps = 0, 0
            curr_train_indexes, curr_dev_indexes = [], []
            for i in are_train_buckets_active[t].nonzero()[0]:
                curr_train_indexes.append(train_indexes_by_buckets[i])
                train_steps += (len(train_indexes_by_buckets[i]) - 1) // self.batch_size + 1
            for i in are_dev_buckets_active[t].nonzero()[0]:
                curr_dev_indexes.append(dev_indexes_by_buckets[i])
                dev_steps += (len(dev_indexes_by_buckets[i]) - 1) // self.batch_size + 1
            train_gen = DataGenerator(X, curr_train_indexes, self.tags_number_,
                                      batch_size=self.batch_size, duplicate_answer=self.use_lm,
                                      fields_to_one_hot={0: self.symbols_number_},
                                      yield_weights=self.to_weigh_loss, weights=weights)
            if dev_steps > 0:
                dev_gen = DataGenerator(X_dev, curr_dev_indexes, self.tags_number_,
                                        batch_size=self.batch_size, shuffle=False,
                                        duplicate_answer=self.use_lm,
                                        fields_to_one_hot={0: self.symbols_number_},
                                        yield_weights=self.to_weigh_loss)
            else:
                dev_gen = None
            # TO_DO: set callback values
            # for callback in self.callbacks:
            #     if isinstance(callback, MultirunEarlyStopping):
            self.model_.fit_generator(
                train_gen, steps_per_epoch=train_steps, epochs=t+1,
                callbacks=self.callbacks, validation_data=dev_gen,
                validation_steps=dev_steps, initial_epoch=t, verbose=1)
            if self.model_.stop_training:
                break

        if model_file is not None:
            self.model_.load_weights(model_file)
        return self

    def _make_active_buckets(self, bucket_dataset_codes):
        are_buckets_active = np.ones(shape=(self.nepochs, len(bucket_dataset_codes)), dtype=int)
        if self.transfer_warmup_epochs > 0:
            for i, code in enumerate(bucket_dataset_codes):
                if code == 0:
                    are_buckets_active[:self.transfer_warmup_epochs, i] = 0
                else:
                    are_buckets_active[self.transfer_warmup_epochs:, i] = 0
        # only those epochs which have at least one bucket are counted
        # has_active_buckets_in_epoch = np.max(are_buckets_active, axis=1).astype(bool)
        # are_buckets_active = are_buckets_active[has_active_buckets_in_epoch]
        return are_buckets_active

    def predict(self, data, labels=None, dataset_codes=None,
                beam_width=1, return_probs=False, return_basic_probs=False,
                predict_with_basic=False):
        if self.morpho_dict is not None and not hasattr(self, "word_tag_mapper_"):
            self._make_word_tag_mapper(data)
        X_test, indexes_by_buckets, datasets_by_buckets =\
            self.transform(data, labels=labels, bucket_size=BUCKET_SIZE)
        answer, probs = [None] * len(data), [None] * len(data)
        basic_probs = [None] * len(data)
        fields_number = len(X_test[0])-int(labels is not None)
        # for k, (curr_bucket, bucket_indexes) in enumerate(zip(X_test[::-1], indexes_by_buckets[::-1])):
        for k, bucket_indexes in enumerate(indexes_by_buckets[::-1]):
            X_curr = make_batch([X_test[i][:fields_number] for i in bucket_indexes], {0: self.symbols_number_})
            # X_curr = [np.array([X_test[i][j] for i in bucket_indexes])
            #           for j in range(len(X_test[0])-int(labels is not None))]
            if self.use_lm and labels is None:
                if self.verbose > 0 and (k < 3 or k % 10 == 0):
                    print("Bucket {} of {} predicting".format(k+1, len(indexes_by_buckets)))
                batch_answer = self.predict_on_batch(X_curr, beam_width)
                bucket_labels = [x[0] for x in batch_answer[0]]
                bucket_probs = [x[0] for x in batch_answer[1]]
                bucket_basic_probs = [x[0] for x in batch_answer[2]]
                if predict_with_basic:
                    bucket_probs = bucket_basic_probs
            else:
                bucket_probs = self.model_.predict(X_curr, batch_size=64)
                bucket_basic_probs = [None] * len(bucket_indexes)
                if isinstance(bucket_probs, list):
                    bucket_probs, bucket_basic_probs = bucket_probs
                bucket_labels = np.argmax(bucket_probs, axis=-1)
            for curr_labels, curr_probs, curr_basic_probs, index in\
                    zip(bucket_labels, bucket_probs, bucket_basic_probs, bucket_indexes):
                curr_labels = curr_labels[:len(data[index])]
                curr_labels = [self.tags_.symbols_[label] for label in curr_labels]
                answer[index], probs[index] = curr_labels, curr_probs[:len(data[index])]
                basic_probs[index] = curr_basic_probs
        return ((answer, probs, basic_probs) if (return_basic_probs and self.use_lm)
                else (answer, probs) if return_probs else answer)

    def score(self, data, labels, return_basic_probs=False):
        if self.morpho_dict is not None and not hasattr(self, "word_tag_mapper_"):
            self._make_word_tag_mapper(data)
        X_test, indexes_by_buckets, _ = self.transform(data, labels, bucket_size=BUCKET_SIZE)
        probs, basic_probs = [None] * len(data), [None] * len(data)
        for k, (X_curr, bucket_indexes) in enumerate(zip(X_test[::-1], indexes_by_buckets[::-1])):
            X_curr = [np.array([X_test[i][j] for i in bucket_indexes])
                      for j in range(len(X_test[0])-1)]
            y_curr = [np.array(X_test[i][-1]) for i in bucket_indexes]
            bucket_probs = self.model_.predict(X_curr, batch_size=256)
            if self.use_lm:
                bucket_probs, bucket_basic_probs = bucket_probs
            else:
                bucket_basic_probs = [None] * len(bucket_indexes)
            for curr_labels, curr_probs, curr_basic_probs, index in\
                    zip(y_curr, bucket_probs, bucket_basic_probs, bucket_indexes):
                L = len(data[index])
                probs[index] = curr_probs[np.arange(L), curr_labels[:L]]
                if curr_basic_probs is not None:
                    basic_probs[index] = curr_basic_probs[np.arange(L), curr_labels[:L]]
        return (probs, basic_probs) if (return_basic_probs and self.use_lm) else probs

    def predict_on_batch(self, X, beam_width=1, return_log=False):
        m, L = X[0].shape[:2]
        basic_outputs = self._basic_model_(X + [0])
        M = m * beam_width
        for i, elem in enumerate(basic_outputs):
            basic_outputs[i] = np.repeat(elem, beam_width, axis=0)
        # positions[j] --- текущая позиция в symbols[j]
        tags, probs, basic_probs = [[] for _ in range(M)], [[] for _ in range(M)], [[] for _ in range(M)]
        partial_scores = np.zeros(shape=(M,), dtype=float)
        is_active, active_count = np.zeros(dtype=bool, shape=(M,)), m
        is_completed = np.zeros(dtype=bool, shape=(M,))
        is_active[np.arange(0, M, beam_width)] = True
        lm_inputs = np.full(shape=(M, L+1), fill_value=BEGIN, dtype=np.int32)
        lm_inputs = self.tags_.symbol_matrix_[lm_inputs]
        # определяем функцию удлинения входа для языковой модели
        def lm_inputs_func(old, new, i, matrix):
            return np.concatenate([old[:i+1], [matrix[new]], old[i+2:]])
        for i in range(L):
            for j, start in enumerate(range(0, M, beam_width)):
                if np.max(X[0][j, i]) == 0 and is_active[start]:
                    # complete the sequences not completed yet
                    # is_active[start] checks that the group has not yet been completed
                    is_completed[start:start+beam_width] = is_active[start:start+beam_width]
                    is_active[start:start+beam_width] = False
            if not any(is_active):
                break
            active_lm_inputs = lm_inputs[is_active,:i+1]
            # predicting language model probabilities and states
            if not self.use_fusion:
                lm_probs, lm_states = self.lm_.output_and_state_func_([active_lm_inputs, 0])
            else:
                lm_states = self.lm_.hidden_state_func_([active_lm_inputs, 0])[0]
            # keeping only active basic outputs
            active_basic_outputs = [elem[is_active,i:i+1] for elem in basic_outputs]
            if self.use_fusion:
                final_layer_inputs = active_basic_outputs + [lm_states[:,-1:]]
            else:
                final_layer_inputs = active_basic_outputs + [lm_probs[:,-1:], lm_states[:,-1:]]
            final_layer_outputs = self._decoder_(final_layer_inputs + [0])[0][:,0]
            hypotheses_by_groups = [[] for _ in range(m)]
            if beam_width == 1:
                curr_output_tags = np.argmax(final_layer_outputs, axis=-1)
                for r, (j, curr_probs, curr_basic_probs, index) in\
                        enumerate(zip(np.nonzero(is_active)[0], final_layer_outputs,
                                      active_basic_outputs[0], curr_output_tags)):
                    new_score = partial_scores[j] - np.log10(curr_probs[index])
                    hyp = (j, index, new_score, -np.log10(curr_probs[index]),
                           -np.log10(curr_basic_probs[0, index]))
                    hypotheses_by_groups[j] = [hyp]
            else:
                curr_best_scores = [np.inf] * m
                for r, (j, curr_probs, curr_basic_probs) in enumerate(
                        zip(np.nonzero(is_active)[0], final_layer_outputs, active_basic_outputs[0])):
                    group_index = j // beam_width
                    prev_partial_score = partial_scores[j]
                    # переходим к логарифмической шкале
                    curr_probs = -np.log10(np.clip(curr_probs, EPS, 1.0))
                    curr_basic_probs = -np.log10(np.clip(curr_basic_probs, EPS, 1.0))
                    if np.isinf(curr_best_scores[group_index]):
                        curr_best_scores[group_index] = prev_partial_score + np.min(curr_probs)
                    min_log_prob = curr_best_scores[group_index] - prev_partial_score + self.max_diff
                    min_log_prob = min(-np.log10(self.min_prob), min_log_prob)
                    possible_indexes = np.where(curr_probs <= min_log_prob)[0]
                    if len(possible_indexes) == 0:
                        possible_indexes = [np.argmin(curr_probs)]
                    for index in possible_indexes:
                        new_score = prev_partial_score + curr_probs[index]
                        hyp = (j, index, new_score, curr_probs[index], curr_basic_probs[0, index])
                        hypotheses_by_groups[group_index].append(hyp)
            for j, curr_hypotheses in enumerate(hypotheses_by_groups):
                curr_hypotheses = sorted(curr_hypotheses, key=(lambda x:x[2]))[:beam_width]
                group_start = j * beam_width
                is_active[group_start:group_start+beam_width] = False
                group_indexes = np.arange(group_start, group_start+len(curr_hypotheses))
                extend_history(tags, curr_hypotheses, group_indexes, pos=1)
                extend_history(probs, curr_hypotheses, group_indexes, pos=3)
                extend_history(basic_probs, curr_hypotheses, group_indexes, pos=4)
                extend_history(partial_scores, curr_hypotheses, group_indexes,
                               pos=2, func=lambda x, y: y)
                extend_history(lm_inputs, curr_hypotheses, group_indexes, pos=1,
                               func=lm_inputs_func, i=i, matrix=self.tags_.symbol_matrix_)
                is_active[group_indexes] = True
        # здесь нужно переделать words, probs в список
        tags_by_groups, probs_by_groups, basic_probs_by_groups = [], [], []
        for group_start in range(0, M, beam_width):
            # приводим к списку, чтобы иметь возможность сортировать
            active_indexes_for_group = list(np.where(is_completed[group_start:group_start+beam_width])[0])
            tags_by_groups.append([tags[group_start+i] for i in active_indexes_for_group])
            curr_group_probs = [np.array(probs[group_start+i])
                                for i in active_indexes_for_group]
            curr_basic_group_probs = [np.array(basic_probs[group_start+i])
                                      for i in active_indexes_for_group]
            if not return_log:
                curr_group_probs = [np.power(10.0, -elem) for elem in curr_group_probs]
                curr_basic_group_probs =\
                    [np.power(10.0, -elem) for elem in curr_basic_group_probs]
            probs_by_groups.append(curr_group_probs)
            basic_probs_by_groups.append(curr_basic_group_probs)
        return tags_by_groups, probs_by_groups, basic_probs_by_groups

    def _make_tag_embeddings(self, lm):
        # embeddings_weights.shape = (n_symbol_features, tag_embeddings_dim)
        embedding_weights = lm.get_embeddings_weights()
        if embedding_weights is None:
            return None
        # embeddings_weights.shape = (n_symbols, tag_embeddings_dim)
        embedding_weights = np.dot(lm.vocabulary_.symbol_matrix_, embedding_weights)
        self.tag_embeddings_dim_ = embedding_weights.shape[1]
        self.tag_embeddings_ = np.zeros(shape=(self.tags_number_, self.tag_embeddings_dim_))
        for i, tag in enumerate(self.tags_.symbols_):
            lm_index = lm.vocabulary_.toidx(tag)
            self.tag_embeddings_[i] = embedding_weights[lm_index]
        if self.normalize_lm_embeddings:
            self.tag_embeddings_ /= np.linalg.norm(self.tag_embeddings_, axis=1)[:,None]
        return self

    def build(self):
        # word_inputs = kl.Input(shape=(None, MAX_WORD_LENGTH+2), dtype="int32")
        word_inputs = kl.Input(shape=(None, MAX_WORD_LENGTH+2, self.symbols_number_), dtype="int32")
        inputs, basic_inputs = [word_inputs], [word_inputs]
        word_outputs = self._build_word_cnn(word_inputs)
        if self.word_dropout > 0.0:
            word_outputs = kl.Dropout(self.word_dropout)(word_outputs)
        if hasattr(self, "lm_"):
            if not self.use_fusion:
                lm_inputs = kl.Input(shape=(None, self.tags_number_), dtype="float")
                inputs.append(lm_inputs)
            lm_state_inputs = kl.Input(shape=(None, self.lm_state_dim_), dtype="float")
            inputs.append(lm_state_inputs)
        if len(self.word_vectorizers) > 0:
            additional_word_inputs = [kl.Input(shape=(None, vectorizer.dim), dtype="float")
                                      for vectorizer, _ in self.word_vectorizers]
            inputs.extend(additional_word_inputs)
            basic_inputs.extend(additional_word_inputs)
            additional_word_embeddings = [kl.Dense(dense_dim)(additional_word_inputs[i])
                                          for i, (_, dense_dim) in enumerate(self.word_vectorizers)]
            word_outputs = kl.Concatenate()([word_outputs] + additional_word_embeddings)
        additional_word_tag_inputs = [kl.Input(shape=(None, self.tags_number_), dtype="float")
                                      for vectorizer in self.word_tag_vectorizers]
        inputs.extend(additional_word_tag_inputs)
        basic_inputs.extend(additional_word_tag_inputs)
        pre_outputs, states = self._build_basic_network(word_outputs, additional_word_tag_inputs)
        loss = (leader_loss(self.leader_loss_weight) if self.use_leader_loss
                else "categorical_crossentropy")
        compile_args = {"optimizer": ko.nadam(clipnorm=5.0),
                        "loss": loss, "metrics": ["accuracy"]}
        if hasattr(self, "lm_"):
            position_inputs = kl.Lambda(positions_func)(word_inputs)
            if self.use_fusion:
                lm_state_inputs = TemporalDropout(lm_state_inputs, self.lm_dropout)
                fusion_inputs = kl.concatenate([states, lm_state_inputs, position_inputs])
                fusion_state_units = kl.TimeDistributed(
                    kl.Dense(self.fusion_state_units, activation="relu"))(fusion_inputs)
                final_outputs = kl.TimeDistributed(
                    kl.Dense(self.tags_number_, activation="softmax",
                             activity_regularizer=self.fusion_regularizer),
                    name="p_output")(fusion_state_units)
                decoder_inputs = [pre_outputs, states, position_inputs, lm_state_inputs]
            else:
                if self.use_rnn_for_weight_state:
                    first_gate_inputs = kl.Bidirectional(kl.LSTM(
                        self.weight_state_rnn_units, dropout=self.lstm_dropout,
                        return_sequences=True))(word_outputs)
                else:
                    first_gate_inputs = word_outputs
                lm_inputs = TemporalDropout(lm_inputs, self.lm_dropout)
                gate_inputs = kl.concatenate([first_gate_inputs, lm_state_inputs, position_inputs])
                gate_layer = WeightedCombinationLayer(name="p_output",
                                                      first_threshold=self.probs_threshold,
                                                      second_threshold=self.lm_probs_threshold,
                                                      use_dimension_bias=self.use_dimension_bias,
                                                      use_intermediate_layer=self.use_intermediate_activation_for_weights,
                                                      intermediate_dim=self.intermediate_units_for_weights)
                final_outputs = gate_layer([pre_outputs, lm_inputs, gate_inputs])
                decoder_inputs = [pre_outputs, word_outputs, position_inputs, lm_inputs, lm_state_inputs]
            outputs = [final_outputs, pre_outputs]
            loss_weights = [1, self.base_model_weight]
            compile_args["loss_weights"] = loss_weights
        else:
            outputs = pre_outputs
        self._compile_args = compile_args
        self.model_ = Model(inputs, outputs)
        self.model_.compile(**compile_args)
        self._embedder_ = kb.Function(basic_inputs + [kb.learning_phase()], [word_outputs, states])
        if hasattr(self, "lm_"):
            self._basic_model_ = kb.Function(basic_inputs + [kb.learning_phase()], decoder_inputs[:3])
            self._decoder_ = kb.Function(decoder_inputs + [kb.learning_phase()], [final_outputs])
        if self.verbose > 0:
            print(self.model_.summary())
        return self

    def _build_word_cnn(self, inputs):
        # inputs = kl.Lambda(kb.one_hot, arguments={"num_classes": self.symbols_number_},
        #                    output_shape=lambda x: tuple(x) + (self.symbols_number_,))(inputs)
        inputs = kl.Lambda(kb.cast, arguments={"dtype": "float32"})(inputs)
        char_embeddings = kl.Dense(self.char_embeddings_size, use_bias=False)(inputs)
        conv_outputs = []
        self.char_output_dim_ = 0
        for window_size, filters_number in zip(self.char_window_size, self.char_filters):
            curr_output = char_embeddings
            curr_filters_number = (min(self.char_filter_multiple * window_size, 200)
                                   if filters_number is None else filters_number)
            for _ in range(self.char_conv_layers - 1):
                curr_output = kl.Conv2D(curr_filters_number, (1, window_size),
                                        padding="same", activation="relu",
                                        data_format="channels_last")(curr_output)
                if self.conv_dropout > 0.0:
                    curr_output = kl.Dropout(self.conv_dropout)(curr_output)
            curr_output = kl.Conv2D(curr_filters_number, (1, window_size),
                                    padding="same", activation="relu",
                                    data_format="channels_last")(curr_output)
            conv_outputs.append(curr_output)
            self.char_output_dim_ += curr_filters_number
        if len(conv_outputs) > 1:
            conv_output = kl.Concatenate(axis=-1)(conv_outputs)
        else:
            conv_output = conv_outputs[0]
        highway_input = kl.Lambda(kb.max, arguments={"axis": -2})(conv_output)
        if self.intermediate_dropout > 0.0:
            highway_input = kl.Dropout(self.intermediate_dropout)(highway_input)
        for i in range(self.char_highway_layers - 1):
            highway_input = Highway(activation="relu")(highway_input)
            if self.highway_dropout > 0.0:
                highway_input = kl.Dropout(self.highway_dropout)(highway_input)
        if self.char_highway_layers > 0:
            highway_output = Highway(activation="relu")(highway_input)
        else:
            highway_output = highway_input
        return highway_output

    def _build_basic_network(self, word_outputs, additional_embeddings=None):
        """
        Creates the basic network architecture,
        transforming word embeddings to intermediate outputs
        """
        lstm_outputs = word_outputs
        for j in range(self.word_lstm_layers - 1):
            lstm_outputs = kl.Bidirectional(
                kl.LSTM(self.word_lstm_units[j], return_sequences=True,
                        dropout=self.lstm_dropout, name="word_lstm_{}".format(j+1)))(lstm_outputs)
        lstm_outputs = kl.Bidirectional(
            kl.LSTM(self.word_lstm_units[-1], return_sequences=True, dropout=self.lstm_dropout,
                    name="word_lstm_{}".format(self.word_lstm_layers)))(lstm_outputs)
        if hasattr(self, "tag_embeddings_"):
            pre_outputs = self.tag_embeddings_output_layer(lstm_outputs)
        else:
            if len(self.word_tag_vectorizers) > 0:
                pre_outputs = kl.TimeDistributed(kl.Dense(self.tags_number_))(lstm_outputs)
                for elem in additional_embeddings:
                    pre_outputs = WeightedSum()([pre_outputs, elem])
                pre_outputs = kl.TimeDistributed(kl.Activation(activation="softmax"), name="p")(pre_outputs)
                if self.regularizer is not None:
                    pre_outputs = kl.ActivityRegularization(l2=self.regularizer.l2)(pre_outputs)
            else:
                pre_outputs = kl.TimeDistributed(
                    kl.Dense(self.tags_number_, activation="softmax",
                             activity_regularizer=self.regularizer),
                name="p")(lstm_outputs)
        return pre_outputs, lstm_outputs

    def _freeze_output_network(self):
        for layer in self.model_.layers:
            if layer.name.startswith("word_lstm") or layer.name == "p":
                layer.trainable = False
        self.model_.compile(**self._compile_args)
        return

    def tag_embeddings_output_layer(self, lstm_outputs):
        outputs_embeddings = kl.TimeDistributed(
            kl.Dense(self.tag_embeddings_dim_, use_bias=False))(lstm_outputs)
        if self.normalize_lm_embeddings:
            norm_layer = kl.Lambda(kb.l2_normalize, arguments={"axis": -1})
            outputs_embeddings = kl.TimeDistributed(norm_layer)(outputs_embeddings)
        score_layer = kl.Lambda(kb.dot, arguments={"y": kb.constant(self.tag_embeddings_.T)})
        scores = kl.TimeDistributed(score_layer)(outputs_embeddings)
        probs = kl.TimeDistributed(kl.Activation("softmax"), name="p")(scores)
        return probs


def extend_history(histories, hyps, indexes, start=0, pos=None,
                   history_pos=0, value=None, func="append", **kwargs):
    to_append = ([elem[pos] for elem in hyps]
                 if value is None else [value] * len(hyps))
    if func == "append":
        func = lambda x, y: x + [y]
    elif func == "sum":
        func = lambda x, y: x + y
    elif not callable(func):
        raise ValueError("func must be 'append', 'sum' or a callable object")
    group_histories = [func(histories[elem[history_pos]], value, **kwargs)
                       for elem, value in zip(hyps, to_append)]
    for i, index in enumerate(indexes):
        histories[start+index] = group_histories[i]