import sys
from collections import defaultdict
import itertools
import ujson as json

import tensorflow as tf
import keras.backend.tensorflow_backend as kbt

import keras.layers as kl
from keras import Model
import keras.activations as kact
from keras.callbacks import EarlyStopping, ModelCheckpoint

from common.common import *
from common.read import read_tags_infile
from common.vocabulary import Vocabulary, FeatureVocabulary
from common.generate import sample_redundant_data, DataGenerator
from common.losses import MulticlassSigmoidLoss, MulticlassSigmoidAccuracy


def extract_lemmatization_pattern(word, lemma, allow_infixes=False, add_one=False):
    stem_length, infixes = None, []
    if not allow_infixes:
        if word.startswith(lemma):
            stem_length = len(lemma) + int(add_one)
        return stem_length
    curr_stem_length = 0
    for i in range(len(lemma)):
        while curr_stem_length < len(word) and word[curr_stem_length] != lemma[i]:
            infixes.append(curr_stem_length)
            curr_stem_length += 1
        if curr_stem_length == len(word):
            return None
        curr_stem_length += 1
    if add_one:
        curr_stem_length += 1
        infixes = [i+1 for i in infixes]
    return [curr_stem_length, infixes]


def clear_data(data, targets, threshold=None):
    indices = [i for i, elem in enumerate(targets) if elem is not None]
    data = [data[i] for i in indices]
    targets = [targets[i] for i in indices]
    if threshold is not None:
        data, targets = sample_redundant_data(data, targets, threshold=threshold)
    return data, targets


class Lemmatizer:

    BATCH_SIZE, NEPOCHS = 16, 20

    def __init__(self, models_number=1, allow_infixes=False, count_threshold=None,
                 use_tags=False, predict_tags=False, model_params=None):
        self.models_number = models_number
        self.allow_infixes = allow_infixes
        self.model_params = model_params or dict()
        self.count_threshold = count_threshold
        self.use_tags = use_tags
        self.predict_tags = predict_tags

    def transform(self, words):
        answer = []
        for word in words:
            curr_word = [0] * (len(word) + 2)
            curr_word[0], curr_word[-1] = BEGIN, END
            for i, a in enumerate(word, 1):
                curr_word[i] = self.symbols_.toidx(a)
            answer.append(curr_word)
        return answer

    def make_targets(self, words, lemmas):
        answer = []
        for word, lemma in zip(words, lemmas):
            pattern = extract_lemmatization_pattern(word, lemma, allow_infixes=self.allow_infixes, add_one=True)
            if pattern is not None and self.allow_infixes:
                infixes = np.zeros(shape=(len(word) + 2,), dtype=int)
                infixes[pattern[1]] = 1
                pattern[1] = infixes
            answer.append(pattern)
        return answer

    def train(self, words, lemmas, dev_words=None, dev_lemmas=None,
              tags=None, dev_tags=None, nepochs=20, batch_size=16,
              patience=-1):
        self.symbols_ = Vocabulary().train(words)
        data = self.transform(words)
        targets = self.make_targets(words, lemmas)
        data, targets = clear_data(data, targets, threshold=self.count_threshold)
        if tags is not None:
            self.tags_ = FeatureVocabulary(min_count=3).train([tags])
            if predict_tags:
                tag_data = [self.tags_.toidx(x) for x in tags]
            else:
                tag_data = [self.tags_.to_vector(x, return_vector=True) for x in tags]
        else:
            tag_data, dev_tag_data = None, None
        if dev_words is not None:
            dev_data = self.transform(dev_words)
            dev_targets = self.make_targets(dev_words, dev_lemmas)
            dev_data, dev_targets = clear_data(dev_data, dev_targets)
            if tags is not None:
                if predict_tags:
                    dev_tag_data = [self.tags_.toidx(x) for x in dev_tags]
                else:
                    dev_tag_data = [self.tags_.to_vector(x, return_vector=True) for x in dev_tags]
        else:
            dev_data, dev_targets = None, None
        self.build(**self.model_params)
        self._train_model(data, targets, dev_data, dev_targets,
                          tag_data=tag_data, dev_tag_data=dev_tag_data,
                          nepochs=nepochs, batch_size=batch_size, patience=patience)
        return self

    def _train_model(self, data, targets, dev_data=None, dev_targets=None,
                     tag_data=None, dev_tag_data=None,
                     nepochs=20, batch_size=16, patience=-1):
        additional_data, additional_dev_data = [], []
        pad_additional_data = []
        additional_targets, additional_dev_targets = [], []
        additional_classes_number, use_length_for_additional_targets = [], []
        if self.allow_infixes:
            additional_targets.append([elem[1] for elem in targets])
            additional_classes_number.append(None)
            targets = [elem[0] for elem in targets]
        if self.use_tags:
            additional_data.append(tag_data)
            pad_additional_data.append(False)
        elif self.predict_tags:
            additional_targets.append(tag_data)
            additional_classes_number.append(self.tags_.symbols_number_)
        for model in self.models_:
            train_gen = DataGenerator(data, targets, additional_data=additional_data,
                                      pad_additional_data=pad_additional_data,
                                      additional_targets=additional_targets,
                                      classes_number=DataGenerator.POSITIONS_AS_CLASSES,
                                      additional_classes_number=additional_classes_number,
                                      shuffle=True, nepochs=None, batch_size=batch_size)
            callbacks = []
            if dev_data is not None:
                if self.allow_infixes:
                    additional_dev_targets.append([elem[1] for elem in dev_targets])
                    dev_targets = [elem[0] for elem in dev_targets]
                if self.use_tags:
                    additional_dev_data.append(dev_tag_data)
                elif self.predict_tags:
                    additional_dev_targets.append(dev_tag_data)
                dev_gen = DataGenerator(dev_data, dev_targets, additional_data=additional_dev_data,
                                        pad_additional_data=pad_additional_data,
                                        additional_targets=additional_dev_targets,
                                        classes_number=DataGenerator.POSITIONS_AS_CLASSES,
                                        additional_classes_number=additional_classes_number,
                                        shuffle=False, batch_size=batch_size)
                dev_steps = dev_gen.steps_per_epoch
                if patience >= 0:
                    monitor = "val_acc" if not self.allow_infixes else "val_ends_acc"
                    callback = EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True)
                    callbacks.append(callback)
            else:
                dev_gen, dev_steps = None, None
            model.fit_generator(train_gen, steps_per_epoch=train_gen.steps_per_epoch, epochs=nepochs,
                                validation_data=dev_gen, validation_steps=dev_steps,
                                callbacks=callbacks)
        return self

    def _build_model(self, symbol_embeddings_size=32, conv_layers=2,
                     filter_width=5, filters_number=192,
                     tag_embeddings_size=64, tag_state_size=128,
                     tag_loss_weight=1.0, use_lstm=False):
        if isinstance(filter_width, int):
            filter_width = [filter_width]
        if isinstance(filters_number, int):
            filters_number = [filters_number]
        symbol_inputs = kl.Input(shape=(None,), dtype="float32")
        inputs = [symbol_inputs]
        symbol_embeddings = kl.Embedding(self.symbols_.symbols_number_, symbol_embeddings_size)(symbol_inputs)
        if self.use_tags:
            tag_inputs = kl.Input(shape=(self.tags_.symbol_vector_size_,), dtype="float32")
            inputs.append(tag_inputs)
            tag_embeddings = kl.Dense(tag_embeddings_size)(tag_inputs)
            # tag_embeddings = kl.Lambda(kb.expand_dims, arguments={"axis": 1})(tag_embeddings)
            tiled_tag_embeddings = kl.Lambda(repeat_, arguments={"k": kb.shape(symbol_embeddings)[1]})(tag_embeddings)
            symbol_embeddings = kl.Concatenate()([symbol_embeddings, tiled_tag_embeddings])
        conv_outputs = []
        for f, w in zip(filters_number, filter_width):
            curr_conv_output = symbol_embeddings
            for _ in range(conv_layers):
                curr_conv_output = kl.Conv1D(f, w, padding="same", activation="relu")(curr_conv_output)
                curr_conv_output = kl.BatchNormalization()(curr_conv_output)
            conv_outputs.append(curr_conv_output)
        conv_output = kl.Concatenate()(conv_outputs) if len(conv_outputs) > 1 else conv_outputs[0]
        total_filters_number = sum(filters_number)
        similarities = kl.Dense(1)(conv_output)
        similarities = kl.Lambda(lambda x: x[...,0], output_shape=(lambda x: x[:-1]))(similarities)
        probs = kl.Softmax(name="ends")(similarities)
        outputs = [probs]
        loss, metrics, loss_weights = ["categorical_crossentropy"], {"ends": "accuracy"}, [1.0]
        if self.allow_infixes:
            infix_similarities = kl.Dense(1)(conv_output)
            infix_similarities = kl.Lambda(lambda x: x[..., 0],
                                           output_shape=(lambda x: x[:-1]))(infix_similarities)
            infix_probs = kl.Activation("sigmoid", name="deletions")(infix_similarities)
            outputs.append(infix_probs)
            # вставить правильные метрики
            loss.append(MulticlassSigmoidLoss())
            loss_weights.append(1.0)
            # metrics["deletions"] = MulticlassSigmoidAccuracy()
        if self.predict_tags:
            tag_states = kl.Dense(tag_state_size)(conv_output)
            tag_states = kl.GlobalMaxPooling1D()(tag_states)
            tag_probs = kl.Dense(self.tags_.symbols_number_, activation="softmax", name="tag_probs")(tag_states)
            outputs.append(tag_probs)
            loss.append("categorical_crossentropy")
            loss_weights.append(tag_loss_weight)
            # metrics["tag_probs"] = "accuracy"
        model = Model(inputs, outputs)
        model.compile(optimizer="adam", loss=loss, metrics=metrics, loss_weights=loss_weights)
        return model

    def build(self, **kwargs):
        self.models_ = [None] * self.models_number
        for i in range(self.models_number):
            self.models_[i] = self._build_model(**kwargs)
        print(self.models_[0].summary())
        return self

    def predict(self, words, tags=None, batch_size=16, model_indexes=None):
        if model_indexes is None:
            model_indexes = list(range(self.models_number))
        elif isinstance(model_indexes, int):
            model_indexes = [model_indexes]
        data = self.transform(words)
        if self.use_tags:
            tag_data = [self.tags_.to_vector(x, return_vector=True) for x in tags]
            additional_data = [tag_data]
            pad_additional_data = [False]
        else:
            additional_data, pad_additional_data = [], []
        answer = [None] * len(words)
        test_gen = DataGenerator(data, additional_data=additional_data,
                                 pad_additional_data=pad_additional_data,
                                 yield_indexes=True, yield_targets=False,
                                 batch_size=batch_size, nepochs=1)
        for batch, indexes in test_gen:
            predictions = [self.models_[i].predict(batch) for i in model_indexes]
            if self.allow_infixes:
                curr_probs = np.mean([elem[0] for elem in predictions], axis=0)
                curr_deletion_probs = np.mean([elem[1] for elem in predictions], axis=0)
            elif self.predict_tags:
                curr_probs = np.mean([elem[0] for elem in predictions], axis=0)
            else:
                curr_probs = np.mean(predictions, axis=0)
            positions = np.argmax(curr_probs, axis=-1) - 1
            if self.allow_infixes:
                deletions = [list(np.where(elem >= 0.5)[0] - 1) for elem in curr_deletion_probs]
            else:
                deletions = [[] for _ in positions]
            for index, pos, curr_deletions in zip(indexes, positions, deletions):
                word = words[index]
                if self.allow_infixes:
                    answer[index] = "".join(a for i, a in enumerate(word[:pos]) if i not in curr_deletions)
                else:
                    answer[index] = word[:pos]
        return answer



read_params = {"to_lower": False, "append_case": True, "return_source_words": True, "return_lemmas": True}
default_values = {"use_tags": False, "allow_infixes": False, "models_number": 1,
                  "model_params": dict(), "train_params": dict()}

def read_config(infile):
    with open(infile, "r", encoding="utf8") as fin:
        config = json.load(fin)
    for key, value in default_values.items():
        if key not in config:
            config[key] = value
    return config


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    kbt.set_session(tf.Session(config=config))
    config = read_config(sys.argv[1])
    use_tags, predict_tags = config["use_tags"], config["predict_tags"]
    tag_sents, word_sents, lemma_sents = read_tags_infile(config["train_file"], **read_params)
    # преобразование в список слов
    words = list(itertools.chain.from_iterable(word_sents))
    lemmas = list(itertools.chain.from_iterable(lemma_sents))
    tags = list(itertools.chain.from_iterable(tag_sents)) if use_tags or predict_tags else None
    if "dev_file" in config:
        dev_tag_sents, dev_word_sents, dev_lemma_sents = read_tags_infile(config["dev_file"], **read_params)
        dev_words = list(itertools.chain.from_iterable(dev_word_sents))
        dev_lemmas = list(itertools.chain.from_iterable(dev_lemma_sents))
        if use_tags or predict_tags:
            dev_tags = list(itertools.chain.from_iterable(dev_tag_sents))
        else:
            dev_tags = None
    else:
        dev_words, dev_lemmas, dev_tags = None, None, None
    # обучение модели
    lemmatizer = Lemmatizer(allow_infixes=config["allow_infixes"],
                            use_tags=use_tags, predict_tags=not use_tags and config["predict_tags"],
                            models_number=config["models_number"], model_params=config["model_params"])
    lemmatizer.train(words, lemmas, dev_words, dev_lemmas,
                     tags=tags, dev_tags=dev_tags, **config["train_params"])
    # измерение качества
    if "test_file" in config:
        test_tag_sents, test_word_sents, test_lemma_sents = read_tags_infile(config["test_file"], **read_params)
        test_words = list(itertools.chain.from_iterable(test_word_sents))
        test_lemmas = list(itertools.chain.from_iterable(test_lemma_sents))
        test_tags = list(itertools.chain.from_iterable(test_tag_sents)) if use_tags else None
        pred_lemmas = lemmatizer.predict(test_words, tags=test_tags)
        equal = sum(int(x==y) for x, y in zip(test_lemmas, pred_lemmas))
        print("Качество лемматизации: {:.2f}, {} из {}.".format(
            100 * equal / len(test_lemmas), equal, len(test_lemmas)))
