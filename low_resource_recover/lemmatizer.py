from collections import defaultdict
import itertools

import keras.layers as kl
from keras import Model
import keras.activations as kact
from keras.callbacks import EarlyStopping, ModelCheckpoint

from common.common import *
from common.read import read_tags_infile
from common.vocabulary import Vocabulary
from common.generate import sample_redundant_data, DataGenerator


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
    return curr_stem_length, infixes


def clear_data(data, targets, threshold=None):
    indices = [i for i, elem in enumerate(targets) if elem is not None]
    data = [data[i] for i in indices]
    targets = [targets[i] for i in indices]
    if threshold is not None:
        data, targets = sample_redundant_data(data, targets, threshold=threshold)
    return data, targets


class Lemmatizer:

    BATCH_SIZE, NEPOCHS = 16, 20

    def __init__(self, models_number=1, allow_infixes=False, count_threshold=None, model_params=None):
        self.models_number = models_number
        self.allow_infixes = allow_infixes
        self.model_params = model_params or dict()
        self.count_threshold = count_threshold

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
            answer.append(pattern)
        return answer

    def train(self, words, lemmas, dev_words=None, dev_lemmas=None,
              nepochs=20, batch_size=16):
        self.symbols_ = Vocabulary().train(words)
        data = self.transform(words)
        targets = self.make_targets(words, lemmas)
        data, targets = clear_data(data, targets, threshold=self.count_threshold)
        if dev_words is not None:
            dev_data = self.transform(dev_words)
            dev_targets = self.make_targets(dev_words, dev_lemmas)
        else:
            dev_data, dev_targets = None, None
        self.build(**self.model_params)
        self._train_model(data, targets, dev_data, dev_targets, nepochs=nepochs, batch_size=batch_size)
        return self

    def _train_model(self, data, targets, dev_data=None, dev_targets=None,
                     nepochs=20, batch_size=16, patience=5):
        model: Model
        for model in self.models_:
            train_gen = DataGenerator(data, targets, positions_are_classes=True,
                                      shuffle=True, nepochs=None, batch_size=batch_size)
            callbacks = []
            if dev_data is not None:
                dev_gen = DataGenerator(dev_data, dev_targets, positions_are_classes=True,
                                        shuffle=False, batch_size=batch_size)
                dev_steps = dev_gen.steps_per_epoch
                monitor = "val_acc" if not self.allow_infixes else None
                callback = EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True)
                callbacks.append(callback)
            else:
                dev_gen, dev_steps = None, None
            model.fit_generator(train_gen, steps_per_epoch=train_gen.steps_per_epoch, epochs=nepochs,
                                validation_data=dev_gen, validation_steps=dev_steps,
                                callbacks=callbacks)
        return model

    def _build_model(self, symbol_embeddings_size=32, conv_layers=2,
                     filter_width=5, filters_number=192, use_lstm=False):
        if isinstance(filter_width, int):
            filter_width = [filter_width]
        if isinstance(filters_number, int):
            filters_number = [filters_number]
        symbol_inputs = kl.Input(shape=(None,), dtype="float32")
        inputs = [symbol_inputs]
        symbol_embeddings = kl.Embedding(self.symbols_.symbols_number_, symbol_embeddings_size)(symbol_inputs)
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
        probs = kl.Softmax()(similarities)
        outputs = [probs]
        loss, metrics = ["categorical_crossentropy"], ["accuracy"]
        if self.allow_infixes:
            infix_similarities = kl.Dense(total_filters_number)(conv_output)
            infix_probs = kl.Activation("sigmoid")(infix_similarities)
            outputs.append(infix_probs)
            # вставить правильные метрики
            loss.append(["binary_crossentropy"])
            metrics.append(["accuracy"])
        model = Model(inputs, outputs)
        model.compile(optimizer="adam", loss=loss, metrics=metrics)
        return model

    def build(self):
        self.models_ = [None] * self.models_number
        for i in range(self.models_number):
            self.models_[i] = self._build_model()
        print(self.models_[0].summary())
        return self

    def predict(self, words, batch_size=16):
        data = self.transform(words)
        answer = [None] * len(words)
        test_gen = DataGenerator(data, yield_indexes=True, yield_targets=False,
                                 batch_size=batch_size, nepochs=1)
        for batch, indexes in test_gen:
            curr_probs = [model.predict(batch) for model in self.models_]
            curr_probs = np.mean(curr_probs, axis=0)
            positions = np.argmax(curr_probs, axis=-1) - 1
            for index, pos in zip(indexes, positions):
                word = words[index]
                if self.allow_infixes:
                    raise NotImplementedError
                else:
                    answer[index] = word[:pos]
        return answer



read_params = {"to_lower": False, "append_case": True, "return_source_words": True, "return_lemmas": True}

if __name__ == "__main__":
    train_file = "data/low-resource/evn/splitted/evn.train.ud"
    dev_file = "data/low-resource/evn/splitted/evn.dev.ud"
    test_file = "data/low-resource/test/gold.evn.test.ud"
    # чтение данных
    tag_sents, word_sents, lemma_sents = read_tags_infile(train_file, **read_params)
    dev_tag_sents, dev_word_sents, dev_lemma_sents = read_tags_infile(dev_file, **read_params)
    # преобразование в список слов
    words = list(itertools.chain.from_iterable(word_sents))
    lemmas = list(itertools.chain.from_iterable(lemma_sents))
    dev_words = list(itertools.chain.from_iterable(dev_word_sents))
    dev_lemmas = list(itertools.chain.from_iterable(dev_lemma_sents))
    # обучение модели
    lemmatizer = Lemmatizer(models_number=3).train(words, lemmas, dev_words, dev_lemmas, nepochs=2)
    # измерение качества
    test_tag_sents, test_word_sents, test_lemma_sents = read_tags_infile(test_file, **read_params)
    test_words = list(itertools.chain.from_iterable(test_word_sents))
    test_lemmas = list(itertools.chain.from_iterable(test_lemma_sents))
    pred_lemmas = lemmatizer.predict(test_words)
    equal = sum(int(x==y) for x, y in zip(test_lemmas, pred_lemmas))
    print("Качество лемматизации: {:.2f}, {} из {}.".format(
        100 * equal / len(test_lemmas), equal, len(test_lemmas)))
