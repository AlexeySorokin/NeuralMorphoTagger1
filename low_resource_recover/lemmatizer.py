from collections import defaultdict
import itertools

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
                 use_tags=False, model_params=None):
        self.models_number = models_number
        self.allow_infixes = allow_infixes
        self.model_params = model_params or dict()
        self.count_threshold = count_threshold
        self.use_tags = use_tags

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
              tags=None, dev_tags=None, nepochs=20, batch_size=16):
        self.symbols_ = Vocabulary().train(words)
        data = self.transform(words)
        targets = self.make_targets(words, lemmas)
        data, targets = clear_data(data, targets, threshold=self.count_threshold)
        if tags is not None:
            self.tags_ = FeatureVocabulary().train([tags])
            tag_data = [self.tags_.toidx(x) for x in tags]
        if dev_words is not None:
            dev_data = self.transform(dev_words)
            dev_targets = self.make_targets(dev_words, dev_lemmas)
            dev_data, dev_targets = clear_data(dev_data, dev_targets)
        else:
            dev_data, dev_targets = None, None
        self.build(**self.model_params)
        self._train_model(data, targets, dev_data, dev_targets, nepochs=nepochs, batch_size=batch_size)
        return self

    def _train_model(self, data, targets, dev_data=None, dev_targets=None,
                     nepochs=20, batch_size=16, patience=5):
        additional_targets, additional_dev_targets = [], []
        additional_classes_number, use_length_for_additional_targets = [], []
        if self.allow_infixes:
            additional_targets.append([elem[1] for elem in targets])
            additional_classes_number.append(None)
            targets = [elem[0] for elem in targets]
        for model in self.models_:
            train_gen = DataGenerator(data, targets, additional_targets=additional_targets,
                                      classes_number=DataGenerator.POSITIONS_AS_CLASSES,
                                      additional_classes_number=additional_classes_number,
                                      shuffle=True, nepochs=None, batch_size=batch_size)
            callbacks = []
            if dev_data is not None:
                if self.allow_infixes:
                    additional_dev_targets.append([elem[1] for elem in dev_targets])
                    dev_targets = [elem[0] for elem in dev_targets]
                dev_gen = DataGenerator(dev_data, dev_targets, additional_targets=additional_dev_targets,
                                        classes_number=DataGenerator.POSITIONS_AS_CLASSES,
                                        additional_classes_number=additional_classes_number,
                                        shuffle=False, batch_size=batch_size)
                dev_steps = dev_gen.steps_per_epoch
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
        probs = kl.Softmax(name="ends")(similarities)
        outputs = [probs]
        loss, metrics = ["categorical_crossentropy"], {"ends": "accuracy"}
        if self.allow_infixes:
            infix_similarities = kl.Dense(1)(conv_output)
            infix_similarities = kl.Lambda(lambda x: x[..., 0],
                                           output_shape=(lambda x: x[:-1]))(infix_similarities)
            infix_probs = kl.Activation("sigmoid", name="deletions")(infix_similarities)
            outputs.append(infix_probs)
            # вставить правильные метрики
            loss.append(MulticlassSigmoidLoss())
            metrics["deletions"] = MulticlassSigmoidAccuracy()
        model = Model(inputs, outputs)
        model.compile(optimizer="adam", loss=loss, metrics=metrics)
        return model

    def build(self):
        self.models_ = [None] * self.models_number
        for i in range(self.models_number):
            self.models_[i] = self._build_model()
        print(self.models_[0].summary())
        return self

    def predict(self, words, batch_size=16, model_indexes=None):
        if model_indexes is None:
            model_indexes = list(range(self.models_number))
        elif isinstance(model_indexes, int):
            model_indexes = [model_indexes]
        data = self.transform(words)
        answer = [None] * len(words)
        test_gen = DataGenerator(data, yield_indexes=True, yield_targets=False,
                                 batch_size=batch_size, nepochs=1)
        for batch, indexes in test_gen:
            predictions = [self.models_[i].predict(batch) for i in model_indexes]
            if self.allow_infixes:
                curr_probs = np.mean([elem[0] for elem in predictions], axis=0)
                curr_deletion_probs = np.mean([elem[1] for elem in predictions], axis=0)
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
USE_TAGS = True

if __name__ == "__main__":
    use_tags = USE_TAGS
    train_file = ["data/low-resource/evn/splitted/evn.train.ud", "data/low-resource/evn/splitted/evn.dev.ud"]
    dev_file = "data/low-resource/test/gold.evn.test.ud"
    test_file = "data/low-resource/test/gold.evn.test.ud"
    # чтение данных
    tag_sents, word_sents, lemma_sents = read_tags_infile(train_file, **read_params)
    dev_tag_sents, dev_word_sents, dev_lemma_sents = read_tags_infile(dev_file, **read_params)
    # преобразование в список слов
    words = list(itertools.chain.from_iterable(word_sents))
    lemmas = list(itertools.chain.from_iterable(lemma_sents))
    tags = list(itertools.chain.from_iterable(tag_sents)) if use_tags else None
    dev_words = list(itertools.chain.from_iterable(dev_word_sents))
    dev_lemmas = list(itertools.chain.from_iterable(dev_lemma_sents))
    dev_tags = list(itertools.chain.from_iterable(dev_tag_sents)) if use_tags else None
    # обучение модели
    lemmatizer = Lemmatizer(allow_infixes=False, use_tags=use_tags).train(
        words, lemmas, dev_words, dev_lemmas, tags=tags, dev_tags=dev_tags, nepochs=25)
    # измерение качества
    test_tag_sents, test_word_sents, test_lemma_sents = read_tags_infile(test_file, **read_params)
    test_words = list(itertools.chain.from_iterable(test_word_sents))
    test_lemmas = list(itertools.chain.from_iterable(test_lemma_sents))
    test_tags = list(itertools.chain.from_iterable(test_tag_sents)) if use_tags else None
    pred_lemmas = lemmatizer.predict(test_words, tags=test_tags)
    equal = sum(int(x==y) for x, y in zip(test_lemmas, pred_lemmas))
    print("Качество лемматизации: {:.2f}, {} из {}.".format(
        100 * equal / len(test_lemmas), equal, len(test_lemmas)))
