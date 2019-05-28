from collections import defaultdict
import itertools
import sys

import numpy as np
from keras.callbacks import Callback, EarlyStopping

from common.common import PAD, to_one_hot


class CustomCallback(Callback):

    def __init__(self):
        super(CustomCallback, self).__init__()

    def on_train_begin(self, logs=None):
        self.verbose = self.params['verbose']
        self.epochs = self.params['epochs']
        self.train_losses = []
        self.val_losses = []
        self.best_loss = np.inf

    def on_epoch_begin(self, epoch, logs=None):
        print('Epoch %d/%d' % (epoch + 1, self.epochs))

    def on_epoch_end(self, epoch, logs=None):
        self.train_losses.append(logs["loss"])
        self.val_losses.append(logs["val_loss"])
        print("loss: {:.4f}, val_loss: {:.4f}".format(self.train_losses[-1], self.val_losses[-1]))
        if self.val_losses[-1] < self.best_loss:
            self.best_loss = self.val_losses[-1]
            print(", best loss\n")
        else:
            print("\n")


class MultirunEarlyStopping(EarlyStopping):

    def on_train_begin(self, logs=None):
        wait, stopped_epoch = self.wait, self.stopped_epoch
        best = getattr(self, "best", np.Inf if self.monitor_op == np.less else -np.Inf)
        super().on_train_begin(logs=logs)
        self.wait, self.stopped_epoch = wait, stopped_epoch
        self.best = best


def _make_many_hot_array(data, classes_number):
    flat_data, shape = np.ravel(data), np.shape(data)
    if flat_data.dtype != "object":
        flat_data = np.reshape(flat_data, (-1, 1))
        shape = shape[:-1]
    answer = np.zeros(shape=(len(flat_data), classes_number), dtype=np.int32)
    for i, elem in enumerate(flat_data):
        answer[i, elem] = 1
    answer = answer.reshape(shape + (-1,))
    return answer


def make_batch(data, fields_to_one_hot=None):
    if fields_to_one_hot is None:
        fields_to_one_hot = dict()
    fields_number = len(data[0])
    answer = [None] * fields_number
    for k in range(fields_number):
        if k in fields_to_one_hot:
            answer[k] = _make_many_hot_array([elem[k] for elem in data], fields_to_one_hot[k])
        else:
            answer[k] = np.array([elem[k] for elem in data])
    return answer


class TaggingDataGenerator:

    def __init__(self, X, indexes_by_buckets, output_symbols_number,
                 batch_size=None, epochs=None, active_buckets=None,
                 use_last=False, has_answer=True, shift_answer=False,
                 duplicate_answer=False, answers_number=1,
                 fields_number=None, fields_to_one_hot=None,
                 has_multiple_labels=False,
                 shuffle=True, weights=None, yield_weights=True):
        self.X = X
        self.indexes_by_buckets = indexes_by_buckets
        self.output_symbols_number = output_symbols_number
        self.batch_size = batch_size
        self.epochs = epochs if epochs is not None else sys.maxsize
        self.active_buckets = np.array(active_buckets) if active_buckets is not None else None
        self.has_answer = has_answer
        self.shift_answer = shift_answer
        self.duplicate_answer = duplicate_answer
        self.answers_number = answers_number
        self.fields_to_one_hot = dict(fields_to_one_hot or [])
        self.has_multiple_labels = has_multiple_labels
        self.shuffle = shuffle
        self.weights = weights if weights is not None else np.ones(shape=(len(X)), dtype=float)
        self.yield_weights = yield_weights
        self._initialize(use_last, fields_number)

    @property
    def steps_on_epoch(self):
        return len(self.batches_indexes)

    @property
    def has_multiple_outputs(self):
        return (isinstance(self.answer_index, list))

    def _initialize(self, use_last=False, fields_number=None):
        # batch starts
        if self.batch_size is None: # outputs each bucket as a single batch
            self.batches_indexes = [(i, 0) for i in range(len(self.indexes_by_buckets))]
        else:
            self.batches_indexes = [(i, j) for i, bucket in enumerate(self.indexes_by_buckets)
                                    for j in range(0, len(bucket), self.batch_size)]
        if use_last:
            self.answer_index = 0
        else:
            if self.answers_number == 1:
                self.answer_index = -1
            else:
                self.answer_index = [-k - 1 for k in range(self.answers_number)][::-1]
        # number of fields to output
        if self.has_multiple_outputs:
            self.fields_number = len(self.X[0]) - self.answers_number
        else:
            self.fields_number = fields_number or (len(self.X[0]) - int(self.has_answer and not use_last))
        # self.answer_index = 0 if use_last else -1 if self.has_answer else None
        # epochs counter
        self.curr_epoch, self.step = 0, 0
        return self

    def _is_bucket_active(self, epoch, bucket):
        if self.active_buckets is None:
            return True
        epoch = min(epoch, len(self.active_buckets)-1)
        return self.active_buckets[epoch, bucket]

    def _calculate_data_size(self):
        self.total_data_length, self.total_array_size = 0, 0
        for i, curr_indexes in enumerate(self.indexes_by_buckets):
            if self._is_bucket_active(self.curr_epoch, i):
                for j in curr_indexes:
                    if isinstance(self.answer_index, list):
                        answer_index = self.answer_index[0]
                    else:
                        answer_index = self.answer_index
                    self.total_array_size += np.count_nonzero(self.X[j][answer_index] != PAD) - 1
                self.total_data_length += len(curr_indexes)
        return

    def _on_epoch_start(self):
        if self.shuffle:
            for elem in self.indexes_by_buckets:
                np.random.shuffle(elem)
            np.random.shuffle(self.batches_indexes)
        self._calculate_data_size()
        self.curr_batch_indexes = [(i, start) for i, start in self.batches_indexes
                                   if self._is_bucket_active(self.curr_epoch, i)]
        return

    def _make_labels(self, labels):
        if self.has_multiple_outputs:
            y_to_yield = [to_one_hot(curr_labels, k)
                          for curr_labels, k in zip(labels, self.output_symbols_number)]
            y_to_yield = np.concatenate(y_to_yield, axis=-1)
        else:
            y_to_yield = to_one_hot(labels, self.output_symbols_number)
            if self.has_multiple_labels:
                y_to_yield = np.max(y_to_yield, axis=-2)
        if self.duplicate_answer:
            y_to_yield = [y_to_yield, y_to_yield]
        return y_to_yield

    def _make_weights(self, labels, weights):
        weights_to_yield = np.ones_like(labels[:,0], dtype=np.float32)
        if np.ndim(weights_to_yield) == 2:
            weights_to_yield = weights_to_yield[:,0]
        weights_to_yield *= weights
        if self.yield_weights and self.total_array_size > 0:
            # the fraction of current batch in the whole data
            weights_to_yield *= self.total_data_length * labels.shape[1] / self.total_array_size
        if self.duplicate_answer:
            weights_to_yield = [weights_to_yield, weights_to_yield]
        return weights_to_yield

    def __next__(self):
        if self.step == 0:
            self._on_epoch_start()
        i, start = self.curr_batch_indexes[self.step]
        bucket_size = len(self.indexes_by_buckets[i])
        end = min(bucket_size, start + self.batch_size) if self.batch_size is not None else bucket_size
        bucket_indexes = self.indexes_by_buckets[i][start:end]
        X_curr = [self.X[j][:self.fields_number] for j in bucket_indexes]
        to_yield = make_batch(X_curr, self.fields_to_one_hot)
        if self.has_answer:
            if self.has_multiple_outputs:
                bucket_labels = [np.array([self.X[j][i] for j in bucket_indexes])
                                 for i in self.answer_index]
            else:
                bucket_labels = np.array([self.X[j][self.answer_index] for j in bucket_indexes])
            if self.shift_answer:  # shifting data to train a language model
                if self.has_multiple_outputs:
                    raise NotImplementedError
                padding = np.full(shape=(len(bucket_labels), 1), fill_value=PAD)
                bucket_labels = np.hstack((bucket_labels[:, 1:], padding))
            y_to_yield = self._make_labels(bucket_labels)
            weights_to_yield = self._make_weights(bucket_labels, self.weights[bucket_indexes])
            answer = (to_yield, y_to_yield, weights_to_yield)
            # answer = (to_yield, y_to_yield)
        else:
            answer = to_yield
        self.step += 1
        if self.step == len(self.curr_batch_indexes):
            self.step = 0
            self.curr_epoch += 1
        return answer

    def __iter__(self):
        return self


class DataGenerator:

    POSITIONS_AS_CLASSES = 0

    def __init__(self, data, targets=None, additional_data=None,
                 additional_targets=None, use_first_item_length=True,
                 yield_targets=True, yield_indexes=False,
                 symbols_number=None, classes_number=None,
                 additional_symbols_number=None,
                 additional_classes_number=None,
                 batch_size=16, shuffle=False, nepochs=None):
        self.data = data
        self.targets = targets
        self.additional_data = additional_data or []
        self.additional_targets = additional_targets or []
        self.use_first_item_length = use_first_item_length
        self.yield_targets = yield_targets
        self.yield_indexes = yield_indexes
        self.symbols_number = symbols_number
        self.classes_number = classes_number
        self.additional_symbols_number = additional_symbols_number
        self.additional_classes_number = additional_classes_number
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.nepochs = nepochs
        self._initialize()

    def _initialize(self):
        if not isinstance(self.additional_symbols_number, list):
            self.additional_symbols_number = [self.additional_symbols_number] * len(self.additional_data)
        self.indexes = []
        ordered_indexes = np.argsort([len(x) for x in self.data])
        for i in range(0, len(ordered_indexes), self.batch_size):
            self.indexes.append(ordered_indexes[i:i + self.batch_size])
        self.step = 0
        self.epoch = 0

    @property
    def steps_per_epoch(self):
        return len(self.indexes)

    def __iter__(self):
        return self

    def _make_batch(self, data, indexes, classes_number=None, use_length=True):
        first_item = np.array(data[0])
        if first_item.ndim > 0 and use_length:
            # dtype = first_item.dtype if len(first_item) > 0 else int
            L = max(len(data[index]) for index in indexes)
            shape = (len(indexes), L) + np.shape(first_item)[1:]
            answer = np.zeros(shape=shape, dtype=first_item.dtype)
            for i, index in enumerate(indexes):
                answer[i, :len(data[index])] = data[index]
        else:
            answer = np.array([data[i] for i in indexes], dtype=first_item.dtype)
        if classes_number is not None:
            answer = to_one_hot(answer, classes_number)
        return answer

    def __next__(self):
        if self.epoch == self.nepochs:
            raise StopIteration()
        if self.shuffle and self.step == 0:
            np.random.shuffle(self.indexes)
        curr_indexes = self.indexes[self.step]
        curr_batch = self._make_batch(self.data, curr_indexes, self.symbols_number)
        curr_additional_batch = [self._make_batch(elem, curr_indexes, n)
                                 for elem, n in zip(self.additional_data, self.additional_symbols_number)]
        answer = [[curr_batch] + curr_additional_batch]
        if self.yield_targets and self.targets is not None:
            classes_number = self.classes_number
            if classes_number == self.POSITIONS_AS_CLASSES:
                classes_number = curr_batch.shape[1]
            curr_targets = self._make_batch(self.targets, curr_indexes, classes_number)
            if len(self.additional_targets) > 0:
                additional_classes_number = [n if n != self.POSITIONS_AS_CLASSES else curr_batch.shape[1]
                                             for n in self.additional_classes_number]
                curr_additional_targets = [
                    self._make_batch(elem, curr_indexes, n)
                    for elem, n in zip(self.additional_targets, additional_classes_number)
                ]
                curr_targets = [curr_targets] + curr_additional_targets
            answer.append(curr_targets)
        if self.yield_indexes:
            answer.append(curr_indexes)
        self.step += 1
        if self.step == self.steps_per_epoch:
            self.step = 0
            self.epoch += 1
        return tuple(answer)


def generate_data(X, indexes_by_buckets, output_symbols_number,
                  batch_size=None, epochs=None, active_buckets=None,
                  use_last=True, has_answer=True,
                  shift_answer=False, shuffle=True, yield_weights=True,
                  duplicate_answer=False, fields_number=None,
                  fields_to_one_hot=None, weights=None):
    if weights is None:
        weights = np.ones(shape=(len(X)))
    if fields_number is None:
        fields_number = len(X[0]) - int(has_answer and not use_last)
    if fields_to_one_hot is None:
        fields_to_one_hot = dict()
    fields_to_one_hot = dict(fields_to_one_hot)
    answer_index = 0 if use_last else -1 if has_answer else None
    if batch_size is None:
        batches_indexes = [(i, 0) for i in range(len(indexes_by_buckets))]
    else:
        batches_indexes = list(itertools.chain.from_iterable(
            (((i, j) for j in range(0, len(bucket), batch_size))
             for i, bucket in enumerate(indexes_by_buckets))))
    total_arrays_size = sum(np.count_nonzero(X[j][answer_index] != PAD) - 1
                            for elem in indexes_by_buckets for j in elem)
    total_data_length = sum(len(elem) for elem in indexes_by_buckets)
    curr_epoch = 0
    if epochs is None:
        epochs = np.inf
    count = 0
    while True:
        if shuffle:
            for elem in indexes_by_buckets:
                np.random.shuffle(elem)
            np.random.shuffle(batches_indexes)
        curr_epoch += 1
        for i, start in batches_indexes:
            if active_buckets is not None and not active_buckets[i, curr_epoch]:
                continue
            bucket_size = len(indexes_by_buckets[i])
            end = min(bucket_size, start + batch_size) if batch_size is not None else bucket_size
            bucket_indexes = indexes_by_buckets[i][start:end]
            # TO DO: fix one-hot generation of data
            to_yield = make_batch([X[j][:fields_number] for j in bucket_indexes], fields_to_one_hot)
            count += 1
            if has_answer:
                indexes_to_yield = np.array([X[j][answer_index] for j in bucket_indexes])
                if shift_answer:
                    padding = np.full(shape=(end - start, 1), fill_value=PAD)
                    indexes_to_yield = np.hstack((indexes_to_yield[:,1:], padding))
                # y_to_yield = to_one_hot(indexes_to_yield, output_symbols_number)
                y_to_yield = to_one_hot(indexes_to_yield, output_symbols_number)
                weights_to_yield = np.ones(shape=(end - start,), dtype=np.float32)
                weights_to_yield *= weights[bucket_indexes]
                if yield_weights:
                    weights_to_yield *= total_data_length * indexes_to_yield.shape[1]
                    weights_to_yield /= total_arrays_size
                if duplicate_answer:
                    y_to_yield = [y_to_yield, y_to_yield]
                    weights_to_yield = [weights_to_yield, weights_to_yield]
                yield (to_yield, y_to_yield, weights_to_yield)
            else:
                yield to_yield


def sample_redundant_data(data, targets, threshold):
    indices_by_words = defaultdict(list)
    new_indices = []
    for i, (word, target) in enumerate(zip(data, targets)):
        indices_by_words[(word, target)].append(i)
        for (word, target), curr_indices in indices_by_words.items():
            m = len(curr_indices)
            if m > threshold:
                number_of_indices_to_select = threshold + int(np.log2(m - threshold))
                indices_to_select = np.random.permutation(curr_indices)[:number_of_indices_to_select]
            else:
                indices_to_select = curr_indices[:]
            new_indices.extend(indices_to_select)
    data = [data[i] for i in new_indices]
    targets = [targets[i] for i in new_indices]
    return data, targets