import sys
import numpy as np

from neural_LM.common import make_batch, PAD, to_one_hot
from keras.callbacks import EarlyStopping

class MultirunEarlyStopping(EarlyStopping):

    def on_train_begin(self, logs=None):
        wait, stopped_epoch = self.wait, self.stopped_epoch
        best = getattr(self, "best", np.Inf if self.monitor_op == np.less else -np.Inf)
        super().on_train_begin(logs=logs)
        self.wait, self.stopped_epoch = wait, stopped_epoch
        self.best = best


class DataGenerator:

    def __init__(self, X, indexes_by_buckets, output_symbols_number,
                 batch_size=None, epochs=None, active_buckets=None,
                 use_last=False, has_answer=True, shift_answer=False,
                 duplicate_answer=False, answers_number=1,
                 fields_number=None, fields_to_one_hot=None,
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
        if self.duplicate_answer:
            y_to_yield = [y_to_yield, y_to_yield]
        return y_to_yield

    def _make_weights(self, labels, weights):
        if self.has_multiple_outputs:
            labels = labels[0]
        weights_to_yield = np.ones_like(labels[:,0], dtype=np.float32) * weights
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
