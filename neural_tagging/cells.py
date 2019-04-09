import numpy as np
import tensorflow as tf

import keras.backend as kb
import keras.layers as kl
import keras.activations as kact
import keras.regularizers as kreg
import keras.initializers as kinit
from keras.engine.topology import InputSpec
from keras.metrics import categorical_accuracy, binary_accuracy

INFTY = -100
from common.common import PAD


class DistanceMatcher(kl.Layer):

    def __init__(self, input_dim, units, activation=None,
                 normalize=False, kernel_initializer='glorot_uniform',
                 kernel_regularizer=None, activity_regularizer=None, **kwargs):
        if 'input_shape' not in kwargs:
            kwargs['input_shape'] = (input_dim,)
        super(DistanceMatcher, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.units = units
        self.activation = kact.get(activation)
        self.normalize = normalize
        self.kernel_initializer = kinit.get(kernel_initializer)
        self.kernel_regularizer = kreg.get(kernel_regularizer)
        self.activity_regularizer = kreg.get(activity_regularizer)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.input_dim})

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer)
        self.built = True

    def call(self, inputs):
        xx = kb.expand_dims(kb.sum(inputs * inputs, axis=-1), -1)
        aa = kb.expand_dims(kb.sum(self.kernel * self.kernel, axis=-2), -2)
        xa = kb.dot(inputs, self.kernel)
        answer = xx + aa - 2 * xa
        if self.normalize:
            answer /= self.input_dim
        answer = kb.sqrt(answer)
        answer = kb.expand_dims(kb.max(answer, axis=-1), axis=-1) - answer
        if self.activation is not None:
            answer = self.activation(answer)
        return answer

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2 and input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)



class Highway(kl.Layer):

    def __init__(self, activation=None, bias_initializer=-1, **kwargs):
        super(Highway, self).__init__(**kwargs)
        # self.output_dim = output_dim
        self.activation = kact.get(activation)
        self.bias_initializer = bias_initializer
        if isinstance(self.bias_initializer, int):
            self.bias_initializer = kinit.constant(self.bias_initializer)
        self.input_spec = [InputSpec(min_ndim=2)]

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.gate_kernel = self.add_weight(
            shape=(input_dim, input_dim), initializer='uniform', name='gate_kernel')
        self.gate_bias = self.add_weight(
            shape=(input_dim,), initializer=self.bias_initializer, name='gate_bias')
        self.dense_kernel = self.add_weight(
            shape=(input_dim, input_dim), initializer='uniform', name='dense_kernel')
        self.dense_bias = self.add_weight(
            shape=(input_dim,), initializer=self.bias_initializer, name='dense_bias')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        gate = kb.dot(inputs, self.gate_kernel)
        gate = kb.bias_add(gate, self.gate_bias, data_format="channels_last")
        gate = self.activation(gate)
        new_value = kb.dot(inputs, self.dense_kernel)
        new_value = kb.bias_add(new_value, self.dense_bias, data_format="channels_last")
        return gate * new_value + (1.0 - gate) * inputs

    def compute_output_shape(self, input_shape):
        return input_shape

def weighted_sum(first, second, sigma, first_threshold=-np.inf, second_threshold=np.inf):
    logit_probs = first * sigma + second * (1.0 - sigma)
    infty_tensor = kb.ones_like(logit_probs) * INFTY
    logit_probs = kb.switch(kb.greater(first, first_threshold), logit_probs, infty_tensor)
    logit_probs = kb.switch(kb.greater(second, second_threshold), logit_probs, infty_tensor)
    # logit_probs = kb.batch_dot(first_normalized, sigma) + kb.batch_dot(second_normalized, 1.0 - sigma)
    return logit_probs


class WeightedCombinationLayer(kl.Layer):

    """
    A class for weighted combination of probability distributions
    """

    def __init__(self, first_threshold=None, second_threshold=None,
                 use_dimension_bias=False, use_intermediate_layer=False,
                 intermediate_dim=64, intermediate_activation=None,
                 from_logits=False, return_logits=False,
                 bias_initializer=1.0, **kwargs):
        # if 'input_shape' not in kwargs:
        #     kwargs['input_shape'] = [(None, input_dim,), (None, input_dim)]
        super(WeightedCombinationLayer, self).__init__(**kwargs)
        self.first_threshold = first_threshold if first_threshold is not None else INFTY
        self.second_threshold = second_threshold if second_threshold is not None else INFTY
        self.use_dimension_bias = use_dimension_bias
        self.use_intermediate_layer = use_intermediate_layer
        self.intermediate_dim = intermediate_dim
        self.intermediate_activation = kact.get(intermediate_activation)
        self.from_logits = from_logits
        self.return_logits = return_logits
        self.bias_initializer = bias_initializer
        self.input_spec = [InputSpec(), InputSpec(), InputSpec()]

    def build(self, input_shape):
        assert len(input_shape) == 3
        assert input_shape[0] == input_shape[1]
        assert input_shape[0][:-1] == input_shape[2][:-1]

        input_dim, features_dim = input_shape[0][-1], input_shape[2][-1]
        if self.use_intermediate_layer:
            self.first_kernel = self.add_weight(
                shape=(features_dim, self.intermediate_dim),
                initializer="random_uniform", name='first_kernel')
            self.first_bias = self.add_weight(
                shape=(self.intermediate_dim,),
                initializer="random_uniform", name='first_bias')
            features_dim = self.intermediate_dim
        self.features_kernel = self.add_weight(
            shape=(features_dim, 1), initializer="random_uniform", name='kernel')
        self.features_bias = self.add_weight(
            shape=(1,), initializer=kinit.Constant(self.bias_initializer), name='bias')
        if self.use_dimension_bias:
            self.dimensions_bias = self.add_weight(
                shape=(input_dim,), initializer="random_uniform", name='dimension_bias')
        super(WeightedCombinationLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, list) and len(inputs) == 3
        first, second, features = inputs[0], inputs[1], inputs[2]
        if not self.from_logits:
            first = kb.clip(first, 1e-10, 1.0)
            second = kb.clip(second, 1e-10, 1.0)
            first_, second_ = kb.log(first), kb.log(second)
        else:
            first_, second_ = first, second
        # embedded_features.shape = (M, T, 1)
        if self.use_intermediate_layer:
            features = kb.dot(features, self.first_kernel)
            features = kb.bias_add(features, self.first_bias, data_format="channels_last")
            features = self.intermediate_activation(features)
        embedded_features = kb.dot(features, self.features_kernel)
        embedded_features = kb.bias_add(
            embedded_features, self.features_bias, data_format="channels_last")
        if self.use_dimension_bias:
            tiling_shape = [1] * (kb.ndim(first)-1) + [kb.shape(first)[-1]]
            embedded_features = kb.tile(embedded_features, tiling_shape)
            embedded_features = kb.bias_add(
                embedded_features, self.dimensions_bias, data_format="channels_last")
        sigma = kb.sigmoid(embedded_features)

        result = weighted_sum(first_, second_, sigma,
                              self.first_threshold, self.second_threshold)
        probs = kb.softmax(result)
        if self.return_logits:
            return [probs, result]
        return probs

    def compute_output_shape(self, input_shape):
        first_shape = input_shape[0]
        if self.return_logits:
            return [first_shape, first_shape]
        return first_shape


def TemporalDropout(inputs, dropout=0.0):
    """
    Drops with :dropout probability temporal steps of input 3D tensor
    """
    # TO DO: adapt for >3D tensors
    if dropout == 0.0:
        return inputs
    inputs_func = lambda x: kb.ones_like(inputs[:, :, 0:1])
    inputs_mask = kl.Lambda(inputs_func)(inputs)
    inputs_mask = kl.Dropout(dropout)(inputs_mask)
    tiling_shape = [1, 1, kb.shape(inputs)[2]] + [1] * (kb.ndim(inputs) - 3)
    inputs_mask = kl.Lambda(kb.tile, arguments={"n": tiling_shape},
                            output_shape=inputs._keras_shape[1:])(inputs_mask)
    answer = kl.Multiply()([inputs, inputs_mask])
    return answer


def first_sigmoid_acc(y_true, y_pred):
    max_values = kb.max(y_pred, axis=-1)
    first_positive_indexes = kb.argmax(kb.cast(y_pred >= 0.5, kb.floatx()), axis=-1)
    first_pred = kb.switch(max_values >= 0.5, first_positive_indexes, kb.argmax(y_pred, axis=-1))
    return kb.cast(kb.equal(kb.argmax(y_true, axis=-1), first_pred), kb.floatx())


def _make_matrix(matrix=None, unknown_index=None, start_index=0, classes_number=None):
    if matrix is not None:
        matrix = kb.constant(matrix)
    elif classes_number is not None:
        matrix = np.zeros(shape=(classes_number, classes_number))
        if unknown_index is not None:
            matrix[start_index:, unknown_index] = 1
        matrix = kb.constant(matrix)
    return matrix

class AmbigiousCategoricalEntropy:

    def __init__(self, matrix=None, unknown_index=None, start_index=0, classes_number=None, min_pred=0.0):
        self.matrix = _make_matrix(matrix, unknown_index, start_index, classes_number)
        self.min_pred = min_pred

    def __call__(self, y_true, y_pred):
        if self.matrix is not None:
            y_pred_large = kb.switch(y_pred > self.min_pred, y_pred, kb.zeros_like(y_pred))
            y_pred += kb.dot(y_pred_large, self.matrix)
        scores = -kb.log(kb.sum(y_true * y_pred, axis=-1))
        return scores

class AmbigiousAccuracy:

    def __init__(self, matrix=None, unknown_index=None, start_index=0, classes_number=None, min_pred=0.0):
        self.matrix = _make_matrix(matrix, unknown_index, start_index, classes_number)
        self.min_pred = min_pred
        self.__name__ = "ambigious_accuracy"

    def __call__(self, y_true, y_pred):
        """
        точность для нескольких правильных ответов
        """
        # оставляем только максимальный класс
        y_max_pred = kb.expand_dims(kb.max(y_pred, axis=-1), -1)
        y_pred = kb.cast(y_pred >= y_max_pred, kb.floatx())
        if self.matrix is not None:
            # домножаем на матрицу
            y_pred = kb.minimum(kb.dot(y_pred, self.matrix), kb.cast(kb.ones_like(y_pred), kb.floatx()))
        scores = kb.max(y_true * y_pred, axis=-1)
        return scores


def leader_loss(weight):
    def _leader_loss(y_true, y_pred):
        corr_pred = kb.sum(y_true * y_pred, axis=-1)
        corr_pred = kb.expand_dims(corr_pred, -1)
        tiling_shape = [1] * (kb.ndim(y_true) - 1) + [kb.shape(y_pred)[-1]]
        corr_pred = kb.tile(corr_pred, tiling_shape)
        y_diff = kb.log(y_pred) - kb.log(corr_pred)
        y_diff = kb.maximum(y_diff, 0.0)
        y_diff *= y_diff
        y_diff = kb.sum(y_diff, axis=-1)
        return kb.categorical_crossentropy(y_true, y_pred) + weight * y_diff
    return _leader_loss


def positions_func(inputs):
    """
    A layer filling i-th column of a 2D tensor with
    1+ln(1+i) when it contaings a meaningful symbol
    and with 0 when it contains PAD
    """
    if kb.ndim(inputs) == 4:
        inputs = kb.argmax(inputs, axis=-1)
    position_inputs = kb.cumsum(kb.ones_like(inputs, dtype="float32"), axis=1)
    position_inputs *= kb.cast(kb.not_equal(inputs, PAD), "float32")
    return kb.log(1.0 + position_inputs)

