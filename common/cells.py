import numpy as np

from keras import layers as kl, activations as kact, initializers as kinit, backend as kb
from keras.engine import InputSpec
import keras.initializers as kint
from keras.constraints import UnitNorm, NonNeg

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


def build_word_cnn(inputs, symbols_number=None, char_embeddings_size=16,
                   char_window_size=5, char_filters=None, char_filter_multiple=25,
                   char_conv_layers=1, highway_layers=1,
                   dropout=0.0, intermediate_dropout=0.0,
                   highway_dropout=0.0, from_one_hot=True):
    # inputs = kl.Lambda(kb.one_hot, arguments={"num_classes": self.symbols_number_},
    #                    output_shape=lambda x: tuple(x) + (self.symbols_number_,))(inputs)
    if from_one_hot:
        dim = int(kb.ndim(inputs))
        inputs = kl.Lambda(kb.cast, arguments={"dtype": "float32"})(inputs)
        char_embeddings = kl.Dense(char_embeddings_size, use_bias=False)(inputs)
    else:
        dim = int(kb.ndim(inputs)) + 1
        char_embeddings = kl.Embedding(symbols_number, char_embeddings_size)(inputs)
    conv_outputs = []
    char_output_dim_ = 0
    if isinstance(char_window_size, int):
        char_window_size = [char_window_size]
    if char_filters is None or isinstance(char_filters, int):
        char_filters = [char_filters] * len(char_window_size)
    for window_size, filters_number in zip(char_window_size, char_filters):
        curr_output = char_embeddings
        curr_filters_number = (min(char_filter_multiple * window_size, 200)
                               if filters_number is None else filters_number)
        ConvLayer = kl.Conv1D if dim == 3 else kl.Conv2D
        conv_params = {"padding": "same", "activation": "relu", "data_format": "channels_last"}
        conv_window_size = window_size if dim == 3 else (1, window_size)
        for _ in range(char_conv_layers - 1):
            curr_output = ConvLayer(curr_filters_number, conv_window_size, **conv_params)(curr_output)
            if dropout > 0.0:
                curr_output = kl.Dropout(dropout)(curr_output)
        curr_output = ConvLayer(curr_filters_number, conv_window_size, **conv_params)(curr_output)
        conv_outputs.append(curr_output)
        char_output_dim_ += curr_filters_number
    if len(conv_outputs) > 1:
        conv_output = kl.Concatenate(axis=-1)(conv_outputs)
    else:
        conv_output = conv_outputs[0]
    highway_input = kl.Lambda(kb.max, arguments={"axis": -2})(conv_output)
    if intermediate_dropout > 0.0:
        highway_input = kl.Dropout(intermediate_dropout)(highway_input)
    for i in range(highway_layers - 1):
        highway_input = Highway(activation="relu")(highway_input)
        if highway_dropout > 0.0:
            highway_input = kl.Dropout(highway_dropout)(highway_input)
    if highway_layers > 0:
        highway_output = Highway(activation="relu")(highway_input)
    else:
        highway_output = highway_input
    return highway_output


class BiaffineAttention(kl.Layer):

    def __init__(self, use_first_bias=False, use_second_bias=False,
                 initializer="identity", **kwargs):
        super(BiaffineAttention, self).__init__(**kwargs)
        self.use_first_bias = use_first_bias
        self.use_second_bias = use_second_bias
        self.initializer = initializer

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[0][:-2] == input_shape[1][:-2]

        self.input_dim = (input_shape[0][-1], input_shape[1][-1])
        if self.initializer == "identity":
            if self.input_dim[0] != self.input_dim[1]:
                raise ValueError("Input dimensions must be equal in case of 'identity' initializer.")
            self.kernel_initializer = kint.Identity(gain=1.0 / np.sqrt(self.input_dim[0]))
        else:
            self.kernel_initializer = kint.glorot_uniform()
        self.kernel = self.add_weight(shape=self.input_dim, initializer=self.kernel_initializer, name='kernel')
        if self.use_first_bias:
            self.first_bias = self.add_weight(
                shape=(self.input_dim[0], 1), initializer="glorot_uniform", name="first_bias")
        if self.use_second_bias:
            self.second_bias = self.add_weight(
                shape=(self.input_dim[1], 1), initializer="glorot_uniform", name="second_bias")
        self.input_spec = [InputSpec(ndim=3, axes={-1: self.input_dim[0]}),
                           InputSpec(ndim=3, axes={-1: self.input_dim[1]})]
        self.built = True

    def call(self, inputs, **kwargs):
        # inputs[0].shape = inputs[1].shape = (B, L, D)
        first = kb.dot(inputs[0], self.kernel)  # first_rie = sum_d x_{rid} a_{de}
        answer = kb.batch_dot(first, inputs[1], axes=[2, 2])  # answer_{rij} = sum_e first_{rie} y_{rje}
        # answer_{.ij} = sum_{d,e} x_{.id} a_{de} y_{.je}; ANSWER. = X A Y^T
        if self.use_first_bias:
            first_bias_term = kb.dot(inputs[0], self.first_bias)
            answer += first_bias_term
        if self.use_second_bias:
            # second_bias_term.shape = (B, 1, L)
            second_bias_term = kb.dot(inputs[1], self.second_bias)
            second_bias_term = kb.permute_dimensions(second_bias_term, [0, 2, 1])
            answer += second_bias_term
        return answer

    def compute_output_shape(self, input_shape):
        first_shape, second_shape = input_shape
        answer = tuple(first_shape[:-1]) + (second_shape[-2],)
        return answer


class BiaffineLayer(kl.Layer):

    def __init__(self, labels_number, input_dim,  use_first_bias=False,
                 use_second_bias=False, use_label_bias=False, activation=None,
                 **kwargs):
        super(BiaffineLayer, self).__init__(**kwargs)
        self.labels_number = labels_number
        self.input_dim = input_dim
        self.use_first_bias = use_first_bias
        self.use_second_bias = use_second_bias
        self.use_label_bias = use_label_bias
        self.activation = kact.get(activation)

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[0][:-1] == input_shape[1][:-1]

        first_input_dim, second_input_dim = input_shape[0][-1], input_shape[1][-1]
        kernel_shape = (first_input_dim, second_input_dim * self.labels_number)
        self.kernel = self.add_weight(shape=kernel_shape, initializer="glorot_uniform", name='kernel')
        if self.use_first_bias:
            self.first_bias = self.add_weight(shape=(first_input_dim, self.labels_number),
                                              initializer="glorot_uniform", name="first_bias")
        if self.use_second_bias:
            self.second_bias = self.add_weight(shape=(second_input_dim, self.labels_number),
                                               initializer="glorot_uniform", name="second_bias")
        if self.use_label_bias:
            self.label_bias = self.add_weight(
                shape=(self.labels_number,), initializer="glorot_uniform", name="second_bias")
        self.input_dim = [first_input_dim, second_input_dim]
        self.input_spec = [InputSpec(min_ndim=2, axes={-1: first_input_dim}),
                           InputSpec(min_ndim=2, axes={-1: second_input_dim})]
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = [kb.shape(inputs[0])[i] for i in range(kb.ndim(inputs[0]))]
        first_input = kb.reshape(inputs[0], [-1, self.input_dim[0]])
        second_input = kb.reshape(inputs[1], [-1, self.input_dim[1]])
        first = kb.reshape(kb.dot(first_input, self.kernel),
                           shape=[-1, self.input_dim[1], self.labels_number])
        answer = kb.batch_dot(first, second_input, axes=[1, 1])
        if self.use_first_bias:
            answer += kb.dot(first_input, self.first_bias)
        if self.use_second_bias:
            answer += kb.dot(second_input, self.second_bias)
        if self.use_label_bias:
            answer = kb.bias_add(answer, self.label_bias)
        if self.activation is not None:
            answer = self.activation(answer)
        answer = kb.reshape(answer, input_shape[:-1] + [self.labels_number])
        return answer

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        answer = tuple(input_shape[:-1]) + (self.labels_number,)
        return answer


class WeightedSum(kl.Layer):

    def __init__(self, n, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.n = n

    def build(self, input_shape):
        assert len(input_shape) == self.n
        for curr_shape in input_shape[1:]:
            assert curr_shape == input_shape[0]

        self.kernel = self.add_weight(name="weights", initializer=kint.Constant(1.0 / self.n),
                                      shape=(self.n,), constraint=NonNeg())
        self.built = True

    def call(self, inputs, **kwargs):
        answer = inputs[0] * self.kernel[0]
        for i in range(1, self.n):
            answer += inputs[i] * self.kernel[i]
        answer /= kb.sum(self.weights)
        return answer

    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
# метрики

class MultilabelSigmoidLoss:

    def __init__(self, alpha=1.0, beta=1.0, axis=-1):
        self.alpha = alpha
        self.beta = beta
        self.axis= axis

    def __call__(self, y_true, y_pred):
        y_pred = kb.clip(y_pred, kb.epsilon(), 1.0 - kb.epsilon())
        positive_loss = -y_true * kb.log(y_pred)
        negative_loss = -(1 - y_true) * kb.log(1.0 - y_pred)
        positive_loss = kb.max(positive_loss, axis=self.axis)
        negative_loss = kb.max(negative_loss, axis=self.axis)
        loss = self.alpha * positive_loss + self.beta * negative_loss
        return loss


class MultilabelSigmoidAccuracy:

    def __init__(self, threshold=0.5, axis=-1):
        self.threshold = kb.constant(threshold)
        self.axis = axis
        self.__name__ = "multilabel_sigmoid_accuracy"

    def __call__(self, y_true, y_pred):
        is_positive = kb.cast(kb.greater(y_pred, self.threshold), dtype=kb.floatx())
        are_equal = kb.cast(kb.equal(y_true, is_positive), dtype=kb.floatx())
        return kb.min(are_equal, axis=self.axis)



