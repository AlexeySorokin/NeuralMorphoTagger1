from keras import layers as kl, activations as kact, initializers as kinit, backend as kb
from keras.engine import InputSpec


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


def build_word_cnn(inputs, char_embeddings_size=16, char_window_size=5,
                   char_filters=None, char_filter_multiple=25,
                   char_conv_layers=1, highway_layers=1,
                   dropout=0.0, intermediate_dropout=0.0, highway_dropout=0.0):
    # inputs = kl.Lambda(kb.one_hot, arguments={"num_classes": self.symbols_number_},
    #                    output_shape=lambda x: tuple(x) + (self.symbols_number_,))(inputs)
    inputs = kl.Lambda(kb.cast, arguments={"dtype": "float32"})(inputs)
    char_embeddings = kl.Dense(char_embeddings_size, use_bias=False)(inputs)
    conv_outputs = []
    char_output_dim_ = 0
    for window_size, filters_number in zip(char_window_size, char_filters):
        curr_output = char_embeddings
        curr_filters_number = (min(char_filter_multiple * window_size, 200)
                               if filters_number is None else filters_number)
        for _ in range(char_conv_layers - 1):
            curr_output = kl.Conv2D(curr_filters_number, (1, window_size),
                                    padding="same", activation="relu",
                                    data_format="channels_last")(curr_output)
            if dropout > 0.0:
                curr_output = kl.Dropout(dropout)(curr_output)
        curr_output = kl.Conv2D(curr_filters_number, (1, window_size),
                                padding="same", activation="relu",
                                data_format="channels_last")(curr_output)
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