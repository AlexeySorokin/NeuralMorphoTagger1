"""
Contains implementation of cells and layers used in NeuralLM construction
"""
import numpy as np

import keras.backend as kb
import keras.layers as kl
from keras.engine import Layer, Model
from keras.engine.topology import InputSpec

from keras import initializers
from keras import regularizers
from keras import constraints

from common.common import distributed_dot_softmax, distributed_transposed_dot,\
    batch_transpose, mask_future_attention, expand_on_edges
if kb.backend() == "theano":
    from .cells_theano import make_history_theano, make_context_theano
elif kb.backend() == "tensorflow":
    from common.common_tensorflow import *


def make_history(X, h, pad, flatten=False):
    if kb.backend() == "theano":
        answer = make_history_theano(X, h, pad, flatten=flatten)
    else:
        answer = batch_shifted_fill(X, h, pad, flatten=flatten)
    if not hasattr(answer, "_keras_shape") and hasattr(X, "_keras_shape"):
        if len(X._keras_shape) == 2:
            new_shape = X._keras_shape + (h,)
        elif not flatten:
            new_shape = X._keras_shape[:-1] + (h, X._keras_shape[-1])
        elif X._keras_shape[-1] is not None:
            new_shape = X._keras_shape[:-1] + (h * X._keras_shape[-1],)
        else:
            new_shape = X._keras_shape[:-1] + (None,)
        answer._keras_shape = new_shape
    return answer


def distributed_cell(inputs):
    """
    Creates a functional wrapper over RNN cell,
    applying it on each timestep without propagating hidden states over timesteps

    """
    assert len(inputs) == 2
    shapes = [elem._keras_shape for elem in inputs]
    # no shape validation, assuming all dims of inputs[0] and inputs[1] are equal
    input_dim, units, ndims = shapes[0][-1], shapes[1][-1], len(shapes[0])
    if ndims > 3:
        dims_order = (1,) + tuple(range(2, ndims)) + (2,)
        inputs = [kl.Permute(dims_order)(inputs[0]), kl.Permute(dims_order)(inputs[0])]
    first_shape, second_shape = shapes[0][2:], shapes[1][2:]
    cell = kl.GRUCell(units, input_shape=first_shape, implementation=0)
    if not cell.built:
        cell.build(first_shape)
    concatenated_inputs = kl.Concatenate()(inputs)
    def timestep_func(x):
        cell_inputs = x[...,:input_dim]
        cell_states = x[...,None,input_dim:]
        cell_output = cell.call(cell_inputs, cell_states)
        return cell_output[0]
    func = kl.TimeDistributed(kl.Lambda(timestep_func, output_shape=second_shape))
    answer = func(concatenated_inputs)
    if ndims > 3:
        reverse_dims_order = (1, ndims-1) + tuple(range(2,ndims-1))
        answer = kl.Permute(reverse_dims_order)(answer)
    return answer


class WeightedSum(Layer):

    def __init__(self, sigma=0.0, axis=-1, **kwargs):
        self.sigma = sigma
        self.axis = axis
        super().__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[0] == input_shape[1]
        self.w = self.add_weight(name="w", shape=(1,),
                                 initializer=initializers.Constant(self.sigma), trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, list) and len(inputs) == 2
        first, second = inputs
        w = expand_number_to_shape(self.w, second)
        answer = first + w * second
        return answer



class LayerNorm1D(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=input_shape[1:],
                                     initializer=initializers.Ones(),
                                     trainable=True)

        self.beta = self.add_weight(name='beta',
                                    shape=input_shape[1:],
                                    initializer=initializers.Zeros(),
                                    trainable=True,)

        super().build(input_shape)

    def call(self, x, **kwargs):
        mean = kb.mean(x, axis=-1, keepdims=True)
        std = kb.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape

class AttentionCell(Layer):
    """
    A layer collecting in each position a weighted sum of previous words embeddings
    where weights in the sum are calculated using attention
    """

    def __init__(self, left, input_dim, output_dim, right=0,
                 merge="concatenate", use_bias=False,
                 embeddings_initializer='uniform', embeddings_regularizer=None,
                 activity_regularizer=None, embeddings_constraint=None, **kwargs):
        if 'input_shape' not in kwargs:
            kwargs['input_shape'] = [(None, input_dim,), (None, output_dim)]
        super(AttentionCell, self).__init__(**kwargs)
        self.left = left
        self.output_dim = output_dim
        self.right = right
        self.merge = merge
        self.use_bias = use_bias
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.input_dim = input_dim
        self.input_spec = [InputSpec(shape=(None, input_dim)),
                           InputSpec(shape=(None, None, output_dim))]

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 2 and len(input_shape[1])
        self.M = self.add_weight(shape=(self.input_dim, self.output_dim),
                                 initializer=self.embeddings_initializer,
                                 name='attention_embedding_1', dtype=self.dtype,
                                 regularizer=self.embeddings_regularizer,
                                 constraint=self.embeddings_constraint)
        self.C = self.add_weight(shape=(self.input_dim, self.output_dim),
                                 initializer=self.embeddings_initializer,
                                 name='attention_embedding_2', dtype=self.dtype,
                                 regularizer=self.embeddings_regularizer,
                                 constraint=self.embeddings_constraint)
        if self.use_bias:
            self.T = self.add_weight(shape=(self.left, self.output_dim),
                                     initializer=self.embeddings_initializer,
                                     name='bias', dtype=self.dtype,
                                     regularizer=self.embeddings_regularizer,
                                     constraint=self.embeddings_constraint)
        super(AttentionCell, self).build(input_shape)

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, list) and len(inputs) == 2
        symbols, encodings = inputs[0], inputs[1]
        # contexts.shape = (M, T, left)
        contexts = make_history(symbols, self.left, symbols[:,:1])
        # M.shape = C.shape = (M, T, left, output_dim)
        M = kb.gather(self.M, contexts) # input embeddings
        C = kb.gather(self.C, contexts) # output embeddings
        if self.use_bias:
            M += self.T
        # p.shape = (M, T, input_dim)
        p = distributed_dot_softmax(M, encodings)
        # p._keras_shape = M._keras_shape[:2] + (self.)
        compressed_context = distributed_transposed_dot(C, p)
        if self.merge in ["concatenate", "sum"] :
            output_func = (kl.Concatenate() if self.merge == "concatenate"
                           else kl.Merge(mode='sum'))
            output = output_func([compressed_context, encodings])
        elif self.merge == "attention":
            output = compressed_context
        elif self.merge == "sigmoid":
            output = distributed_cell([compressed_context, encodings])
        return [output, p]

    def compute_output_shape(self, input_shape):
        first_shape, second_shape = input_shape
        if self.merge == "concatenate":
            output_shape = second_shape[:2] + (2*second_shape[2],)
        # elif self.merge in ["sum", "attention", "sigmoid"]:
        else:
            output_shape = second_shape
        p_shape = second_shape[:2] + (self.input_dim,)
        return [output_shape, p_shape]


class AttentionCell3D(Layer):
    """
    Attention cell applicable to 3D data
    """

    def __init__(self, left, input_dim, output_dim, right=0,
                 merge="concatenate", use_bias=False,
                 embeddings_dropout=0.0,
                 embeddings_initializer='uniform', embeddings_regularizer=None,
                 activity_regularizer=None, embeddings_constraint=None, **kwargs):
        if 'input_shape' not in kwargs:
            kwargs['input_shape'] = [(None, input_dim,), (None, output_dim)]
        super(AttentionCell3D, self).__init__(**kwargs)
        self.left = left
        self.output_dim = output_dim
        self.right = right
        self.merge = merge
        self.use_bias = use_bias
        self.embeddings_dropout = embeddings_dropout
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.input_dim = input_dim
        self.input_spec = [InputSpec(shape=(None, None, input_dim)),
                           InputSpec(shape=(None, None, output_dim))]

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 3 and len(input_shape[1]) == 3
        self.M = self.add_weight(shape=(self.input_dim, self.output_dim),
                                 initializer=self.embeddings_initializer,
                                 name='3Dattention_embedding_1', dtype=self.dtype,
                                 regularizer=self.embeddings_regularizer,
                                 constraint=self.embeddings_constraint)
        self.C = self.add_weight(shape=(self.input_dim, self.output_dim),
                                 initializer=self.embeddings_initializer,
                                 name='3Dattention_embedding_2', dtype=self.dtype,
                                 regularizer=self.embeddings_regularizer,
                                 constraint=self.embeddings_constraint)
        if self.use_bias:
            self.T = self.add_weight(shape=(self.left, self.output_dim),
                                     initializer=self.embeddings_initializer,
                                     name='bias', dtype=self.dtype,
                                     regularizer=self.embeddings_regularizer,
                                     constraint=self.embeddings_constraint)
        super(AttentionCell3D, self).build(input_shape)

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, list) and len(inputs) == 2
        symbols, encodings = inputs[0], inputs[1]
        # dropout masks
        self._generate_dropout_mask(inputs[1])
        # contexts.shape = (M, T, left, input_dim)
        contexts = make_history(symbols, self.left, symbols[:,:1])
        # M.shape = C.shape = (M, T, left, output_dim)
        M = kb.dot(contexts, self.M) # input embeddings
        C = kb.dot(contexts, self.C) # output embeddings
        if self.use_bias:
            M += self.T
        if self.embeddings_dropout > 0.0:
            M = M * self._dropout_mask[0]
            C = C * self._dropout_mask[1]
        p = distributed_dot_softmax(M, encodings)
        compressed_context = distributed_transposed_dot(C, p)
        if self.merge in ["concatenate", "sum"] :
            output_func = (kl.Concatenate() if self.merge == "concatenate"
                           else kl.Merge(mode='sum'))
            output = output_func([compressed_context, encodings])
        elif self.merge == "attention":
            output = compressed_context
        elif self.merge == "sigmoid":
            output = distributed_cell([compressed_context, encodings])
        return [output, p]

    def compute_output_shape(self, input_shape):
        first_shape, second_shape = input_shape
        if self.merge == "concatenate":
            output_shape = second_shape[:2] + (2*second_shape[2],)
        # elif self.merge in ["sum", "attention", "sigmoid"]:
        else:
            output_shape = second_shape
        p_shape = second_shape[:2] + (self.input_dim,)
        return [output_shape, p_shape]

    def _generate_dropout_mask(self, inputs, training=None):
        if 0 < self.embeddings_dropout < 1:
            ones = kb.expand_dims(kb.ones_like(inputs), -2)
            ones = kb.repeat_elements(ones, self.left, -2)

            def dropped_inputs():
                return kb.dropout(ones, self.embeddings_dropout)

            self._dropout_mask = [kb.in_train_phase(dropped_inputs,
                                                    ones, training=training)
                                  for _ in range(2)]
        else:
            self._dropout_mask = None


def scaled_attention(Q, K, V, scale=1.0, attend_future=True):
    logits = kb.batch_dot(Q, K, axes=(2, 2)) / scale
    if not attend_future:
        logits = mask_future_attention(logits)
    probs = kb.softmax(logits)
    answer = kb.batch_dot(probs, V, axes=(2,1))
    return [answer, probs]


def scaled_attention_with_bias(Q, K, V, K_bias=None, V_bias=None,
                               scale=1.0, attend_future=True):
    logits = kb.batch_dot(Q, K, axes=(2, 2))
    if K_bias is not None:
        # logits_(ij) += dot(q_i, K_bias_(j-i))
        logits = batch_add_offset_bias(logits, Q, K_bias)
    logits /= scale
    if not attend_future:
        logits = mask_future_attention(logits)
    probs = kb.softmax(logits)
    answer = kb.batch_dot(probs, V, axes=(2,1))
    if V_bias is not None:
        # answer_(ij) += p_{ij} * V_bias_(j-i)
        answer = batch_add_offset_bias(answer, probs, V_bias, transpose_bias=False)
    return [answer, probs]

def add_position_bias(x, bias):
    T = kb.shape(x)[1]
    offset = bias.shape[0] // 2  # starting in the middle of the bias
    bias = expand_on_edges(bias, T, T)  # extending position bias matrix to deal with clipping
    bias = fill_by_slices(bias, T + offset, T, T, reverse=True)
    return kb.bias_add(x, bias, data_format="channels_last")


def self_attention(queries, keys, values, W_query, W_key, W_value,
                   key_bias, value_bias, input_dim, head_dim,
                   attend_future=True):
    queries = kb.dot(queries, W_query)
    keys = kb.dot(keys, W_key)
    values = kb.dot(values, W_value)
    # separating by heads
    to_concatenate, probs = [], []
    for start in range(0, input_dim, head_dim):
        curr, curr_probs = scaled_attention_with_bias(
            queries[:, :, start:start+head_dim], keys[:, :, start:start+head_dim],
            values[:, :, start:start+head_dim], K_bias=key_bias, V_bias=value_bias,
            scale=kb.sqrt(kb.constant(head_dim)), attend_future=attend_future)
        to_concatenate.append(curr)
        probs.append(curr_probs)
    if len(to_concatenate) > 1:
        answer = kb.concatenate(to_concatenate, axis=-1)
        probs = kb.concatenate([kb.expand_dims(x, 0) for x in probs], axis=0)
    else:
        answer, probs = to_concatenate[0], kb.expand_dims(probs[0], 0)
    return answer, probs


class SelfAttentionEncoder(Layer):
    """
    Realizes self-attention encoder mechanism as in Shaw et al.(2018)
    """

    def __init__(self, input_dim, attend_future=True, heads=1,
                 position_clip=5, use_projection=True, use_residual=True,
                 **kwargs):
        if 'input_shape' not in kwargs:
            kwargs['input_shape'] = (None, input_dim)
        super(SelfAttentionEncoder, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.attend_future = attend_future
        self.heads = heads
        self.position_clip = position_clip
        self.use_projection = use_projection
        self.use_residual = use_residual
        self.input_spec = [InputSpec(ndim=3, axes={-1: self.input_dim})]
        if self.input_dim % self.heads != 0:
            raise ValueError("Number of heads must divide input dimension")
        self.head_dim = self.input_dim // self.heads

    def build(self, input_shape):
        self.query_weight = self.add_weight(shape=(self.input_dim, self.input_dim),
                                            name='sae_query', dtype=self.dtype,
                                            initializer="glorot_uniform")
        self.key_weight = self.add_weight(shape=(self.input_dim, self.input_dim),
                                          name='sae_key', dtype=self.dtype,
                                          initializer="glorot_uniform")
        self.key_position_bias = self.add_weight(shape=(2*self.position_clip+1, self.head_dim,),
                                                 name='sae_key_position_bias', dtype=self.dtype,
                                                 initializer="glorot_uniform")
        self.value_weight = self.add_weight(shape=(self.input_dim, self.input_dim),
                                            name='sae_value', dtype=self.dtype,
                                            initializer="glorot_uniform")
        self.value_position_bias = self.add_weight(shape=(2*self.position_clip + 1, self.head_dim,),
                                                   name='sae_value_position_bias', dtype=self.dtype,
                                                   initializer="glorot_uniform")
        if self.use_projection:
            self.projection_weight = self.add_weight(shape=(self.input_dim, self.input_dim),
                                                     name='sae_projection', dtype=self.dtype,
                                                     initializer="glorot_uniform")
        super(SelfAttentionEncoder, self).build(input_shape)

    def call(self, inputs, **kwargs):
        answer, probs = self_attention(
            inputs, inputs, inputs, self.query_weight, self.key_weight,
            self.value_weight, self.key_position_bias, self.value_position_bias,
            self.input_dim, self.head_dim, attend_future=self.attend_future)
        if self.use_projection:
            answer = kb.dot(answer, self.projection_weight)
        if self.use_residual:
            answer += inputs
        return [answer, probs]

    def compute_output_shape(self, input_shape):
        first_shape = input_shape
        second_shape = input_shape[:-1] + (input_shape[1],)
        return [first_shape, second_shape]


class SelfAttentionDecoder(Layer):
    """
    Realizes self-attention decoder mechanism as in Shaw et al.(2018)
    """

    def __init__(self, input_dim, attend_future=True, attend_encoder_future=True,
                 heads=1, position_clip=5, use_projection=True, use_residual=True,
                 **kwargs):
        if 'input_shape' not in kwargs:
            kwargs['input_shape'] = (None, input_dim)
        super(SelfAttentionDecoder, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.attend_future = attend_future
        self.attend_encoder_future = attend_encoder_future
        self.heads = heads
        self.position_clip = position_clip
        self.use_projection = use_projection
        self.use_residual = use_residual
        self.input_spec = [InputSpec(ndim=3, axes={-1: self.input_dim}),
                           InputSpec(ndim=3, axes={-1: self.input_dim})]
        if self.input_dim % self.heads != 0:
            raise ValueError("Number of heads must divide input dimension")
        self.head_dim = self.input_dim // self.heads

    def build(self, input_shape):
        self.query_weight = self.add_weight(shape=(2, self.input_dim, self.input_dim),
                                            initializer="glorot_uniform",
                                            name='sad_query', dtype=self.dtype)
        self.key_weight = self.add_weight(shape=(2, self.input_dim, self.input_dim),
                                          initializer="glorot_uniform",
                                          name='sad_key', dtype=self.dtype)
        self.key_position_bias = self.add_weight(shape=(2, 2*self.position_clip+1, self.head_dim,),
                                                 initializer="glorot_uniform",
                                                 name='sad_key_position_bias', dtype=self.dtype)
        self.value_weight = self.add_weight(shape=(2, self.input_dim, self.input_dim),
                                            initializer="glorot_uniform",
                                            name='sad_value', dtype=self.dtype)
        self.value_position_bias = self.add_weight(shape=(2, 2*self.position_clip + 1, self.head_dim,),
                                                   initializer="glorot_uniform",
                                                   name='sad_value_position_bias', dtype=self.dtype)
        if self.use_projection:
            self.projection_weight = self.add_weight(shape=(self.input_dim, self.input_dim),
                                                     initializer="glorot_uniform",
                                                     name='sad_projection', dtype=self.dtype)
        super(SelfAttentionDecoder, self).build(input_shape)

    def call(self, inputs, **kwargs):
        decodings, encodings = inputs
        answer, first_probs = self_attention(
            decodings, decodings, decodings, self.query_weight[0], self.key_weight[0],
            self.value_weight[0], self.key_position_bias[0], self.value_position_bias[0],
            self.input_dim, self.head_dim, attend_future=self.attend_future)
        answer, second_probs = self_attention(
            answer, encodings, encodings, self.query_weight[1], self.key_weight[1],
            self.value_weight[1], self.key_position_bias[1], self.value_position_bias[1],
            self.input_dim, self.head_dim, attend_future=self.attend_encoder_future)
        if self.use_projection:
            answer = kb.dot(answer, self.projection_weight)
        if self.use_residual:
            answer += decodings
        return [answer, first_probs, second_probs]

    def compute_output_shape(self, input_shape):
        first_shape = input_shape[0]
        second_shape = input_shape[0][:-1] + (input_shape[0][1],)
        return [first_shape, second_shape, second_shape]


class TransposedWrapper(kl.Layer):

    def __init__(self, layer, feature_matrix=None, **kwargs):
        super(TransposedWrapper, self).__init__(**kwargs)
        self.layer = layer
        if feature_matrix is not None:
            self.feature_matrix = kb.constant(feature_matrix)
            self.output_dim = feature_matrix.shape[0]
        else:
            self.feature_matrix = None
            self.output_dim = self.layer.kernel._keras_shape[0]
            # self.output_dim = kb.shape(self.layer.kernel)[0]

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
        super(TransposedWrapper, self).build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        ref_embeddings = self.layer.kernel
        if self.feature_matrix is not None:
            ref_embeddings = kb.dot(self.feature_matrix, ref_embeddings)
        res = kb.dot(inputs, kb.transpose(ref_embeddings))
        return res

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.output_dim,)
