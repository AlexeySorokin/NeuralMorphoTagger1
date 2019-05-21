"""
File containing common operation with theano structures
required by NeuralLM, but possibly useful for other modules
"""

import numpy as np

import keras.backend as kb

if kb.backend() == "tensorflow":
    from common.common_tensorflow import generate_future_mask


EPS = 1e-15

AUXILIARY = ['PAD', 'BEGIN', 'END', 'UNKNOWN']
AUXILIARY_CODES = PAD, BEGIN, END, UNKNOWN = 0, 1, 2, 3


# def to_one_hot(x, k):
#     """
#     Takes an array of integers and transforms it
#     to an array of one-hot encoded vectors
#     """
#     unit = np.eye(k, dtype=int)
#     return unit[x]

def to_one_hot(indices, num_classes):
    """
    Theano implementation for numpy arrays

    :param indices: np.array, dtype=int
    :param num_classes: int, число классов
    :return: answer, np.array, shape=indices.shape+(num_classes,)
    """
    shape = indices.shape
    indices = np.ravel(indices)
    answer = np.zeros(shape=(indices.shape[0], num_classes), dtype=int)
    answer[np.arange(indices.shape[0]), indices] = 1
    return answer.reshape(shape+(num_classes,))

def repeat_(x, k):
    tile_factor = [1, k] + [1] * (kb.ndim(x) - 1)
    return kb.tile(x[:,None,:], tile_factor)


def batch_transpose(x):
    """
    Transposes two last dimensions of x
    """
    d = kb.ndim(x)
    pattern = tuple(range(d-2)) + (d-1, d-2)
    return kb.permute_dimensions(x, pattern)


def mask_future_attention(x, mask_value=-np.inf):
    """
    Masks with mask_value the elements x[..., i, j] with j > i
    """
    x_shape = kb.shape(x)
    mask = generate_future_mask(x_shape[-1])
    for i in range(kb.ndim(x)-2):
        mask = kb.expand_dims(mask, 0)
    tiling_mask = [kb.shape(x)[i] for i in range(kb.ndim(x)-2)] + [1, 1]
    mask = kb.tile(mask, tiling_mask)
    mask_tensor = kb.ones_like(x, dtype=x.dtype) * mask_value
    answer = kb.switch(mask, x, mask_tensor)
    return answer


def expand_on_edges(a, L, R):
    """
    Expands a by L copies of a[0] on the left and R copies of a[-1] on the right
    :param a:
    :param L:
    :param R:
    :return:
    """
    left = kb.tile(a[0], [L] + [kb.shape(a)[i] for i in range(1, kb.ndim(a))])
    right = kb.tile(a[-1], [R] + [kb.shape(a)[i] for i in range(1, kb.ndim(a))])
    return kb.concatenate([left, a, right], axis=0)


def distributed_transposed_dot(C, P):
    """
    Calculates for each timestep the weighted sum
    of the embeddings C_ according to the probability distribution P_

    C: a tensor of embeddings of shape
        batch_size * timesteps * history * embeddings_shape or
        timesteps * history * embeddings_shape
    P: a tensor of attention probabilities of shape
        batch_size * timesteps * history or
        timesteps * history

    Returns:
    ---------------
    answer: a tensor of weighted embeddings of shape
        batch_size * timesteps * embeddings_shape or
        timesteps * embeddings_shape
    """
    p_dims_number = int(kb.ndim(P))
    C_shape = tuple((kb.shape(C)[i] for i in range(kb.ndim(C))))
    # new_P_shape = (-1,) + tuple(kb.shape(P)[2:])
    C_ = kb.reshape(C, (-1,) + C_shape[p_dims_number-1:])
    P_ = kb.reshape(P, (-1, kb.shape(P)[p_dims_number-1]))
    answer_shape = C_shape[:p_dims_number-1] + C_shape[p_dims_number:]
    answer = kb.reshape(kb.batch_dot(C_, P_, axes=[1, 1]), answer_shape)
    if not hasattr(answer, "_keras_shape") and hasattr(C, "_keras_shape"):
        answer._keras_shape = C._keras_shape[:-2] + (C._keras_shape[-1],)
    return answer

def distributed_dot_softmax(M, H):
    """
    Obtains a matrix m of recent embeddings
    and the hidden state h of the lstm
    and calculates the attention distribution over embeddings
    p = softmax(<m, h>)

    M: embeddings tensor of shape
        (batch_size, timesteps, history_length, units) or
        (timesteps, history_length, units)
    H: hidden state tensor of shape
        (batch_size, timesteps, units) or
        (timesteps, units)
    :return:
    """
    # flattening all dimensions of M except the last two ones
    M_shape = kb.print_tensor(kb.shape(M))
    M_shape = tuple((M_shape[i] for i in range(kb.ndim(M))))
    new_M_shape = (-1,) + M_shape[-2:]
    H_shape = kb.print_tensor(kb.shape(H))
    new_H_shape = (-1, H_shape[-1])
    M_ = kb.reshape(M, new_M_shape)
    # new_H_shape = kb.concatenate([np.array([-1]), kb.shape(H)[-2:]], axis=0)
    H_ = kb.reshape(H, new_H_shape)
    energies = kb.batch_dot(M_, H_, axes=[2, 1])
    # Tensor representing shape is not iterable with tensorflow backend
    answer = kb.reshape(kb.softmax(energies), M_shape[:-1])
    if not hasattr(answer, "_keras_shape") and hasattr(M, "_keras_shape"):
        answer._keras_shape = M._keras_shape[:-1]
    return answer







