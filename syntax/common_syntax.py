import sys, os
# sys.path.append(os.getcwd())
# for elem in sys.path:
#     print(elem)
from common.common import BEGIN, END, PAD

from deeppavlov import build_model, configs
from deeppavlov.core.common.params import from_params
from deeppavlov.core.commands.utils import parse_config
from deeppavlov.models.embedders.glove_embedder import GloVeEmbedder


def load_embeddings(embedder_mode, **kwargs):
    if embedder_mode == "elmo":
        embedder_func = load_elmo
    elif embedder_mode == "glove":
        embedder_func = load_glove
    elif embedder_mode is not None:
        raise ValueError("Wrong embedder mode: {}".format(embedder_mode))
    else:
        return None
    return embedder_func(**kwargs)

def load_elmo(elmo_output_names=("lstm_outputs1",)):
    config = parse_config(getattr(configs.elmo_embedder, "elmo_ru-news"))
    elmo_config = config["chainer"]["pipe"][-1]
    elmo_config['elmo_output_names'] = elmo_output_names
    embedder = from_params(elmo_config)
    return embedder

def load_glove(load_path):
    embedder = GloVeEmbedder(load_path=load_path, pad_zero=True)
    return embedder

def pad_data(words, heads=None, deps=None, to_pad_heads=True):
    if words[0][0] != "<s>":
        words = [["<s>"] + elem + ["</s>"] for elem in words]
        if heads is not None and to_pad_heads:
            heads = [[0] + elem + [len(elem)+1] for elem in heads]
        if deps is not None:
            deps = [['BEGIN'] + elem + ["END"] for elem in deps]
    answer = [words]
    if heads is not None:
        answer.append(heads)
    if deps is not None:
        answer.append(deps)
    return tuple(answer) if len(answer) > 1 else answer[0]


def make_indexes_for_syntax(heads, deps=None, dep_vocab=None, to_pad=True):
    dep_indexes, head_indexes, dep_codes = [], [], []
    for head_sent in heads:
        L = len(head_sent)
        if to_pad:
            dep_indexes.append(list(range(L+2)))
            head_indexes.append([0] + head_sent + [L+1])
        else:
            dep_indexes.append(list(range(L)))
            head_indexes.append(head_sent)
    if deps is not None:
        for dep_sent in deps:
            curr_dep_codes = [dep_vocab.toidx(x) for x in dep_sent]
            if to_pad:
                curr_dep_codes = [BEGIN] + curr_dep_codes + [END]
            dep_codes.append(curr_dep_codes)
        return dep_indexes, head_indexes, dep_codes
    return dep_indexes, head_indexes


def reverse_heads(sent):
    answer = [[] for _ in  range(len(sent) + 2)]
    for i, elem in enumerate(sent):
        answer[elem].append(i)
    return answer