from common.common import BEGIN, END, PAD

from deeppavlov import build_model, configs
from deeppavlov.core.common.params import from_params
from deeppavlov.core.commands.utils import parse_config


def load_elmo(elmo_output_names=("lstm_outputs1",)):
    config = parse_config(getattr(configs.elmo_embedder, "elmo_ru-news"))
    elmo_config = config["chainer"]["pipe"][-1]
    elmo_config['elmo_output_names'] = elmo_output_names
    embedder = from_params(elmo_config)
    return embedder


def pad_data(words, heads=None, deps=None):
    if words[0][0] != "<s>":
        words = [["<s>"] + elem + ["</s>"] for elem in words]
        if heads is not None:
            heads = [[0] + elem + [len(elem)+1] for elem in heads]
        if deps is not None:
            deps = [['BEGIN'] + elem + ["END"] for elem in deps]
    answer = [words]
    if heads is not None:
        answer.append(heads)
    if deps is not None:
        answer.append(deps)
    return tuple(answer) if len(answer) > 1 else answer[0]


def make_indexes_for_syntax(heads, deps=None, dep_vocab=None):
    dep_indexes, head_indexes, dep_codes = [], [], []
    for head_sent in heads:
        L = len(head_sent)
        dep_indexes.append(list(range(L+2)))
        head_indexes.append([0] + head_sent + [L+1])
    if deps is not None:
        for dep_sent in deps:
            dep_codes.append([BEGIN] + [dep_vocab.toidx(x) for x in dep_sent] + [END])
        return dep_indexes, head_indexes, dep_codes
    return dep_indexes, head_indexes