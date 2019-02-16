from collections import defaultdict
import inspect
import itertools

from .common import *
from .UD_preparation.read_tags import descr_to_feats
from neural_LM.UD_preparation.extract_tags_from_UD import make_UD_pos_and_tag, make_full_UD_tag


def vocabulary_from_json(info, use_features=False, decompose_labels=False):
    cls = DecomposingVocabulary if decompose_labels else FeatureVocabulary if use_features else Vocabulary
    info_to_initialize = dict(elem for elem in info.items() if elem[0][-1] != "_")
    vocab = cls(**info_to_initialize)
    args = dict()
    for attr, val in info.items():
        # if attr[-1] == "_" and not isinstance(getattr(cls, attr, None), property):
        if attr in ["symbols_", "symbol_codes_"]:
            setattr(vocab, attr, val)
        elif attr == "tokens_" and use_features:
            args = {"tokens": val}
        elif attr[-1] == "_" and not isinstance(getattr(cls, attr, None), property):
            setattr(vocab, attr, val)
    if hasattr(vocab, "symbols_") and not hasattr(vocab, "symbol_codes_"):
        vocab.symbol_codes_ = {x: i for i, x in enumerate(vocab.symbols_)}
    if use_features and not decompose_labels:
        vocab._make_features(**args)
    return vocab

class Vocabulary:

    def __init__(self, character=False, min_count=1):
        self.character = character
        self.min_count = min_count

    def train(self, text):
        symbols = defaultdict(int)
        for elem in text:
            if self.character:
                curr_symbols = [symbol for x in elem for symbol in x]
            else:
                curr_symbols = elem
            for x in curr_symbols:
                symbols[x] += 1
        symbols = [x for x, count in symbols.items() if count >= self.min_count]
        self.symbols_ = AUXILIARY + sorted(symbols)
        self.symbol_codes_ = {x: i for i, x in enumerate(self.symbols_)}
        return self

    def toidx(self, x):
        return self.symbol_codes_.get(remove_token_field(x), UNKNOWN)

    def fromidx(self, label):
        return self.symbols_[label]

    @property
    def symbols_number_(self):
        return len(self.symbols_)

    def jsonize(self):
        info = {attr: val for attr, val in inspect.getmembers(self)
                if (not(attr.startswith("__") or inspect.ismethod(val))
                    and (attr[-1] != "_" or attr in ["symbols_", "symbol_codes_", "tokens_"]))}
        return info


def remove_token_field(x):
    splitted = x.split("token=")
    return splitted[0][:-1] if (len(splitted) > 1) else splitted[0]


def remove_token_fields(text):
    new_text = []
    for sent in text:
        new_text.append([remove_token_field(x) for x in sent])
    return new_text


class FeatureVocabulary(Vocabulary):

    def __init__(self, character=False, use_tokens=False, min_count=1):
        super().__init__(character=character, min_count=min_count)
        self.use_tokens = use_tokens

    def _disambig(self, x):
        symbol, feats = make_UD_pos_and_tag(x, return_mode="items")
        feat_values = defaultdict(list)
        if len(feats) == 0:
            return [x]
        for key, value in feats:
            feat_values[key].append(value)
        feat_keys = sorted(feat_values.keys())
        possible_feat_values = list(itertools.product(*(feat_values[key] for key in feat_keys)))
        possible_feats = [list(zip(feat_keys, elem)) for elem in possible_feat_values]
        return [make_full_UD_tag(symbol, elem, mode="items") for elem in possible_feats]

    def toidx(self, x, return_single=True):
        possible_tags = self._disambig(x)
        if len(possible_tags) == 1:
            return super().toidx(x)
        elif return_single:
            return super().toidx(possible_tags[0])
        else:
            return [super().toidx(elem) for elem in possible_tags]

    def train(self, text, tokens=None):
        if self.use_tokens:
            if self.character:
                raise ValueError("use_tokens cannot be True with character=True")
            text = remove_token_fields(text)
        else:
            tokens = None
        text_to_train = [list(itertools.chain.from_iterable(
            (self._disambig(x) for x in sent))) for sent in text]
        super().train(text_to_train)
        self._make_features(tokens=tokens)
        return self

    def _make_features(self, tokens=None):
        labels = set()
        # first pass determines the set of feature-value pairs
        for symbol in self.symbols_[4:]:
            symbol, feats = make_UD_pos_and_tag(symbol, return_mode="items")
            labels.add(symbol)
            for feature, value in feats:
                if feature != "token":
                    labels.add("{}_{}_{}".format(symbol, feature, value))
        labels = sorted(labels, key=(lambda x: (x.count("_"), x)))
        self.symbol_labels_ = AUXILIARY + labels
        self.symbol_labels_codes_ = {x: i for i, x in enumerate(self.symbol_labels_)}
        # second pass: constructing symbol-feature matrix
        self.symbol_matrix_ = np.zeros(shape=(len(self.symbols_), len(self.symbol_labels_)))
        for i, symbol in enumerate(self.symbols_):
            if symbol in AUXILIARY:
                codes = [i]
            else:
                symbol, feats = make_UD_pos_and_tag(symbol, return_mode="items")
                curr_labels = {symbol} | {"{}_{}_{}".format(symbol, *x) for x in feats}
                codes = [self.symbol_labels_codes_[label] for label in curr_labels]
            self.symbol_matrix_[i, codes] = 1
        if tokens is not None:
            self.tokens_ = sorted(tokens)
            self.token_codes_ = {token: i for i, token in enumerate(self.tokens_)}
            self.symbol_matrix_ = np.hstack([self.symbol_matrix_,
                                             np.zeros(shape=(self.symbols_number_,
                                                             len(self.tokens_)), dtype=int)])
        else:
            self.tokens_, self.token_codes_ = None, None
        return self

    def get_feature_code(self, x):
        return self.symbol_labels_codes_.get(x, UNKNOWN)

    def get_token_code(self, x):
        if x not in self.token_codes_:
            return None
        return self.token_codes_[x] + len(self.symbol_labels_)

    @property
    def symbol_vector_size_(self):
        return len(self.symbol_labels_) + (len(self.tokens_) if self.tokens_ is not None else 0)

    def jsonize(self):
        return super().jsonize()

UNKNOWN_VALUE = 1

class DecomposingVocabulary:

    def __init__(self, min_count=1):
        self.min_count = min_count

    def train(self, text):
        tags, feat_values = set(), defaultdict(set)
        feats_for_tags = defaultdict(set)
        for sent in text:
            for tag in sent:
                tag, feats = make_UD_pos_and_tag(tag, return_mode="items")
                tags.add(tag)
                for feat, value in feats:
                    feat_values[feat].add(value)
                    feats_for_tags[tag].add(feat)
        self.tags_ = ["PAD", "UNKNOWN_VALUE"] + sorted(tags)
        self.tag_codes_ = {tag: code for code, tag in enumerate(self.tags_)}
        self.feats_for_tags_ = [sorted(feats_for_tags[tag]) for tag in self.tags_]
        self.feats_= sorted(feat_values.keys())
        self.feat_codes_ = {feat: code for code, feat in enumerate(self.feats_)}
        self.feat_values_ = [(["PAD"] + sorted(feat_values[key])) for key in self.feats_]
        self.feat_value_codes_ = [{value: code for code, value in enumerate(elem)} for elem in self.feat_values_]
        return self

    @property
    def feats_number_(self):
        return len(self.feats_)

    def toidx(self, x):
        tag, feats = make_UD_pos_and_tag(x, return_mode="items")
        answer = np.zeros(shape=(self.feats_number_+1), dtype=int)
        answer[0] = tag_code = self.tag_codes_.get(tag, UNKNOWN_VALUE)
        for feat, value in feats:
            feat_code = self.feat_codes_.get(feat)
            if feat_code is None or feat not in self.feats_for_tags_[tag_code]:
                continue
            answer[feat_code+1] = self.feat_value_codes_[feat_code].get(value, PAD)
        return answer

    def fromidx(self, label):
        tag = self.tags_[label[0]]
        tag_index = self.tag_codes_[tag]
        feats, values = [], []
        for i, (feat, index) in enumerate(zip(self.feats_, label[1:])):
            if feat in self.feats_for_tags_[tag_index] and index > 0:
                feats.append(feat)
                values.append(self.feat_values_[i][index])
        if len(feats) == 0:
            return tag
        else:
            feat_repr = "|".join("{}={}".format(*elem) for elem in zip(feats, values))
            return "{},{}".format(tag, feat_repr)


    @property
    def symbols_number_(self):
        return [len(self.tags_)] + [len(elem) for elem in self.feat_values_]

    def jsonize(self):
        info = {attr: val for attr, val in inspect.getmembers(self)
                if (not(attr.startswith("__") or inspect.ismethod(val)))}
        return info
