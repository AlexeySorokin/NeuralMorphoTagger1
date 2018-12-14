import sys
from collections import defaultdict
import ujson as json
from itertools import product
from functools import reduce
from abc import abstractmethod

import numpy as np
from scipy.stats.contingency import chi2_contingency

from neural_tagging.suffix_guesser import load_guesser
from neural_LM.UD_preparation.extract_tags_from_UD import make_UD_pos_and_tag
from read import read_unimorph_infile


def read_ud_infile(infile):
    answer = defaultdict(lambda: defaultdict(int))
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line == "" or line[0] == "#":
                continue
            splitted = line.split("\t")
            if not splitted[0].isdigit():
                continue
            word, lemma = splitted[1:3]
            if lemma[0].islower():
                word = word[0].lower() + word[1:]
            tag = ",".join([splitted[3], splitted[5]]) if splitted[5] != "_" else splitted[3]
            answer[word][tag] += 1
    return answer


def extract_features(tag):
    splitted = tag.split(";")
    answer = [splitted[0]] + ["{}_{}".format(splitted[0], x) for x in splitted[1:]]
    return answer


def load_vectorizer(cls, infile):
    cls = eval(cls)
    with open(infile, "r", encoding="utf8") as fin:
        params = json.load(fin)
        init_params = {key: value for key, value in params.items() if key[-1] != "_"}
        vectorizer = cls(**init_params)
        for key, value in params.items():
            if key[-1] == "_":
                setattr(vectorizer, key, value)
    return vectorizer


class GuessingVectorizer:

    def __init__(self, guesser=None, threshold=0.5):
        self._make_guesser(guesser, threshold)

    def _make_guesser(self, guesser=None, threshold=0.5):
        self.guesser = guesser
        self.threshold = threshold
        if self.guesser is not None:
            self.guesser_ = load_guesser(self.guesser)
        else:
            self.guesser_ = None

    @abstractmethod
    def _get_basic_code(self, item):
        raise NotImplementedError()

    @abstractmethod
    def _tags_to_codes(self, item):
        raise NotImplementedError()

    def __getitem__(self, item):
        codes = self._get_basic_code(item)
        # print(item, codes)
        if codes is None and self.guesser_ is not None:
            tags = [tag for tag, value in self.guesser_.predict(item) if value >= self.threshold]
            # print(item, tags, sep="\n")
            codes = self._tags_to_codes(tags)
            # print(codes)
            # sys.exit()
        # print("")

        return codes



class UnimorphVectorizer(GuessingVectorizer):

    ATTRS = ["feats_", "feat_codes_", "word_codes_"]

    def __init__(self, guesser=None, threshold=0.5):
        super().__init__(guesser, threshold)

    def train(self, infile, save_file=None):
        data = read_unimorph_infile(infile)
        feats = set()
        for tags in data.values():
            for tag in tags:
                feats.update(extract_features(tag))
        self.feats_ = sorted(feats)
        self.feat_codes_ = {x: i for i, x in enumerate(self.feats_)}
        self.word_codes_ = dict()
        for word, tags in data.items():
            curr_tag_codes = []
            for tag in tags:
                coded_tag = [self.feat_codes_[x] for x in extract_features(tag)]
                curr_tag_codes.append(coded_tag)
            self.word_codes_[word] = curr_tag_codes
        if save_file is not None:
            to_save = dict()
            for key in self.ATTRS:
                to_save[key] = getattr(self, key)
            with open(save_file, "w", encoding="utf8") as fout:
                json.dump(to_save, fout)
        return self

    @property
    def dim(self):
        return len(self.feat_codes_)

    def _tags_to_codes(self, tags):
        return [[self.feat_codes_[x] for x in extract_features(tag)] for tag in tags]

    def _get_basic_code(self, item):
        codes = self.word_codes_.get(item)
        if codes is None:
            codes = self.word_codes_.get(item.lower())
        return codes


class MatchingVectorizer(GuessingVectorizer):

    def __init__(self, guesser=None, threshold=0.5,
                 pos_count_threshold=100, count_threshold=10,
                 pos_prob_threshold=0.5, prob_threshold=0.9,
                 significance_threshold=0.001, key_significance_threshold=0.001,
                 max_ud_tags_number=25, verbose=0):
        super().__init__(guesser, threshold)
        self.pos_count_threshold = pos_count_threshold
        self.count_threshold = count_threshold
        self.pos_prob_threshold = pos_prob_threshold
        self.prob_threshold = prob_threshold
        self.significance_threshold = significance_threshold
        self.key_significance_threshold = key_significance_threshold
        self.max_ud_tags_number = max_ud_tags_number
        self.verbose = verbose

    def train(self, unimorph_infile, ud_infile, save_file=None):
        unimorph_data = read_unimorph_infile(unimorph_infile)
        self._unimorph_data = unimorph_data
        unimorph_tags = {tag for tags in unimorph_data.values() for tag in tags}
        self.unimorph_tags_ = sorted(unimorph_tags)
        self.unimorph_tags_codes_ = {tag: i for i, tag in enumerate(self.unimorph_tags_)}
        ud_data = read_ud_infile(ud_infile)
        ud_tags = {tag for tags in ud_data.values() for tag in tags}
        self.ud_tags_ = sorted(ud_tags)
        self.ud_tags_codes_ = {tag: i for i, tag in enumerate(self.ud_tags_)}
        unimorph_pos_to_ud, unimorph_feats_to_ud = self.make_feat_matches(unimorph_data, ud_data)
        self.uni_to_ud_ = self.make_tag_matches(unimorph_pos_to_ud, unimorph_feats_to_ud)
        self.word_codes_ = dict()
        for word, tags in unimorph_data.items():
            curr_tags = set()
            for tag in tags:
                code = self.unimorph_tags_codes_[tag]
                ud_codes = self.uni_to_ud_.get(code, [])
                curr_tags.update(ud_codes)
            self.word_codes_[word] = list(curr_tags)
        with open("log_1.out", "w", encoding="utf8") as fout:
            for key, values in sorted(self.uni_to_ud_.items()):
                for value in values:
                    fout.write("{}\t{}\n".format(self.unimorph_tags_[key], self.ud_tags_[value]))
                fout.write("\n")
        return self

    def make_feat_matches(self, unimorph_data, ud_data):
        # первый проход: считаем вероятности uni_feat -> ud_feat
        # и собираем значения признаков
        unimorph_pos_tags, ud_pos_tags = defaultdict(int), defaultdict(int)
        unimorph_feats, ud_feats = defaultdict(int), defaultdict(int)
        uni_ud_tag_counts = defaultdict(lambda: defaultdict(int))
        uni_ud_pos_counts = defaultdict(lambda: defaultdict(int))
        unimorph_ud_pairs = []
        for word, curr_ud_tags in ud_data.items():
            curr_unimorph_tags = unimorph_data.get(word)
            if curr_unimorph_tags is None:
                continue
            # выделение признаков
            curr_unimorph_tags = [elem.split(";") for elem in curr_unimorph_tags]
            curr_unimorph_tags = [(elem[0], elem[1:]) for elem in curr_unimorph_tags]
            curr_unimorph_feats = [feat for elem in curr_unimorph_tags for feat in elem[1]]
            curr_ud_tags = [(make_UD_pos_and_tag(x)) + (count,) for x, count in curr_ud_tags.items()]
            curr_ud_tags = [(elem[0], (elem[1].split("|") if elem[1] != "_" else []), elem[2]) for elem in curr_ud_tags]
            for curr_unimorph_pos, _ in curr_unimorph_tags:
                unimorph_pos_tags[curr_unimorph_pos] += len(curr_ud_tags)
            for feat in curr_unimorph_feats:
                unimorph_feats[feat] += len(curr_ud_tags)
            for elem in curr_ud_tags:
                ud_pos_tags[elem[0]] += 1
                for ud_feat in elem[1]:
                    ud_feats[ud_feat] += 1
            for pos, feats, count in curr_ud_tags:
                for curr_unimorph_pos, curr_unimorph_feats in curr_unimorph_tags:
                    uni_ud_pos_counts[curr_unimorph_pos][pos] += 1
                    for unimorph_feat in curr_unimorph_feats:
                        for ud_feat in feats:
                            uni_ud_tag_counts[unimorph_feat][ud_feat] += 1
            for ud_tag in curr_ud_tags:
                unimorph_ud_pairs.append([curr_unimorph_tags, ud_tag])
        # нормализация
        for uni_pos, curr_data in uni_ud_pos_counts.items():
            for ud_pos, count in curr_data.items():
                curr_data[ud_pos] /= unimorph_pos_tags[uni_pos]
        for uni_feat, curr_data in uni_ud_tag_counts.items():
            for ud_feat, count in curr_data.items():
                curr_data[ud_feat] /= unimorph_feats[uni_feat]
        # выделение правильных соответствий
        for i, (curr_unimorph_tags, ud_tag) in enumerate(unimorph_ud_pairs):
            if len(curr_unimorph_tags) == 1:
                unimorph_ud_pairs[i] = (curr_unimorph_tags[0], ud_tag)
                continue
            best_score, best_index = np.inf, None
            for j, (uni_pos, uni_feats) in enumerate(curr_unimorph_tags):
                score = -np.log(uni_ud_pos_counts[uni_pos][ud_tag[0]])
                if len(ud_tag[1]) > 0:
                    for uni_feat in uni_feats:
                        feat_score = max(uni_ud_tag_counts[uni_feat][ud_feat] for ud_feat in ud_tag[1])
                        score -= np.log(feat_score)
                if score < best_score:
                    best_score, best_index = score, j
            unimorph_ud_pairs[i] = (curr_unimorph_tags[best_index], ud_tag)
        # сохраняем признаки и их кодировку
        all_tags = [unimorph_pos_tags, ud_pos_tags, unimorph_feats, ud_feats]
        all_tag_codes = [dict() for _ in range(4)]
        for i in range(4):
            all_tags[i] = sorted(all_tags[i])
            all_tag_codes[i] = {tag: code for code, tag in enumerate(all_tags[i])}
        unimorph_pos_tags, ud_pos_tags, unimorph_feats, ud_feats = all_tags
        uni_pos_codes, ud_pos_codes, uni_feat_codes, ud_feat_codes = all_tag_codes
        # составление матриц сопряжённости
        uni_ud_pos_matrix = np.zeros(shape=(len(unimorph_pos_tags), len(ud_pos_tags)), dtype=int)
        uni_ud_tag_matrix = np.zeros(shape=(len(unimorph_feats), len(ud_feats)), dtype=int)
        uni_feats_counts = np.zeros_like(unimorph_feats, dtype=int)
        ud_feats_counts = np.zeros_like(ud_feats, dtype=int)
        for (uni_pos, curr_uni_feats), (ud_pos, curr_ud_feats, count) in unimorph_ud_pairs:
            uni_pos_code, ud_pos_code = uni_pos_codes[uni_pos], ud_pos_codes[ud_pos]
            uni_ud_pos_matrix[uni_pos_code,ud_pos_code] += 1
            curr_uni_feats = np.array([uni_feat_codes[x] for x in curr_uni_feats], dtype=int)
            curr_ud_feats = np.array([ud_feat_codes[x] for x in curr_ud_feats], dtype=int)
            uni_ud_tag_matrix[curr_uni_feats[:,None], curr_ud_feats[None, :]] += 1
            uni_feats_counts[curr_uni_feats] += 1
            ud_feats_counts[curr_ud_feats] += 1
        # извлечение соответствующих пар
        possible_pos_pairs = self.extract_frequent(
            uni_ud_pos_matrix, mode="pos", to_print=(unimorph_pos_tags, ud_pos_tags))
        total_counts = (ud_feats_counts, uni_feats_counts, len(unimorph_ud_pairs))
        ud_feat_keys = defaultdict(list)
        for i, elem in enumerate(ud_feats):
            key, value = elem.split("=")
            ud_feat_keys[key].append(i)
        possible_tag_pairs = self.extract_frequent(uni_ud_tag_matrix, total_counts=total_counts,
                                                   keys=ud_feat_keys, to_print=(unimorph_feats, ud_feats))
        possible_pos_pairs = {unimorph_pos_tags[i]: [(ud_pos_tags[j], score) for j, score in value]
                              for i, value in possible_pos_pairs.items()}
        possible_tag_pairs = {unimorph_feats[i]: [(ud_feats[j], score) for j, score in value]
                              for i, value in possible_tag_pairs.items()}
        return possible_pos_pairs, possible_tag_pairs

    def extract_frequent(self, matrix, mode="tag", axis=None, total_counts=None, keys=None, to_print=None):
        m, n = matrix.shape
        if axis is None:
            axis = [0, 1]
        if isinstance(axis, int):
            axis = [axis]
        are_frequent = []
        if total_counts is None:
            total_counts = [np.sum(matrix, axis=axe) for axe in range(np.ndim(matrix))]
            total_counts.append(np.sum(matrix))
        if keys is not None:
            if to_print is not None:
                self.row_names = to_print[0]
            are_keys_significant = self._find_significant_keys(matrix, total_counts[0], keys)
        else:
            are_keys_significant = np.ones_like(matrix, dtype=bool)
        prob_threshold = self.prob_threshold if mode == "tag" else self.pos_prob_threshold
        count_threshold = self.count_threshold if mode == "tag" else self.pos_count_threshold
        for axe in axis:
            threshold = total_counts[axe] * prob_threshold
            threshold = np.expand_dims(threshold, axis=axe)
            are_frequent.append((matrix >= np.maximum(threshold, count_threshold)))
        are_frequent = np.max(are_frequent, axis=0)
        are_frequent *= are_keys_significant
        # mask = np.expand_dims((np.sum(matrix, axis=1) >= self.count_threshold), axis=1)
        # are_frequent *= mask
        answer = defaultdict(list)
        for i, j in zip(*np.nonzero(are_frequent)):
            curr_total_sum = total_counts[-1] - total_counts[1][i] - total_counts[0][j] + matrix[i,j]
            curr_matrix = [[matrix[i,j], total_counts[0][j] - matrix[i, j]],
                           [total_counts[1][i] - matrix[i, j], curr_total_sum]]
            try:
                score = chi2_contingency(curr_matrix, lambda_="log-likelihood")[1]
                if score < self.significance_threshold:
                    answer[i].append((j, score))
            except:
                pass
            if to_print is not None and self.verbose:
                print("{} {}".format(to_print[0][i], to_print[1][j], score), end=" ")
                print(" ".join(str(x) for x in np.ravel(curr_matrix)), "{:.4f}".format(score))
        return answer

    def _find_significant_keys(self, matrix, total_counts, keys):
        answer = np.zeros_like(matrix, dtype=bool)
        for key, indexes in keys.items():
            possible_rows = np.where(np.max(matrix[:,indexes], axis=1) >= self.count_threshold)[0]
            indexes = np.array(indexes)[(total_counts[indexes] > 0)]
            for r in possible_rows:
                curr_matrix = [matrix[r, indexes], total_counts[indexes] - matrix[r, indexes]]
                try:
                    score = chi2_contingency(curr_matrix)[1]
                    if hasattr(self, "row_names") and self.verbose:
                        print(self.row_names[r], key, "{:.4f}".format(score))
                    if score < self.key_significance_threshold:
                        answer[r, indexes] = True
                except:
                    pass
        return answer

    def make_tag_matches(self, pos_pairs, feat_pairs):
        answer = dict()
        indexes_by_keys = defaultdict(set)
        for i, tag in enumerate(self.ud_tags_):
            pos, feats = make_UD_pos_and_tag(tag, return_list=True)
            indexes_by_keys[pos].add(i)
            for feat in feats:
                indexes_by_keys[feat].add(i)
        for i, unimorph_tag in enumerate(self.unimorph_tags_):
            curr_answer = set()
            unimorph_tag = unimorph_tag.split(";")
            unimorph_pos, unimorph_feats = unimorph_tag[0], unimorph_tag[1:]
            possible_ud_pos = pos_pairs.get(unimorph_pos)
            if possible_ud_pos is None:
                continue
            curr_ud_data = [possible_ud_pos]
            for feat in unimorph_feats:
                curr_ud_feats = feat_pairs.get(feat, [])
                curr_values_by_keys = defaultdict(list)
                for feat, score in curr_ud_feats:
                    key, value = feat.split("=")
                    curr_values_by_keys[key].append((key, value, score))
                curr_ud_data.extend(curr_values_by_keys.values())
            for elem in product(*curr_ud_data):
                ud_tag = self._disambiguate_features(elem)
                possible_indexes = reduce(lambda x,y: x & y, [indexes_by_keys[feat] for feat in ud_tag])
                curr_answer.update(possible_indexes)
            if len(curr_answer) > 0 and len(curr_answer) < self.max_ud_tags_number:
                answer[i] = list(curr_answer)
        return answer

    def _disambiguate_features(self, feats):
        pos, _ = feats[0]
        possible_values = dict()
        for key, value, score in feats[1:]:
            if key not in possible_values:
                possible_values[key] = (value, score)
            elif score < possible_values[key][1]:
                possible_values[key] = (value, score)
        return [pos] + ["{}={}".format(key, value) for key, (value, _) in possible_values.items()]

    def _get_basic_code(self, item):
        codes = self.word_codes_.get(item)
        if codes is None:
            codes = self.word_codes_.get(item.lower())
        return codes

    def _tags_to_codes(self, tags):
        curr_tags = set()
        for tag in tags:
            code = self.unimorph_tags_codes_[tag]
            ud_codes = self.uni_to_ud_.get(code, [])
            curr_tags.update(ud_codes)
        return list(curr_tags)


if __name__ == '__main__':
    unimorph_infile = "/home/alexeysorokin/data/Data/UniMorph/portuguese"
    ud_infile = '/home/alexeysorokin/data/Data/UD2.0/UD_Portuguese/pt-ud-train.conllu'
    matcher = MatchingVectorizer(verbose=1, guesser="neural_tagging/dump/guessers/portuguese.guess")
    matcher.train(unimorph_infile, ud_infile)
    print(matcher["segundo"])
    # print(matcher["gráfnak"])