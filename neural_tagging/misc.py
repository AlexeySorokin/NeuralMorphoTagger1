from collections import defaultdict
from itertools import chain
from heapdict import heapdict
import inspect
import ujson as json

import numpy as np

from neural_LM.UD_preparation.extract_tags_from_UD import *


def load_tag_normalizer(infile):
    with open(infile, "r", encoding="utf8") as fin:
        data = json.load(fin)
    tag_normalizer = TagNormalizer(data["max_error"])
    for key, value in data.items():
        if key == "_trie":
            value = [defaultdict(dict, elem) for elem in value]
        elif key == "feats":
            value = defaultdict(lambda: defaultdict(int), value)
        elif key == "feats_by_pos":
            value = defaultdict(set, value)
        elif key == "labels":
            value = {(elem[0], tuple(tuple(x) for x in elem[1])) for elem in value}
        elif key == "max_error":
            continue
        setattr(tag_normalizer, key, value)
    return tag_normalizer

class TagNormalizer:

    def __init__(self, max_error=2, use_most_frequent_value=True,
                 return_all=False, transform_to_normalized=True):
        self.max_error = max_error
        self.use_most_frequent_value = use_most_frequent_value
        self.return_all = return_all
        self.transform_to_normalized = transform_to_normalized
        self.label_mapping = dict()

    @property
    def nodes_number(self):
        return len(self._trie)

    def to_json(self, outfile, label_file=None):
        data = dict()
        for (attr, val) in inspect.getmembers(self):
            if not (attr.startswith("__") or inspect.ismethod(val)
                    or isinstance(getattr(TagNormalizer, attr, None), property)):
                data[attr] = val
        with open(outfile, "w", encoding="utf8") as fout:
            json.dump(data, fout)
        if label_file is not None:
            with open(label_file, "w", encoding="utf8") as fout:
                for old_tag, new_tag in sorted(self.label_mapping.items()):
                    fout.write("{}\t{}\n".format(old_tag, new_tag))

    def train(self, labels, counts=None):
        while isinstance(labels[0], list):
            labels = list(chain.from_iterable(labels))
        labels = [make_UD_pos_and_tag(label, return_mode="items", normalize_punct=False) for label in labels]
        if counts is None:
            counts = [1] * len(labels)
        self.labels = set(labels)
        self.pos = sorted(set(elem[0] for elem in labels))
        self.feats_by_pos = defaultdict(set)
        self.feats = defaultdict(lambda: defaultdict(int))
        for (pos, tag), count in zip(labels, counts):
            for key, value in tag:
                self.feats[key][value] += count
                self.feats_by_pos[pos].add(key)
        self.max_values = {feat: max(values.keys(), key=lambda x: values[x])
                           for feat, values in self.feats.items()}
        self._make_trie(labels)
        return self

    def _get_node(self, source, key, value=None):
        curr = self._trie[source].get(key)
        if curr is None:
            curr = self._add_node(source, key, value)
        else:
            if value is not None:
                curr = curr.get(value)
                if curr is None:
                    curr = self._add_node(source, key, value)
        return curr

    def _add_node(self, source, key, value=None):
        if value is None:
            self._trie[source][key] = self.nodes_number
        else:
            self._trie[source][key][value] = self.nodes_number
        self._trie.append(defaultdict(dict))
        self._counts.append(0)
        return self.nodes_number - 1

    def _make_trie(self, labels):
        self._trie = [defaultdict(dict)]
        self._counts = [0]
        for pos, tag in labels:
            curr = self._get_node(0, pos)
            for key, value in tag:
                curr = self._get_node(curr, key, value)
            self._counts[curr] += 1
        self._make_trie_counts()
        return self

    def _make_trie_counts(self):
        self._trie_counts = self._counts[:]
        order = list(self._trie[0].values())
        while(len(order) > 0):
            curr = order.pop()
            order.extend(child for elem in self._trie[curr].values() for child in elem.values())
        for curr in order[::-1]:
            node = self._trie[curr]
            for key, data in node.items():
                for value, child in data.items():
                    self._trie_counts[curr] = max(self._trie_counts[child], self._counts[curr])
        return self

    def _output_tag(self, tag):
        if not self.transform_to_normalized:
            return tag
        elif self.return_all:
            to_choose = np.array(self.label_mapping[tag])
            return np.random.choice(to_choose)
        else:
            return self.label_mapping[tag]

    def transform(self, tag, mode=None):
        if isinstance(tag, list):
            return [self.transform(x, mode=mode) for x in tag]
        if tag in self.label_mapping:
            return self._output_tag(tag)
        pos, feats = make_UD_pos_and_tag(tag, return_mode="items", normalize_punct=False)
        if pos not in self.pos:
            return pos if mode == "UD" else (pos, tuple())
        answer = []
        for key, value in feats:
            if key not in self.feats:
                continue
            elif value not in self.feats[key] and self.use_most_frequent_value:
                value = self.max_values[key]
            answer.append((key, value))
        if (pos, tuple(answer)) not in self.labels:
            new_answer = self._search_trie(pos, answer)
        else:
            new_answer = None
        if new_answer is not None and len(new_answer) > 0:
            answer = new_answer
        elif self.return_all:
            answer = [answer]
        if mode == "UD":
            if self.return_all:
                answer = [make_full_UD_tag(pos, elem, mode="items") for elem in answer]
            else:
                answer = make_full_UD_tag(pos, answer, mode="items")
        else:
            answer = (pos, tuple(answer))
            if self.return_all:
                answer = [answer]
        self.label_mapping[tag] = answer
        return self._output_tag(tag)


    def _find_matching_tag(self, tag, return_cost=False, mode="UD"):
        pos, feats = make_UD_pos_and_tag(tag, return_mode="items", normalize_punct=False)
        new_feats = []
        for key, value in feats:
            if key not in self.feats:
                continue
            elif value not in self.feats[key]:
                value = self.max_values[key]
            new_feats.append((key, value))
        feat_answer = self._search_trie(pos, new_feats, return_cost=return_cost)
        if not self.return_all:
            feat_answer = [feat_answer] if feat_answer is not None else []
        if len(feat_answer) > 0:
            answer = []
            for elem in feat_answer:
                if return_cost:
                    elem, cost = elem
                if mode == "UD":
                    curr_answer = make_full_UD_tag(pos, elem, mode="items")
                else:
                    curr_answer = (pos, tuple(elem))
                if return_cost:
                    curr_answer = (curr_answer, cost)
                answer.append(curr_answer)
            if not self.return_all:
                answer = answer[0]
            return answer
        else:
            return [] if self.return_all else None

    def _search_trie(self, pos, feats, return_cost=False):
        curr = self._trie[0].get(pos)
        if curr is None:
            return [] if self.return_all else None
        key = (curr, 0, tuple())
        value = (0, 0)
        agenda = heapdict({key: value})
        answer, min_cost = [], None
        while len(agenda) > 0:
            (curr, index, data), (cost, freq) = agenda.popitem()
            if min_cost is not None:
                if cost > min_cost[0]:
                    break
                elif not self.return_all and freq > min_cost[1]:
                    break
            node = self._trie[curr]
            if index == len(feats):
                if self._counts[curr] > 0:
                    new_full_cost = (cost, freq)
                    to_append = ((data, cost) if return_cost else data)
                    if min_cost is None or new_full_cost < min_cost:
                        answer, min_cost = [to_append], new_full_cost
                    elif self.return_all and cost == min_cost[0]:
                        answer.append(to_append)
                        min_cost = min(new_full_cost, min_cost)
                is_feat_possible = 1
            else:
                feat, value = feats[index]
                feat_data = node.get(feat)
                if feat_data is not None:
                    child = feat_data.get(value)
                    # if child is None:
                    #     for other, other_child in feat_data.items():
                    #         if value in other.split(","):
                    #             child = other_child
                    #             break
                    if child is not None:
                        new_data = data + (feats[index],)
                        agenda[(child, index+1, new_data)] = (cost, -self._trie_counts[child])
                is_feat_possible = index < len(feats) and int(feat in self.feats_by_pos[pos])
            if cost <= self.max_error - is_feat_possible:
                if index < len(feats):
                    agenda[(curr, index+1, data)]  = (cost+is_feat_possible, -self._trie_counts[curr])
                if cost == self.max_error:
                    continue
                for other_feat, feat_data in node.items():
                    if index == len(feats) or (other_feat < feat and (index == 0 or other_feat > feats[index-1][0])):
                        for value, child in feat_data.items():
                            new_data = data + ((other_feat, value),)
                            agenda[(child, index, new_data)] = (cost + 1, -self._trie_counts[child])
        return answer if self.return_all else answer[0] if len(answer) > 0 else None


if __name__ == "__main__":
    tags = read_tags_infile("/home/alexeysorokin/data/Data/UD2.3/UD_Belarusian-HSE/be_hse-ud-train.conllu")
    tags += read_tags_infile("/home/alexeysorokin/data/Data/UD2.3/UD_Belarusian-HSE/be_hse-ud-dev.conllu")
    normalizer = TagNormalizer().train(tags)
    with open("labels.out", "w", encoding="utf8") as fout:
        for tag in sorted(normalizer.labels):
            fout.write(make_full_UD_tag(*tag, mode="items") + "\n")
    other_tags = read_tags_infile("/home/alexeysorokin/data/Data/UD2.3/UD_Ukrainian-IU/uk_iu-ud-dev.conllu")
    other_tags = list(chain.from_iterable(other_tags))
    counts, other = defaultdict(int), set()
    for tag in other_tags:
        new_tag = normalizer.transform(tag)
        key = "keep" if new_tag == tag else "known" if new_tag in normalizer.labels else "other"
        counts[key] += 1
        if key == "other":
            other.add(make_full_UD_tag(*new_tag, mode="items"))
    for key, count in counts.items():
        print(key, count)
    print(len(other))


