from collections import defaultdict
from itertools import chain
from heapdict import heapdict
import inspect
import ujson as json

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

    def __init__(self, max_error=2):
        self.max_error = max_error

    @property
    def nodes_number(self):
        return len(self._trie)

    def to_json(self, outfile):
        data = dict()
        for (attr, val) in inspect.getmembers(self):
            if not (attr.startswith("__") or inspect.ismethod(val)
                    or isinstance(getattr(TagNormalizer, attr, None), property)):
                data[attr] = val
        with open(outfile, "w", encoding="utf8") as fout:
            json.dump(data, fout)

    def train(self, labels):
        if isinstance(labels[0], list):
            labels = list(chain.from_iterable(labels))
        labels = [make_UD_pos_and_tag(label, return_mode="items") for label in labels]
        self.labels = set(labels)
        self.pos = sorted(set(elem[0] for elem in labels))
        self.feats_by_pos = defaultdict(set)
        self.feats = defaultdict(lambda: defaultdict(int))
        for pos, tag in labels:
            for key, value in tag:
                self.feats[key][value] += 1
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

    def transform(self, tag, mode=None):
        pos, feats = make_UD_pos_and_tag(tag, return_mode="items")
        if pos not in self.pos:
            return pos if mode == "UD" else (pos, tuple())
        answer = []
        for key, value in feats:
            if key not in self.feats:
                continue
            elif value not in self.feats[key]:
                value = self.max_values[key]
            answer.append((key, value))
        if (pos, tuple(answer)) not in self.labels:
            new_answer = self._search_trie(pos, answer)
            if new_answer is not None:
                answer = new_answer
        if mode == "UD":
            answer = make_full_UD_tag(pos, answer, mode="items")
            return answer
        return (pos, tuple(answer))

    def _search_trie(self, pos, feats):
        curr = self._trie[0][pos]
        key = (curr, 0, tuple())
        value = (0, 0)
        agenda = heapdict({key: value})
        final_answer, min_cost = [], None
        while len(agenda) > 0:
            (curr, index, data), (cost, freq) = agenda.popitem()
            if min_cost is not None and (cost, freq) >= min_cost:
                break
            node = self._trie[curr]
            if index == len(feats):
                if self._counts[curr] > 0 and (min_cost is None or (cost, freq) < min_cost):
                    final_answer, min_cost = data, (cost, freq)
            else:
                feat, value = feats[index]
                feat_data = node.get(feat)
                if feat_data is not None:
                    child = feat_data.get(value)
                    if child is not None:
                        new_data = data + (feats[index],)
                        agenda[(child, index+1, new_data)] = (cost, -self._trie_counts[child])
            if cost < self.max_error:
                if index < len(feats):
                    agenda[(curr, index+1, data)]  = (cost+1, -self._trie_counts[curr])
                for other_feat, feat_data in node.items():
                    if index == len(feats) or other_feat < feat:
                        for value, child in feat_data.items():
                            new_data = data + ((other_feat, value),)
                            agenda[(child, index, new_data)] = (cost + 1, -self._trie_counts[child])
        return final_answer


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


