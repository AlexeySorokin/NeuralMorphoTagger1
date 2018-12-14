from collections import defaultdict, deque, OrderedDict
import copy
import inspect
import ujson as json

import numpy as np

from read import read_unimorph_infile


class TrieNode:

    def __init__(self, parent=None, count=0, class_counts=None):
        self.parent = parent
        self.count = count
        self.class_counts = class_counts or defaultdict(int)
        self.exact_count = 0
        self.exact_class_counts = defaultdict(int)
        self.children = dict()

    def to_json(self):
        answer = {key: getattr(self, key) for key in ["count", "exact_count", "children", "probs"]}
        for key in ["class_counts", "exact_class_counts"]:
            answer[key] = dict(getattr(self, key))
        return answer


def load_node(data):
    node = TrieNode()
    for key, value in data.items():
        if key in ["class_counts", "exact_class_counts"]:
            value = defaultdict(int, value)
        elif key == "probs":
            value = {int(label): prob for label, prob in value.items()}
        setattr(node, key, value)
    return node


def load_trie(data):
    trie = [load_node(elem) for elem in data]
    for i, elem in enumerate(trie):
        for a, child in elem.children.items():
            trie[child].parent = elem
    return trie


def load_guesser(infile):
    with open(infile, "r", encoding="utf8") as fin:
        json_data = json.load(fin)
    args = {key: value for key, value in json_data.items() if not key.endswith("_")}
    guesser = SuffixGuesser(**args)
    args = {key: value for key, value in json_data.items() if key[-1] == "_"}
    for key, value in args.items():
        if key == "trie_":
            guesser.trie_ = load_trie(value)
        else:
            setattr(guesser, key, value)
    return guesser

class SuffixGuesser:

    INF = 1000

    def __init__(self, max_suffix_length=8, min_suffix_count=10, min_impurity=1e-6,
                 threshold=0.5, min_suffix_to_guess=3):
        self.max_suffix_length = max_suffix_length
        self.min_suffix_count = min_suffix_count
        self.min_impurity = min_impurity
        self.threshold = threshold
        self.min_suffix_to_guess = min_suffix_to_guess

    @property
    def root(self):
        return self.trie_[0]

    @property
    def total_count(self):
        return self.trie_[0].count

    def __len__(self):
        return len(self.trie_)

    def save(self, outfile):
        info = dict()
        for (attr, val) in inspect.getmembers(self):
            if not (attr.startswith("__") or inspect.ismethod(val) or
                    isinstance(getattr(SuffixGuesser, attr, None), property)
                    or attr.isupper() or attr == "trie_"):
                info[attr] = val
            elif attr == "trie_":
                info[attr] = [node.to_json() for node in self.trie_]
        with open(outfile, "w", encoding="utf8") as fout:
            json.dump(info, fout)


    def _get_tag_index(self, tag):
        index = self.tag_codes_.get(tag)
        if index is None:
            index = self.tag_codes_[tag] = len(self.tags_)
            self.tags_.append(tag)
        return index

    def add_node(self, word, indexes, weights=None, count_start=0):
        if weights is None:
            weights = [1.0] * len(indexes)
        curr = self.root
        for i, a in enumerate(word):
            if a not in curr.children:
                curr.children[a] = len(self)
                self.trie_.append(TrieNode(curr))
            if i >= count_start:
                curr.count += 1
            for index, weight in zip(indexes, weights):
                curr.class_counts[index] += weight
            curr = self.trie_[curr.children[a]]
        if len(word) >= count_start:
            curr.count += 1
        for index, weight in zip(indexes, weights):
            curr.class_counts[index] += weight
        return

    def find_node(self, word):
        """
        Находит номер максимального узла, до которого можно дойти в боре по суффиксу word
        """
        curr = 0
        for i, a in enumerate(word[:-min(len(word), self.max_suffix_length)-1:-1]):
            node = self.trie_[curr]
            child = node.children.get(a)
            if child is not None:
                curr = child
            else:
                break
        return curr

    def train(self, data):
        self.make_trie(data)
        self.prune_trie()
        self.make_probs(data)
        return self

    def make_trie(self, data, build=False):
        self.trie_ = [TrieNode()]
        self.tags_, self.tag_codes_ = [], dict()
        for lemma, forms in data:
            curr_tags_count = defaultdict(int)
            for form, tags in forms.items():
                for tag in tags:
                    curr_tags_count[tag] += 1
            curr_suffixes = set()
            for form, tags in forms.items():
                form = "^" + form
                weights = [1.0 / curr_tags_count[tag] for tag in tags] if build else None
                tags = [self._get_tag_index(tag) for tag in tags]
                count_start = self.INF if "" in curr_suffixes else 0
                curr_suffixes.add("")
                suff = ""
                for i in range(1, min(self.max_suffix_length, len(form)) + 1):
                    suff += form[-i]
                    if count_start == self.INF and suff not in curr_suffixes:
                        count_start = i
                    curr_suffixes.add(suff)
                self.add_node(suff, tags, weights, count_start)
        return self

    def make_probs(self, data):
        for j, (lemma, forms) in enumerate(data):
            if j % 1000 == 0:
                print(j, lemma, end=" ")
            print("")
            curr_forms_data = defaultdict(set)
            for form, tags in forms.items():
                form = "^" + form
                tags = [self._get_tag_index(tag) for tag in tags]
                index = self.find_node(form)
                curr_forms_data[index].update(tags)
            for index, tags in curr_forms_data.items():
                curr = self.trie_[index]
                curr.exact_count += 1
                for tag in tags:
                    curr.exact_class_counts[tag] += 1
        for node in self.trie_:
            if node.exact_count > 0:
                node.probs = {label: count / node.exact_count
                              for label, count in node.exact_class_counts.items()}
            else:
                node.probs = dict()
        return self

    def _make_order(self, to_keep=None):
        if to_keep is None:
            to_keep = [True] * len(self.trie_)
        answer, queue = [0], deque([self.root])
        while len(queue) > 0:
            curr = queue.pop()
            for a, child in curr.children.items():
                if to_keep[child]:
                    queue.append(self.trie_[child])
                    answer.append(child)
        return answer

    def prune_trie(self):
        to_keep = [node.count >= self.min_suffix_count for node in self.trie_]
        # print(np.count_nonzero(to_keep))
        # order = self._make_order()
        # if self.min_impurity > 0.0:
        #     for index in order[:0:-1]:
        #         curr = self.trie_[index]
        #         if any(to_keep[child] for a, child in curr.children.items()):
        #             continue
        #         parent = curr.parent
        #         score = 0.0
        #         for label, parent_count in parent.class_counts.items():
        #             curr_count = curr.class_counts.get(label, 0)
        #             curr_score = impurity_score(curr_count, curr.count, parent_count, parent.count)
        #             score = max(score, curr_score)
        #         score *= parent.count / self.total_count
        #         if score < self.min_impurity:
        #             to_keep[index] = False
        new_trie_indexes = dict()
        for i, flag in enumerate(to_keep):
            if flag:
                new_trie_indexes[i] = len(new_trie_indexes)
        self.trie_ = [self.trie_[i] for i in new_trie_indexes]
        for node in self.trie_:
            node.children = {a: new_trie_indexes[index] for a, index in node.children.items()
                             if index in new_trie_indexes}
        return self

    def predict(self, word):
        curr = self.root
        length = 0
        for a in word[::-1]:
            child = curr.children.get(a)
            if child is not None:
                length += 1
                curr = self.trie_[child]
            else:
                break
        if length >= self.min_suffix_to_guess and curr.exact_count >= self.min_suffix_count:
            return [(self.tags_[index], prob) for index, prob in curr.probs.items()
                    if prob >= self.threshold]
        else:
            return []

    def __str__(self):
        return self._to_string("", self.root, 0)

    def _to_string(self, letter, root, offset):
        class_string = " ".join("{}:{}".format(x[0], int(x[1])) for x in sorted(root.class_counts.items()))
        answer = " " * offset + "{} {} {}\n".format(letter[::-1], root.count, class_string)
        for a, child in sorted(root.children.items()):
            answer += self._to_string(letter+a, self.trie_[child], offset+1)
        return answer

    def probs_str(self):
        return self._probs_to_string("", self.root, 0)

    def _probs_to_string(self, letter, root, offset):
        class_string = " ".join("{}:{:.3f}".format(x[0], x[1]) for x in sorted(root.probs.items()))
        answer = " " * offset + "{} {} {}\n".format(letter[::-1], root.exact_count, class_string)
        for a, child in sorted(root.children.items()):
            answer += self._probs_to_string(letter+a, self.trie_[child], offset+1)
        return answer


def impurity_score(count, total, parent_count, parent_total):
    q = total / parent_total
    prob, parent_prob = count / total, parent_count / parent_total
    score = parent_prob * (1.0 - parent_prob)
    if parent_total > total:
        other_prob = (parent_count - count) / (parent_total - total)
        score -= q * other_prob * (1.0 - other_prob)
    score -= q * prob * (1.0 - prob)
    return score

def test():
    infile = "/home/alexeysorokin/data/Data/UniMorph/hungarian"
    data = read_unimorph_infile(infile, by_lemmas=True, to_list=True)
    m = int(len(data) * 0.9)
    guesser = SuffixGuesser(min_impurity=1e-3)
    guesser.train(data)
    guesser.save("models/guessers/hu")
    guesser = load_guesser("models/guessers/hu")
    for form, tags in data[m][1].items():
        print(form)
        print("\t".join(sorted(tags)))
        for tag, value in sorted(guesser.predict(form)):
            print("{}:{:.2f}".format(tag, value), end=" ")
        print("\n")

if __name__ == "__main__":
    test()