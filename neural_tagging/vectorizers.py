from collections import defaultdict

def read_unimorph_infile(infile):
    answer = defaultdict(list)
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            splitted = line.split("\t")
            if len(splitted) != 3:
                continue
            _, word, tag = splitted
            answer[word].append(tag)
    return answer


def extract_features(tag):
    splitted = tag.split(";")
    answer = [splitted[0]] + ["{}_{}".format(splitted[0], x) for x in splitted[1:]]
    return answer


class UnimorphVectorizer:

    def __init__(self):
        pass

    def train(self, infile):
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
        return self

    @property
    def dim(self):
        return len(self.feat_codes_)

    def __getitem__(self, item):
        codes = self.word_codes_.get(item)
        if codes is None:
            codes = self.word_codes_.get(item.lower())
        return codes

