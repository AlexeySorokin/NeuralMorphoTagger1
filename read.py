from collections import defaultdict
from collections.__init__ import defaultdict


def read_substitution_file(infile):
    answer = []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                continue
            answer.append(line.split("\t")[-2:])
    return answer

def read_unimorph_infile(infiles, by_lemmas=False, to_list=False,
                         forms_to_add=None, pos_mapper=None):
    if isinstance(infiles, str):
        infiles = [infiles]
    forms_to_add = forms_to_add or dict()
    answer = defaultdict((lambda: defaultdict(list)) if by_lemmas else list)
    pos_tags_by_lemmas = defaultdict(set)
    for infile in infiles:
        with open(infile, "r", encoding="utf8") as fin:
            for line in fin:
                line = line.strip()
                splitted = line.split("\t")
                if len(splitted) != 3:
                    continue
                lemma, word, tag = splitted
                pos_tag = pos_mapper(tag) if pos_mapper is not None else None
                if pos_tag is not None:
                    pos_tags_by_lemmas[lemma].add(pos_tag)
                if by_lemmas:
                    answer[lemma][word].append(tag)
                else:
                    answer[word].append(tag)
    for lemma, curr_pos_tags in pos_tags_by_lemmas.items():
        for pos_tag in curr_pos_tags:
            source_tag = forms_to_add.get(pos_tag)
            if source_tag is not None:
                dest = answer[lemma][lemma] if by_lemmas else answer[lemma]
                if source_tag not in dest:
                    dest.append(source_tag)
    if to_list:
        answer = list(answer.items())
    return answer


def read_ud_infile(infiles):
    if isinstance(infiles, str):
        infiles = [infiles]
    answer = defaultdict(lambda: defaultdict(int))
    for infile in infiles:
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