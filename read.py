from collections.__init__ import defaultdict


def read_unimorph_infile(infile, by_lemmas=False, to_list=False):
    answer = defaultdict((lambda: defaultdict(list)) if by_lemmas else list)
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            splitted = line.split("\t")
            if len(splitted) != 3:
                continue
            lemma, word, tag = splitted
            if by_lemmas:
                answer[lemma][word].append(tag)
            else:
                answer[word].append(tag)
    if to_list:
        answer = list(answer.items())
    return answer