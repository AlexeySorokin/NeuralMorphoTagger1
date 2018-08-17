from collections import defaultdict

def read_dictionary(infile):
    answer = defaultdict(set)
    with open(infile, "r", encoding="utf8") as fin:
        for i, line in enumerate(fin):
            line = line.strip()
            if line == "":
                continue
            splitted = line.split()
            word = splitted[0]
            if len(splitted) >= 3:
                pos, tag = splitted[1:]
                tag = "{},{}".format(pos, tag) if tag != "_" else pos
            else:
                tag = splitted[1]
            answer[word].add(tag)
    answer = {word: list(tags) for word, tags in answer.items()}
    return answer