import sys
import getopt
import numpy as np

sys.path.append("/home/alexeysorokin/data/neural_tagging/")
from neural_LM.UD_preparation.extract_tags_from_UD import read_tags_infile

BUCKETS_NUMBER = 1
SEED = 174

def read_column_data(infile):
    answer, texts = [], []
    with open(infile, "r", encoding="utf8") as fin:
        curr_sent, curr_raw_sent = [], []
        for line in fin:
            line = line.strip()
            if line == "":
                if len(curr_sent) > 0:
                    answer.append(curr_sent)
                    texts.append(curr_raw_sent)
                curr_sent, curr_raw_sent = [], []
                continue
            curr_sent.append(line.split("\t"))
            curr_raw_sent.append(line)
        if len(curr_sent) > 0:
            answer.append(curr_sent)
            texts.append(curr_raw_sent)
    return answer, texts


def split_data(data, split, return_indexes=False, shuffle=True):
    lengths = [len(x[0]) for x in data]
    indexes = np.argsort(lengths)
    bucket_levels = [0] + [int(len(lengths) * (i+1) / BUCKETS_NUMBER) for i in range(BUCKETS_NUMBER)]
    bucket_indexes = [indexes[start:bucket_levels[i+1]] for i, start in enumerate(bucket_levels[:-1])]
    train_indexes, test_indexes = [], []
    for curr_bucket_indexes in bucket_indexes:
        L = len(curr_bucket_indexes)
        np.random.shuffle(curr_bucket_indexes)
        level = int(L * (1.0 - split))
        train_indexes.extend(curr_bucket_indexes[:level])
        test_indexes.extend(curr_bucket_indexes[level:])
    if shuffle:
        np.random.shuffle(train_indexes)
        np.random.shuffle(test_indexes)
    train_data = [data[i] for i in train_indexes]
    test_data = [data[i] for i in test_indexes]
    # np.random.shuffle(train_data)
    # np.random.shuffle(test_data)
    answer = (train_data, test_data)
    if return_indexes:
        answer += ((train_indexes, test_indexes),)
    return answer

def output_data(outfile, data, index=None):
    with open(outfile, "w", encoding="utf8") as fout:
        for elem in data:
            if index is not None:
                elem = elem[index]
            fout.write("\n".join(elem) + "\n\n")

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "b:s:m:a")
    buckets_number, seed, morph_file = BUCKETS_NUMBER, SEED, None
    allow_multiple_tags = False
    for opt, val in opts:
        if opt == "-b":
            buckets_number = int(val)
        elif opt == "-s":
            seed = int(val)
        elif opt == "-m":
            morph_file = val
        elif opt == "-a":
            allow_multiple_tags = True
    np.random.seed(seed)
    if len(args) != 4:
        sys.exit("Usage: split.py [-b BUCKETS_NUMBER] [-m morph_file] split infile train_file test_file")
    split, infile, train_file, test_file = args
    split = float(split)
    data, texts = read_tags_infile(infile, return_source_text=True, read_words=True,
                                   wrap=False, allow_multiple_tags=allow_multiple_tags)
    data = [elem + (text,) for elem, text in zip(data, texts)]
    train_data, test_data, (train_indexes, test_indexes) = split_data(data, split, return_indexes=True)
    index = 2
    print("morph_file", morph_file)
    if morph_file is not None:
        _, morph_sents = read_column_data(morph_file)
        print(len(data), len(morph_sents))
        for i, (sent, morph_sent) in enumerate(zip(data, morph_sents)):
            if len(sent[0]) != len(morph_sent):
                for elem in zip(*sent[:2]):
                    print(i, *elem)
                for elem in morph_sent:
                    print(i, *elem)
                print("")
        train_data = [morph_sents[i] for i in train_indexes]
        # np.random.shuffle(train_data)
        test_data = [morph_sents[i] for i in test_indexes]
        # np.random.shuffle(test_data)
        index = None
    print(train_indexes[:10])
    output_data(train_file, train_data, index=index)
    output_data(test_file, test_data, index=index)

