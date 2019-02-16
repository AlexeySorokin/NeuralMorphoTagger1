import sys
import getopt
import numpy as np

from neural_LM.UD_preparation.extract_tags_from_UD import read_tags_infile

BUCKETS_NUMBER = 1
SEED = 174

def split_data(data, split):
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
    train_data = [data[i] for i in train_indexes]
    test_data = [data[i] for i in test_indexes]
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    return train_data, test_data

def output_data(outfile, data):
    with open(outfile, "w", encoding="utf8") as fout:
        for elem in data:
            fout.write("\n".join(elem[2]) + "\n\n")

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "b:s:")
    buckets_number, seed = BUCKETS_NUMBER, SEED
    for opt, val in opts:
        if opt == "-b":
            buckets_number = int(val)
        elif opt == "-s":
            seed = int(val)
    np.random.seed(seed)
    if len(args) != 4:
        sys.exit("Usage: split.py [-b BUCKETS_NUMBER] split infile train_file test_file")
    split, infile, train_file, test_file = args
    split = float(split)
    data, texts = read_tags_infile(infile, return_source_text=True, read_words=True, wrap=False)
    data = [elem + (text,) for elem, text in zip(data, texts)]
    train_data, test_data = split_data(data, split)
    output_data(train_file, train_data)
    output_data(test_file, test_data)

