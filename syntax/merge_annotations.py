import sys
import os
import argparse

from common.read import read_UD_file

parser = argparse.ArgumentParser()
parser.add_argument("source_file")
parser.add_argument("tag_file")
parser.add_argument("outfile")
parser.add_argument("-m", "--measure_quality", action="store_true",
                    help="measure quality of automatic annotation")


if __name__ == "__main__":
    args = parser.parse_args()
    # source_file, tag_file, outfile = sys.argv[1:4]
    source_data = read_UD_file(args.source_file)
    tag_data = read_UD_file(args.tag_file)
    if args.measure_quality:
        total, total_sent = sum(len(x) for x in source_data), len(source_data)
        corr, corr_sent = 0, 0
        for first_sent, second_sent in zip(source_data, tag_data):
            has_errors = False
            for first, second in zip(first_sent, second_sent):
                if first[3] == second[3] and first[5] == second[5]:
                    corr += 1
                else:
                    has_errors = True
            corr_sent += int(not has_errors)
    print("{:.2f} {:.2f}".format(100 * corr / total, 100 * corr_sent / total_sent))
    assert len(source_data) == len(tag_data), "{} != {}".format(len(source_data), len(tag_data))
    assert all(len(x) == len(y) for x, y in zip(source_data, tag_data))
    for first_sent, second_sent in zip(source_data, tag_data):
        for first, second in zip(first_sent, second_sent):
            first[3], first[5] = second[3], second[5]
    with open(args.outfile, "w", encoding="utf8") as fout:
        for sent in source_data:
            for elem in sent:
                fout.write("\t".join(elem) + "\n")
            fout.write("\n")

