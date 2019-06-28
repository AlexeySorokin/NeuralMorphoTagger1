import sys
import os
import getopt
import ujson as json

import tensorflow as tf
import keras.backend.tensorflow_backend as kbt

from syntax.network import StrangeSyntacticParser, evaluate_heads, load_parser
from syntax.common_syntax import load_elmo, load_glove
from common.read import read_syntax_infile, read_UD_file, read_tags_infile

DEFAULT_DICT_PARAMS = ["model_params", "embedder_params"]
DEFAULT_NONE_PARAMS = ["load_file", "save_file", "model_file", "train_file", "dev_file", "test_file", "outfile"]
DEFAULT_PARAMS = {"use_tags": True,
                  "train_params": {"nepochs": 20, "patience": 3},
                  "max_train_sents": -1, "max_dev_sents": -1, "max_test_sents": -1}


def read_parser_config(infile):
    with open(infile, "r", encoding="utf8") as fin:
        config = json.load(fin)
    for param in DEFAULT_DICT_PARAMS:
        if param not in config:
            config[param] = dict()
    for param in DEFAULT_NONE_PARAMS:
        if param not in config:
            config[param] = None
    for param, value in DEFAULT_PARAMS.items():
        if param not in config:
            config[param] = value
    return config


def dump_output(outfile, sents, tags, heads, deps, pred_heads, pred_deps, pred_probs=None):
    with open(outfile, "w", encoding="utf8") as fout:
        for i, (sent, tag_sent, head_sent, dep_sent, pred_head_sent, pred_dep_sent) in\
                enumerate(zip(sents, tags, heads, deps, pred_heads, pred_deps)):
            pred_probs_sent = pred_probs[i] if pred_probs is not None else None
            sent = ["ROOT"] + sent
            for j, word in enumerate(sent[1:]):
                format_string = "{}\t{}\t{}\t{}:{}\t{}"
                has_error = (head_sent[j] != pred_head_sent[j] or dep_sent[j] != pred_dep_sent[j])
                if has_error:
                    format_string += "\t{}:{}\t{}\tERROR"
                fout.write((format_string + "\n").format(
                    j+1, word, tag_sent[j], pred_head_sent[j], sent[pred_head_sent[j]],
                    pred_dep_sent[j], head_sent[j], sent[head_sent[j]], dep_sent[j]))
                if has_error and pred_probs_sent is not None:
                    fout.write("{:.1f}\t{:.1f}\n".format(100 * pred_probs_sent[j+1, pred_head_sent[j]],
                                                         100 * pred_probs_sent[j+1, head_sent[j]]))
            fout.write("\n")


SHORT_OPTS, LONG_OPTS = 'ltT', ["no-load", "no-train", "no-test"]

if __name__ == "__main__":
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
    kbt.set_session(tf.Session(config=tf_config))
    to_load, to_train, to_test = True, True, True
    opts, args = getopt.getopt(sys.argv[1:], SHORT_OPTS)
    for opt, val in opts:
        if opt in ["-l", "--no-load"]:
            to_load = False
        if opt in ["-t", "--no-train"]:
            to_train = False
        if opt in ["-T", "--no-test"]:
            to_test = False
    config = read_parser_config(args[0])
    # train_params = {"nepochs": 20, "patience": 3}
    if to_load and config["load_file"] is not None and os.path.exists(config["load_file"]):
        to_build = False
        parser = load_parser(config["load_file"])
    else:
        to_build = True
        # embedder_func = (load_elmo if config["embedder_mode"] == "elmo" else
        #                  load_glove if config["embedder_mode"] == "glove" else
        #                  None)
        # embedder = embedder_func(config["embedder_params"]) if embedder_func is not None else None
        parser = StrangeSyntacticParser(**config["model_params"])
    if to_train and config["train_file"] is not None:
        train_file, dev_file = config["train_file"], config["dev_file"]
        # train_file = "/home/alexeysorokin/data/Data/UD2.3/UD_Russian-SynTagRus/ru_syntagrus-ud-train.conllu"
        sents, heads, deps = read_syntax_infile(train_file, max_sents=config["max_train_sents"],
                                                to_shuffle=False, to_lower=True, to_process_word=False)
        if config["use_tags"]:
            tags = read_tags_infile(train_file, max_sents=config["max_train_sents"], to_shuffle=False)
        else:
            tags = None
        # dev_file = "/home/alexeysorokin/data/Data/UD2.3/UD_Russian-SynTagRus/ru_syntagrus-ud-dev.conllu"
        if dev_file is not None:
            dev_sents, dev_heads, dev_deps = read_syntax_infile(dev_file, max_sents=config["max_dev_sents"],
                                                                to_lower=True, to_shuffle=False,
                                                                to_process_word=False)
            if config["use_tags"]:
                dev_tags = read_tags_infile(dev_file, max_sents=config["max_dev_sents"], to_shuffle=False)
            else:
                dev_tags = None
        else:
            dev_sents, dev_heads, dev_deps, dev_tags = [None] * 4
        parser.train(sents, heads, deps, dev_sents, dev_heads, dev_deps,
                     tags=tags, dev_tags=dev_tags, to_build=to_build,
                     save_file=config["save_file"], model_file=config["model_file"],
                     train_params=config["train_params"])
    if to_test and config["test_file"] is not None:
        test_file = config["test_file"]
        # test_file = "/home/alexeysorokin/data/Data/UD2.3/UD_Russian-SynTagRus/ru_syntagrus-ud-test.conllu"
        test_sents, test_heads, test_deps = read_syntax_infile(test_file, max_sents=config["max_test_sents"],
                                                               to_lower=True, to_shuffle=False,
                                                               to_process_word=False)
        if config["use_tags"]:
            test_tags = read_tags_infile(test_file, max_sents=config["max_test_sents"], to_shuffle=False)
        else:
            test_tags = None
        pred_heads, pred_deps, pred_probs = parser.predict(test_sents, test_tags, return_probs=True)
        test_both = [list(zip(*elem)) for elem in zip(test_heads, test_deps)]
        pred_both = [list(zip(*elem)) for elem in zip(pred_heads, pred_deps)]
        print(evaluate_heads(test_both, pred_both))
        print(evaluate_heads(test_heads, pred_heads))
        print(evaluate_heads(test_deps, pred_deps))
        if config["outfile"] is not None:
            dump_output(config["outfile"], test_sents, test_tags, test_heads,
                        test_deps, pred_heads, pred_deps, pred_probs)
