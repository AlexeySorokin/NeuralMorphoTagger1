import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import copy
import random
import inspect
from collections import defaultdict

import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as kbt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from common.generate import MultirunEarlyStopping
from neural_LM.UD_preparation.extract_tags_from_UD import read_tags_infile, make_UD_pos_and_tag
from neural_tagging.neural_tagging_1 import CharacterTagger, load_tagger
from neural_LM import load_lm
from neural_tagging.misc import TagNormalizer, load_tag_normalizer
from read import read_substitution_file


DEFAULT_NONE_PARAMS = ["model_file", "test_files", "outfiles", "train_files",
                       "dev_files", "dump_file", "save_file", "load_file", "lm_file",
                       "prediction_files", "comparison_files",
                       "gh_outfiles", "gh_comparison_files"]
DEFAULT_PARAMS = {}
DEFAULT_LIST_PARAMS = ["vectorizers", "additional_train_files",
                       "additional_dev_files", "additional_test_files"]
DEFAULT_DICT_PARAMS = ["model_params", "read_params", "predict_params", "vocabulary_files",
                       "train_read_params", "dev_read_params", "test_read_params",
                       "train_params", "checkpoints", "normalizer_params"]
DEFAULT_DICT_WITH_KEY_PARAMS = {"dev_split_params": ["shuffle", "validation_split"]}


TRAIN_KEYS = ["batch_size", "validation_split", "nepochs",
              "transfer_warmup_epochs", "freeze_after_transfer", "batch_params"]

def read_config(infile):
    with open(infile, "r", encoding="utf8") as fin:
        from_json = json.load(fin)
    params = dict()
    for param in DEFAULT_NONE_PARAMS:
        params[param] = from_json.get(param)
    for param in DEFAULT_LIST_PARAMS:
        params[param] = from_json.get(param, [])
    for param in DEFAULT_DICT_PARAMS:
        params[param] = from_json.get(param, dict())
    for param, default_value in DEFAULT_PARAMS.items():
        params[param] = from_json.get(param, default_value)
    for param, keys in DEFAULT_DICT_WITH_KEY_PARAMS.items():
        value = from_json.get(param)
        if value is not None:
            for key in keys:
                if key not in value:
                    raise KeyError("Field {} must be present in parameter {}.".format(key, param))
        params[param] = value
    for param, value in from_json.items():
        if param not in params:
            params[param] = value
    for key in TRAIN_KEYS:
        if key in params["model_params"]:
            params["train_params"][key] = params["model_params"].pop(key)
    return params


def make_file_params_list(param, k, name="params"):
    if isinstance(param, str):
        param = [param]
    elif param is None:
        param = [None] * k
    if len(param) != k:
        Warning("You should pass the same number of {0} as test_files, "
                "setting {0} to None".format(name))
        param = [None] * k
    return param


def calculate_answer_probs(vocab, probs, labels):
    answer = [None] * len(labels)
    for i, (curr_probs, curr_labels) in enumerate(zip(probs, labels)):
        m = len(curr_labels)
        curr_label_indexes = [vocab.toidx(label) for label in curr_labels]
        answer[i] = curr_probs[np.arange(m), curr_label_indexes]
    return answer


def output_predictions(outfile, data, labels):
    with open(outfile, "w", encoding="utf8") as fout:
        for sent, sent_labels in zip(data, labels):
            for j, (word, label) in enumerate(zip(sent, sent_labels), 1):
                if not isinstance(label, list):
                    label = [label]
                for curr_label in label:
                    pos, tag = make_UD_pos_and_tag(curr_label)
                    fout.write("{}\t{}\t_\t{}\t_\t{}\n".format(j, word, pos, tag))
            fout.write("\n")


def output_results(outfile, data, pred_labels, corr_labels,
                   probs, corr_probs, basic_probs=None,
                   corr_basic_probs=None, lm_probs=None, corr_lm_probs=None):
    has_lm_probs = lm_probs is not None
    has_basic_probs = basic_probs is not None
    fields_number = 2 * (1 + int(has_basic_probs) + int(has_lm_probs))
    format_string = "\t".join(["{:.3f}"] * fields_number) + "\n"
    with open(outfile, "w", encoding="utf8") as fout:
        for i, (sent, sent_pred_labels, sent_labels, sent_probs, sent_corr_probs)\
                in enumerate(zip(data, pred_labels, corr_labels, probs, corr_probs)):
            is_correct = (sent_pred_labels == sent_labels)
            total_prob = -np.sum(np.log(sent_probs))
            total_corr_prob = -np.sum(np.log(sent_corr_probs))
            total_basic_prob = has_basic_probs and -np.sum(np.log(basic_probs[i]))
            total_corr_basic_prob = has_basic_probs and -np.sum(np.log(corr_basic_probs[i]))
            lm_prob = has_lm_probs and lm_probs[i][1]
            corr_lm_prob = has_lm_probs and corr_lm_probs[i][1]
            fout.write(format_string.format(
                total_prob, total_corr_prob, total_basic_prob,
                total_corr_basic_prob, lm_prob, corr_lm_prob))
            if not is_correct:
                fout.write("INCORRECT\n")
            for j, (word, pred_tag, corr_tag, pred_prob, corr_prob) in\
                    enumerate(zip(sent, sent_pred_labels,
                                  sent_labels, sent_probs, sent_corr_probs)):
                curr_format_string =\
                    "{0}\t{1}\t{2}" + ("\tERROR\n" if pred_tag != corr_tag else "\n")
                fout.write(curr_format_string.format("".join(word), pred_tag, corr_tag))
                basic_prob = has_basic_probs and basic_probs[i][j]
                corr_basic_prob = has_basic_probs and corr_basic_probs[i][j]
                lm_prob = has_lm_probs and lm_probs[i][0][j]
                corr_lm_prob = has_lm_probs and corr_lm_probs[i][0][j]
                fout.write(format_string.format(
                    100*pred_prob, 100*corr_prob, 100*basic_prob,
                    100*corr_basic_prob, 100*lm_prob, 100*corr_lm_prob))
            fout.write("\n")


def are_equal(test, pred):
    if isinstance(test, list):
        return all(a in test for a in pred) or test == ["UNKN"]
    return test == pred

def make_output(cls, test_data, test_labels, predictions, probs, basic_probs=None,
                lm=None, outfile=None, comparison_file=None, gold_history=False):
    return_basic_probs = (basic_probs is not None)
    corr, total, corr_sent = 0, 0, 0
    for pred, test in zip(predictions, test_labels):
        total += len(test)
        curr_corr = sum(are_equal(x, y) for x, y in zip(test, pred))
        corr += curr_corr
        corr_sent += int(len(test) == curr_corr)
    print("Точность {:.2f}: {} из {} меток".format(100 * corr / total, corr, total))
    print("Точность по предложениям {:.2f}: {} из {} предложений".format(
        100 * corr_sent / len(test_labels), corr_sent, len(test_labels)))
    if outfile is not None:
        with open(outfile, "w", encoding="utf8") as fout:
            for sent, pred, test in zip(test_data, predictions, test_labels):
                for r, (word, pred_tag, corr_tag) in enumerate(zip(sent, pred, test), 1):
                    if isinstance(corr_tag, list):
                        wrong_tags = [x for x in corr_tag if x not in pred_tag]
                        has_errors = not are_equal(corr_tag, pred_tag)
                        for curr_pred_tag in pred_tag:
                            has_error = has_errors and curr_pred_tag not in corr_tag
                            format_string = "{0}\t{1}\t{2}" + ("\tERROR\n" if has_error else "\n")
                            fout.write(format_string.format(r, "".join(word), curr_pred_tag))
                        # if len(wrong_tags) > 0:
                        #     fout.write("\n".join(wrong_tags) + "\n")
                        fout.write("\n".join(corr_tag) + "\n")
                    else:
                        format_string = "{0}\t{1}\t{2}\t{3}" + ("\tERROR\n" if pred_tag != corr_tag else "\n")
                        fout.write(format_string.format(r, "".join(word), corr_tag, pred_tag))
                fout.write("\n")
    if comparison_file is not None:
        # считаем вероятности правильных слов
        if hasattr(cls, "lm_") and not gold_history:
            prediction_probs = probs
            prediction_basic_probs = basic_probs
            corr_probs = cls.score(test_data, test_labels,
                                   return_basic_probs=return_basic_probs)
            if return_basic_probs:
                corr_probs, corr_basic_probs = corr_probs
            else:
                corr_basic_probs = None
        else:
            prediction_probs = calculate_answer_probs(cls.tags_, probs, predictions)
            corr_probs = calculate_answer_probs(cls.tags_, probs, test_labels)
            if return_basic_probs:
                prediction_basic_probs = calculate_answer_probs(cls.tags_, basic_probs, predictions)
                corr_basic_probs = calculate_answer_probs(cls.tags_, basic_probs, test_labels)
            else:
                prediction_basic_probs, corr_basic_probs = None, None
        if lm is not None:
            prediction_lm_probs = lm.predict(
                [[x] for x in predictions], return_letter_scores=True, batch_size=256)
            corr_lm_probs = lm.predict(
                [[x] for x in test_labels], return_letter_scores=True, batch_size=256)
            if not all(((len(x) + 1) == len(y[0]) and len(y[0]) == len(z[0]))
                       for x, y, z in zip(test_data, prediction_lm_probs, corr_lm_probs)):
                # to prevent index errors we do not print language model scores
                # (not sure yet there length is always correct, possibly a hidden bug)
                prediction_lm_probs, corr_lm_probs = None, None
        else:
            prediction_lm_probs, corr_lm_probs = None, None
        output_results(comparison_file, test_data, predictions, test_labels,
                       prediction_probs, corr_probs, prediction_basic_probs,
                       corr_basic_probs, prediction_lm_probs, corr_lm_probs)


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.25
    kbt.set_session(tf.Session(config=config))
    random.seed(167)
    if len(sys.argv[1:]) != 1:
        sys.exit("Usage: main.py <config json file>")
    params = read_config(sys.argv[1])
    callbacks = []
    word_dictionary, normalizer = None, None
    if "stop_callback" in params:
        stop_callback = MultirunEarlyStopping(**params["stop_callback"])
        callbacks.append(stop_callback)
    if "LR_callback" in params:
        lr_callback = ReduceLROnPlateau(**params["LR_callback"])
        callbacks.append(lr_callback)
    if len(callbacks) == 0:
        callbacks = None
    params["model_params"]["callbacks"] = callbacks
    params["predict_params"]["return_probs"] = True
    if "tag_normalizer_load_file" in params:
        normalizer = load_tag_normalizer(params["tag_normalizer_load_file"])
    if params.get("to_train", True) and params["train_files"] is not None:
        if params.get("to_load", False) and params["load_file"] is not None:
            cls = load_tagger(params["load_file"])
        else:
            cls = CharacterTagger(**params["model_params"])
        train_read_params = copy.deepcopy(params["read_params"])
        train_read_params.update(params["train_read_params"])
        train_data = []
        for train_file in params["train_files"]:
            train_data += read_tags_infile(train_file, read_words=True, **train_read_params)
        train_data, train_labels = [x[0] for x in train_data], [x[1] for x in train_data]
        if params["dev_files"] is not None:
            dev_read_params = copy.deepcopy(params["read_params"])
            dev_read_params.update(params["dev_read_params"])
            dev_data = []
            for dev_file in params["dev_files"]:
                dev_data += read_tags_infile(dev_file, read_words=True, **dev_read_params)
            dev_data, dev_labels = [x[0] for x in dev_data], [x[1] for x in dev_data]
        else:
            dev_data, dev_labels = None, None
        if len(params["additional_train_files"]) > 0:
            if normalizer is None:
                normal_tags = train_labels + (dev_labels if dev_labels is not None else [])
                normalizer_params = params["normalizer_params"]
                normalizer = TagNormalizer(**normalizer_params).train(normal_tags)
            additional_train_datasets = defaultdict(list)
            additional_read_params = params.get("additional_read_params", train_read_params)
            if isinstance(additional_read_params, dict):
                additional_read_params = [additional_read_params] * len(params["additional_train_files"])
            for i, (train_file, code) in enumerate(params["additional_train_files"]):
                curr_data = read_tags_infile(train_file, read_words=True, **additional_read_params[i])
                additional_train_datasets[code].append(curr_data)
            additional_train_data, additional_train_labels = [], []
            for code, datasets in sorted(additional_train_datasets.items()):
                curr_data = [x[0] for dataset in datasets for x in dataset]
                curr_labels = [x[1] for dataset in datasets for x in dataset]
                additional_train_data.append(curr_data)
                curr_labels = [[normalizer.transform(x, mode="UD") for x in elem] for elem in curr_labels]
                additional_train_labels.append(curr_labels)
            if "tag_normalizer_save_file" in params:
                normalizer.to_json(params["tag_normalizer_save_file"], params.get("tag_normalizer_mapping_file"))
            if not normalizer.transform_to_normalized:
                label_mapping = normalizer.label_mapping
            else:
                label_mapping = None
        else:
            additional_train_data, additional_train_labels = None, None
            label_mapping = None
        if "word_substitution_file" in params:
            words_to_substitute = read_substitution_file(params["word_substitution_file"])
        else:
            words_to_substitute = None
        cls.train(train_data, train_labels, dev_data, dev_labels,
                  additional_train_data, additional_train_labels,
                  label_mapping=label_mapping,
                  words_to_substitute=words_to_substitute,
                  train_params=params["train_params"],
                  model_file=params["model_file"], save_file=params["save_file"],
                  lm_file=params["lm_file"], checkpoints=params["checkpoints"],
                  **params["vocabulary_files"])
    elif params["load_file"] is not None:
        cls, train_data = load_tagger(params["load_file"]), None
    else:
        raise ValueError("Either train_file or load_file should be given")
    if params["save_file"] is not None and params["dump_file"] is not None:
        cls.to_json(params["save_file"], params["dump_file"])
    test_files, test_dataset_codes = [], []
    if params["test_files"] is not None:
        # defining output files
        test_files = params["test_files"]
        if isinstance(test_files, str):
            test_files = [test_files]
        test_dataset_codes = [0] * len(test_files)
    if len(params["additional_test_files"]) > 0:
        for infile, code in params["additional_test_files"]:
            test_files.append(infile)
            test_dataset_codes.append(code)
    if len(test_files) > 0:
        test_read_params = copy.deepcopy(params["read_params"])
        test_read_params.update(params["test_read_params"])
        prediction_files = make_file_params_list(params["prediction_files"], len(test_files),
                                                 name="prediction_files")
        outfiles = make_file_params_list(params["outfiles"], len(test_files),
                                                 name="outfiles")
        comparison_files = make_file_params_list(params["comparison_files"], len(test_files),
                                                 name="comparison_files")
        gh_outfiles = make_file_params_list(params["gh_outfiles"], len(test_files),
                                            name="gold_history_outfiles")
        gh_comparison_files = make_file_params_list(params["gh_comparison_files"], len(test_files),
                                                    name="gold_history_comparison_files")
        # loading language model if available
        lm = (cls.lm_ if hasattr(cls, "lm_") else
              load_lm(params["lm_file"]) if params["lm_file"] is not None else None)
        for (test_file, dataset_code, prediction_file, outfile,
             comparison_file, gh_outfile, gh_comparison_file) in zip(
                    test_files, test_dataset_codes, prediction_files, outfiles,
                    comparison_files, gh_outfiles, gh_comparison_files):
            test_data, source_data = read_tags_infile(
                test_file, read_words=True, return_source_words=True, **test_read_params)
            if not test_read_params.get("read_only_words", False):
                test_data, test_labels = [x[0] for x in test_data], [x[1] for x in test_data]
                if normalizer is not None and dataset_code > 0:
                    test_labels = [[normalizer.transform(x, mode="UD") for x in elem]
                                   for elem in test_labels]
            else:
                test_labels = None
            cls_predictions = cls.predict(test_data, dataset_codes=dataset_code, **params["predict_params"])
            predictions, probs = cls_predictions[:2]
            basic_probs = cls_predictions[2] if len(cls_predictions) > 2 else None
            if prediction_file is not None:
                output_predictions(prediction_file, source_data, predictions)
            if test_labels is not None:
                make_output(cls, test_data, test_labels, predictions,
                            probs, basic_probs, lm, outfile, comparison_file)
                if hasattr(cls, "lm_") and gh_outfile is not None:
                    print("Using gold history:")
                    cls_predictions = cls.predict(test_data, test_labels, dataset_codes=dataset_code,
                                                  **params["predict_params"])
                    predictions, probs = cls_predictions[:2]
                    basic_probs = cls_predictions[2] if len(cls_predictions) > 2 else None
                    make_output(cls, test_data, test_labels, predictions, probs, basic_probs,
                                lm, gh_outfile, gh_comparison_file, gold_history=True)

