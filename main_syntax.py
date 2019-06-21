import tensorflow as tf
import keras.backend.tensorflow_backend as kbt


from syntax.network import load_syntactic_parser, StrangeSyntacticParser, evaluate_heads, load_parser
from syntax.common import load_elmo, load_glove
from common.read import read_syntax_infile, read_UD_file, read_tags_infile

USE_TAGS = True
MAX_SENTS, MAX_DEV_SENTS, MAX_TEST_SENTS = -1, -1, -1
LOAD_FILE = "syntax/models/model-child.json" # "syntax/models/model.json"
TO_TRAIN, TO_TEST = True, True
SAVE_FILE = "syntax/models/model-child.json"
MODEL_FILE = "syntax/models/model-child.hdf5"
EMBEDDER_MODE = "elmo"
OUTFILE = "syntax/dump/analysis-model-child.out"
TO_PREDICT_CHILDREN = True

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


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    kbt.set_session(tf.Session(config=config))
    train_params = {"nepochs": 18, "patience": 3}
    if LOAD_FILE is not None:
        to_build = False
        parser = load_parser(LOAD_FILE)
    else:
        to_build = True
        embedder = (load_elmo() if EMBEDDER_MODE == "elmo" else
                    load_glove("dump/embedders/glove_ru_100000.vec") if EMBEDDER_MODE == "glove" else
                    None)
        parser = StrangeSyntacticParser(embedder=embedder, use_char_model=True,
                                        use_tags=USE_TAGS, to_predict_children=TO_PREDICT_CHILDREN,
                                        use_joint_model=True, train_params=train_params,
                                        model_params={"lstm_layers": 1, "lstm_size": 128},
                                        char_layer_params={"char_window_size": [1, 2, 3, 4, 5, 6, 7],
                                                           "char_embeddings_size": 32, "char_filter_multiple": 25}
                                       )
    if TO_TRAIN:
        train_file = "/home/alexeysorokin/data/Data/UD2.3/UD_Russian-SynTagRus/ru_syntagrus-ud-train.conllu"
        sents, heads, deps = read_syntax_infile(train_file, max_sents=MAX_SENTS, to_shuffle=False,
                                                to_lower=True, to_process_word=False)
        tags = read_tags_infile(train_file, max_sents=MAX_SENTS, to_shuffle=False) if USE_TAGS else None
        dev_file = "/home/alexeysorokin/data/Data/UD2.3/UD_Russian-SynTagRus/ru_syntagrus-ud-dev.conllu"
        dev_sents, dev_heads, dev_deps = read_syntax_infile(dev_file, max_sents=MAX_DEV_SENTS,
                                                            to_lower=True, to_shuffle=False,
                                                            to_process_word=False)
        dev_tags = read_tags_infile(dev_file, max_sents=MAX_DEV_SENTS, to_shuffle=False) if USE_TAGS else False
        parser.train(sents, heads, deps, dev_sents, dev_heads, dev_deps,
                     tags=tags, dev_tags=dev_tags, to_build=to_build,
                     save_file=SAVE_FILE, model_file=MODEL_FILE, train_params=train_params)
    if TO_TEST:
        test_file = "/home/alexeysorokin/data/Data/UD2.3/UD_Russian-SynTagRus/ru_syntagrus-ud-test.conllu"
        test_sents, test_heads, test_deps = read_syntax_infile(
            test_file, max_sents=MAX_TEST_SENTS, to_lower=True, to_shuffle=False, to_process_word=False)
        test_tags = read_tags_infile(test_file, max_sents=MAX_TEST_SENTS, to_shuffle=False) if USE_TAGS else None
        pred_heads, pred_deps, pred_probs = parser.predict(test_sents, test_tags, return_probs=True)
        test_both = [list(zip(*elem)) for elem in zip(test_heads, test_deps)]
        pred_both = [list(zip(*elem)) for elem in zip(pred_heads, pred_deps)]
        print(evaluate_heads(test_both, pred_both))
        print(evaluate_heads(test_heads, pred_heads))
        print(evaluate_heads(test_deps, pred_deps))
        if OUTFILE is not None:
            dump_output(OUTFILE, test_sents, test_tags, test_heads,
                        test_deps, pred_heads, pred_deps, pred_probs)
