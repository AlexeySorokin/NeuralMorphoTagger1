from syntax.network import load_syntactic_parser, StrangeSyntacticParser, evaluate_heads, load_elmo
from common.read import read_syntax_infile, read_UD_file, read_tags_infile

USE_TAGS = True

if __name__ == "__main__":
    embedder = load_elmo()
    parser = StrangeSyntacticParser(embedder=embedder, use_char_model=False, use_tags=USE_TAGS,
                                    use_joint_model=True, train_params={"nepochs": 2})
                                    # head_train_params={"nepochs": 2}, dep_train_params={"nepochs": 2},
                                    # char_layer_params={"char_window_size": [1, 2, 3, 4, 5, 6, 7],
                                    #             "char_filter_multiple": 25})
    train_file = "/home/alexeysorokin/data/Data/UD2.3/UD_Russian-SynTagRus/ru_syntagrus-ud-train.conllu"
    sents, heads, deps = read_syntax_infile(train_file, to_shuffle=False, to_lower=True, to_process_word=False)
    tags = read_tags_infile(train_file, to_shuffle=False) if USE_TAGS else None
    dev_file = "/home/alexeysorokin/data/Data/UD2.3/UD_Russian-SynTagRus/ru_syntagrus-ud-dev.conllu"
    dev_sents, dev_heads, dev_deps = read_syntax_infile(
        dev_file, to_lower=True, to_shuffle=False, to_process_word=False)
    dev_tags = read_tags_infile(dev_file, to_shuffle=False) if USE_TAGS else False
    parser.train(sents, heads, deps, dev_sents, dev_heads, dev_deps, tags=tags, dev_tags=dev_tags)
    test_file = "/home/alexeysorokin/data/Data/UD2.3/UD_Russian-SynTagRus/ru_syntagrus-ud-test.conllu"
    test_sents, test_heads, test_deps = read_syntax_infile(
        test_file, to_lower=True, to_shuffle=False, to_process_word=False)
    test_tags = read_tags_infile(test_file, to_shuffle=False) if USE_TAGS else None
    pred_heads, pred_deps = parser.predict(test_sents, test_tags)
    print(evaluate_heads(test_heads, pred_heads))
    print(evaluate_heads(test_deps, pred_deps))
    # parser = load_syntactic_parser("syntax/config/config_load_basic.json")
    # infile = "/home/alexeysorokin/data/Other/Gapping/results/test_gold.out"
    # outfile = "/home/alexeysorokin/data/Other/Gapping/results/test_gold_new.out"
    # sents = read_UD_file(infile)
    # word_sents = [[elem[1] for elem in sent] for sent in sents]
    # heads, deps = parser.predict(word_sents)
    # for sent, head_sent, dep_sent in zip(sents, heads, deps):
    #     for i, (head, dep) in enumerate(zip(head_sent, dep_sent)):
    #         sent[i][6:8] = str(head), dep
    # with open(outfile, "w", encoding="utf8") as fout:
    #     for sent in sents:
    #         for elem in sent:
    #             fout.write("\t".join(elem) + "\n")
    #         fout.write("\n")
