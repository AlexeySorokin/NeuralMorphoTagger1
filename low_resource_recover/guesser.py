from common.read import read_morphemes_infile, read_tags_infile

def read_mixed_data(infile, morpheme_infile, tokenize=False):
    tag_data = read_tags_infile(infile, read_words=True, to_process_word=False)
    morpheme_data = read_morphemes_infile(morpheme_infile, tokenize=tokenize)
    joint_data = [first + (second,) for first, second in zip(tag_data, morpheme_data)]
    joint_data = [(x, y) + tuple(z) for x, y, z in joint_data]
    return tag_data, morpheme_data

if __name__ == "__main__":
    infile = "C:/Programming/neural_tagging/data/low-resource/sel/splitted/sel.train.ud"
    morpheme_file = "C:/Programming/neural_tagging/data/low-resource/sel/splitted/sel.train.morph"
    tag_data, morpheme_data = read_mixed_data(infile, morpheme_file)
    bad_indexes = [i for i in range(len(tag_data)) if len(tag_data[i][0]) != len(morpheme_data[i])]
    print(len(bad_indexes))
