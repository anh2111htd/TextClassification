import numpy as np
import pandas as pd
import csv


def parse_label(label):
    return int(label.strip()[-1])


def get_sent_pair_panda_df(filepath):
    """
    load text file into pandas df, then to torch text
    :param filepath:
    :return:
    """
    with open(filepath, 'r', encoding="utf-8") as f:
        data = [line.strip().split('\t') for line in f]
        data_source = list(map(lambda x: x[0], data))
        data_target = list(map(lambda x: x[1], data))
        data_label = list(map(lambda x: parse_label(x[2]), data))
    full_df = pd.DataFrame({"source": data_source, "target": data_target, "label": data_label})
    return full_df


def init_word_embedding(vocab, w2v_file, embed_size):
    word_embeddings = np.zeros((len(vocab), embed_size)).astype(np.double)
    if w2v_file is not None:
        glove_embeddings = pd.read_table(w2v_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
        for i, word in enumerate(vocab):
            try:
                word_embeddings[i] = glove_embeddings.loc[word].values
            except KeyError:
                word_embeddings[i] = np.random.normal(size=(embed_size,))
    return word_embeddings.astype(np.float32)
