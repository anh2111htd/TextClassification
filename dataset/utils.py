import numpy as np
import pandas as pd
import csv


def tokenize_char(sent, sent_tokenizer, max_word_len):
    word_tokens = sent_tokenizer(sent)
    char_tokens = []
    for word in word_tokens:
        word = list(word[:max_word_len])
        word += ["<pad>"] * (max_word_len - len(word))
        char_tokens += word
    return char_tokens


def get_sent_pair_panda_df(file_path, label_map_fn, sent_tokenizer, max_word_len):
    with open(file_path, 'r', encoding="utf-8") as f:
        data = [line.strip().split('\t') for line in f]
        data_source = list(map(lambda x: x[0], data))
        data_target = list(map(lambda x: x[1], data))
        data_source_char = list(map(lambda x: tokenize_char(x[0], sent_tokenizer, max_word_len), data))
        data_target_char = list(map(lambda x: tokenize_char(x[1], sent_tokenizer, max_word_len), data))
        data_label = list(map(lambda x: label_map_fn(x[2]), data))
    full_df = pd.DataFrame({"source": data_source, "target": data_target,
                            "source_char": data_source_char, "target_char": data_target_char,
                            "label": data_label})
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


def save_list_as_text(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for word in data:
            f.write("{}\n".format(word))


def load_text_as_list(output_path):
    with open(output_path, 'r', encoding="utf-8") as f:
        return f.read().splitlines()
