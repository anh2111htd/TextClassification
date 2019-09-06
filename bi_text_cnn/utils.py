import torch
from torchtext import data
from torchtext.vocab import Vectors
import spacy
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.validate_iterator = None
        self.vocab = []
        self.word_embeddings = {}

    def parse_label(self, label):
        """
        parse string labels into integers
        :param label: string of form '__label__0'
        :return: integer, 0 in this case
        """
        return int(label.strip()[-1])

    def get_panda_df(self, filepath):
        """
        load text file into pandas df, then to torch text
        :param filepath:
        :return:
        """
        with open(filepath, 'r', encoding="utf-8") as f:
            data = [line.strip().split('\t') for line in f]
            data_source = list(map(lambda x: x[0], data))
            data_target = list(map(lambda x: x[1], data)) 
            data_label = list(map(lambda x: self.parse_label(x[2]), data))
        full_df = pd.DataFrame({"source": data_source, "target": data_target, "label": data_label})
        return full_df

    def load_data(self, w2v_file, train_file, test_file, val_file=None):
        """
        Load data from file
        Set up train, test, val iterators
        Create word embeddings and vocabulary
        :param w2v_file:
        :param train_file:
        :param test_file:
        :param val_file:
        :param split_ratio:
        :return:
        """
        NLP = spacy.load('en')
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]

        SOURCE_TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        TARGET_TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [("source", SOURCE_TEXT), ("target", TARGET_TEXT), ("label", LABEL)]

        train_df = self.get_panda_df(train_file)
        train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, datafields)

        test_df = self.get_panda_df(test_file)
        test_example = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
        test_data = data.Dataset(test_example, datafields)

        if val_file:
            val_df = self.get_panda_df(val_file)
            val_example = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
            val_data = data.Dataset(val_example, datafields)
        else:
            train_data, val_data = train_data.split(split_ratio=self.config.split_ratio)

        SOURCE_TEXT.build_vocab(train_data, vectors=Vectors(w2v_file))
        TARGET_TEXT.build_vocab(train_data, vectors=Vectors(w2v_file))
        SOURCE_TEXT.vocab.extend(TARGET_TEXT.vocab)
        self.word_embeddings = SOURCE_TEXT.vocab.vectors
        self.vocab = SOURCE_TEXT.vocab

        self.train_iterator = data.BucketIterator(
            train_data,
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.source) + len(x.target),
            repeat=False,
            shuffle=True
        )

        self.validate_iterator, self.test_iterator = data.BucketIterator.splits(
            (val_data, test_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.source) + len(x.target),
            repeat=False,
            shuffle=False)

        print("Loaded {} training examples".format(len(train_data)))
        print("Loaded {} test examples".format(len(test_data)))
        print("Loaded {} validation examples".format(len(val_data)))


def evaluate_model(model, iterator):
    all_preds = []
    all_y = []
    for idx, batch in enumerate(iterator):
        if torch.cuda.is_available():
            x_source, x_target = batch.source.cuda(), batch.target.cuda()
        else:
            x_source, x_target = batch.source, batch.target
        y_pred = model(x_source, x_target) 
        predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    return score
