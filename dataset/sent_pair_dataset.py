import dataset.utils as dataset_utils
import spacy
from torchtext import data
from torchtext.vocab import Vectors


class SentPairConfig(object):
    def __init__(self, max_source_len, max_target_len, train_test_ratio,
                 batch_size):
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.train_test_ratio = train_test_ratio
        self.batch_size = batch_size


class SentPairDataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.validate_iterator = None
        self.vocab = []
        self.word_embeddings = {}
        self.preprocessor = spacy.load('en')

    @staticmethod
    def fetch_sent_pair_batch_fn(batch, device):
        x_source, x_target = batch.source.to(device), batch.target.to(device)
        return x_source, x_target

    def tokenize(self, sent):
        return [x.text for x in self.preprocessor.tokenizer(sent) if x.text != " "]

    def load_data(self, w2v_file, train_file, test_file, val_file=None):
        source_field = data.Field(sequential=True, tokenize=self.tokenize,
                                  lower=True, fix_length=self.config.max_source_len)
        target_field = data.Field(sequential=True, tokenize=self.tokenize,
                                  lower=True, fix_length=self.config.max_target_len)
        label_field = data.Field(sequential=False, use_vocab=False)
        data_fields = [("source", source_field), ("target", target_field), ("label", label_field)]

        train_df = dataset_utils.get_sent_pair_panda_df(train_file)
        train_examples = [data.Example.fromlist(i, data_fields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, data_fields)

        test_df = dataset_utils.get_sent_pair_panda_df(test_file)
        test_example = [data.Example.fromlist(i, data_fields) for i in test_df.values.tolist()]
        test_data = data.Dataset(test_example, data_fields)

        if val_file:
            val_df = dataset_utils.get_sent_pair_panda_df(val_file)
            val_example = [data.Example.fromlist(i, data_fields) for i in val_df.values.tolist()]
            val_data = data.Dataset(val_example, data_fields)
        else:
            train_data, val_data = train_data.split(split_ratio=self.config.train_test_ratio)

        source_field.build_vocab(train_data, vectors=Vectors(w2v_file))
        target_field.build_vocab(train_data, vectors=Vectors(w2v_file))
        source_field.vocab.extend(target_field.vocab)
        self.word_embeddings = source_field.vocab.vectors
        self.vocab = source_field.vocab

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