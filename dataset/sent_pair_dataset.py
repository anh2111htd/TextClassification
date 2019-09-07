import dataset.utils as dataset_utils
import spacy
from torchtext import data


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
        self.infer_iterator = None
        self.vocab = []
        self.word_embeddings = None
        self.preprocessor = spacy.load('en')

    @staticmethod
    def fetch_sent_pair_batch_fn(batch, device):
        x_source, x_target = batch.source.to(device), batch.target.to(device)
        return x_source, x_target

    def tokenize(self, sent):
        return [x.text for x in self.preprocessor.tokenizer(sent) if x.text != " "]

    def load_infer_data(self, infer_file, existing_vocab, label_map_fn):
        source_field = data.Field(sequential=True, tokenize=self.tokenize,
                                  lower=True, fix_length=self.config.max_source_len)
        target_field = data.Field(sequential=True, tokenize=self.tokenize,
                                  lower=True, fix_length=self.config.max_target_len)
        data_fields = [("source", source_field), ("target", target_field)]

        infer_df = dataset_utils.get_sent_pair_panda_df(infer_file, label_map_fn)
        infer_examples = [data.Example.fromlist(i, data_fields) for i in infer_df.values.tolist()]
        infer_data = data.Dataset(infer_examples, data_fields)
        source_field.build_vocab(existing_vocab)
        target_field.build_vocab(existing_vocab)

        self.infer_iterator = data.BucketIterator(
            infer_data,
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.source) + len(x.target),
            repeat=False,
            shuffle=False)

    def load_data(self, train_file, test_file, embed_size, vocab_output_path, label_map_fn,
                  w2v_file=None, val_file=None):
        source_field = data.Field(sequential=True, tokenize=self.tokenize,
                                  lower=True, fix_length=self.config.max_source_len)
        target_field = data.Field(sequential=True, tokenize=self.tokenize,
                                  lower=True, fix_length=self.config.max_target_len)
        label_field = data.Field(sequential=False, use_vocab=False)
        data_fields = [("source", source_field), ("target", target_field), ("label", label_field)]

        train_df = dataset_utils.get_sent_pair_panda_df(train_file, label_map_fn)
        train_examples = [data.Example.fromlist(i, data_fields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, data_fields)

        test_df = dataset_utils.get_sent_pair_panda_df(test_file, label_map_fn)
        test_example = [data.Example.fromlist(i, data_fields) for i in test_df.values.tolist()]
        test_data = data.Dataset(test_example, data_fields)

        if val_file:
            val_df = dataset_utils.get_sent_pair_panda_df(val_file, label_map_fn)
            val_example = [data.Example.fromlist(i, data_fields) for i in val_df.values.tolist()]
            val_data = data.Dataset(val_example, data_fields)
        else:
            train_data, val_data = train_data.split(split_ratio=self.config.train_test_ratio)

        source_field.build_vocab(train_data)
        target_field.build_vocab(train_data)
        label_field.build_vocab(train_data)
        source_field.vocab.extend(target_field.vocab)
        self.vocab = source_field.vocab
        self.word_embeddings = dataset_utils.init_word_embedding(self.vocab.itos, w2v_file, embed_size)

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

        dataset_utils.save_list_as_text(self.vocab.itos, vocab_output_path)
        print("Saved vocab file to {}".format(vocab_output_path))
