from dataset import SentPairDataset, SentPairConfig, save_list_as_text, load_text_as_list
from functools import partial
from sent_pair.cnn import SentPairCNN, SentPairCNNConfig
from sent_pair.bidaf import BiDAF, BiDAFConfig
from tasks.utils import *
import torch
from training.basics import BasicConfig, run_basics, infer_basics


class SyntheticConfig(object):
    task_name = "synthetic"
    train_file = "data/synthetic.train"
    test_file = "data/synthetic.test"
    w2v_file = None
    vocab_output_path = "data/synthetic_vocab.txt"
    char_vocab_output_path = "data/synthetic_char_vocab.txt"
    infer_output_path = "data/synthetic_infer_result.txt"
    max_source_len = 15
    max_target_len = 15
    max_word_len = 4
    train_test_ratio = 0.8
    batch_size = 8
    embed_size = 10
    num_classes = 4


def idx2label(idx):
    return "label_{}".format(idx)


def label2idx(label):
    return int(label[-1])


def use_sent_pair_cnn(vocab):
    model_config = SentPairCNNConfig.get_common(
        embed_size=SyntheticConfig.embed_size,
        output_size=SyntheticConfig.num_classes,
        max_sen_len=max(SyntheticConfig.max_source_len, SyntheticConfig.max_target_len)
    )
    model = SentPairCNN(
        config=model_config,
        vocab_size=len(vocab))
    fetch_batch_fn = SentPairCNN.fetch_sent_pair_batch_fn
    return model, fetch_batch_fn


def use_bidaf(word_vocab, char_vocab):
    model_config = BiDAFConfig.get_common(
        embed_size=SyntheticConfig.embed_size,
        output_size=SyntheticConfig.num_classes
    )
    model = BiDAF(
        config=model_config,
        w_vocab_size=len(word_vocab),
        c_vocab_size=len(char_vocab)
    )

    return model, partial(BiDAF.fetch_sent_pair_batch_fn, SyntheticConfig.max_word_len)


def run_synthetic_classification():

    # Sentence Pair Dataset config

    config = SentPairConfig(max_source_len=SyntheticConfig.max_source_len, max_target_len=SyntheticConfig.max_target_len,
                            train_test_ratio=SyntheticConfig.train_test_ratio, batch_size=SyntheticConfig.batch_size,
                            max_word_len=SyntheticConfig.max_word_len)
    stance_dataset = SentPairDataset(config)
    stance_dataset.load_data(train_file=SyntheticConfig.train_file, test_file=SyntheticConfig.test_file,
                             w2v_file=SyntheticConfig.w2v_file, embed_size=SyntheticConfig.embed_size,
                             vocab_output_path=SyntheticConfig.vocab_output_path,
                             char_vocab_output_path=SyntheticConfig.char_vocab_output_path,
                             label_map_fn=label2idx)

    # Run config
    run_config = BasicConfig.get_common()

    # model, fetch_batch_fn = use_sent_pair_cnn(stance_dataset.vocab)
    model, fetch_batch_fn = use_bidaf(stance_dataset.vocab, stance_dataset.char_vocab)

    exp_name = get_exp_name(SyntheticConfig.task_name, SentPairCNN.get_name())

    return run_basics(
        exp_name=exp_name,
        model=model,
        dataset=stance_dataset,
        fetch_batch_fn=fetch_batch_fn,
        config=run_config
    )


def infer_synthetic_classification(infer_model_path, infer_file):
    config = SentPairConfig(max_source_len=SyntheticConfig.max_source_len, max_target_len=SyntheticConfig.max_target_len,
                            train_test_ratio=SyntheticConfig.train_test_ratio, batch_size=SyntheticConfig.batch_size,
                            max_word_len=SyntheticConfig.max_word_len)
    stance_dataset = SentPairDataset(config)

    # Run config
    run_config = BasicConfig.get_common()
    loaded_vocab = load_text_as_list(SyntheticConfig.vocab_output_path)
    loaded_char_vocab = load_text_as_list(SyntheticConfig.char_vocab_output_path)

    # model, fetch_batch_fn = use_sent_pair_cnn(loaded_vocab)
    model, fetch_batch_fn = use_bidaf(loaded_vocab, loaded_char_vocab)

    loaded_state_dict = torch.load(infer_model_path)["state_dict"]
    model.load_state_dict(loaded_state_dict)
    stance_dataset.load_infer_data(infer_file=infer_file,
                                   existing_vocab=loaded_vocab, existing_char_vocab=loaded_char_vocab,
                                   label_map_fn=label2idx)
    predictions = infer_basics(
        model=model,
        dataset=stance_dataset,
        fetch_batch_fn=fetch_batch_fn,
        config=run_config
    )
    predictions = [idx2label(pred) for pred in predictions]

    save_list_as_text(predictions, SyntheticConfig.infer_output_path)
