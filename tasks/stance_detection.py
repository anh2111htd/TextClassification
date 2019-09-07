from dataset import SentPairDataset, SentPairConfig
from sent_pair_cnn import SentPairCNN, SentPairCNNConfig
from tasks.utils import *
from training.basics import BasicConfig, run_basics


def run_stance_detection():

    # Sentence Pair Dataset config
    task_name = "stance"
    train_file = "data/stance.train"
    test_file = "data/stance.test"
    w2v_file = "data/glove.6B.50d.txt"
    max_source_len = 50
    max_target_len = 50
    train_test_ratio = 0.5
    batch_size = 8
    embed_size = 50
    config = SentPairConfig(max_source_len=max_source_len, max_target_len=max_target_len,
                            train_test_ratio=train_test_ratio, batch_size=batch_size)
    stance_dataset = SentPairDataset(config)
    stance_dataset.load_data(train_file=train_file, test_file=test_file,
                             w2v_file=w2v_file, embed_size=embed_size)
    vocab_size = len(stance_dataset.vocab)

    # Run config
    run_config = BasicConfig.get_common()
    num_classes = 4
    model_config = SentPairCNNConfig.get_common(
        embed_size=embed_size,
        output_size=num_classes,
        max_sen_len=max(max_source_len, max_target_len)
    )
    model = SentPairCNN(
        config=model_config,
        vocab_size=vocab_size)
    fetch_batch_fn = SentPairDataset.fetch_sent_pair_batch_fn
    exp_name = get_exp_name(task_name, SentPairCNN.get_name())

    run_basics(
        exp_name=exp_name,
        model=model,
        dataset=stance_dataset,
        fetch_batch_fn=fetch_batch_fn,
        config=run_config
    )
