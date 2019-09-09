import common
import numpy as np
import os


def generate_synthetic_sent_pair_dataset(train_size, num_classes, max_source_len, max_target_len, vocab_size):
    train_dataset = []
    for _ in range(train_size):
        label = np.random.randint(0, num_classes)
        source_len = np.random.randint(2, max_source_len + 1)
        target_len = np.random.randint(2, max_target_len + 1)
        source_seed = np.random.randint(1, vocab_size)
        target_seed = np.random.randint(1, vocab_size)
        source_sent, target_sent = [source_seed] * source_len, [target_seed] * target_len
        for i in range(1, min(source_len, target_len)):
            # if source_sent[i - 1] % 3 == 0:
            #     source_sent[i] = abs(source_sent[i - 1] + target_sent[i - 1] + label) % vocab_size
            #     target_sent[i] = abs(target_sent[i - 1] + label) % vocab_size
            # elif source_sent[i - 1] % 3 == 1:
            #     source_sent[i] = abs(source_sent[i - 1] - target_sent[i - 1] - label) % vocab_size
            #     target_sent[i] = abs(target_sent[i - 1] - label) % vocab_size
            # elif source_sent[i - 1] % 3 == 2:
            source_sent[i] = abs((source_sent[i - 1] + target_sent[i - 1]) * label) % vocab_size
            target_sent[i] = abs(target_sent[i - 1] * label) % vocab_size
        train_dataset.append([" ".join([str(x) for x in source_sent]), " ".join([str(x) for x in target_sent]),
                              "label_{}".format(label)])
    return train_dataset


def generate_synthetic_sent_pair(data_dir="data/"):
    vocab_size = 100
    max_source_len = 30
    max_target_len = 30
    num_classes = 4
    train_size = 2000
    test_size = 200
    train_dataset = generate_synthetic_sent_pair_dataset(train_size, num_classes,
                                                         max_source_len, max_target_len, vocab_size)
    test_dataset = generate_synthetic_sent_pair_dataset(test_size, num_classes,
                                                        max_source_len, max_target_len, vocab_size)
    common.write_csv(train_dataset, None, os.path.join(data_dir, "synthetic.train"), delimiter="\t")
    common.write_csv(test_dataset, None, os.path.join(data_dir, "synthetic.test"), delimiter="\t")

