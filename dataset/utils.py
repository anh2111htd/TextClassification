import pandas as pd


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
