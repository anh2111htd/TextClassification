import torch
from torch import nn
import torch.nn.functional as f_torch


class CharEmbeddingConfig(object):
    def __init__(self, c_embed_size, out_chs, filters):
        self.c_embed_size = c_embed_size
        self.out_chs = out_chs
        self.filters = filters


class CharEmbedding(nn.Module):
    def __init__(self, c_vocab_size, config):
        super(CharEmbedding, self).__init__()
        self.c_embed_size = config.c_embed_size
        self.c_embedding = nn.Embedding(c_vocab_size, config.c_embed_size)
        self.conv = nn.ModuleList([nn.Conv2d(1, config.out_chs, (f[0], f[1])) for f in config.filters])
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        # x: (batch_size, sent_len, word_len)
        input_shape = x.size()
        word_len = x.size(2)
        x = x.view(-1, word_len)  # (batch_size * sent_len, word_len)
        x = self.c_embedding(x)
        x = x.view(*input_shape, -1)  # (batch_size, sent_len, word_len, c_embed_size)
        x = x.sum(2)  # (batch_size, sent_len, c_embed_size) # add all char embedding

        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, c_embed_size) - insert conv input channel = 1
        x = [f_torch.relu(conv(x)) for conv in self.conv]
        # (batch_size, out_chs, sent_len, c_embed_size-filter_w+1), stride == 1
        x = [xx.view((xx.size(0), xx.size(2), xx.size(3), xx.size(1))) for xx in x]
        x = [torch.sum(xx, 2) for xx in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)

        return x

