from torch import nn
import torch


class WordEmbeddingConfig(object):
    def __init__(self, freeze_w_embed, w_embed_size):
        self.freeze_w_embed = freeze_w_embed
        self.w_embed_size = w_embed_size


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, config):
        super(WordEmbedding, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, config.w_embed_size)

    def init_word_embeddings(self, word_embeddings, freeze_w_embed=False):
        word_embeddings_tensor = torch.from_numpy(word_embeddings)
        self.word_embeddings.from_pretrained(word_embeddings_tensor, freeze=freeze_w_embed)

    def forward(self, x):
        return self.word_embeddings(x)
