import torch
from torch import nn
import torch.nn.functional as f_torch
from sent_pair.bidaf.char_embedding import CharEmbedding, CharEmbeddingConfig
from sent_pair.bidaf.word_embedding import WordEmbedding, WordEmbeddingConfig
from sent_pair.bidaf.highway import Highway, HighwayConfig


class BiDAFConfig(object):
    def __init__(self, freeze_w_embed, w_embed_size, c_embed_size, c_out_chs, c_filters,
                 output_size, hw_n_layers=2, dropout=0.2):
        self.freeze_w_embed = freeze_w_embed
        self.w_embed_size = w_embed_size
        self.c_embed_size = c_embed_size
        self.c_out_chs = c_out_chs
        self.c_filters = c_filters
        self.output_size = output_size
        self.hw_n_layers = hw_n_layers
        self.dropout = dropout

    @staticmethod
    def get_common(embed_size, output_size):
        return BiDAFConfig(
            freeze_w_embed=False,
            w_embed_size=embed_size,
            c_embed_size=5,
            c_out_chs=5,
            c_filters=[[1, 5]],
            output_size=output_size,
            hw_n_layers=1,
            dropout=0.3
        )


class BiDAF(nn.Module):
    def __init__(self, config, c_vocab_size, w_vocab_size):
        super(BiDAF, self).__init__()
        self.config = config
        self.loss = nn.CrossEntropyLoss()
        self.w_embed_size = config.w_embed_size
        self.c_embed_size = config.c_embed_size
        self.embed_size = self.w_embed_size + self.c_embed_size
        self.char_embed_net = CharEmbedding(c_vocab_size, BiDAF.get_char_embed_config(config))
        self.word_embed_net = WordEmbedding(w_vocab_size, BiDAF.get_word_embed_config(config))
        self.highway_net = Highway(self.embed_size, BiDAF.get_highway_config(config))
        self.context_embed_net = nn.GRU(self.embed_size, self.embed_size, bidirectional=True,
                                        dropout=config.dropout, batch_first=True)
        self.linear_weight = nn.Linear(6 * self.embed_size, 1, bias=False)
        self.encoder = nn.GRU(8 * self.embed_size, self.embed_size, num_layers=2, bidirectional=True,
                              dropout=config.dropout, batch_first=True)
        self.output_layer = nn.Linear(4 * self.embed_size, self.config.output_size)
        self.softmax = nn.Softmax(dim=1)

    def build_contextual_embd(self, char_seq, word_seq):
        char_embed_seq = self.char_embed_net(char_seq)
        word_embed_seq = self.word_embed_net(word_seq)
        embed_seq = torch.cat((char_embed_seq, word_embed_seq), 2)  # concat word and char embedding
        embed_seq = self.highway_net(embed_seq)
        context_embed_seq, last_hidden_state = self.context_embed_net(embed_seq)
        return context_embed_seq

    def forward(self, source_char, source_word, target_char, target_word):
        batch_size = source_word.size(0)
        source_len = source_word.size(1)
        target_len = target_word.size(1)

        source_embedding = self.build_contextual_embd(source_char, source_word)
        target_embedding = self.build_contextual_embd(target_char, target_word)

        # (batch_size, source_len, target_len, 2 * embed_size)
        shape = (batch_size, source_len, target_len, 2 * self.embed_size)
        # (batch_size, source_len, 1, 2 * embed_size)
        source_embedding_ex = source_embedding.unsqueeze(2)
        # (batch_size, source_len, target_len, 2 * embed_size)
        source_embedding_ex = source_embedding_ex.expand(shape)
        # (batch_size, 1, target_len, 2 * embed_size)
        target_embedding_ex = target_embedding.unsqueeze(1)
        # (batch_size, source_len, target_len, 2 * embed_size)
        target_embedding_ex = target_embedding_ex.expand(shape)
        element_wise_mul = torch.mul(source_embedding_ex, target_embedding_ex)
        # (batch_size, source_len, target_len, 6 * embed_size)
        cat_data = torch.cat((source_embedding_ex, target_embedding_ex, element_wise_mul), 3)
        # (batch_size, source_len, target_len)
        combined_seq_rep = self.linear_weight(cat_data).view(batch_size, source_len, target_len)

        # source-to-target attention
        source_target_attn = f_torch.softmax(combined_seq_rep, dim=-1)  # (batch_size, target_len, 2 * embed_size)
        # (batch_size, source_len, 2 * embed_size)
        # = bmm( (batch_size, source_len, target_len), (batch_size, target_len, 2 * embed_size) )
        source_target = torch.bmm(source_target_attn, target_embedding)
        # target-to-source attention
        # (batch_size, source_len)
        target_source_attn = f_torch.softmax(torch.max(combined_seq_rep, 2)[0], dim=-1)
        # = bmm( (batch_size, 1, source_len), (batch_size, source_len, 2 * embed_size) )
        target_source = torch.bmm(target_source_attn.unsqueeze(1), source_embedding)
        target_source = target_source.repeat(1, target_len, 1)

        # target aware representation of each source word
        # (batch_size, source_len, 8 * embed_size)
        target_aware_source_rep = torch.cat((source_embedding, source_target,
                                             source_embedding.mul(source_target),
                                             source_embedding.mul(target_source)), 2)
        # last_hidden_state: (batch_size, 2, 2 * embed_size)
        encoder_output, last_hidden_state = self.encoder(target_aware_source_rep)
        # (batch_size, 4 * self.embed_size)
        combined_encoded_rep = last_hidden_state.view(batch_size, 4 * self.embed_size)
        output = self.softmax(self.output_layer(combined_encoded_rep))
        return output

    @staticmethod
    def get_char_embed_config(config):
        return CharEmbeddingConfig(
            c_embed_size=config.c_embed_size,
            out_chs=config.c_out_chs,
            filters=config.c_filters
        )

    @staticmethod
    def get_word_embed_config(config):
        return WordEmbeddingConfig(
            freeze_w_embed=config.freeze_w_embed,
            w_embed_size=config.w_embed_size
        )

    @staticmethod
    def get_highway_config(config):
        return HighwayConfig(
            hw_n_layers=config.hw_n_layers
        )

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight)

    @classmethod
    def get_name(cls):
        return "bidaf"

    @staticmethod
    def fetch_sent_pair_batch_fn(max_word_len, batch, device):
        batch_size = len(batch)
        x_source, x_target = batch.source.to(device).permute(1, 0), batch.target.to(device).permute(1, 0)
        x_source_char, x_target_char = batch.source_char.to(device), batch.target_char.to(device)
        x_source_char = x_source_char.view(batch_size, -1, max_word_len)
        x_target_char = x_target_char.view(batch_size, -1, max_word_len)
        return x_source_char, x_source, \
            x_target_char, x_target

    def init_word_embeddings(self, word_embeddings, freeze=False):
        self.word_embed_net.init_word_embeddings(word_embeddings, freeze)
