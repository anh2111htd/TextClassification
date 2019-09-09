import torch
from torch import nn


class SentPairCNNConfig(object):
    def __init__(self, freeze_embed, embed_size, num_channels, kernel_sizes, output_size,
                 max_sen_len, dropout_keep, read_out_size):
        self.freeze_embed = freeze_embed
        self.embed_size = embed_size
        self.num_channels = num_channels
        self.kernel_size = kernel_sizes
        self.output_size = output_size
        self.max_sen_len = max_sen_len
        self.dropout_keep = dropout_keep
        self.read_out_size = read_out_size

    @staticmethod
    def get_common(embed_size, output_size, max_sen_len):
        return SentPairCNNConfig(
            freeze_embed=False,
            embed_size=embed_size,
            num_channels=100,
            kernel_sizes=[3, 4, 5],
            output_size=output_size,
            max_sen_len=max_sen_len,
            dropout_keep=0.8,
            read_out_size=16
        )


class SentPairCNN(nn.Module):

    def __init__(self, config, vocab_size):
        super(SentPairCNN, self).__init__()
        self.config = config
        self.loss = nn.CrossEntropyLoss()
        self.word_embeddings = nn.Embedding(vocab_size, self.config.embed_size)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.embed_size, out_channels=self.config.num_channels,
                      kernel_size=self.config.kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[0] + 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.embed_size, out_channels=self.config.num_channels,
                      kernel_size=self.config.kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[1] + 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.embed_size, out_channels=self.config.num_channels,
                      kernel_size=self.config.kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[2] + 1)
        )

        self.dropout = nn.Dropout(self.config.dropout_keep)

        # Fully-Connected Layer
        self.fc1 = nn.Linear(self.config.num_channels * len(self.config.kernel_size), self.config.read_out_size)
        self.fc2 = nn.Linear(self.config.read_out_size * 2, self.config.output_size)

        # Softmax non-linearity
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x_source, x_target):
        embedded_source, embedded_target = self.word_embeddings(x_source).permute(1, 2, 0), \
                                           self.word_embeddings(x_target).permute(1, 2, 0)
        source_read_out = self.forward_single(embedded_source)
        target_read_out = self.forward_single(embedded_target)
        final_read_out = torch.cat((source_read_out, target_read_out), 1)
        output = self.softmax(self.fc2(final_read_out))
        return output

    def forward_single(self, embedded_sent): 
        conv_out1 = self.conv1(embedded_sent).squeeze(2)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)

        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        final_out = self.fc1(final_feature_map)
        return self.relu(final_out)

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight)

    @classmethod
    def get_name(cls):
        return "sent-pair-cnn"

    @staticmethod
    def fetch_sent_pair_batch_fn(batch, device):
        x_source, x_target = batch.source.to(device), batch.target.to(device)
        return x_source, x_target

    def init_word_embeddings(self, word_embeddings):
        word_embeddings_tensor = torch.from_numpy(word_embeddings)
        self.word_embeddings.from_pretrained(word_embeddings_tensor, freeze=self.config.freeze_embed)
