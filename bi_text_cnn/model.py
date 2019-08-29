import torch
from torch import nn
import numpy as np
from bi_text_cnn.utils import *


class BiTextCNN(nn.Module):
    def __init__(self, config, vocab_size, word_embedings):
        super(BiTextCNN, self).__init__()
        self.config = config
        self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(word_embedings, requires_grad=False)
        self.optimizer = None
        self.loss_op = None

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
        self.fc = nn.Linear(self.config.num_channels * len(self.config.kernel_size), self.config.output_size)

        # Softmax non-linearity
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

    def forward(self, x_source, x_target):
        embedded_source, embedded_target = self.embeddings(x_source).permute(1, 2, 0), \
                                           self.embeddings(x_target).permute(1, 2, 0)
        source_read_out = self.forward_single(embedded_source)
        target_read_out = self.forward_single(embedded_target)
        final_read_out = torch.cat((source_read_out, target_read_out), 1)
        output = self.softmax(self.fc(final_read_out))
        return output

    def forward_single(self, embedded_sent):
        conv_out1 = self.conv1(embedded_sent).squeeze(2)  # shape=(64, num_channels, 1) (squeeze 1)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)

        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        final_out = self.fc(final_feature_map)
        return self.relu(final_out)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []

        # Reduce learning rate as number of epochs increase
        if (epoch == int(self.config.max_epochs / 3)) or (epoch == int(2 * self.config.max_epochs / 3)):
            self.reduce_lr()

        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x_source, x_target = batch.source.cuda(), batch.target.cuda()
                y = (batch.label - 1).type(torch.cuda.LongTensor)
            else:
                x_source, x_target = batch.source, batch.target
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x_source, x_target)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()

            if i % 100 == 0:
                print("Iter: {}".format(i + 1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []

                # Evalute Accuracy on validation set
                val_accuracy = evaluate_model(self, val_iterator)
                print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                self.train()

        return train_losses, val_accuracies
