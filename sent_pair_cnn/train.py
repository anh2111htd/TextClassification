from sent_pair_cnn.model import *
from sent_pair_cnn.utils import *
import torch.optim as optim


def train_sent_pair_cnn(sent_pair_dataset):
    config = Config()

    model = SentPairCNN(config, len(sent_pair_dataset.vocab), sent_pair_dataset.word_embeddings)
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    cross_entropy_loss = nn.CrossEntropyLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(cross_entropy_loss)

    train_losses = []
    val_accuracies = []

    for i in range(config.max_epochs):
        train_loss, val_accuracy = model.run_epoch(sent_pair_dataset.train_iterator, sent_pair_dataset.validate_iterator, i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

    train_acc = evaluate_model(model, sent_pair_dataset.train_iterator)
    val_acc = evaluate_model(model, sent_pair_dataset.validate_iterator)
    test_acc = evaluate_model(model, sent_pair_dataset.test_iterator)

    print('Final Training Accuracy: {:.4f}'.format(train_acc))
    print('Final Validation Accuracy: {:.4f}'.format(val_acc))
    print('Final Test Accuracy: {:.4f}'.format(test_acc))

