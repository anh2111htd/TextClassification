from bi_text_cnn.config import Config
from bi_text_cnn.model import *
from bi_text_cnn.utils import *
import torch.optim as optim


def train_text_cnn(train_file="../data/data.train", test_file="../data/data.test",
                   w2v_file="../data/glove.840B.300d.txt"):
    config = Config()

    dataset = Dataset(config)
    dataset.load_data(w2v_file=w2v_file, train_file=train_file, test_file=test_file)

    model = BiTextCNN(config, len(dataset.vocab), dataset.word_embeddings)
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    NLLLoss = nn.NLLLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(NLLLoss)
    ##############################################################

    train_losses = []
    val_accuracies = []

    for i in range(config.max_epochs):
        print("Epoch: {}".format(i))
        train_loss, val_accuracy = model.run_epoch(dataset.train_iterator, dataset.validate_iterator, i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

    train_acc = evaluate_model(model, dataset.train_iterator)
    val_acc = evaluate_model(model, dataset.validate_iterator)
    test_acc = evaluate_model(model, dataset.test_iterator)

    print('Final Training Accuracy: {:.4f}'.format(train_acc))
    print('Final Validation Accuracy: {:.4f}'.format(val_acc))
    print('Final Test Accuracy: {:.4f}'.format(test_acc))

