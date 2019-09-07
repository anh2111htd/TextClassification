import numpy as np
import os
from training import Evaluator, save_model_checkpoint
import torch
from torch.utils.tensorboard import SummaryWriter


class BasicConfig(object):
    def __init__(self, grad_accumulate_steps, max_grad_norm, inform_every_batch_num, save_every_epoch_num,
                 eval_every_epoch_num, use_cuda, learning_rate, epoch_num, metrics, log_dir, ckpt_dir):
        self.grad_accumulate_steps = grad_accumulate_steps
        self.max_grad_norm = max_grad_norm
        self.inform_every_batch_num = inform_every_batch_num
        self.save_every_epoch_num = save_every_epoch_num
        self.eval_every_epoch_num = eval_every_epoch_num
        self.use_cuda = use_cuda
        self.learning_rate = learning_rate
        self.epoch_num = epoch_num
        self.metrics = metrics
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir

    @staticmethod
    def get_common(use_cuda=False):
        grad_accumulate_steps = 4
        max_grad_norm = 10.
        inform_every_batch_num = 50
        save_every_epoch_num = 5
        eval_every_epoch_num = 2
        learning_rate = 1e-3
        epoch_num = 50
        metrics = ["accuracy", "precision", "recall", "f1"]
        log_dir = "exp_log/"
        ckpt_dir = "exp_ckpt/"
        return BasicConfig(
            grad_accumulate_steps=grad_accumulate_steps,
            max_grad_norm=max_grad_norm,
            inform_every_batch_num=inform_every_batch_num,
            save_every_epoch_num=save_every_epoch_num,
            eval_every_epoch_num=eval_every_epoch_num,
            use_cuda=use_cuda,
            learning_rate=learning_rate,
            epoch_num=epoch_num,
            metrics=metrics,
            log_dir=log_dir,
            ckpt_dir=ckpt_dir,
        )


def run_basics(exp_name, model, dataset, fetch_batch_fn, config):
    device = torch.device("cuda:0" if config.use_cuda else "cpu")
    writer = SummaryWriter(os.path.join(config.log_dir, exp_name))

    best_evaluator = Evaluator(predictions=[], labels=[])
    model.init_word_embeddings(dataset.word_embeddings)
    model.apply(model.init_weights)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer.zero_grad()

    for epoch in range(config.epoch_num):
        loss, train_evaluator = train_fn(model, optimizer, dataset.train_iterator,
                                         fetch_batch_fn, device, config)
        print("Finish training {} epochs, loss: {}".format(epoch, loss))
        train_result = train_evaluator.evaluate(config.metrics)
        writer.add_scalars("train", train_result, epoch)

        if epoch % config.eval_every_epoch_num == 0:
            validate_loss, validate_evaluator = eval_fn(model, dataset.validate_iterator,
                                                        fetch_batch_fn, device, config)
            print("Finish validating {} epochs, loss: {}".format(epoch, validate_loss))
            validate_result = validate_evaluator.evaluate(config.metrics)
            writer.add_scalars("validate", validate_result, epoch)

            if validate_evaluator.is_better_than(best_evaluator, config.metrics):
                print("Best evaluator is updated.")
                best_evaluator = validate_evaluator
                save_model_checkpoint(state=dict(
                    epoch=epoch,
                    state_dict=dict([(key, value.to("cpu")) for key, value in model.state_dict().items()]),
                    ),
                    is_best=True,
                    output_dir=config.ckpt_dir,
                    exp_name=exp_name,
                    step=epoch
                )
        if epoch % config.save_every_epoch_num == 0:
            print("Saving model checkpoint.")
            save_model_checkpoint(state=dict(
                epoch=epoch,
                state_dict=dict([(key, value.to("cpu")) for key, value in model.state_dict().items()]),
                ),
                is_best=False,
                output_dir=config.ckpt_dir,
                exp_name=exp_name,
                step=epoch
            )

    writer.close()


def train_fn(model, optimizer, train_iterator, fetch_batch_fn, device, config):
    model.train()
    total_loss = 0
    batch_idx = 0
    predictions, labels = [], []

    for batch_idx, batch in enumerate(train_iterator):
        with torch.set_grad_enabled(True):
            label = (batch.label - 1).type(torch.LongTensor).to(device)
            logit = model(*fetch_batch_fn(batch, device))
            pred = torch.max(logit.cpu().data, 1)[1] + 1
            loss = model.loss(logit, label)
            predictions.extend(pred.numpy())
            labels.extend(batch.label.numpy())
            loss.backward()

            if batch_idx % config.grad_accumulate_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item()

        if batch_idx % config.inform_every_batch_num == 0:
            print("Finish training {} batches.".format(batch_idx))

        batch_idx += 1

    loss_per_batch = total_loss / batch_idx
    train_evaluator = Evaluator(np.array(predictions).flatten(), labels)

    return loss_per_batch, train_evaluator


def eval_fn(model, eval_iterator, fetch_batch_fn, device, config):
    model.eval()
    total_loss = 0
    batch_idx = 0
    predictions, labels = [], []

    for batch_idx, batch in enumerate(eval_iterator):
        with torch.set_grad_enabled(False):
            label = (batch.label - 1).type(torch.LongTensor).to(device)
            logit = model(*fetch_batch_fn(batch, device))
            pred = torch.max(logit.cpu().data, 1)[1] + 1
            loss = model.loss(logit, label)
            predictions.extend(pred.numpy())
            labels.extend(batch.label.numpy())

        total_loss += loss.item()

        if batch_idx % config.inform_every_batch_num == 0:
            print("Finish evaluating {} batches.".format(batch_idx))

        batch_idx += 1

    loss_per_batch = total_loss / batch_idx
    eval_evaluator = Evaluator(np.array(predictions).flatten(), labels)

    return loss_per_batch, eval_evaluator
