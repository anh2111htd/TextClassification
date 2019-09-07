import torch
import numpy as np
from sklearn.metrics import accuracy_score


def evaluate_model(model, iterator):
    all_preds = []
    all_y = []
    for idx, batch in enumerate(iterator):
        if torch.cuda.is_available():
            x_source, x_target = batch.source.cuda(), batch.target.cuda()
        else:
            x_source, x_target = batch.source, batch.target
        y_pred = model(x_source, x_target) 
        predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    return score
