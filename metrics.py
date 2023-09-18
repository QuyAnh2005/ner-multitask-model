import torch
from sklearn.metrics import f1_score
from utils import PAD_token

def flat_accuracy(output, target):
    preds = output.argmax(dim=2)
    correct = (torch.flatten(preds) == torch.flatten(target)).float()
    acc = correct.sum() / len(correct)
    return acc

def f1_score_seq(output, target):
    preds = output.argmax(dim=2)
    score = 0.0
    for idx in range(len(preds)):
        ground_truth = [token for token in target[idx] if token != PAD_token]
        pred = preds[idx][:len(ground_truth)]
        score += f1_score(ground_truth, pred, average='micro')
    return score / len(preds)
