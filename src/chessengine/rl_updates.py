"""Outcome fitting with an optional explicit penalty toward a frozen initializer."""

import torch


def value_loss(prediction, outcome, initial_prediction=None, anchor_weight=0.0):
    if anchor_weight < 0:
        raise ValueError("anchor_weight must be nonnegative")
    loss = ((prediction - outcome) ** 2).mean()
    if anchor_weight:
        if initial_prediction is None:
            raise ValueError("Anchoring requires frozen initial predictions")
        loss = loss + anchor_weight * ((prediction - initial_prediction.detach()) ** 2).mean()
    return loss


def frozen_predictions(model, inputs):
    model.eval()
    with torch.no_grad():
        return model(inputs).squeeze(-1)
