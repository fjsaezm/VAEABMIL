import torch
import numpy as np
from .predict import predict


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
)


def expected_calibration_error(y_true, y_pred, n_bins=10, thr=0.5):
    """
    Compute the Expected Calibration Error (ECE).

    Arguments:
        y_true: Array of ground-truth labels, of shape (n_samples,).
        y_pred: Array of predictions, of shape (n_samples,).
        n_bins: Number of bins to use for reliability diagram.
        thr: Threshold to use for binary predictions.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(y_pred > bin_lower, y_pred <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin] == (y_pred[in_bin] > thr))
            avg_confidence_in_bin = np.mean(y_pred[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def auprc(labels, preds, pos_label=1):
    """
    Compute the Area Under the Precision-Recall Curve (AUPRC).

    Arguments:
        labels: Array of ground-truth labels, of shape (n_samples,).
        preds: Array of predictions, of shape (n_samples,).
        pos_label: Label of the positive class.
    """
    precision, recall, _ = precision_recall_curve(labels, preds, pos_label=pos_label)
    return auc(recall, precision)


def fpr_at_thr_tpr(y_true, y_pred, pos_label=1, thr=0.8):
    """
    Return the FPR when TPR is at minimum thr.

    Arguments:
        y_true: Array of ground-truth labels, of shape (n_samples,).
        y_pred: Array of predictions, of shape (n_samples,).
        pos_label: Label of the positive class.
        thr: Minimum TPR value to consider.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=pos_label)

    if all(tpr < thr):
        # No threshold allows TPR >= thr
        return 0
    elif all(tpr >= thr):
        # All thresholds allow TPR >= thr, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x >= thr]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == thr
        return np.interp(thr, tpr, fpr)


def compute_optimal_thr(y_true, y_pred):
    """
    Compute optimal threshold for y_pred

    Arguments:
        y_true: Array of ground-truth labels, of shape (n_samples,).
        y_pred: Array of predictions, of shape (n_samples,).
    """
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    # idx = np.argmax(tpr - fpr, axis=0)
    idx = np.argmax(tpr * (1 - fpr), axis=0)
    optimal_thr = thr[idx]
    return optimal_thr


def evaluate(Y_true, Y_logits_pred, desc="Test"):

    Y_pred = np.where(Y_logits_pred > 0, 1, 0)  # (batch_size, 1)

    metrics = {}
    metrics["bag/bce_loss"] = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.from_numpy(Y_logits_pred).float(), torch.from_numpy(Y_true).float()
    ).item()
    try:
        metrics["bag/auroc"] = roc_auc_score(Y_true, Y_logits_pred)
    except ValueError:
        metrics["bag/auroc"] = 0.0
    metrics["bag/auprc"] = auprc(Y_true, Y_logits_pred)
    metrics["bag/fpr90"] = fpr_at_thr_tpr(Y_true, Y_logits_pred, thr=0.90)

    metrics["bag/acc"] = accuracy_score(Y_true, Y_pred)
    metrics["bag/prec"] = precision_score(Y_true, Y_pred, zero_division=0)
    metrics["bag/rec"] = recall_score(Y_true, Y_pred, zero_division=0)
    metrics["bag/f1"] = f1_score(Y_true, Y_pred, zero_division=0)
    metrics["bag/ece"] = expected_calibration_error(Y_true, Y_pred)

    metrics = {f"{desc.lower()}/{k}": v for k, v in metrics.items()}

    return metrics


def predict_and_eval(model, dataloader, device, desc="Test", set="Test"):
    Y_true, _, Y_logits_pred, _, _, _ = predict(
        model, dataloader, device=device, desc=desc
    )
    return evaluate(Y_true, Y_logits_pred, desc=set)
