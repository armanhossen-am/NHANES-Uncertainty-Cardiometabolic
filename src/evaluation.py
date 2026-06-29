from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    brier_score_loss,
    confusion_matrix,
)


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, y_prob),
    }


def bootstrap_auc_ci(y_true, y_prob, n_bootstrap=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    aucs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)

        if len(np.unique(y_true[idx])) < 2:
            continue

        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))

    lower = np.percentile(aucs, 2.5)
    upper = np.percentile(aucs, 97.5)

    return lower, upper


def bootstrap_auprc_ci(y_true, y_prob, n_bootstrap=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    auprcs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)

        if len(np.unique(y_true[idx])) < 2:
            continue

        auprcs.append(average_precision_score(y_true[idx], y_prob[idx]))

    lower = np.percentile(auprcs, 2.5)
    upper = np.percentile(auprcs, 97.5)

    return lower, upper


def evaluate_model(
    y_true,
    y_prob,
    model_name,
    scenario,
    outcome,
    threshold=0.5,
    n_bootstrap=1000,
):
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)

    auroc_low, auroc_high = bootstrap_auc_ci(
        y_true, y_prob, n_bootstrap=n_bootstrap
    )

    auprc_low, auprc_high = bootstrap_auprc_ci(
        y_true, y_prob, n_bootstrap=n_bootstrap
    )

    row = {
        "scenario": scenario,
        "outcome": outcome,
        "model": model_name,
        **metrics,
        "auroc_ci_low": auroc_low,
        "auroc_ci_high": auroc_high,
        "auprc_ci_low": auprc_low,
        "auprc_ci_high": auprc_high,
        "threshold": threshold,
        "n_test": len(y_true),
        "prevalence_test": np.mean(y_true),
    }

    return row
