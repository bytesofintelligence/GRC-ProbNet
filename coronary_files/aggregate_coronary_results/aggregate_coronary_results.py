import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

FOLDS = [1, 2, 3, 4, 5]
BASE_DIR = "output/asoca-coronary"
PRED_FILE = "classification/predictions.csv"
OUT_DIR = BASE_DIR


def specificity_score(y_true, y_pred):
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def main():
    dfs = []
    for fold in FOLDS:
        path = os.path.join(BASE_DIR, f"fold_{fold}", PRED_FILE)
        df = pd.read_csv(path)
        df["fold"] = fold
        df = df.rename(columns={"pred_label": "pred", "pred_prob": "prob"})
        df["label"] = (df["label"] == "Diseased").astype(int)
        df["pred"] = (df["pred"] == "Diseased").astype(int)
        dfs.append(df)

    all_preds = pd.concat(dfs, ignore_index=True)
    all_preds = all_preds.sort_values("patient_id").reset_index(drop=True)

    # Sanity checks
    assert all_preds["patient_id"].nunique() == 40, (
        f"Expected 40 unique patient_ids, got {all_preds['patient_id'].nunique()}"
    )
    for fold in FOLDS:
        fold_ids = all_preds.loc[all_preds["fold"] == fold, "patient_id"]
        assert fold_ids.nunique() == len(fold_ids), (
            f"Fold {fold} has duplicate patient_ids in its test set"
        )

    # Overall metrics across all 40 samples
    y_true = all_preds["label"].values
    y_pred = all_preds["pred"].values
    y_prob = all_preds["prob"].values
    overall = compute_metrics(y_true, y_pred, y_prob)

    # Per-fold metrics
    fold_metrics = {}
    for fold in FOLDS:
        subset = all_preds[all_preds["fold"] == fold].sort_values("patient_id")
        fold_metrics[fold] = compute_metrics(
            subset["label"].values,
            subset["pred"].values,
            subset["prob"].values,
        )

    # Mean ± std across folds
    metric_names = list(next(iter(fold_metrics.values())).keys())
    fold_summary = {}
    for metric in metric_names:
        values = [fold_metrics[f][metric] for f in FOLDS]
        fold_summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "per_fold": {f: float(fold_metrics[f][metric]) for f in FOLDS},
        }

    results = {
        "overall": {k: float(v) for k, v in overall.items()},
        "fold_summary": fold_summary,
    }

    os.makedirs(OUT_DIR, exist_ok=True)

    json_path = os.path.join(OUT_DIR, "aggregate_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Build flat CSV rows: one row = overall, one row per fold
    rows = []
    overall_row = {"split": "overall", **{k: round(v, 4) for k, v in overall.items()}}
    rows.append(overall_row)
    for fold in FOLDS:
        row = {"split": f"fold_{fold}", **{k: round(v, 4) for k, v in fold_metrics[fold].items()}}
        rows.append(row)
    mean_row = {
        "split": "mean",
        **{m: round(fold_summary[m]["mean"], 4) for m in metric_names},
    }
    std_row = {
        "split": "std",
        **{m: round(fold_summary[m]["std"], 4) for m in metric_names},
    }
    rows.extend([mean_row, std_row])

    csv_path = os.path.join(OUT_DIR, "aggregate_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # Print summary
    print("=== Overall (40 test samples) ===")
    for metric, value in overall.items():
        print(f"  {metric:20s}: {value:.4f}")

    print("\n=== Per-fold metrics ===")
    header = f"{'metric':20s}" + "".join(f"  fold_{f}" for f in FOLDS) + "  mean±std"
    print(header)
    for metric in metric_names:
        vals = [fold_metrics[f][metric] for f in FOLDS]
        row_str = f"{metric:20s}" + "".join(f"  {v:.4f}" for v in vals)
        row_str += f"  {np.mean(vals):.4f}±{np.std(vals, ddof=1):.4f}"
        print(row_str)

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
