"""Train, evaluate, and save the loan repayment model.

Usage (from the project root):
    python train_model.py
"""

from __future__ import annotations

import logging

from utils.data_loader import DataError, load_dataset, profile_dataset
from utils.model_utils import save_artifacts, train_model
from utils.config import MODEL_PATH, REPORT_PATH


def main() -> int:
    """Run the full training workflow and print a summary."""
    logging.basicConfig(level=logging.WARNING)
    try:
        df = load_dataset()
    except DataError as exc:
        print(f"Error: {exc}")
        return 1

    profile = profile_dataset(df)
    print(f"Rows: {profile['rows']:,}  Columns: {profile['columns']}")
    print(f"Missing values: { {k: v for k, v in profile['missing'].items() if v} }")
    print(f"Duplicate rows: {profile['duplicate_rows']}")
    print(f"Repayment rate: {profile['positive_rate']:.1%}")
    print(f"Rows with open_acc > total_acc: {profile['open_exceeds_total_rows']:,}")

    print("\nTraining candidates (5-fold CV on the training split)...")
    pipeline, report = train_model(df)

    for name, scores in report["cv_results"].items():
        print(f"  {name:<24} CV ROC-AUC {scores['cv_auc_mean']:.3f} ± {scores['cv_auc_std']:.3f}")
    print(f"\nSelected: {report['selected_model']}")
    print(report["selection_rationale"])

    metrics = report["test_metrics"]
    print("\nHeld-out test results")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "balanced_accuracy"):
        print(f"  {key:<18} {metrics[key]:.3f}")
    print(f"  {'majority baseline':<18} {report['baseline_accuracy']:.3f} (accuracy)")
    print(f"  confusion matrix   {metrics['confusion_matrix']}")

    if report["signal_warning"]:
        print(
            "\nWARNING: test ROC-AUC is below 0.60. The features carry little "
            "predictive signal, so predictions should not be relied on."
        )

    save_artifacts(pipeline, report)
    print(f"\nSaved model  -> {MODEL_PATH}")
    print(f"Saved report -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
