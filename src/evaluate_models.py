"""
evaluate_models.py
==================

세 모델(Random Forest / XGBoost / Optimized XGBoost)을 **동일한 테스트 데이터**로
평가하고, 분포 변화(distribution shift) 데이터에 대해서도 별도 평가하는 모듈입니다.

계산 지표
--------
- Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1
- Confusion Matrix
- Class별 Precision / Recall / F1

결과는 CSV(모델 비교표)와 PNG(그래프 5종)로 저장됩니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_generator import DEFECT_NAMES, N_CLASSES
from src.visualization import (
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_macro_f1_comparison,
    plot_model_metric_comparison,
    plot_per_class_recall,
)

MODEL_DISPLAY = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost (baseline)",
    "optimized_xgboost": "XGBoost (Optuna optimized)",
}


def default_results_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# 지표 계산
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """요구 지표 전체를 계산합니다."""
    labels = list(range(N_CLASSES))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro",
                                                  labels=labels, zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro",
                                           labels=labels, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted",
                                      labels=labels, zero_division=0)),
    }


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Class별 Precision / Recall / F1 을 계산합니다."""
    labels = list(range(N_CLASSES))
    p = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return pd.DataFrame({
        "class": labels,
        "class_name": [DEFECT_NAMES[i] for i in labels],
        "precision": p,
        "recall": r,
        "f1": f,
        "support": [int((np.asarray(y_true) == i).sum()) for i in labels],
    })


def evaluate_model(model, X: pd.DataFrame, y: np.ndarray) -> Dict:
    """단일 모델 평가 (지표 + 혼동행렬 + class별 지표)."""
    y_pred = model.predict(X)
    y_pred = np.asarray(y_pred).astype(int).ravel()
    metrics = compute_metrics(y, y_pred)
    cm = confusion_matrix(y, y_pred, labels=list(range(N_CLASSES)))
    pc = per_class_metrics(y, y_pred)
    return {"metrics": metrics, "confusion_matrix": cm, "per_class": pc, "y_pred": y_pred}


def evaluate_all_models(
    models: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    X_shift: Optional[pd.DataFrame] = None,
    y_shift: Optional[np.ndarray] = None,
) -> Dict:
    """세 모델을 동일 테스트 데이터로 평가하고, 분포 변화 데이터도 별도 평가합니다."""
    results: Dict = {"test": {}, "shift": {}}
    rows: List[Dict] = []

    for name, model in models.items():
        ev = evaluate_model(model, X_test, y_test)
        results["test"][name] = ev
        rows.append({"model": name, "dataset": "test", **ev["metrics"]})
        print(f"    [평가:test ] {MODEL_DISPLAY.get(name, name):32s} "
              f"acc={ev['metrics']['accuracy']:.4f} macroF1={ev['metrics']['macro_f1']:.4f}")

    if X_shift is not None and y_shift is not None:
        for name, model in models.items():
            ev = evaluate_model(model, X_shift, y_shift)
            results["shift"][name] = ev
            rows.append({"model": name, "dataset": "distribution_shift", **ev["metrics"]})
            print(f"    [평가:shift] {MODEL_DISPLAY.get(name, name):32s} "
                  f"acc={ev['metrics']['accuracy']:.4f} macroF1={ev['metrics']['macro_f1']:.4f}")

    results["comparison"] = pd.DataFrame(rows)
    return results


# ---------------------------------------------------------------------------
# 결과 저장
# ---------------------------------------------------------------------------
def save_evaluation_results(
    results: Dict,
    results_dir: Optional[str | Path] = None,
) -> Dict[str, Path]:
    """비교표 CSV, class별 지표 CSV, 그래프 5종을 저장합니다."""
    results_dir = Path(results_dir) if results_dir is not None else default_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Path] = {}

    comparison: pd.DataFrame = results["comparison"]

    # 1) 모델 비교표 (테스트셋)
    cmp_test = comparison[comparison["dataset"] == "test"].drop(columns=["dataset"])
    p1 = results_dir / "model_comparison.csv"
    cmp_test.to_csv(p1, index=False, encoding="utf-8-sig")
    saved["model_comparison_csv"] = p1
    print(f"    [저장] {p1}")

    # 2) 분포 변화 평가표
    if (comparison["dataset"] == "distribution_shift").any():
        cmp_shift = comparison[comparison["dataset"] == "distribution_shift"].drop(columns=["dataset"])
        p2 = results_dir / "model_comparison_distribution_shift.csv"
        cmp_shift.to_csv(p2, index=False, encoding="utf-8-sig")
        saved["shift_csv"] = p2
        print(f"    [저장] {p2}")

    # 3) Class별 지표 CSV
    pc_rows = []
    for ds in ["test", "shift"]:
        for name, ev in results.get(ds, {}).items():
            tmp = ev["per_class"].copy()
            tmp.insert(0, "model", name)
            tmp.insert(1, "dataset", "test" if ds == "test" else "distribution_shift")
            pc_rows.append(tmp)
    if pc_rows:
        pc_all = pd.concat(pc_rows, ignore_index=True)
        p3 = results_dir / "per_class_metrics.csv"
        pc_all.to_csv(p3, index=False, encoding="utf-8-sig")
        saved["per_class_csv"] = p3
        results["per_class_all"] = pc_all
        print(f"    [저장] {p3}")

    # 4) 그래프: Accuracy / Macro F1 / 혼동행렬 / Class별 Recall / (이력은 optimize 모듈)
    saved["accuracy_png"] = plot_accuracy_comparison(comparison, results_dir / "accuracy_comparison.png")
    saved["macro_f1_png"] = plot_macro_f1_comparison(comparison, results_dir / "macro_f1_comparison.png")
    saved["combined_png"] = plot_model_metric_comparison(
        comparison, ("accuracy", "macro_f1"), "test",
        results_dir / "model_performance_comparison.png")

    # 혼동행렬: 최적화 모델 기준
    main_model = "optimized_xgboost" if "optimized_xgboost" in results["test"] \
        else list(results["test"].keys())[-1]
    cm = results["test"][main_model]["confusion_matrix"]
    saved["confusion_png"] = plot_confusion_matrix(
        cm, main_model, results_dir / "confusion_matrix.png", dataset="test")

    # 분포 변화 혼동행렬도 저장
    if "shift" in results and results["shift"]:
        cm_shift = results["shift"][main_model]["confusion_matrix"]
        saved["confusion_shift_png"] = plot_confusion_matrix(
            cm_shift, main_model,
            results_dir / "confusion_matrix_distribution_shift.png",
            dataset="shift")

    # Class별 Recall 비교
    if "per_class_all" in results:
        recall_df = results["per_class_all"][["model", "dataset", "class", "recall"]]
        saved["recall_png"] = plot_per_class_recall(
            recall_df, results_dir / "per_class_recall.png")

    return saved


if __name__ == "__main__":
    from src.train_models import (
        load_training_frame,
        load_models,
        split_stratified,
    )

    df = load_training_frame()
    X_train, X_test, y_train, y_test = split_stratified(df)
    models = load_models()
    if not models:
        raise SystemExit("모델 파일이 없습니다. 먼저 run_pipeline.py 를 실행하세요.")
    res = evaluate_all_models(models, X_test, y_test)
    save_evaluation_results(res)
