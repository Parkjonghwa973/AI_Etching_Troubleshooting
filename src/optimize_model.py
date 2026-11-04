"""
optimize_model.py
=================

Optuna 를 사용해 XGBoost 하이퍼파라미터를 최적화하는 모듈입니다.

- 최적화 목표: **교차검증 Macro F1-score 최대화**
- 최적화는 항상 **훈련 데이터 내부의 교차검증(Stratified K-Fold)** 으로만 수행하며
  테스트 데이터는 사용하지 않습니다.
- 최적화 이력(trial 마다 점수, 파라미터)을 CSV 로 저장하고 그래프로 시각화합니다.
- 최적화 전/후 성능을 비교하며, 실제 향상되지 않았다면 "향상됨"으로 표시하지 않습니다.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data_generator import N_CLASSES, RANDOM_STATE
from src.train_models import build_xgboost
from src.visualization import plot_optimization_history, save_figure

import matplotlib.pyplot as plt

DEFAULT_N_TRIALS = 30
DEFAULT_N_SPLITS = 3
OPTUNA_LOG_LEVEL = 30  # warning 이상만 출력


# ---------------------------------------------------------------------------
# Optuna import (없으면 안내)
# ---------------------------------------------------------------------------
try:
    import optuna
except Exception as exc:  # pragma: no cover
    warnings.warn(f"optuna 를 불러오지 못했습니다: {exc}")
    optuna = None  # type: ignore


# ---------------------------------------------------------------------------
# 탐색 공간 / 목적 함수
# ---------------------------------------------------------------------------
def suggest_params(trial: "optuna.Trial") -> Dict:
    """최적화 대상 하이퍼파라미터 8개를 제안합니다."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 900, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
    }


def create_objective(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = RANDOM_STATE,
):
    """목적 함수 팩토리 (Stratified K-Fold CV Macro F1)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial: "optuna.Trial") -> float:
        params = suggest_params(trial)
        model = build_xgboost(params)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring="f1_macro", n_jobs=1,
        )
        return float(np.mean(scores))

    return objective


def baseline_cv_score(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> float:
    """최적화 전(기본 파라미터) XGBoost 의 교차검증 Macro F1 을 계산합니다."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = build_xgboost()
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=1)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 최적화 실행
# ---------------------------------------------------------------------------
def run_optimization(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_trials: int = DEFAULT_N_TRIALS,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = RANDOM_STATE,
    show_progress: bool = True,
) -> Dict:
    """Optuna 최적화를 실행하고 결과 dict 를 반환합니다.

    Returns
    -------
    dict : {
        "best_params", "best_score", "baseline_score", "improved",
        "improvement", "trials", "study"
    }
    """
    if optuna is None:
        raise ImportError("optuna 가 설치되어 있지 않습니다. pip install optuna")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\n  [4단계] Optuna 하이퍼파라미터 최적화 시작 (trials={n_trials}, "
          f"{n_splits}-Fold Stratified CV)")
    baseline = baseline_cv_score(X_train, y_train, n_splits, random_state)
    print(f"    최적화 전 교차검증 Macro F1 (기본 XGBoost): {baseline:.4f}")

    sampler = optuna.samplers.TPESampler(seed=random_state, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name="xgb_etching_macro_f1")
    study.optimize(create_objective(X_train, y_train, n_splits, random_state),
                   n_trials=n_trials, show_progress_bar=show_progress)

    best_score = float(study.best_value)
    best_params = dict(study.best_params)

    # 성능 비교: 실제로 향상되지 않았다면 improved=False
    improvement = best_score - baseline
    improved = improvement > 1e-6

    trials = [
        {
            "trial": t.number + 1,
            "macro_f1": t.value,
            **{k: t.params[k] for k in sorted(t.params)},
        }
        for t in study.trials
    ]

    print(f"    최적화 전 CV Macro F1 : {baseline:.4f}")
    print(f"    최적화 후 CV Macro F1 : {best_score:.4f}  (Δ = {improvement:+.4f})")
    if improved:
        print("    [결과] 성능이 향상되었습니다.")
    else:
        print("    [결과] 성능이 유의미하게 향상되지 않았습니다. "
              "(기본 파라미터를 유지하는 것이 안전할 수 있습니다)")

    return {
        "best_params": best_params,
        "best_score": best_score,
        "baseline_score": baseline,
        "improved": bool(improved),
        "improvement": float(improvement),
        "trials": trials,
        "study": study,
        "n_trials": n_trials,
        "n_splits": n_splits,
    }


# ---------------------------------------------------------------------------
# 결과 저장
# ---------------------------------------------------------------------------
def default_results_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "results"


def save_optimization_results(
    result: Dict,
    results_dir: Optional[str | Path] = None,
) -> Dict[str, Path]:
    """최적화 이력 CSV, 요약 JSON, 이력 그래프 PNG 를 저장합니다."""
    results_dir = Path(results_dir) if results_dir is not None else default_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Path] = {}

    # 1) trial 이력 CSV
    trials_df = pd.DataFrame(result["trials"])
    hist_csv = results_dir / "optimization_history.csv"
    trials_df.to_csv(hist_csv, index=False, encoding="utf-8-sig")
    saved["history_csv"] = hist_csv
    print(f"    [저장] {hist_csv}")

    # 2) 요약 JSON (최적 파라미터 / 전후 성능 / 향상 여부)
    summary = {
        "n_trials": result["n_trials"],
        "n_splits": result["n_splits"],
        "baseline_cv_macro_f1": round(result["baseline_score"], 5),
        "best_cv_macro_f1": round(result["best_score"], 5),
        "improvement": round(result["improvement"], 5),
        "improved": result["improved"],
        "best_params": result["best_params"],
    }
    json_path = results_dir / "optimization_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    saved["summary_json"] = json_path
    print(f"    [저장] {json_path}")

    # 3) 이력 그래프
    fig_path = results_dir / "optimization_history.png"
    plot_optimization_history(
        [t["macro_f1"] for t in result["trials"]],
        baseline_score=result["baseline_score"],
        path=fig_path,
    )
    saved["history_png"] = fig_path
    return saved


def compare_models_before_after(
    baseline_cv: float,
    optimized_cv: float,
) -> pd.DataFrame:
    """최적화 전/후 성능 비교표를 만듭니다 (향상되지 않으면 명시)."""
    delta = optimized_cv - baseline_cv
    status = "Improved" if delta > 1e-6 else "Not improved"
    return pd.DataFrame([
        {
            "model": "XGBoost (baseline)",
            "cv_macro_f1": round(baseline_cv, 4),
            "delta_vs_baseline": 0.0,
            "status": "reference",
        },
        {
            "model": "XGBoost (Optuna optimized)",
            "cv_macro_f1": round(optimized_cv, 4),
            "delta_vs_baseline": round(delta, 4),
            "status": status,
        },
    ])


if __name__ == "__main__":
    from src.train_models import load_training_frame, split_stratified

    df = load_training_frame()
    X_train, X_test, y_train, y_test = split_stratified(df)
    res = run_optimization(X_train, y_train, n_trials=10)
    save_optimization_results(res)
    print(compare_models_before_after(res["baseline_score"], res["best_score"]))
