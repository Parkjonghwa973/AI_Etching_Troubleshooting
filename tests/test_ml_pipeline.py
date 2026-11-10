"""
test_ml_pipeline.py
===================

머신러닝 학습 / 공정 조건 최적화 모듈에 대한 pytest 테스트입니다.

검증 항목
--------
- 모델 학습이 정상적으로 수행되는가? (Random Forest / XGBoost)
- 예측 확률이 유효한가? (6클래스, 합 = 1)
- 공정 최적화 결과가 지정된 변수 범위를 벗어나지 않는가?
- 최적화가 Normal 확률을 낮추지(해치지) 않는가?
- Optuna 최적화 결과 구조가 정상적인가? (간단 트라이얼)
- SHAP 형상 정규화 함수가 다양한 입력 형태를 처리하는가?
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_generator import (  # noqa: E402
    INPUT_COLUMNS,
    N_CLASSES,
    PROCESS_BOUNDS,
    generate_dataset,
)
from src.process_optimizer import (  # noqa: E402
    bounds_array,
    optimize_process,
    result_to_dataframe,
    train_distribution_stats,
)
from src.train_models import (  # noqa: E402
    FEATURE_COLUMNS,
    build_random_forest,
    build_xgboost,
    split_stratified,
)

N_TRAIN = 800


@pytest.fixture(scope="module")
def data():
    return generate_dataset(n_samples=N_TRAIN, random_state=42)


@pytest.fixture(scope="module")
def split(data):
    return split_stratified(data, test_size=0.25, random_state=42)


@pytest.fixture(scope="module")
def xgb_model(split):
    """빠른 학습을 위해 소형 XGBoost 사용."""
    X_train, _, y_train, _ = split
    model = build_xgboost({"n_estimators": 60, "max_depth": 4, "learning_rate": 0.2})
    model.fit(X_train, y_train)
    return model


@pytest.fixture(scope="module")
def rf_model(split):
    X_train, _, y_train, _ = split
    model = build_random_forest({"n_estimators": 40, "random_state": 42})
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# 1) 모델 학습
# ---------------------------------------------------------------------------
def test_random_forest_trains(rf_model, split):
    _, X_test, _, y_test = split
    preds = rf_model.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(np.unique(preds)).issubset(set(range(N_CLASSES)))
    assert rf_model.score(X_test, y_test) > 0.3, "무작위 추측보다는 성능이 있어야 합니다"


def test_xgboost_trains(xgb_model, split):
    _, X_test, _, y_test = split
    preds = xgb_model.predict(X_test)
    assert len(preds) == len(y_test)
    assert xgb_model.score(X_test, y_test) > 0.3


def test_predict_proba_valid(xgb_model, split):
    _, X_test, _, _ = split
    probs = xgb_model.predict_proba(X_test.head(20).to_numpy())
    assert probs.shape == (20, N_CLASSES)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert (probs >= 0).all()


def test_stratified_split_preserves_distribution(data):
    X_train, X_test, y_train, y_test = split_stratified(data, test_size=0.2, random_state=42)
    train_ratio = pd.Series(y_train).value_counts(normalize=True).sort_index()
    test_ratio = pd.Series(y_test).value_counts(normalize=True).sort_index()
    assert np.allclose(train_ratio.values, test_ratio.values, atol=0.03)
    assert abs(len(X_train) / len(data) - 0.8) < 0.01


# ---------------------------------------------------------------------------
# 2) 공정 조건 최적화
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["random", "de"])
def test_optimization_stays_within_bounds(xgb_model, data, method):
    stats = train_distribution_stats(data[FEATURE_COLUMNS])
    current = {c: float(np.mean(PROCESS_BOUNDS[c])) for c in INPUT_COLUMNS}
    result = optimize_process(xgb_model, current, method=method, n_iter=400,
                              train_stats=stats, random_state=42)

    for col, (lo, hi) in PROCESS_BOUNDS.items():
        val = result["recommended"][col]
        assert lo - 1e-6 <= val <= hi + 1e-6, f"{col}={val} 가 범위 {lo}~{hi} 를 벗어남"
    assert result["within_bounds"] is True


def test_optimization_output_structure(xgb_model, data):
    stats = train_distribution_stats(data[FEATURE_COLUMNS])
    current = {c: float(np.mean(PROCESS_BOUNDS[c])) for c in INPUT_COLUMNS}
    result = optimize_process(xgb_model, current, method="random", n_iter=800,
                              train_stats=stats, random_state=42)

    for key in ["original", "recommended", "defect_before", "defect_after",
                "probabilities_before", "probabilities_after",
                "normal_prob_before", "normal_prob_after", "parameter_changes",
                "warnings"]:
        assert key in result, f"결과 dict 에 {key} 가 없습니다"

    assert 0.0 <= result["normal_prob_before"] <= 1.0
    assert 0.0 <= result["normal_prob_after"] <= 1.0
    assert result["defect_before"] in ["Normal", "Undercut", "Bowing", "Etch Stop", "Footing", "Punch-through"]
    assert isinstance(result["warnings"], list) and len(result["warnings"]) > 0


def test_optimization_does_not_hurt_normal_probability(xgb_model, data):
    """최적화 후 Normal 확률은 최소한 현재 조건보다 낮지 않아야 합니다."""
    stats = train_distribution_stats(data[FEATURE_COLUMNS])
    current = {c: float(np.mean(PROCESS_BOUNDS[c])) for c in INPUT_COLUMNS}
    result = optimize_process(xgb_model, current, method="random", n_iter=1500,
                              train_stats=stats, random_state=42)
    assert result["normal_prob_after"] >= result["normal_prob_before"] - 1e-6


def test_recommendation_dataframe(xgb_model, data):
    stats = train_distribution_stats(data[FEATURE_COLUMNS])
    current = {c: float(np.mean(PROCESS_BOUNDS[c])) for c in INPUT_COLUMNS}
    result = optimize_process(xgb_model, current, method="random", n_iter=300,
                              train_stats=stats, random_state=42)
    table = result_to_dataframe(result)
    assert list(table.columns) == ["Parameter", "Original", "Recommended", "Change", "Change (%)"]
    assert len(table) == len(INPUT_COLUMNS)


def test_out_of_distribution_warning():
    """학습 분포 밖의 조건에는 경고가 포함되어야 합니다."""
    from src.process_optimizer import ood_penalty

    fake_stats = {
        "mean": np.array([0.5, 125, 52.5, 175, 900, 60.0]),
        "std": np.array([0.1, 10, 5, 20, 100, 5.0]),
        "min": np.array([0.1, 100, 40, 150, 800, 50.0]),
        "max": np.array([0.9, 150, 65, 200, 1000, 70.0]),
    }
    outside = np.array([0.99, 500.0, 200.0, 900.0, 5000.0, 200.0])
    penalty, warns = ood_penalty(outside, fake_stats)
    assert penalty > 0
    assert len(warns) > 0

    inside = np.array([0.5, 125, 52.5, 175, 900, 60.0])
    penalty_in, warns_in = ood_penalty(inside, fake_stats)
    assert penalty_in == 0.0
    assert warns_in == []


# ---------------------------------------------------------------------------
# 3) Optuna / SHAP 유틸
# ---------------------------------------------------------------------------
def test_optuna_smoke(split):
    optuna = pytest.importorskip("optuna")
    from src.optimize_model import run_optimization

    X_train, _, y_train, _ = split
    result = run_optimization(X_train, y_train, n_trials=3, n_splits=2,
                              random_state=42, show_progress=False)
    assert "best_params" in result
    assert 0.0 <= result["best_score"] <= 1.0
    assert len(result["trials"]) == 3
    assert isinstance(result["improved"], bool)


def test_shap_normalization_shapes():
    from src.explain_model import normalize_shap_output

    n, f, c = 10, 6, N_CLASSES
    # (1) list form (구버전)
    lst = [np.random.RandomState(i).rand(n, f) for i in range(c)]
    assert normalize_shap_output(lst, f).shape == (n, f, c)
    # (2) 3D form (신버전)
    arr = np.random.RandomState(0).rand(n, f, c)
    assert normalize_shap_output(arr, f).shape == (n, f, c)
    # (3) 2D form (단일 클래스)
    arr2 = np.random.RandomState(1).rand(n, f)
    assert normalize_shap_output(arr2, f).shape == (n, f, c)
