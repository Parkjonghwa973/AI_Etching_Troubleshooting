"""
train_models.py
===============

3개의 머신러닝 모델을 학습하고 저장하는 모듈입니다.

- Model A : Random Forest
- Model B : XGBoost (기본 하이퍼파라미터)
- Model C : Hyperparameter Optimized XGBoost (Optuna 결과 사용)

데이터 분할
----------
- 훈련 80% / 테스트 20%, **Stratified split** (random_state 고정)
- 모델 최적화(하이퍼파라미터 탐색)에는 훈련 데이터 내부의 교차검증만 사용하며,
  테스트 데이터는 최종 평가 전에는 사용하지 않습니다.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover
    warnings.warn(f"xgboost 를 불러오지 못했습니다: {exc}")
    XGBClassifier = None  # type: ignore

from src.data_generator import (
    INPUT_COLUMNS,
    LABEL_COLUMN,
    N_CLASSES,
    RANDOM_STATE,
)

TEST_SIZE = 0.2

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

XGB_BASE_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.10,
    subsample=0.90,
    colsample_bytree=0.90,
    min_child_weight=1,
    reg_alpha=0.0,
    reg_lambda=1.0,
    objective="multi:softprob",
    num_class=N_CLASSES,
    eval_metric="mlogloss",
    tree_method="hist",
    n_jobs=4,
    random_state=RANDOM_STATE,
)

FEATURE_COLUMNS: List[str] = list(INPUT_COLUMNS)


# ---------------------------------------------------------------------------
# 데이터 준비
# ---------------------------------------------------------------------------
def load_training_frame(path: Optional[str | Path] = None) -> pd.DataFrame:
    """데이터셋 로드 (없으면 생성)."""
    from src.data_generator import default_data_path, generate_dataset, load_dataset

    path = Path(path) if path is not None else default_data_path()
    if not path.exists():
        print("    [INFO] 데이터 파일이 없어 새로 생성합니다.")
        df = generate_dataset()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return df
    return load_dataset(path)


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """X(입력 6변수), y(불량 라벨)를 분리합니다."""
    X = df[FEATURE_COLUMNS].astype(float)
    y = df[LABEL_COLUMN].astype(int).to_numpy()
    return X, y


def split_stratified(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Stratified 80:20 분할. 테스트셋은 최종 평가 전 사용 금지."""
    X, y = prepare_xy(df)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y, shuffle=True,
    )


# ---------------------------------------------------------------------------
# 모델 빌드 / 학습
# ---------------------------------------------------------------------------
def build_random_forest(params: Optional[Dict] = None) -> RandomForestClassifier:
    p = dict(RF_PARAMS)
    p.update(params or {})
    return RandomForestClassifier(**p)


def build_xgboost(params: Optional[Dict] = None) -> "XGBClassifier":
    if XGBClassifier is None:
        raise ImportError("xgboost 가 설치되어 있지 않습니다. pip install xgboost")
    p = dict(XGB_BASE_PARAMS)
    p.update(params or {})
    return XGBClassifier(**p)


def train_baseline_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> Dict[str, object]:
    """Model A(Random Forest) 와 Model B(XGBoost baseline) 를 학습합니다."""
    models: Dict[str, object] = {}

    print("    [학습] Model A: Random Forest ...", end=" ")
    rf_params = {"random_state": random_state}
    rf = build_random_forest(rf_params)
    rf.fit(X_train, y_train)
    models["random_forest"] = rf
    print("완료")

    print("    [학습] Model B: XGBoost (baseline) ...", end=" ")
    xgb_params = {"random_state": random_state}
    xgb = build_xgboost(xgb_params)
    xgb.fit(X_train, y_train)
    models["xgboost"] = xgb
    print("완료")

    return models


def train_optimized_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    best_params: Optional[Dict] = None,
    random_state: int = RANDOM_STATE,
) -> object:
    """Model C: Optuna 로 찾은 최적 파라미터로 XGBoost 를 학습합니다."""
    params = dict(best_params or {})
    params.update({"random_state": random_state})
    print("    [학습] Model C: XGBoost (optimized) ...", end=" ")
    model = build_xgboost(params)
    model.fit(X_train, y_train)
    print("완료")
    return model


# ---------------------------------------------------------------------------
# 저장 / 로드
# ---------------------------------------------------------------------------
def default_models_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "models"


def save_models(models: Dict[str, object], models_dir: Optional[str | Path] = None) -> Dict[str, Path]:
    """모델을 joblib 로 저장합니다.

    저장 파일: random_forest.pkl / xgboost.pkl / optimized_xgboost.pkl
    """
    models_dir = Path(models_dir) if models_dir is not None else default_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Path] = {}
    for name, model in models.items():
        path = models_dir / f"{name}.pkl"
        joblib.dump(model, path)
        saved[name] = path
        print(f"    [모델 저장] {path}")
    return saved


def load_models(models_dir: Optional[str | Path] = None) -> Dict[str, object]:
    """저장된 모델을 로드합니다 (없는 파일은 건너뜀)."""
    models_dir = Path(models_dir) if models_dir is not None else default_models_dir()
    models: Dict[str, object] = {}
    for name in ["random_forest", "xgboost", "optimized_xgboost"]:
        path = models_dir / f"{name}.pkl"
        if path.exists():
            models[name] = joblib.load(path)
    return models


if __name__ == "__main__":
    df = load_training_frame()
    X_train, X_test, y_train, y_test = split_stratified(df)
    print(f"훈련: {len(X_train)}건 / 테스트: {len(X_test)}건")
    ms = train_baseline_models(X_train, y_train)
    save_models(ms)
