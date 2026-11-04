"""
automl_engine.py
================

AI 기반 자동 머신러닝 최적화(AutoML) 엔진입니다.

기존 `train_models.py` / `optimize_model.py` 를 유지한 채, 추가로 **4개 모델**을
자동 학습·비교하고 Optuna 로 각 모델의 하이퍼파라미터를 자동 최적화합니다.

- 후보 모델: Random Forest / XGBoost / Extra Trees / HistGradientBoosting
- 분석 모드: fast(모델별 5회) / normal(15회) / precise(30회)
- 목표: 교차검증 Macro F1-score 최대화 (불량 유형별 Recall 도 함께 기록)
- 진행률 표시(progress callback) 및 모델별 학습/최적화 시간 기록

과적합 방지 설계 (중요)
----------------------
1) **3-Stage 분할**: 전체 → (학습 60% + 검증 20% + 테스트 20%)로 stratified 분리
2) 하이퍼파라미터 탐색과 모델 선정은 **학습/검증 데이터 내부**에서만 수행
3) **최종 테스트셋은 모델 선택·튜닝에 사용하지 않음** (강건성 평가용)
4) 분포 변화 데이터도 모델 선정에 사용하지 않고 **최종 강건성 평가**에 사용
5) Stratified K-Fold CV (학습 데이터 내부)
6) **Early Stopping**: XGBoost 는 학습 서브셋 내부 홀드아웃으로, HistGradientBoosting 은
   sklearn 내부 validation_fraction 으로 조기 종료 (CV 평가셋은 조기 종료에 사용하지 않음
   → 데이터 누출 방지)
7) 하이퍼파라미터 탐색 범위 제한 및 모델 복잡도 제어(트리 수/깊이 상한)
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data_generator import INPUT_COLUMNS, LABEL_COLUMN, N_CLASSES, RANDOM_STATE
from src.evaluate_models import compute_metrics, per_class_metrics
from src.train_models import build_random_forest, build_xgboost

try:
    import optuna
except Exception as exc:  # pragma: no cover
    warnings.warn(f"optuna 를 불러오지 못했습니다: {exc}")
    optuna = None  # type: ignore

optuna.logging.set_verbosity(optuna.logging.WARNING) if optuna is not None else None

# 분석 모드: 모델별 Optuna 최적화 횟수
MODE_PRESETS: Dict[str, int] = {
    "fast": 5,       # 빠른 분석
    "normal": 15,    # 일반 분석
    "precise": 30,   # 정밀 분석
}
DEFAULT_MODE = "fast"
DEFAULT_N_SPLITS = 3
EARLY_STOPPING_ROUNDS = 15
EARLY_STOP_FRACTION = 0.15

ProgressCallback = Callable[[str, int, int, str], None]


# ---------------------------------------------------------------------------
# 데이터 분할 (데이터 누출 방지)
# ---------------------------------------------------------------------------
def split_train_val_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_fraction_of_train: float = 0.25,
    random_state: int = RANDOM_STATE,
    label_column: str = LABEL_COLUMN,
) -> Dict[str, Any]:
    """학습 60% / 검증 20% / 테스트 20% stratified 3-Stage 분할.

    - 탐색(Optuna CV)과 모델 선정은 train/val 내부에서만 수행
    - test 는 최종 1회 평가 전용 (모델 선택에 사용하지 않음)

    Returns
    -------
    dict : X_train, X_val, X_test, y_train, y_val, y_test,
           feature_names, sizes, ratio
    """
    feature_columns = [c for c in INPUT_COLUMNS if c in df.columns]
    X = df[feature_columns].astype(float)
    y = df[label_column].astype(int).to_numpy()
    idx = np.arange(len(df))

    X_trv, X_te, y_trv, y_te, i_trv, i_te = train_test_split(
        X, y, idx, test_size=test_size, random_state=random_state, stratify=y)

    X_tr, X_va, y_tr, y_va, i_tr, i_va = train_test_split(
        X_trv, y_trv, i_trv, test_size=val_fraction_of_train,
        random_state=random_state + 1, stratify=y_trv)

    # 누출 방지 검증: 인덱스 교집합이 없어야 함
    assert len(set(i_tr) & set(i_va)) == 0, "학습/검증 인덱스 중복(누출)"
    assert len(set(i_tr) & set(i_te)) == 0, "학습/테스트 인덱스 중복(누출)"
    assert len(set(i_va) & set(i_te)) == 0, "검증/테스트 인덱스 중복(누출)"

    return {
        "X_train": X_tr, "y_train": y_tr, "train_index": i_tr,
        "X_val": X_va, "y_val": y_va, "val_index": i_va,
        "X_test": X_te, "y_test": y_te, "test_index": i_te,
        "feature_names": feature_columns,
        "sizes": {"train": len(X_tr), "val": len(X_va), "test": len(X_te)},
        "ratio": {"train": len(X_tr) / len(df),
                  "val": len(X_va) / len(df),
                  "test": len(X_te) / len(df)},
        "n_total": len(df),
    }


def assert_no_leakage(splits: Dict[str, Any]) -> None:
    """데이터 누출 방지 사전 조건 검사 (테스트용 공개 함수)."""
    tr = set(splits["train_index"])
    va = set(splits["val_index"])
    te = set(splits["test_index"])
    if tr & va or tr & te or va & te:
        raise AssertionError("데이터 분할에 누출(중복 인덱스)이 있습니다.")
    if len(tr | va | te) != splits["n_total"]:
        raise AssertionError("분할된 인덱스 합계가 전체 데이터 수와 다릅니다.")


# ---------------------------------------------------------------------------
# 모델 카탈로그
# ---------------------------------------------------------------------------
def _build_rf(params: Dict, seed: int):
    p = dict(
        n_estimators=int(params.get("n_estimators", 300)),
        max_depth=int(params.get("max_depth", 16)),
        min_samples_leaf=int(params.get("min_samples_leaf", 2)),
        max_features=params.get("max_features", "sqrt"),
        n_jobs=-1,
        random_state=seed,
    )
    return build_random_forest(p)


def _build_et(params: Dict, seed: int):
    return ExtraTreesClassifier(
        n_estimators=int(params.get("n_estimators", 300)),
        max_depth=int(params.get("max_depth", 16)),
        min_samples_leaf=int(params.get("min_samples_leaf", 2)),
        max_features=params.get("max_features", "sqrt"),
        n_jobs=-1,
        random_state=seed,
    )


def _build_xgb(params: Dict, seed: int):
    p = dict(
        n_estimators=int(params.get("n_estimators", 300)),
        max_depth=int(params.get("max_depth", 5)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        subsample=float(params.get("subsample", 0.9)),
        colsample_bytree=float(params.get("colsample_bytree", 0.9)),
        min_child_weight=float(params.get("min_child_weight", 3)),
        reg_alpha=float(params.get("reg_alpha", 0.01)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        random_state=seed,
    )
    # xgboost >= 2.0: 조기 종료는 생성자 인자 + fit(eval_set=...) 로 지정
    if "early_stopping_rounds" in params:
        p["early_stopping_rounds"] = int(params["early_stopping_rounds"])
    return build_xgboost(p)


def _build_hgb(params: Dict, seed: int):
    return HistGradientBoostingClassifier(
        learning_rate=float(params.get("learning_rate", 0.1)),
        max_depth=int(params.get("max_depth", 8)),
        max_iter=int(params.get("max_iter", 200)),
        min_samples_leaf=int(params.get("min_samples_leaf", 15)),
        l2_regularization=float(params.get("l2_regularization", 0.0)),
        max_leaf_nodes=int(params.get("max_leaf_nodes", 31)),
        early_stopping=True,            # sklearn 내부 validation_fraction 홀드아웃
        validation_fraction=0.15,
        n_iter_no_change=10,
        random_state=seed,
    )


# (모델명, 표시명, 빌더, 탐색 공간, early-stopping 지원 여부)
def _space_rf(trial) -> Dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 6, 24),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5, 0.8]),
    }


def _space_et(trial) -> Dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 6, 24),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5, 0.8]),
    }


def _space_xgb(trial) -> Dict:
    # 탐색 범위 제한: 과적합 방지(얕은 트리, 강한 정규화, 낮은 학습률)
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
    }


def _space_hgb(trial) -> Dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "max_iter": trial.suggest_int("max_iter", 50, 300, step=10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 40),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 3.0),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
    }


MODEL_CATALOG: Dict[str, Dict] = {
    "random_forest": {
        "display": "Random Forest",
        "builder": _build_rf, "space": _space_rf, "early_stopping": False,
    },
    "xgboost": {
        "display": "XGBoost",
        "builder": _build_xgb, "space": _space_xgb, "early_stopping": True,
    },
    "extra_trees": {
        "display": "Extra Trees",
        "builder": _build_et, "space": _space_et, "early_stopping": False,
    },
    "hist_gradient_boosting": {
        "display": "HistGradientBoosting",
        "builder": _build_hgb, "space": _space_hgb, "early_stopping": True,
    },
}
MODEL_NAMES: List[str] = list(MODEL_CATALOG.keys())


def build_model(model_name: str, params: Optional[Dict] = None, seed: int = RANDOM_STATE):
    """모델명과 파라미터로 학습기(미학습)를 생성합니다."""
    if model_name not in MODEL_CATALOG:
        raise KeyError(f"알 수 없는 모델: {model_name}. 후보: {MODEL_NAMES}")
    return MODEL_CATALOG[model_name]["builder"](params or {}, seed)


# ---------------------------------------------------------------------------
# 학습 (Early Stopping 지원)
# ---------------------------------------------------------------------------
def _carve_early_stop(X: pd.DataFrame, y: np.ndarray, seed: int):
    """학습 서브셋 내부에서 조기 종료용 홀드아웃 분리 (CV 평가셋은 건드리지 않음)."""
    if len(X) < 40:
        return X, y, None, None
    X_fit, X_stop, y_fit, y_stop = train_test_split(
        X, y, test_size=EARLY_STOP_FRACTION, random_state=seed, stratify=y)
    return X_fit, y_fit, X_stop, y_stop


def fit_model(
    model_name: str,
    params: Dict,
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int = RANDOM_STATE,
    early_stop: bool = True,
):
    """모델을 학습합니다. 지원 모델은 내부 홀드아웃으로 Early Stopping 합니다."""
    info = MODEL_CATALOG[model_name]

    if model_name == "xgboost" and early_stop:
        X_fit, y_fit, X_stop, y_stop = _carve_early_stop(X, y, seed)
        p = dict(params)
        p["n_estimators"] = int(p.get("n_estimators", 300))
        p["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
        model = _build_xgb(p, seed)
        model.fit(X_fit, y_fit, eval_set=[(X_stop, y_stop)], verbose=False)
        best_iter = int(getattr(model, "best_iteration", 0) or 0)
        best_iter = max(best_iter, 20)
        # 최종 모델: 전체 학습 데이터로 재학습하되, 찾은 트리 수만큼만 사용
        p_final = dict(params)
        p_final["n_estimators"] = best_iter
        model = _build_xgb(p_final, seed)
        model.fit(X, y, verbose=False)
        model.best_iteration_ = best_iter
        return model

    model = info["builder"](params, seed)
    if model_name == "hist_gradient_boosting":
        # sklearn 내부 validation_fraction 홀드아웃으로 조기 종료
        model.early_stopping = True
        model.validation_fraction = EARLY_STOP_FRACTION
        model.n_iter_no_change = 10
    model.fit(X, y)
    return model


def model_complexity(model_name: str, model) -> int:
    """모델 복잡도 지표 (트리 수 × 깊이 형태의 정수)."""
    try:
        if model_name in ("random_forest", "extra_trees"):
            depth = int(np.mean([est.get_depth() for est in model.estimators_[:30]]))
            return int(model.n_estimators * max(depth, 1))
        if model_name == "xgboost":
            trees = int(getattr(model, "best_iteration_", 0) or model.n_estimators)
            return int(trees * int(model.max_depth))
        if model_name == "hist_gradient_boosting":
            return int(model.n_iter_ * int(model.max_depth or 8))
    except Exception:
        pass
    return 0


def measure_inference_ms(model, X: pd.DataFrame) -> float:
    """검증셋 추론 시간 (100건 기준 ms)."""
    X_use = X.head(min(500, len(X)))
    if len(X_use) == 0:
        return 0.0
    t = time.perf_counter()
    model.predict(X_use)
    return float((time.perf_counter() - t) * 1000.0 / max(len(X_use), 1) * 100.0)


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------
def evaluate_classifier(model, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
    """Accuracy / Macro F1 / Weighted F1 / 클래스별 Recall 등 평가."""
    y_pred = np.asarray(model.predict(X)).astype(int).ravel()
    metrics = compute_metrics(y, y_pred)
    pc = per_class_metrics(y, y_pred)
    return {
        "metrics": metrics,
        "per_class": pc,
        "per_class_recall": {int(r["class"]): float(r["recall"]) for _, r in pc.iterrows()},
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# Optuna 목적 함수 (학습 데이터 내부 Stratified K-Fold CV)
# ---------------------------------------------------------------------------
def make_objective(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = RANDOM_STATE,
):
    """학습 데이터 내부 CV 로만 평가하는 목적 함수 (Macro F1 최대화).

    - 폴드 분할은 stratified (shuffle + seed 고정 → 재현성)
    - Early Stopping 은 폴드 학습 서브셋 내부 홀드아웃에서만 수행 (평가셋 누출 방지)
    """
    from sklearn.metrics import f1_score

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def objective(trial: "optuna.Trial") -> float:
        params = MODEL_CATALOG[model_name]["space"](trial)
        fold_scores: List[float] = []
        fold_recalls: List[np.ndarray] = []
        for tr_i, va_i in cv.split(X_train, y_train):
            Xa = X_train.iloc[tr_i]
            ya = y_train[tr_i]
            Xb = X_train.iloc[va_i]
            yb = y_train[va_i]
            model = fit_model(model_name, params, Xa, ya, seed=seed,
                              early_stop=True)
            y_pred = np.asarray(model.predict(Xb)).astype(int).ravel()
            fold_scores.append(f1_score(yb, y_pred, average="macro", zero_division=0))
            recall = per_class_metrics(yb, y_pred)["recall"].to_numpy()
            fold_recalls.append(recall)
        trial.set_user_attr("macro_f1", float(np.mean(fold_scores)))
        trial.set_user_attr("per_class_recall", np.mean(fold_recalls, axis=0).tolist())
        return float(np.mean(fold_scores))

    return objective


def baseline_model_cv(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = RANDOM_STATE,
) -> float:
    """기본 파라미터의 학습 내부 CV Macro F1 (비교 기준선)."""
    from sklearn.metrics import f1_score

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    params: Dict = {}
    scores = []
    for tr_i, va_i in cv.split(X_train, y_train):
        model = fit_model(model_name, params, X_train.iloc[tr_i],
                          y_train[tr_i], seed=seed, early_stop=True)
        y_pred = model.predict(X_train.iloc[va_i])
        scores.append(f1_score(y_train[va_i], y_pred, average="macro", zero_division=0))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# AutoML 메인
# ---------------------------------------------------------------------------
def run_automl(
    splits: Dict[str, Any],
    mode: str = DEFAULT_MODE,
    n_trials: Optional[int] = None,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = RANDOM_STATE,
    model_names: Optional[Sequence[str]] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """4개 모델을 자동 학습·최적화·검증합니다.

    Parameters
    ----------
    splits : split_train_val_test() 결과
    mode : "fast" | "normal" | "precise"
    n_trials : 모델별 최적화 횟수 (mode 프리셋을 덮어씀)
    progress_cb : callback(stage, current, total, message)

    Returns
    -------
    dict : {
        "mode", "trials_per_model", "n_splits",
        "per_model": {name: {...}}, "history": {name: [trial records]},
        "val_leaderboard": DataFrame, "splits": splits
    }
    """
    if optuna is None:
        raise ImportError("optuna 가 필요합니다: pip install optuna")
    if mode not in MODE_PRESETS:
        raise ValueError(f"mode 는 {list(MODE_PRESETS)} 중 하나여야 합니다: {mode}")
    trials = int(n_trials) if n_trials else MODE_PRESETS[mode]
    names = list(model_names) if model_names else list(MODEL_NAMES)

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]

    def notify(stage: str, cur: int, total: int, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(stage, cur, total, msg)

    per_model: Dict[str, Dict] = {}
    history: Dict[str, List[Dict]] = {}
    total_steps = len(names) * (trials + 1)
    step = 0

    for name in names:
        display = MODEL_CATALOG[name]["display"]
        notify("automl", step, total_steps,
               f"[{display}] 기본 파라미터 CV 계산 중...")

        # 기준선 CV
        try:
            base_cv = baseline_model_cv(name, X_train, y_train, n_splits, seed)
        except Exception as exc:
            base_cv = float("nan")
            warnings.warn(f"{name} 기준선 CV 실패: {exc}")

        # Optuna 최적화 (학습 데이터 내부 CV)
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
        study = optuna.create_study(direction="maximize", sampler=sampler,
                                    study_name=f"automl_{name}")
        t_opt0 = time.perf_counter()

        def _make_callback(display_name: str):
            def _on_trial(study_, trial_):
                nonlocal step
                step += 1
                try:
                    best_txt = f"{study_.best_value:.4f}"
                except Exception:
                    best_txt = "n/a"
                notify("automl", step, total_steps,
                       f"[{display_name}] 최적화 {trial_.number + 1}/{trials} "
                       f"(현재 최고 CV Macro F1={best_txt})")
            return _on_trial

        study.optimize(make_objective(name, X_train, y_train, n_splits, seed),
                       n_trials=trials, show_progress_bar=False,
                       callbacks=[_make_callback(display)])
        opt_time = time.perf_counter() - t_opt0

        best_params = dict(study.best_params)
        best_cv = float(study.best_value)

        # 최종 학습: 학습 데이터 전체로 refit (Early Stopping 내부 홀드아웃 사용)
        notify("automl", step, total_steps, f"[{display}] 최적 파라미터로 최종 학습 중...")
        t_fit0 = time.perf_counter()
        model = fit_model(name, best_params, X_train, y_train, seed=seed, early_stop=True)
        fit_time = time.perf_counter() - t_fit0

        # 검증셋 평가 (모델 선정 기준 — 테스트셋 미사용)
        val_eval = evaluate_classifier(model, X_val, y_val)
        infer_ms = measure_inference_ms(model, X_val)

        history[name] = [
            {"trial": t.number + 1, "macro_f1": t.value, **t.params}
            for t in study.trials
        ]
        per_model[name] = {
            "display": display,
            "model": model,
            "best_params": best_params,
            "baseline_cv_macro_f1": base_cv,
            "best_cv_macro_f1": best_cv,
            "cv_improvement": best_cv - base_cv if base_cv == base_cv else float("nan"),
            "val_metrics": val_eval["metrics"],
            "val_per_class": val_eval["per_class"],
            "val_per_class_recall": val_eval["per_class_recall"],
            "fit_time_sec": fit_time,
            "opt_time_sec": opt_time,
            "complexity": model_complexity(name, model),
            "inference_ms_per_100": infer_ms,
            "n_trials": trials,
            "early_stopping": MODEL_CATALOG[name]["early_stopping"],
        }
        notify("automl", step, total_steps,
               f"[{display}] 완료 — CV={best_cv:.4f}, "
               f"검증 Macro F1={val_eval['metrics']['macro_f1']:.4f}, "
               f"학습 {fit_time:.1f}s / 최적화 {opt_time:.1f}s")

    rows = []
    for name, r in per_model.items():
        m = r["val_metrics"]
        rows.append({
            "model": name,
            "display": r["display"],
            "cv_macro_f1": r["best_cv_macro_f1"],
            "baseline_cv_macro_f1": r["baseline_cv_macro_f1"],
            "val_accuracy": m["accuracy"],
            "val_macro_f1": m["macro_f1"],
            "val_weighted_f1": m["weighted_f1"],
            "worst_class_recall": float(min(r["val_per_class_recall"].values())),
            "fit_time_sec": r["fit_time_sec"],
            "opt_time_sec": r["opt_time_sec"],
            "complexity": r["complexity"],
            "inference_ms_per_100": r["inference_ms_per_100"],
        })
    leaderboard = pd.DataFrame(rows).sort_values(
        "val_macro_f1", ascending=False).reset_index(drop=True)

    return {
        "mode": mode,
        "trials_per_model": trials,
        "n_splits": n_splits,
        "seed": seed,
        "per_model": per_model,
        "history": history,
        "leaderboard": leaderboard,
        "splits": splits,
        "model_names": names,
    }


def automl_to_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
    """리더보드 DataFrame 반환 (호출 편의용)."""
    return result["leaderboard"].copy()


if __name__ == "__main__":
    from src.train_models import load_training_frame

    df = load_training_frame()
    sp = split_train_val_test(df)
    print("분할:", sp["sizes"], sp["ratio"])
    res = run_automl(sp, mode="fast", n_trials=2)
    print(res[["model", "cv_macro_f1", "val_macro_f1", "fit_time_sec", "opt_time_sec"]]
          .to_string(index=False))
