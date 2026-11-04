"""
explain_model.py
================

최적화된 XGBoost 모델에 대해 **SHAP 기반 원인 후보 분석**을 수행하는 모듈입니다.

구현 기능
--------
- Global feature importance (평균 |SHAP|)
- Local feature contribution (개별 샘플의 변수 기여도)
- SHAP summary plot (beeswarm)
- 특정 불량 예측에 대한 변수별 기여도 분석
- 합성 데이터 생성 규칙에 실제로 사용된 변수와 SHAP 결과 비교

.. warning::
    SHAP 값은 **모델 예측에 대한 기여도(contribution)** 일 뿐,
    물리적 인과관계(causality)의 증거가 아닙니다. 결과 해석 시 반드시 유의해야 합니다.

SHAP 버전 차이 처리
------------------
- 구버전(shap < 0.45) : 다중 클래스 TreeExplainer 출력이 클래스별 list[array]
- 신버전(shap >= 0.45) : (n_samples, n_features, n_classes) 3D array
두 형상을 모두 받아들여 (n_samples, n_features, n_classes) 로 정규화합니다.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data_generator import (
    DEFECT_NAMES,
    GROUND_TRUTH_FEATURES,
    N_CLASSES,
    RANDOM_STATE,
)
from src.visualization import (
    plot_feature_importance,
    plot_ground_truth_vs_shap,
    plot_local_contribution,
    plot_shap_summary,
)

CAUSALITY_WARNING = (
    "[주의] SHAP 값은 '모델이 예측을 내리는 데 어떤 변수가 얼마나 기여했는지'를 나타내는 "
    "해석 지표이며, 물리적 인과관계(causality)를 증명하지 않습니다. "
    "SHAP 중요도가 높다고 해서 해당 변수가 불량의 원인이라고 단정할 수 없습니다."
)

try:
    import shap as shap_lib
except Exception as exc:  # pragma: no cover
    warnings.warn(f"shap 를 불러오지 못했습니다: {exc}")
    shap_lib = None  # type: ignore


def default_results_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# SHAP 값 계산 및 형상 정규화
# ---------------------------------------------------------------------------
def compute_shap_values(
    model,
    X: pd.DataFrame,
    max_samples: int = 500,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """TreeExplainer 로 SHAP 값을 계산합니다.

    Returns
    -------
    (shap_values_3d, X_used) :
        shap_values_3d.shape = (n_samples, n_features, n_classes)
    """
    if shap_lib is None:
        raise ImportError("shap 가 설치되어 있지 않습니다. pip install shap")

    X_used = X
    if len(X) > max_samples:
        X_used = X.sample(n=max_samples, random_state=random_state).sort_index()

    explainer = shap_lib.TreeExplainer(model)
    raw = explainer.shap_values(X_used)
    sv3 = normalize_shap_output(raw, n_features=X_used.shape[1])
    return sv3, X_used


def normalize_shap_output(
    shap_values,
    n_features: int,
    n_classes: int = N_CLASSES,
) -> np.ndarray:
    """SHAP 출력 형상 차이를 처리하여 (n, n_features, n_classes) 로 통일합니다.

    지원 형상
    --------
    1) list[array]           (신/구버전 모두, 클래스별 2D array)
    2) ndarray 3D            (shap >= 0.45 : (n, features, classes))
    3) ndarray 2D            (단일 클래스 출력, 회귀/이진 분류)
    """
    if isinstance(shap_values, (list, tuple)):
        if len(shap_values) != n_classes:
            raise ValueError(
                f"SHAP list 길이({len(shap_values)})가 클래스 수({n_classes})와 다릅니다.")
        arr = np.stack([np.asarray(sv) for sv in shap_values], axis=-1)
        return arr

    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        # (n, features, classes) 또는 (n, classes, features) 형태 확인
        if arr.shape[1] == n_features and arr.shape[2] == n_classes:
            return arr
        if arr.shape[1] == n_classes and arr.shape[2] == n_features:
            return np.transpose(arr, (0, 2, 1))
        raise ValueError(f"예상하지 못한 SHAP 3D 형태: {arr.shape}")
    if arr.ndim == 2:
        # 단일 클래스 → 클래스 1차원으로 확장
        out = np.zeros((arr.shape[0], arr.shape[1], n_classes), dtype=float)
        out[:, :, 1] = arr
        return out
    raise ValueError(f"지원하지 않는 SHAP 출력 형태: ndim={arr.ndim}")


# ---------------------------------------------------------------------------
# 분석 함수
# ---------------------------------------------------------------------------
def global_feature_importance(
    shap_values_3d: np.ndarray,
    feature_names: Sequence[str],
    class_index: Optional[int] = None,
) -> pd.DataFrame:
    """Global feature importance: 평균 |SHAP|.

    class_index=None 이면 전체 클래스에 대한 평균(절댓값)을 사용합니다.
    """
    if class_index is None:
        imp = np.abs(shap_values_3d).mean(axis=(0, 2))
    else:
        imp = np.abs(shap_values_3d[:, :, class_index]).mean(axis=0)
    return pd.DataFrame({
        "feature": list(feature_names),
        "mean_abs_shap": imp,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def local_contribution(
    shap_values_3d: np.ndarray,
    feature_names: Sequence[str],
    sample_index: int,
    class_index: int,
    x_row: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """개별 샘플에 대한 변수별 SHAP 기여도 표."""
    vals = shap_values_3d[sample_index, :, class_index]
    df = pd.DataFrame({
        "feature": list(feature_names),
        "shap_value": vals,
        "abs_shap": np.abs(vals),
    })
    if x_row is not None:
        df["feature_value"] = [float(x_row[f]) for f in feature_names]
    return df.sort_values("abs_shap", ascending=False).reset_index(drop=True)


def compare_with_generation_rules(
    importance: pd.DataFrame,
    ground_truth_features: Optional[Dict[int, List[str]]] = None,
) -> pd.DataFrame:
    """SHAP 중요도와 실제 데이터 생성 규칙에서 사용된 변수를 비교합니다."""
    gt = ground_truth_features or GROUND_TRUTH_FEATURES
    gt_set = set()
    for feats in gt.values():
        gt_set.update(feats)
    out = importance.copy()
    out["in_generation_rules"] = out["feature"].isin(gt_set)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def top_features_for_class(
    shap_values_3d: np.ndarray,
    feature_names: Sequence[str],
    class_index: int,
    top_n: int = 3,
) -> List[str]:
    """특정 불량 클래스 예측에 가장 크게 기여하는 변수 목록."""
    imp = np.abs(shap_values_3d[:, :, class_index]).mean(axis=0)
    order = np.argsort(imp)[::-1][:top_n]
    return [feature_names[i] for i in order]


def explain_prediction(
    model,
    x_row: pd.DataFrame,
    feature_names: Sequence[str],
    class_index: Optional[int] = None,
) -> Dict:
    """단일 조건에 대한 예측 + SHAP 기여도를 반환합니다 (대시보드용).

    Returns
    -------
    dict : {"probabilities", "predicted_class", "predicted_name",
            "contributions", "causality_warning"}
    """
    if shap_lib is None:
        raise ImportError("shap 가 설치되어 있지 않습니다.")
    explainer = shap_lib.TreeExplainer(model)
    raw = explainer.shap_values(x_row)
    sv3 = normalize_shap_output(raw, n_features=x_row.shape[1])[0]  # (n_features, n_classes)

    probs = model.predict_proba(x_row.to_numpy())[0]
    predicted = int(np.argmax(probs))
    if class_index is None:
        class_index = predicted

    contrib = pd.DataFrame({
        "feature": list(feature_names),
        "shap_value": sv3[:, class_index],
        "abs_shap": np.abs(sv3[:, class_index]),
        "feature_value": [float(x_row.iloc[0][f]) for f in feature_names],
    }).sort_values("abs_shap", ascending=False).reset_index(drop=True)

    return {
        "probabilities": probs,
        "predicted_class": predicted,
        "predicted_name": DEFECT_NAMES[predicted],
        "explained_class": class_index,
        "contributions": contrib,
        "causality_warning": CAUSALITY_WARNING,
    }


# ---------------------------------------------------------------------------
# 전체 분석 실행
# ---------------------------------------------------------------------------
def run_shap_analysis(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    results_dir: Optional[str | Path] = None,
    max_samples: int = 500,
    top_classes: Sequence[int] = (1, 2, 3, 4),
) -> Dict:
    """SHAP 전체 분석을 수행하고 결과/그래프를 저장합니다."""
    results_dir = Path(results_dir) if results_dir is not None else default_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    print(CAUSALITY_WARNING)
    feature_names = list(X_test.columns)

    sv3, X_used = compute_shap_values(model, X_test, max_samples=max_samples)
    print(f"    [SHAP] 계산 완료: shap_values shape = {sv3.shape} "
          f"(n_samples, n_features, n_classes)")

    results: Dict = {"shap_values": sv3, "X_used": X_used,
                     "causality_warning": CAUSALITY_WARNING}

    # 1) Global feature importance
    imp = global_feature_importance(sv3, feature_names)
    imp_csv = results_dir / "shap_global_importance.csv"
    imp.to_csv(imp_csv, index=False, encoding="utf-8-sig")
    results["global_importance"] = imp
    print(f"    [저장] {imp_csv}")
    plot_feature_importance(
        imp, results_dir / "feature_importance.png",
        title="Global Feature Importance (mean |SHAP value|, Optimized XGBoost)")
    plot_feature_importance(
        imp, results_dir / "feature_importance_shap.png",
        title="Global Feature Importance (mean |SHAP value|, Optimized XGBoost)")

    # 2) SHAP summary plots (전체 + 주요 클래스)
    #    전체용: 클래스 무관 평균 |SHAP| (모든 클래스에 대한 영향의 크기)
    sv_overall = np.abs(sv3).mean(axis=2)[:, :, None]
    _save_summary_safe(sv_overall, X_used, 0,
                       results_dir / "shap_summary.png",
                       "SHAP Summary (all classes, mean |impact|)")
    for ci in top_classes:
        _save_summary_safe(sv3, X_used, ci,
                           results_dir / f"shap_summary_class{ci}.png",
                           f"SHAP Summary — class {ci}: {DEFECT_NAMES[ci]}")

    # 3) 특정 불량 클래스의 주요 기여 변수
    class_top = {}
    rows = []
    for ci in top_classes:
        top = top_features_for_class(sv3, feature_names, ci)
        class_top[ci] = top
        rows.append({"class": ci, "class_name": DEFECT_NAMES[ci],
                     "top_features": ", ".join(top)})
    class_top_df = pd.DataFrame(rows)
    class_top_csv = results_dir / "shap_top_features_by_class.csv"
    class_top_df.to_csv(class_top_csv, index=False, encoding="utf-8-sig")
    results["top_features_by_class"] = class_top_df
    print(f"    [저장] {class_top_csv}")

    # 4) Local feature contribution 예시 (각 클래스 1건)
    local_rows = []
    for ci in range(N_CLASSES):
        # X_used 에서 해당 클래스로 예측된 첫 샘플 탐색
        preds = model.predict(X_used.to_numpy())
        cand = np.where(np.asarray(preds).ravel() == ci)[0]
        if len(cand) == 0:
            continue
        si = int(cand[0])
        lc = local_contribution(sv3, feature_names, si, ci, X_used.iloc[si])
        plot_local_contribution(
            lc, results_dir / f"shap_local_class{ci}.png",
            title=f"Local Contribution — sample #{si}, class {ci}: {DEFECT_NAMES[ci]}")
        lc.insert(0, "class", ci)
        lc.insert(0, "sample_index", int(X_used.index[si]))
        local_rows.append(lc)
    if local_rows:
        local_all = pd.concat(local_rows, ignore_index=True)
        local_csv = results_dir / "shap_local_contributions.csv"
        local_all.to_csv(local_csv, index=False, encoding="utf-8-sig")
        results["local_contributions"] = local_all
        print(f"    [저장] {local_csv}")

    # 5) 생성 규칙 비교
    comparison = compare_with_generation_rules(imp)
    cmp_csv = results_dir / "shap_vs_generation_rules.csv"
    comparison.to_csv(cmp_csv, index=False, encoding="utf-8-sig")
    results["rule_comparison"] = comparison
    print(f"    [저장] {cmp_csv}")
    plot_ground_truth_vs_shap(comparison, results_dir / "shap_vs_generation_rules.png")

    # 6) 학습 데이터 분포 기반 통계 (이상치 경고용)
    results["train_stats"] = {
        "mean": X_train.mean().to_numpy(),
        "std": X_train.std().to_numpy(),
        "min": X_train.min().to_numpy(),
        "max": X_train.max().to_numpy(),
    }
    return results


def _save_summary_safe(sv3, X_used, class_index, path, title) -> Optional[Path]:
    """요구 그래프와 무관하게 SHAP summary 저장 실패 시 경고만 출력."""
    try:
        p = plot_shap_summary(sv3, X_used, class_index, path)
        if p is not None:
            print(f"    [저장] {p}  ({title})")
        return p
    except Exception as exc:  # pragma: no cover
        print(f"    [WARN] SHAP summary 저장 실패 ({path.name}): {exc}")
        return None


if __name__ == "__main__":
    from src.train_models import FEATURE_COLUMNS, load_models, load_training_frame

    df = load_training_frame()
    models = load_models()
    if "optimized_xgboost" not in models:
        raise SystemExit("최적화 모델이 없습니다. run_pipeline.py 를 실행하세요.")
    X = df[FEATURE_COLUMNS]
    run_shap_analysis(models["optimized_xgboost"], X.sample(1000, random_state=0), X)
