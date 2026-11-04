"""
auto_troubleshooter.py
======================

사용자의 현재 공정 조건을 입력하면 **불량 예측 → 원인 후보 분석 → 공정 조건 탐색 →
추천 조건 → 비교 → 식각 프로파일 시뮬레이션 → 개선 가능성 평가**를 자동 수행합니다.

설계 포인트
----------
- 탐색 범위 = **합성 데이터 생성 범위 ∩ 사용자 안전 범위** (교집합)
- 학습 데이터 분포에서 벗어난 조건에는 패널티/경고 (process_optimizer 재사용)
- 정상(Normal) 예측 확률이 높은 조건을 추천 (Random Search / Differential Evolution)
- 원인 후보 분석: SHAP(트리 모델) 우선, 실패 시 **1회 변경 민감도(sensitivity)** 대체
  방법 사용 (모델-agnostic)

.. warning::
    추천 조건의 정상 예측 확률이 높아져도 **실제 공정 불량이 감소한다고 단정할 수
    없습니다.** 모든 결과는 교육용 합성 데이터에 기반한 머신러닝 예측 설명입니다.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data_generator import DEFECT_NAMES, INPUT_COLUMNS, PROCESS_BOUNDS
from src.process_optimizer import (
    OPTIMIZATION_WARNING,
    optimize_process,
    predict_defect,
    result_to_dataframe,
    summary_dataframe,
    train_distribution_stats,
)

CAVEAT = (
    "⚠️ 본 추천은 학습된 머신러닝 모델의 예측에 기반한 교육용 결과이며, "
    "실제 공정에서 불량이 감소한다고 단정할 수 없습니다."
)

METHOD_CHOICES = {"random": "Random Search", "de": "Differential Evolution",
                  "differential_evolution": "Differential Evolution"}


# ---------------------------------------------------------------------------
# 범위: 합성 데이터 범위 ∩ 사용자 안전 범위
# ---------------------------------------------------------------------------
def intersect_bounds(
    safe_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    data_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Tuple[float, float]]:
    """탐색 범위 = 데이터 생성 범위와 안전 범위의 교집합.

    교집합이 비어 있는 변수는 ValueError.
    """
    data_bounds = data_bounds or PROCESS_BOUNDS
    if safe_bounds is None:
        return dict(data_bounds)

    out: Dict[str, Tuple[float, float]] = {}
    for col in INPUT_COLUMNS:
        d_lo, d_hi = data_bounds[col]
        s_lo, s_hi = safe_bounds.get(col, (d_lo, d_hi))
        lo, hi = max(float(d_lo), float(s_lo)), min(float(d_hi), float(s_hi))
        if lo > hi:
            raise ValueError(
                f"'{col}' 안전 범위({s_lo}~{s_hi})와 데이터 범위({d_lo}~{d_hi})의 "
                f"교집합이 비어 있습니다.")
        out[col] = (lo, hi)
    return out


# ---------------------------------------------------------------------------
# 원인 후보 분석 (SHAP 우선, 대체 방법)
# ---------------------------------------------------------------------------
def sensitivity_contribution(
    model,
    conditions: Dict[str, float],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    class_index: Optional[int] = None,
) -> pd.DataFrame:
    """모델-agnostic 대체 방법: 각 변수를 범위의 ±10% 로 바꿔 예측 확률 변화량 계산.

    SHAP 을 지원하지 않는 모델이나 SHAP 오류 시 사용합니다.
    """
    bounds = bounds or PROCESS_BOUNDS
    base = np.array([float(conditions[c]) for c in INPUT_COLUMNS]).reshape(1, -1)
    probs_base = np.asarray(model.predict_proba(base)[0]).ravel()
    target = int(np.argmax(probs_base)) if class_index is None else int(class_index)

    rows = []
    for i, col in enumerate(INPUT_COLUMNS):
        lo, hi = bounds[col]
        span = (hi - lo) * 0.10
        x_up, x_dn = base.copy(), base.copy()
        x_up[0, i] = float(np.clip(base[0, i] + span, lo, hi))
        x_dn[0, i] = float(np.clip(base[0, i] - span, lo, hi))
        p_up = float(np.asarray(model.predict_proba(x_up)[0]).ravel()[target])
        p_dn = float(np.asarray(model.predict_proba(x_dn)[0]).ravel()[target])
        effect = 0.5 * (p_up - p_dn)   # 중앙차분
        rows.append({
            "feature": col,
            "feature_value": float(base[0, i]),
            "contribution": effect,           # + : 해당 변수 증가 시 이 클래스 확률 ↑
            "abs_contribution": abs(effect),
            "direction": "증가 시 확률 ↑" if effect > 0 else ("증가 시 확률 ↓" if effect < 0 else "영향 없음"),
        })
    df = pd.DataFrame(rows).sort_values("abs_contribution", ascending=False)
    df = df.reset_index(drop=True)
    df.attrs["method"] = "sensitivity (1회 변경 민감도)"
    df.attrs["target_class"] = target
    return df


def explain_current_condition(
    model,
    conditions: Dict[str, float],
    feature_names: Optional[Sequence[str]] = None,
    class_index: Optional[int] = None,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """현재 조건에 대한 예측 + 주요 기여 변수/기여 방향 분석."""
    feature_names = list(feature_names or INPUT_COLUMNS)
    x_df = pd.DataFrame([{c: float(conditions[c]) for c in feature_names}])
    probs = np.asarray(model.predict_proba(x_df.to_numpy())[0]).ravel()
    predicted = int(np.argmax(probs)) if class_index is None else int(class_index)

    method, contributions = "", None
    try:
        from src.explain_model import explain_prediction
        out = explain_prediction(model, x_df, feature_names, class_index=predicted)
        contributions = out["contributions"].rename(
            columns={"shap_value": "contribution", "abs_shap": "abs_contribution"})
        if "feature_value" not in contributions.columns:
            contributions["feature_value"] = float(x_df.iloc[0][feature_names[0]])
        contributions["direction"] = contributions["contribution"].apply(
            lambda v: "증가 시 확률 ↑" if v > 0 else ("증가 시 확률 ↓" if v < 0 else "영향 없음"))
        method = "SHAP (TreeExplainer)"
    except Exception:
        contributions = sensitivity_contribution(model, conditions, bounds, predicted)
        method = "sensitivity (SHAP 미지원 모델 대체)"

    return {
        "probabilities": probs,
        "predicted_class": predicted,
        "predicted_name": DEFECT_NAMES[predicted],
        "contributions": contributions[[
            "feature", "feature_value", "contribution",
            "abs_contribution", "direction"]].round(6) if contributions is not None else None,
        "contribution_method": method,
        "top_features": (list(contributions["feature"].head(3))
                         if contributions is not None else []),
    }


# ---------------------------------------------------------------------------
# 개선 가능성 평가
# ---------------------------------------------------------------------------
def assess_improvement(result: Dict[str, Any]) -> Dict[str, Any]:
    """추천 전후 예측을 비교해 개선 가능성을 평가합니다 (단정하지 않음)."""
    before = float(result["normal_prob_before"])
    after = float(result["normal_prob_after"])
    delta = after - before
    defect_fixed = (result["defect_before_id"] != 0 and result["defect_after_id"] == 0)

    if delta > 0.005 and defect_fixed:
        verdict = "개선 후보: 불량 → Normal 로 예측이 전환되고 정상 확률이 증가했습니다."
        level = "개선 가능성 높음 (모델 예측 기준)"
    elif delta > 0.005:
        verdict = "개선 후보: 정상(Normal) 예측 확률이 증가했습니다."
        level = "개선 가능성 있음 (모델 예측 기준)"
    elif abs(delta) <= 0.005:
        verdict = "변화 미미: 정상 예측 확률이 유의미하게 변하지 않았습니다."
        level = "개선 가능성 낮음"
    else:
        verdict = "정상 예측 확률이 감소했습니다. 다른 탐색 설정(방법/횟수)을 고려하세요."
        level = "개선 실패 (모델 예측 기준)"

    return {
        "normal_prob_delta": delta,
        "defect_before": result["defect_before"],
        "defect_after": result["defect_after"],
        "defect_changed": result["defect_before"] != result["defect_after"],
        "defect_to_normal": defect_fixed,
        "verdict": verdict,
        "level": level,
        "caveat": CAVEAT,
    }


# ---------------------------------------------------------------------------
# 메인: 자동 트러블슈팅
# ---------------------------------------------------------------------------
def auto_troubleshoot(
    model,
    current_conditions: Dict[str, float],
    safe_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    method: str = "de",
    n_iter: int = 400,
    train_stats: Optional[Dict[str, np.ndarray]] = None,
    random_state: int = 42,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """현재 조건 분석부터 추천까지 전체 과정을 자동 수행합니다.

    Parameters
    ----------
    model : 선정된 분류기 (predict_proba 필수, 클래스 0 = Normal)
    current_conditions : 6개 공정 변수
    safe_bounds : 사용자 지정 안전 범위 (없으면 데이터 범위만 사용)
    method : "random" 또는 "de"
    n_iter : 탐색 횟수
    train_stats : 학습 데이터 분포 통계 (분포 이탈 패널티용)

    Returns
    -------
    dict : current_analysis / bounds / recommendation / comparison / improvement / ...
    """
    t0 = time.perf_counter()
    feature_names = list(feature_names or INPUT_COLUMNS)
    bounds = intersect_bounds(safe_bounds)
    method_key = method.lower()
    if method_key not in METHOD_CHOICES:
        raise ValueError(f"method 는 'random' 또는 'de' 여야 합니다: {method}")

    # 1) 현재 조건 예측 + 2) 원인 후보 분석
    current_analysis = explain_current_condition(
        model, current_conditions, feature_names, bounds=bounds)

    # 3~4) 공정 조건 탐색 및 추천
    rec = optimize_process(
        model, current_conditions,
        method=method_key,
        n_iter=n_iter,
        bounds=bounds,
        train_stats=train_stats,
        random_state=random_state,
    )

    # 5) 기존/추천 비교표
    comparison = result_to_dataframe(rec)
    summary = summary_dataframe(rec)

    # 6~7) 개선 가능성 평가 (식각 프로파일 시뮬레이션은 validation_engine 담당)
    improvement = assess_improvement(rec)

    return {
        "current_analysis": current_analysis,
        "bounds": bounds,
        "method": rec["method"],
        "recommendation": rec,
        "comparison": comparison,
        "summary": summary,
        "improvement": improvement,
        "parameter_changes": rec["parameter_changes"],
        "probabilities_before": rec["probabilities_before"],
        "probabilities_after": rec["probabilities_after"],
        "normal_prob_before": rec["normal_prob_before"],
        "normal_prob_after": rec["normal_prob_after"],
        "defect_before": rec["defect_before"],
        "defect_after": rec["defect_after"],
        "within_bounds": rec["within_bounds"],
        "warnings": list(rec.get("warnings", [])) + [CAVEAT],
        "elapsed_sec": time.perf_counter() - t0,
        "caveat": CAVEAT,
    }


def build_train_stats_from_frame(X_train: pd.DataFrame) -> Dict[str, np.ndarray]:
    """학습 데이터 프레임으로 분포 통계 생성."""
    return train_distribution_stats(X_train)


if __name__ == "__main__":
    from src.etching_simulator import default_process_conditions
    from src.model_selector import load_best_model

    model = load_best_model()
    out = auto_troubleshoot(model, default_process_conditions(), method="random",
                            n_iter=500)
    print(out["comparison"].to_string(index=False))
    print(out["improvement"])
