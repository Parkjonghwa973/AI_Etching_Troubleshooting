"""
process_optimizer.py
====================

사용자의 현재 공정 조건을 머신러닝 모델로 평가하고, 불량 확률(특히 Normal 확률)을
높이는 **추천 공정 조건**을 탐색하는 모듈입니다.

구현 기능
--------
- 현재 조건의 불량 유형/확률 예측
- Random Search 또는 Differential Evolution(직접 구현)으로 조건 탐색
- 초기 조건에서 지나치게 멀어지지 않도록 **변경량 패널티** 적용
- 학습 데이터 분포에서 벗어난 조건에는 **분포 이탈(OOD) 경고/패널티** 적용
- 추천 조건을 교육용 식각 시뮬레이터에 넣어 전후 형상 지표 비교

.. warning::
    추천 조건은 머신러닝 모델의 예측에 기반한 **교육용 제안**입니다.
    실제 공정에서 불량이 반드시 감소한다고 단정할 수 없습니다.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data_generator import (
    DEFECT_NAMES,
    INPUT_COLUMNS,
    N_CLASSES,
    PROCESS_BOUNDS,
)

OPTIMIZATION_WARNING = (
    "[주의] 본 추천은 학습된 머신러닝 모델의 예측에 기반한 결과입니다. "
    "실제 공정에서 불량이 감소한다고 단정할 수 없으며, 반도체 공정 변경에는 "
    "숙련 엔지니어의 검토와 실제 실험 검증이 필요합니다."
)

OOD_WARNING = (
    "[주의] 추천 조건이 학습 데이터의 분포에서 벗어나 있습니다(out-of-distribution). "
    "모델 예측 신뢰도가 낮을 수 있으니 주의가 필요합니다."
)


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------
def conditions_to_array(conditions: Dict[str, float]) -> np.ndarray:
    """dict → (1, 6) array (INPUT_COLUMNS 순서 고정)."""
    return np.array([[float(conditions[c]) for c in INPUT_COLUMNS]], dtype=float)


def bounds_array(
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> np.ndarray:
    """(6, 2) 탐색 범위 배열."""
    b = bounds or PROCESS_BOUNDS
    return np.array([b[c] for c in INPUT_COLUMNS], dtype=float)


def predict_probabilities(model, conditions: Dict[str, float]) -> np.ndarray:
    """현재 조건의 5-class 예측 확률."""
    x = conditions_to_array(conditions)
    return np.asarray(model.predict_proba(x)[0]).ravel()


def predict_defect(model, conditions: Dict[str, float]) -> Tuple[int, str, np.ndarray]:
    """(예측 불량 유형 id, 이름, 확률) 반환."""
    probs = predict_probabilities(model, conditions)
    lab = int(np.argmax(probs))
    return lab, DEFECT_NAMES[lab], probs


def train_distribution_stats(X_train: pd.DataFrame) -> Dict[str, np.ndarray]:
    """학습 데이터 분포 통계 (분포 이탈 패널티용)."""
    return {
        "mean": X_train.mean().to_numpy(dtype=float),
        "std": np.clip(X_train.std().to_numpy(dtype=float), 1e-6, None),
        "min": X_train.min().to_numpy(dtype=float),
        "max": X_train.max().to_numpy(dtype=float),
    }


def ood_penalty(
    x: np.ndarray,
    stats: Optional[Dict[str, np.ndarray]],
    z_threshold: float = 2.5,
) -> Tuple[float, List[str]]:
    """학습 분포에서 벗어난 정도(0 이상의 패널티)와 경고 메시지 목록.

    - 학습 데이터 min/max 범위 밖: 경고 + 큰 패널티
    - z-score 임계값 초과: 경고 + 선형 패널티
    """
    if stats is None:
        return 0.0, []
    x = np.asarray(x, dtype=float).ravel()
    penalty = 0.0
    warnings_: List[str] = []

    below = x < stats["min"]
    above = x > stats["max"]
    for i, col in enumerate(INPUT_COLUMNS):
        if below[i] or above[i]:
            warnings_.append(f"{col}: 학습 데이터 범위({stats['min'][i]:.2f}~{stats['max'][i]:.2f}) "
                             f"밖의 값 {x[i]:.2f}")
            penalty += 2.0

    z = np.abs((x - stats["mean"]) / stats["std"])
    excess = np.clip(z - z_threshold, 0.0, None)
    penalty += float(excess.mean()) * 3.0
    if np.any(excess > 0) and not warnings_:
        idx = int(np.argmax(excess))
        warnings_.append(f"{INPUT_COLUMNS[idx]}: 학습 분포 평균 대비 z={z[idx]:.2f} 로 멀리 떨어져 있음")

    return penalty, warnings_


def change_penalty(
    x: np.ndarray,
    x0: np.ndarray,
    bnds: np.ndarray,
) -> float:
    """초기 조건 대비 변경량 패널티 (정규화 변경량의 평균)."""
    span = np.clip(bnds[:, 1] - bnds[:, 0], 1e-6, None)
    return float(np.mean(np.abs(np.asarray(x).ravel() - np.asarray(x0).ravel()) / span))


def objective_value(
    x: np.ndarray,
    model,
    x0: np.ndarray,
    bnds: np.ndarray,
    stats: Optional[Dict[str, np.ndarray]],
    change_weight: float,
    ood_weight: float,
) -> float:
    """목적 함수: -(Normal 확률) + 변경량 패널티 + 분포 이탈 패널티 (최소화 대상)."""
    x = np.clip(np.asarray(x, dtype=float).ravel(), bnds[:, 0], bnds[:, 1])
    probs = np.asarray(model.predict_proba(x.reshape(1, -1))[0]).ravel()
    p_normal = float(probs[0])
    ood, _ = ood_penalty(x, stats)
    return -p_normal + change_weight * change_penalty(x, x0, bnds) + ood_weight * ood


# ---------------------------------------------------------------------------
# 탐색 알고리즘
# ---------------------------------------------------------------------------
def random_search(
    model,
    x0: np.ndarray,
    bnds: np.ndarray,
    stats: Optional[Dict[str, np.ndarray]],
    n_iter: int = 3000,
    change_weight: float = 0.25,
    ood_weight: float = 1.0,
    random_state: int = 42,
) -> np.ndarray:
    """Random Search: 주어진 범위 내에서 무작위 후보를 평가해 최선 조건 선택."""
    rng = np.random.default_rng(random_state)
    candidates = rng.uniform(bnds[:, 0], bnds[:, 1], size=(n_iter, bnds.shape[0]))
    candidates[0] = x0  # 현재 조건도 후보에 포함
    best, best_score = x0.copy(), np.inf
    for c in candidates:
        s = objective_value(c, model, x0, bnds, stats, change_weight, ood_weight)
        if s < best_score:
            best, best_score = np.clip(c, bnds[:, 0], bnds[:, 1]).copy(), s
    return best


def differential_evolution(
    model,
    x0: np.ndarray,
    bnds: np.ndarray,
    stats: Optional[Dict[str, np.ndarray]],
    n_iter: int = 400,
    pop_size: int = 24,
    mutation: float = 0.7,
    crossover: float = 0.9,
    change_weight: float = 0.25,
    ood_weight: float = 1.0,
    random_state: int = 42,
) -> np.ndarray:
    """Differential Evolution (rand/1/bin) — numpy 만으로 직접 구현."""
    rng = np.random.default_rng(random_state)
    dim = bnds.shape[0]

    def fit(v: np.ndarray) -> float:
        return objective_value(v, model, x0, bnds, stats, change_weight, ood_weight)

    pop = rng.uniform(bnds[:, 0], bnds[:, 1], size=(pop_size, dim))
    pop[0] = x0
    scores = np.array([fit(ind) for ind in pop])

    for _ in range(n_iter):
        for i in range(pop_size):
            idxs = [j for j in range(pop_size) if j != i]
            a, b, c = pop[rng.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + mutation * (b - c), bnds[:, 0], bnds[:, 1])
            cross_mask = rng.random(dim) < crossover
            if not cross_mask.any():
                cross_mask[rng.integers(0, dim)] = True
            trial = np.where(cross_mask, mutant, pop[i])
            s_trial = fit(trial)
            if s_trial <= scores[i]:
                pop[i], scores[i] = trial, s_trial

    best = int(np.argmin(scores))
    return np.clip(pop[best], bnds[:, 0], bnds[:, 1])


# ---------------------------------------------------------------------------
# 공정 조건 최적화 (공개 API)
# ---------------------------------------------------------------------------
def optimize_process(
    model,
    current_conditions: Dict[str, float],
    method: str = "random",
    n_iter: int = 3000,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    train_stats: Optional[Dict[str, np.ndarray]] = None,
    change_weight: float = 0.25,
    ood_weight: float = 1.0,
    random_state: int = 42,
) -> Dict:
    """현재 공정 조건을 개선할 수 있는 추천 조건을 탐색합니다.

    Parameters
    ----------
    model : 학습된 분류기 (predict_proba 지원, 클래스 0 = Normal)
    current_conditions : 6개 공정 변수 dict
    method : "random" (Random Search) 또는 "de" (Differential Evolution)
    n_iter : 탐색 횟수 (random: 후보 수, de: 세대 수)
    bounds : 탐색 범위 (기본: 데이터 생성 범위)
    train_stats : 학습 데이터 분포 통계 (분포 이탈 패널티/경고용)
    change_weight : 초기 조건 변경량 패널티 계수
    ood_weight : 분포 이탈 패널티 계수

    Returns
    -------
    dict : original / recommended / 확률 / 예측 불량 / 변경량 / 경고 등
    """
    bnds = bounds_array(bounds)
    current = {c: float(np.clip(current_conditions[c], *bnds[i]))
               for i, c in enumerate(INPUT_COLUMNS)}
    x0 = np.array([current[c] for c in INPUT_COLUMNS], dtype=float)

    method_key = method.lower()
    if method_key.startswith("de") or "differential" in method_key:
        method_name = "Differential Evolution"
        x_best = differential_evolution(model, x0, bnds, train_stats,
                                        n_iter=max(int(n_iter // 12), 30),
                                        change_weight=change_weight,
                                        ood_weight=ood_weight,
                                        random_state=random_state)
    else:
        method_name = "Random Search"
        x_best = random_search(model, x0, bnds, train_stats,
                               n_iter=int(n_iter),
                               change_weight=change_weight,
                               ood_weight=ood_weight,
                               random_state=random_state)

    recommended = {c: float(x_best[i]) for i, c in enumerate(INPUT_COLUMNS)}

    # 전후 예측
    lab_before, name_before, probs_before = predict_defect(model, current)
    lab_after, name_after, probs_after = predict_defect(model, recommended)

    # 경고 수집
    warnings_list: List[str] = []
    ood_before, w_before = ood_penalty(x0, train_stats)
    ood_after, w_after = ood_penalty(x_best, train_stats)
    warnings_list += [f"[현재 조건] {w}" for w in w_before]
    warnings_list += [f"[추천 조건] {w}" for w in w_after]
    if ood_after > 0:
        warnings_list.append(OOD_WARNING)
    warnings_list.append(OPTIMIZATION_WARNING)

    changes = {
        "absolute": {c: recommended[c] - current[c] for c in INPUT_COLUMNS},
        "relative_pct": {
            c: (recommended[c] - current[c]) / abs(current[c]) * 100.0
            if abs(current[c]) > 1e-9 else 0.0
            for c in INPUT_COLUMNS
        },
        "normalized_change": change_penalty(x_best, x0, bnds),
    }

    return {
        "method": method_name,
        "original": current,
        "recommended": recommended,
        "defect_before": name_before,
        "defect_before_id": lab_before,
        "defect_after": name_after,
        "defect_after_id": lab_after,
        "probabilities_before": probs_before,
        "probabilities_after": probs_after,
        "normal_prob_before": float(probs_before[0]),
        "normal_prob_after": float(probs_after[0]),
        "parameter_changes": changes,
        "within_bounds": bool(np.all(x_best >= bnds[:, 0] - 1e-9)
                              and np.all(x_best <= bnds[:, 1] + 1e-9)),
        "ood_penalty_before": float(ood_before),
        "ood_penalty_after": float(ood_after),
        "warnings": warnings_list,
        "warning": OPTIMIZATION_WARNING,
    }


# ---------------------------------------------------------------------------
# 결과 정리 / 시뮬레이션 연계
# ---------------------------------------------------------------------------
def result_to_dataframe(result: Dict) -> pd.DataFrame:
    """추천 결과를 사람이 읽기 쉬운 표로 정리합니다."""
    rows = []
    for c in INPUT_COLUMNS:
        rows.append({
            "Parameter": c,
            "Original": round(result["original"][c], 3),
            "Recommended": round(result["recommended"][c], 3),
            "Change": round(result["parameter_changes"]["absolute"][c], 3),
            "Change (%)": round(result["parameter_changes"]["relative_pct"][c], 1),
        })
    return pd.DataFrame(rows)


def summary_dataframe(result: Dict) -> pd.DataFrame:
    """최적화 전후 요약표."""
    return pd.DataFrame([
        {"Item": "Method", "Before": "-", "After": result["method"]},
        {"Item": "Predicted defect",
         "Before": result["defect_before"], "After": result["defect_after"]},
        {"Item": "P(Normal)",
         "Before": f"{result['normal_prob_before']:.3f}",
         "After": f"{result['normal_prob_after']:.3f}"},
    ])


def compare_profiles(
    result: Dict,
    n_points: int = 240,
):
    """추천 조건을 교육용 식각 시뮬레이터에 입력해 전후 형상을 비교합니다.

    Returns
    -------
    (before_profile, after_profile, metrics_table)
    """
    from src.etching_simulator import compute_profile, profile_metrics_table

    before = compute_profile(result["original"], defect_type=result["defect_before_id"],
                             n_points=n_points)
    after = compute_profile(result["recommended"], defect_type=result["defect_after_id"],
                            n_points=n_points)
    table = profile_metrics_table({"Before (original)": before,
                                   "After (recommended)": after})
    return before, after, table


def optimize_both_methods(
    model,
    current_conditions: Dict[str, float],
    n_iter_random: int = 3000,
    n_iter_de: int = 360,
    train_stats: Optional[Dict[str, np.ndarray]] = None,
    random_state: int = 42,
) -> Dict[str, Dict]:
    """Random Search와 Differential Evolution 결과를 모두 반환해 비교합니다."""
    return {
        "random": optimize_process(model, current_conditions, method="random",
                                   n_iter=n_iter_random, train_stats=train_stats,
                                   random_state=random_state),
        "de": optimize_process(model, current_conditions, method="de",
                               n_iter=n_iter_de, train_stats=train_stats,
                               random_state=random_state),
    }


if __name__ == "__main__":
    from src.etching_simulator import default_process_conditions
    from src.train_models import load_models

    models = load_models()
    if "optimized_xgboost" not in models:
        raise SystemExit("모델이 없습니다. run_pipeline.py 를 먼저 실행하세요.")
    cur = default_process_conditions()
    res = optimize_process(models["optimized_xgboost"], cur, method="random")
    print(result_to_dataframe(res).to_string(index=False))
    print(f"예측 불량: {res['defect_before']} → {res['defect_after']}")
    print(f"P(Normal): {res['normal_prob_before']:.3f} → {res['normal_prob_after']:.3f}")
