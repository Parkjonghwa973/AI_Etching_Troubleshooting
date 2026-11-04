"""
validation_engine.py
====================

AI(머신러닝)가 추천한 공정 조건을 **교육용 Etching 식각 시뮬레이터**에 입력하여
기존 조건과 추천 조건의 전후 결과를 비교·검증하는 모듈입니다.

계산 지표
--------
- Etch depth (식각 깊이, nm)
- Top CD (상단 CD, nm)
- Middle CD (중간 CD, nm)
- Bottom CD (하단 CD, nm)
- Sidewall angle (측벽 각도, deg)
- Defect severity (결함 심각도, 0~100)

.. warning::
    시뮬레이터와 합성 데이터 생성 규칙은 **동일한 물리 가정**을 공유하므로,
    이 비교 결과는 "AI 추천 조건의 독립적인 물리 검증"이 아니라
    **동일 가정 내 정합성 확인**입니다. 실제 공정 불량 감소를 증명하지 않습니다.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_generator import INPUT_COLUMNS, compute_defect_severity, compute_indicators
from src.etching_simulator import compute_profile, predict_defect_type

VALIDATION_CAVEAT = (
    "[주의] 식각 시뮬레이터와 합성 데이터 생성 규칙은 동일한 물리 가정을 공유하므로, "
    "이 결과는 독립적인 물리 검증이 아니라 동일 가정 내 정합성 확인입니다. "
    "추천 조건의 실제 공정 불량 감소를 증명하지 않습니다."
)

METRIC_KEYS = [
    ("etch_depth_nm", "Etch depth (nm)"),
    ("top_cd_nm", "Top CD (nm)"),
    ("middle_cd_nm", "Middle CD (nm)"),
    ("bottom_cd_nm", "Bottom CD (nm)"),
    ("sidewall_angle_deg", "Sidewall angle (deg)"),
    ("defect_severity", "Defect severity (0-100)"),
]


def compute_defect_severity_value(conditions: Dict[str, float]) -> float:
    """주어진 조건의 결함 심각도(0~100, 규칙 기반, 노이즈 없음)를 계산합니다."""
    df = pd.DataFrame([{c: float(conditions[c]) for c in INPUT_COLUMNS}])
    ind = compute_indicators(df, noise_scale=0.0, add_noise=False)
    probs = np.asarray(ind.attrs.get("defect_probabilities",
                                     None), dtype=float)
    if probs.ndim != 2 or probs.shape[0] != 1:
        from src.data_generator import defect_probabilities
        probs = defect_probabilities(
            ind,
            np.asarray(ind.attrs["lateral_factor"]),
            np.asarray(ind.attrs["bowing_driver"]),
            df["temperature_c"].to_numpy(),
        )
    return float(compute_defect_severity(ind, probs, noise_scale=0.0)[0])


def compute_condition_metrics(
    conditions: Dict[str, float],
    defect_type: Optional[int] = None,
) -> Dict[str, Any]:
    """조건 하나에 대해 식각 프로파일 지표 + 결함 심각도를 계산합니다."""
    profile = compute_profile(conditions, defect_type=defect_type)
    severity = compute_defect_severity_value(conditions)
    metrics = {
        "etch_depth_nm": float(profile["etch_depth_nm"]),
        "top_cd_nm": float(profile["top_cd_nm"]),
        "middle_cd_nm": float(profile["middle_cd_nm"]),
        "bottom_cd_nm": float(profile["bottom_cd_nm"]),
        "sidewall_angle_deg": float(profile["sidewall_angle_deg"]),
        "defect_severity": severity,
        "max_lateral_deviation_nm": float(profile["max_lateral_deviation_nm"]),
        "defect_type": int(profile["defect_type"]),
        "defect_name": str(profile["defect_name"]),
    }
    return {"metrics": metrics, "profile": profile}


def validate_recommendation(
    before_conditions: Dict[str, float],
    after_conditions: Dict[str, float],
    before_defect: Optional[int] = None,
    after_defect: Optional[int] = None,
) -> Dict[str, Any]:
    """기존 조건과 추천 조건의 전후 결과를 비교합니다.

    Parameters
    ----------
    before_conditions : 기존(현재) 공정 조건
    after_conditions : AI 추천 공정 조건
    before_defect / after_defect : 각 조건의 불량 유형(0~4). None 이면 규칙 기반 예측

    Returns
    -------
    dict : before / after (지표 dict), delta (DataFrame), profile plots 데이터,
           verdict 텍스트, caveat
    """
    t0 = time.perf_counter()
    if before_defect is None:
        before_defect, before_name = predict_defect_type(before_conditions)
    else:
        before_defect = int(before_defect)
        from src.data_generator import DEFECT_NAMES
        before_name = DEFECT_NAMES[before_defect]
    if after_defect is None:
        after_defect, after_name = predict_defect_type(after_conditions)
    else:
        after_defect = int(after_defect)
        from src.data_generator import DEFECT_NAMES
        after_name = DEFECT_NAMES[after_defect]

    before = compute_condition_metrics(before_conditions, before_defect)
    after = compute_condition_metrics(after_conditions, after_defect)

    rows: List[Dict[str, Any]] = []
    for key, label in METRIC_KEYS:
        b = before["metrics"][key]
        a = after["metrics"][key]
        rows.append({
            "metric": label,
            "before": b,
            "after": a,
            "delta": a - b,
            "delta_pct": (a - b) / abs(b) * 100.0 if abs(b) > 1e-9 else np.nan,
        })
    delta_df = pd.DataFrame(rows)

    sev_delta = after["metrics"]["defect_severity"] - before["metrics"]["defect_severity"]
    if sev_delta < -0.5:
        verdict = (f"시뮬레이터 상 결함 심각도가 {sev_delta:+.2f} 감소했습니다. "
                   "(동일 가정 내 결과이며 실제 불량 감소를 의미하지 않습니다.)")
    elif sev_delta > 0.5:
        verdict = (f"시뮬레이터 상 결함 심각도가 {sev_delta:+.2f} 증가했습니다. "
                   "다른 탐색 설정을 고려하세요.")
    else:
        verdict = "시뮬레이터 상 결함 심각도 변화가 미미합니다."

    return {
        "before_conditions": dict(before_conditions),
        "after_conditions": dict(after_conditions),
        "before_defect": before_defect,
        "after_defect": after_defect,
        "before_defect_name": before_name,
        "after_defect_name": after_name,
        "before": before["metrics"],
        "after": after["metrics"],
        "delta": delta_df,
        "severity_delta": float(sev_delta),
        "verdict": verdict,
        "caveat": VALIDATION_CAVEAT,
        "elapsed_sec": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------
def plot_profile_comparison(
    validation: Dict[str, Any],
    path: str | Path,
) -> Path:
    """기존/추천 조건의 2D 식각 프로파일을 나란히 비교합니다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    for ax, prof, title, color in (
        (axes[0], validation["before"], "Before (current)", "#E74C3C"),
        (axes[1], validation["after"], "After (recommended)", "#2E86C1"),
    ):
        y = np.asarray(prof.get("y_nm"))
        left = np.asarray(prof.get("left_nm"))
        right = np.asarray(prof.get("right_nm"))
        ax.plot(left, y, color=color, lw=2)
        ax.plot(right, y, color=color, lw=2)
        ax.axhline(0.0, color="#888888", lw=0.8, ls="--")
        ax.set_title(f"{title}\n({prof['defect_name']})", fontsize=11)
        ax.set_xlabel("Lateral position (nm)")
        ax.set_ylabel("Depth (nm)")
        ax.set_xlim(-700, 700)
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
    fig.suptitle("Before vs After: 2D Etching Profile", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_metric_delta(validation: Dict[str, Any], path: str | Path) -> Path:
    """주요 지표의 전후 차이(delta) 막대 그래프."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    delta_df = validation["delta"]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2E86C1" if d >= 0 else "#E74C3C" for d in delta_df["delta"]]
    ax.barh(delta_df["metric"], delta_df["delta"], color=colors, alpha=0.85)
    ax.axvline(0.0, color="#555555", lw=1)
    ax.set_xlabel("Delta (after - before)")
    ax.set_title("Validation: metric change after recommendation")
    ax.grid(alpha=0.3, axis="x")
    for i, (d, p) in enumerate(zip(delta_df["delta"], delta_df["delta_pct"])):
        if pd.notna(p):
            ax.text(d, i, f" {d:+.2f} ({p:+.1f}%)", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def run_validation(
    before_conditions: Dict[str, float],
    after_conditions: Dict[str, float],
    before_defect: Optional[int] = None,
    after_defect: Optional[int] = None,
    results_dir: str | Path = "results/automl",
) -> Dict[str, Any]:
    """전후 비교를 수행하고 그래프/CSV 를 results_dir 에 저장합니다."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_recommendation(
        before_conditions, after_conditions, before_defect, after_defect)

    profile_png = plot_profile_comparison(
        validation, results_dir / "validation_profile_comparison.png")
    delta_png = plot_metric_delta(
        validation, results_dir / "validation_metric_delta.png")
    csv_path = results_dir / "validation_metrics.csv"
    validation["delta"].to_csv(csv_path, index=False)

    validation["profile_plot"] = profile_png
    validation["delta_plot"] = delta_png
    validation["metrics_csv"] = csv_path
    return validation


if __name__ == "__main__":
    from src.etching_simulator import default_process_conditions

    base = default_process_conditions()
    out = run_validation(base, base)
    print(out["delta"].to_string(index=False))
    print(out["caveat"])
