"""
sensitivity_analysis.py
=======================

6개 공정 변수에 대한 **One-At-A-Time (OAT) 민감도 분석**을 수행합니다.

- 다른 변수는 기준 조건으로 고정
- 한 변수만 PROCESS_BOUNDS 범위에서 변화
- 식각 시뮬레이터로 Etch depth / Top·Middle·Bottom CD / Sidewall angle /
  Defect severity 를 계산
- 변수별 민감도 지수(정규화된 출력 변화량)와 순위를 산출
- 2차원(상위 2변수) 교호작용 히트맵 생성

모든 수치는 합성 공정 모델(`etching_simulator` + `data_generator`)의 실제 출력값입니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_generator import INPUT_COLUMNS, PROCESS_BOUNDS
from src.etching_simulator import default_process_conditions
from src.validation_engine import compute_condition_metrics

DEFAULT_N_POINTS = 30
METRIC_COLS = [
    "etch_depth_nm", "top_cd_nm", "middle_cd_nm", "bottom_cd_nm",
    "sidewall_angle_deg", "defect_severity",
]
METRIC_LABELS = {
    "etch_depth_nm": "Etch depth (nm)",
    "top_cd_nm": "Top CD (nm)",
    "middle_cd_nm": "Middle CD (nm)",
    "bottom_cd_nm": "Bottom CD (nm)",
    "sidewall_angle_deg": "Sidewall angle (deg)",
    "defect_severity": "Defect severity (0-100)",
}


def variable_range(var: str, n_points: int = DEFAULT_N_POINTS) -> np.ndarray:
    """변수 범위를 균등 샘플링합니다."""
    lo, hi = PROCESS_BOUNDS[var]
    return np.linspace(lo, hi, n_points)


def evaluate_condition(conditions: Dict[str, float]) -> Dict[str, float]:
    """단일 조건의 6가지 지표를 계산합니다."""
    out = compute_condition_metrics(conditions)
    return {m: float(out["metrics"][m]) for m in METRIC_COLS}


def sensitivity_one_variable(
    base_conditions: Dict[str, float],
    var: str,
    n_points: int = DEFAULT_N_POINTS,
) -> pd.DataFrame:
    """한 변수를 변화시키며 OAT 민감도를 계산합니다."""
    base = dict(base_conditions)
    values = variable_range(var, n_points)
    rows: List[Dict] = []
    for v in values:
        cond = dict(base)
        cond[var] = float(v)
        metrics = evaluate_condition(cond)
        rows.append({"variable": var, "value": float(v), **metrics})
    return pd.DataFrame(rows)


def run_full_sensitivity(
    base_conditions: Optional[Dict[str, float]] = None,
    n_points: int = DEFAULT_N_POINTS,
    results_dir: str | Path = "results/research_validation",
) -> Dict[str, object]:
    """6개 변수 전체에 대해 OAT 민감도 분석을 수행하고 CSV/PNG 를 저장합니다."""
    base = dict(base_conditions or default_process_conditions())
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_frames: List[pd.DataFrame] = []
    scores: Dict[str, float] = {}

    for var in INPUT_COLUMNS:
        df = sensitivity_one_variable(base, var, n_points)
        all_frames.append(df)
        # 민감도 점수: 각 지표를 정규화([0,1])한 뒤 표준편차의 평균
        norm = df[METRIC_COLS].apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9))
        scores[var] = float(norm.std().mean())

    full_df = pd.concat(all_frames, ignore_index=True)
    full_df.to_csv(results_dir / "sensitivity_oat.csv", index=False)

    # 변수별 민감도 순위표
    summary = pd.DataFrame([
        {"variable": v, "sensitivity_score": scores[v],
         "rank": i + 1}
        for i, (v, _) in enumerate(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))
    ])
    summary.to_csv(results_dir / "sensitivity_ranking.csv", index=False)

    # 변수별 그래프
    for var in INPUT_COLUMNS:
        plot_sensitivity_curve(full_df, var, results_dir / f"sensitivity_{var}.png")

    # 순위 막대 그래프
    plot_ranking(summary, results_dir / "sensitivity_ranking.png")

    # 상위 2변수 교호작용 히트맵
    top2 = summary.sort_values("rank").head(2)["variable"].tolist()
    interaction_png = plot_interaction_heatmap(
        base, top2[0], top2[1], n_points=min(n_points, 25),
        path=results_dir / "sensitivity_interaction_top2.png")

    return {
        "base_conditions": base,
        "full_df": full_df,
        "summary": summary,
        "scores": scores,
        "results_dir": results_dir,
        "interaction_png": interaction_png,
    }


def plot_sensitivity_curve(
    full_df: pd.DataFrame,
    var: str,
    path: str | Path,
) -> Path:
    """변수 하나에 대한 6지표 변화 곡선."""
    path = Path(path)
    df = full_df[full_df["variable"] == var]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for ax, metric in zip(axes, METRIC_COLS):
        ax.plot(df["value"], df[metric], lw=2, color="#2E86C1")
        ax.set_xlabel(var)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(alpha=0.3)
    fig.suptitle(f"OAT Sensitivity: {var}", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_ranking(summary: pd.DataFrame, path: str | Path) -> Path:
    """변수별 민감도 점수 막대 그래프."""
    path = Path(path)
    df = summary.sort_values("sensitivity_score", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["variable"], df["sensitivity_score"], color="#2E86C1", edgecolor="white")
    ax.set_xlabel("Normalized sensitivity score")
    ax.set_title("Variable Sensitivity Ranking (simulator output)")
    ax.grid(alpha=0.3, axis="x")
    for i, (v, s) in enumerate(zip(df["variable"], df["sensitivity_score"])):
        ax.text(s, i, f" {s:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_interaction_heatmap(
    base_conditions: Dict[str, float],
    var1: str,
    var2: str,
    n_points: int = 20,
    metric: str = "defect_severity",
    path: str | Path = "results/research_validation/sensitivity_interaction.png",
) -> Path:
    """두 변수의 2D 그리드로 교호작용(비선형/결합 효과)을 시각화합니다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    v1_vals = variable_range(var1, n_points)
    v2_vals = variable_range(var2, n_points)
    Z = np.zeros((len(v2_vals), len(v1_vals)))
    for i, v2 in enumerate(v2_vals):
        for j, v1 in enumerate(v1_vals):
            cond = dict(base_conditions)
            cond[var1] = float(v1)
            cond[var2] = float(v2)
            Z[i, j] = compute_condition_metrics(cond)["metrics"][metric]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.contourf(v1_vals, v2_vals, Z, levels=20, cmap="viridis")
    fig.colorbar(im, ax=ax, label=METRIC_LABELS[metric])
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.set_title(f"Interaction: {var1} × {var2} on {METRIC_LABELS[metric]}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


if __name__ == "__main__":
    out = run_full_sensitivity()
    print(out["summary"].to_string(index=False))
