"""
visualization.py
================

발표에 바로 활용할 수 있는 그래프를 만드는 모듈입니다.

디자인 규칙
----------
- 흰색 배경
- 파란색 계열 강조색
- 명확한 제목 / 축 단위 표기
- 고해상도(200 dpi) PNG 저장
- 색상만으로 구분하지 않도록 범례·값 라벨 추가
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # 비대화형 백엔드 (Windows/서버 모두 안전)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_generator import DEFECT_NAMES

# 파란색 계열 팔레트
BLUE_PALETTE = ["#0B3C5D", "#1F77B4", "#4C9BE8", "#9ECBEF", "#D6EAF8"]
ACCENT = "#1F77B4"
ACCENT_DARK = "#0B3C5D"
ALERT = "#C0392B"
CLASS_COLORS = ["#2E86C1", "#E74C3C", "#8E44AD", "#E67E22", "#16A085"]

MODEL_DISPLAY = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost (baseline)",
    "optimized_xgboost": "XGBoost (Optuna optimized)",
}


def setup_style() -> None:
    """전역 matplotlib 스타일을 설정합니다."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": "#CCCCCC",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.unicode_minus": False,
        "font.sans-serif": ["Malgun Gothic", "Microsoft YaHei", "SimHei",
                            "DejaVu Sans", "Arial Unicode MS"],
    })


setup_style()


def save_figure(fig: "plt.Figure", path: str | Path) -> Path:
    """PNG로 저장하고 경로를 반환합니다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    [그래프 저장] {path}")
    return path


# ---------------------------------------------------------------------------
# 1) 모델 성능 비교 (Accuracy / Macro F1)
# ---------------------------------------------------------------------------
def _model_order(models: Sequence[str]) -> List[str]:
    order = ["random_forest", "xgboost", "optimized_xgboost"]
    ordered = [m for m in order if m in set(models)]
    ordered += [m for m in models if m not in ordered]
    return ordered


def plot_model_metric_comparison(
    results: pd.DataFrame,
    metric_cols: Sequence[str] = ("accuracy", "macro_f1"),
    dataset: str = "test",
    path: str | Path = "results/model_comparison.png",
) -> Path:
    """모델 Accuracy / Macro F1 비교 막대 그래프."""
    df = results[results["dataset"] == dataset].copy()
    models = _model_order(df["model"].tolist())
    df = df.set_index("model").loc[models]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(models))
    width = 0.36
    labels = {"accuracy": "Accuracy", "macro_f1": "Macro F1-score"}
    for i, (col, c) in enumerate(zip(metric_cols, [ACCENT_DARK, ACCENT])):
        vals = df[col].values
        bars = ax.bar(x + (i - 0.5) * width, vals, width,
                      label=labels.get(col, col), color=c, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score (0 ~ 1)")
    title_ds = "Test Set" if dataset == "test" else "Distribution-Shift Set"
    ax.set_title(f"Model Performance Comparison ({title_ds})")
    ax.legend(loc="lower right", frameon=True)
    ax.axhline(1.0, color="#999999", linewidth=0.8, linestyle="--")
    return save_figure(fig, path)


def plot_accuracy_comparison(results: pd.DataFrame,
                             path: str | Path = "results/accuracy_comparison.png") -> Path:
    """1) Model Accuracy Comparison (요구 그래프 1)."""
    return plot_model_metric_comparison(results, ("accuracy",), "test", path)


def plot_macro_f1_comparison(results: pd.DataFrame,
                             path: str | Path = "results/macro_f1_comparison.png") -> Path:
    """2) Macro F1-score Comparison (요구 그래프 2)."""
    return plot_model_metric_comparison(results, ("macro_f1",), "test", path)


# ---------------------------------------------------------------------------
# 3) Confusion Matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str = "optimized_xgboost",
    path: str | Path = "results/confusion_matrix.png",
    dataset: str = "test",
) -> Path:
    """3) Confusion Matrix 히트맵 (요구 그래프 3)."""
    class_names = [f"{i}: {DEFECT_NAMES[i]}" for i in range(cm.shape[0])]
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 10})
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    title_ds = "Test Set" if dataset == "test" else "Distribution-Shift Set"
    ax.set_title(f"Confusion Matrix — {MODEL_DISPLAY.get(model_name, model_name)} ({title_ds})")
    fig.tight_layout()
    return save_figure(fig, path)


def plot_per_class_recall(
    per_class: pd.DataFrame,
    path: str | Path = "results/per_class_recall.png",
) -> Path:
    """4) Class별 Recall 비교 (요구 그래프 4).

    per_class columns: model, dataset, class, recall
    """
    df = per_class[per_class["dataset"] == "test"].copy()
    models = _model_order(df["model"].unique())
    classes = sorted(df["class"].unique())

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(classes))
    width = 0.8 / max(len(models), 1)
    colors = [ACCENT_DARK, ACCENT, "#9ECBEF"]
    for i, m in enumerate(models):
        sub = df[df["model"] == m].set_index("class").reindex(classes)
        vals = sub["recall"].values
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width,
                      label=MODEL_DISPLAY.get(m, m),
                      color=colors[i % len(colors)], edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}: {DEFECT_NAMES[int(c)]}" for c in classes], fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall (0 ~ 1)")
    ax.set_title("Per-class Recall Comparison (Test Set)")
    ax.legend(loc="upper right", frameon=True)
    return save_figure(fig, path)


# ---------------------------------------------------------------------------
# 5) Optimization History
# ---------------------------------------------------------------------------
def plot_optimization_history(
    trial_values: Sequence[float],
    baseline_score: Optional[float] = None,
    path: str | Path = "results/optimization_history.png",
) -> Path:
    """5) Optuna 최적화 이력 (요구 그래프 5)."""
    values = np.asarray(trial_values, dtype=float)
    trials = np.arange(1, len(values) + 1)
    best_so_far = np.maximum.accumulate(values)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(trials, values, s=28, color="#9ECBEF", alpha=0.75,
               label="Trial score (CV Macro F1)", edgecolors="white")
    ax.plot(trials, best_so_far, color=ACCENT_DARK, linewidth=2,
            label="Best score so far")
    if baseline_score is not None:
        ax.axhline(baseline_score, color=ALERT, linestyle="--", linewidth=1.6,
                   label=f"Baseline XGBoost CV F1 ({baseline_score:.3f})")
    ax.set_xlabel("Optuna trial number")
    ax.set_ylabel("CV Macro F1-score")
    ax.set_title("Hyperparameter Optimization History (Optuna)")
    ax.legend(loc="lower right", frameon=True)
    ax.set_ylim(min(values.min(), 0) - 0.02, min(values.max() + 0.05, 1.0))
    return save_figure(fig, path)


# ---------------------------------------------------------------------------
# Feature importance / SHAP
# ---------------------------------------------------------------------------
def plot_feature_importance(
    importance: pd.DataFrame,
    path: str | Path = "results/feature_importance.png",
    value_column: str = "mean_abs_shap",
    title: str = "Global Feature Importance (mean |SHAP value|)",
    top_n: int = 12,
) -> Path:
    """Feature importance 수평 막대 그래프."""
    df = importance.sort_values(value_column, ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [ACCENT if i >= len(df) - 3 else "#9ECBEF" for i in range(len(df))]
    bars = ax.barh(df["feature"], df[value_column], color=colors, edgecolor="white")
    for b, v in zip(bars, df[value_column]):
        ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.3f}",
                va="center", fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_shap_summary(
    shap_values_3d: np.ndarray,
    X: pd.DataFrame,
    class_index: int,
    path: str | Path,
    max_display: int = 10,
) -> Optional[Path]:
    """SHAP summary(beeswarm) plot. SHAP 이 설치되어 있으면 그립니다."""
    try:
        import shap as shap_lib
    except Exception as exc:  # pragma: no cover
        print(f"    [WARN] SHAP 라이브러리를 불러오지 못했습니다: {exc}")
        return None
    sv = shap_lib.Explanation(
        values=shap_values_3d[:, :, class_index],
        base_values=np.zeros(shap_values_3d.shape[0]),  # shap>=0.45: None 방지
        data=X.to_numpy(),
        feature_names=list(X.columns),
    )
    fig = plt.figure()
    shap_lib.summary_plot(sv, show=False, max_display=max_display)
    plt.title(f"SHAP Summary — class {class_index}: {DEFECT_NAMES[class_index]}")
    plt.xlabel("SHAP value (impact on model output)")
    return save_figure(plt.gcf(), path)


def plot_local_contribution(
    contribution: pd.DataFrame,
    path: str | Path,
    title: str,
) -> Path:
    """개별 샘플의 변수별 SHAP 기여도 수평 막대 그래프."""
    df = contribution.sort_values("abs_shap", ascending=True)
    colors = [ALERT if s > 0 else "#2E86C1" for s in df["shap_value"]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(df["feature"], df["shap_value"], color=colors, edgecolor="white")
    for b, v in zip(bars, df["shap_value"]):
        ax.text(v + (0.002 if v >= 0 else -0.002), b.get_y() + b.get_height() / 2,
                f" {v:+.3f}", va="center", fontsize=9,
                ha="left" if v >= 0 else "right")
    ax.set_xlabel("SHAP value (pushes prediction toward this class if > 0)")
    ax.set_title(title)
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save_figure(fig, path)


# ---------------------------------------------------------------------------
# 식각 프로파일
# ---------------------------------------------------------------------------
def plot_etching_profiles(
    profiles: Dict[str, Dict[str, object]],
    path: str | Path = "results/etching_profiles.png",
    title: str = "Etching Profile Comparison (Educational Model)",
) -> Path:
    """5가지 불량 유형의 2D 식각 단면 비교."""
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for i, (name, p) in enumerate(profiles.items()):
        y = p["y_nm"]
        ax.plot(p["right_nm"], y, color=CLASS_COLORS[i % len(CLASS_COLORS)],
                linewidth=2, label=name)
        ax.plot(p["left_nm"], y, color=CLASS_COLORS[i % len(CLASS_COLORS)],
                linewidth=2)
    ax.axhline(0, color="#666666", linewidth=1.0, linestyle="--", label="Wafer surface (y = 0)")
    ax.set_xlabel("Lateral position (nm)")
    ax.set_ylabel("Depth (nm, negative = down)")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True, title="Defect type")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return save_figure(fig, path)


def plot_profile_single(
    profile: Dict[str, object],
    path: str | Path,
    title: Optional[str] = None,
    reference: Optional[Dict[str, object]] = None,
) -> Path:
    """단일 프로파일 (선택적으로 Normal 기준 형상과 비교)."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    y = profile["y_nm"]
    ax.fill_betweenx(y, profile["left_nm"], profile["right_nm"],
                     color="#D6EAF8", alpha=0.9, label="Etched region")
    ax.plot(profile["right_nm"], y, color=ACCENT_DARK, linewidth=2,
            label=f"Predicted: {profile['defect_name']}")
    ax.plot(profile["left_nm"], y, color=ACCENT_DARK, linewidth=2)
    if reference is not None:
        ax.plot(reference["right_nm"], reference["y_nm"], color=ALERT,
                linewidth=1.8, linestyle="--", label="Normal reference")
        ax.plot(reference["left_nm"], reference["y_nm"], color=ALERT,
                linewidth=1.8, linestyle="--")
    ax.axhline(0, color="#666666", linewidth=1.0, linestyle=":", label="Wafer surface")
    ax.set_xlabel("Lateral position (nm)")
    ax.set_ylabel("Depth (nm, negative = down)")
    ax.set_title(title or f"Etching Profile — {profile['defect_name']} (Educational Model)")
    ax.legend(loc="upper right", frameon=True)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return save_figure(fig, path)


def plot_before_after_profiles(
    before: Dict[str, object],
    after: Dict[str, object],
    path: str | Path,
    title: str = "Before / After Optimization — Etching Profile",
) -> Path:
    """공정 조건 최적화 전후 프로파일 비교."""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(before["right_nm"], before["y_nm"], color=ALERT, linewidth=2.2,
            label=f"Before ({before['defect_name']})")
    ax.plot(before["left_nm"], before["y_nm"], color=ALERT, linewidth=2.2)
    ax.plot(after["right_nm"], after["y_nm"], color=ACCENT_DARK, linewidth=2.2,
            linestyle="--", label=f"After ({after['defect_name']})")
    ax.plot(after["left_nm"], after["y_nm"], color=ACCENT_DARK, linewidth=2.2,
            linestyle="--")
    ax.axhline(0, color="#666666", linewidth=1.0, linestyle=":", label="Wafer surface")
    ax.set_xlabel("Lateral position (nm)")
    ax.set_ylabel("Depth (nm, negative = down)")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return save_figure(fig, path)


# ---------------------------------------------------------------------------
# 예측 확률 / 기타
# ---------------------------------------------------------------------------
def plot_prediction_probabilities(
    probabilities: Sequence[float],
    path: str | Path,
    title: str = "Predicted Defect Probabilities",
) -> Path:
    """불량 유형별 예측 확률 막대 그래프."""
    names = [DEFECT_NAMES[i] for i in range(len(probabilities))]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, np.asarray(probabilities), color=CLASS_COLORS[:len(names)],
                  edgecolor="white")
    for b, v in zip(bars, probabilities):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Predicted probability (0 ~ 1)")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_ground_truth_vs_shap(
    comparison: pd.DataFrame,
    path: str | Path = "results/shap_vs_generation_rules.png",
) -> Path:
    """SHAP 중요도와 실제 데이터 생성 규칙 변수의 비교 그래프."""
    df = comparison.sort_values("mean_abs_shap", ascending=True)
    colors = ["#1F77B4" if f else "#BDC3C7" for f in df["in_generation_rules"]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(df["feature"], df["mean_abs_shap"], color=colors, edgecolor="white",
            label="mean |SHAP|")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("SHAP Importance vs. Data-generation Rules")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#1F77B4"),
        plt.Rectangle((0, 0), 1, 1, color="#BDC3C7"),
    ]
    ax.legend(handles, ["Used in generation rules", "Not explicitly used"],
              loc="lower right", frameon=True)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save_figure(fig, path)
