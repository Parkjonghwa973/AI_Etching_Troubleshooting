"""
model_selector.py
=================

AutoML 이 만든 후보 모델들 중 **최적 모델을 자동 선정**하는 모듈입니다.

선정 기준
--------
1) **1차 기준**: 검증(Validation) 데이터 Macro F1-score 최대
2) 동점 또는 차이가 작으면(tie_tolerance 내) 다음을 순차적으로 고려
   - 클래스별 Recall (최저 Recall 이 높은 모델 우선 — 불량 유형 누락 방지)
   - 모델 복잡도 (낮을수록 우선)
   - 추론 시간 (빠를수록 우선)

.. note::
    선정은 **학습·검증 데이터 내부**에서만 수행하며, 최종 테스트셋은
    선정 이후 최종 1회 평가에만 사용합니다.

산출물
------
- models/best_model.pkl : 선정된 모델 (joblib)
- results/automl/model_selection.json : 선정 기준과 실제 점수
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.data_generator import DEFECT_NAMES
from src.evaluate_models import compute_metrics

# 검증 Macro F1 차이가 이 값 이내면 "동점"으로 보고 2차 기준 적용
DEFAULT_TIE_TOLERANCE = 0.01


def build_selection_table(automl_result: Dict[str, Any]) -> pd.DataFrame:
    """모델별 선정 후보 표 (검증 성능 + 2차 기준)."""
    rows: List[Dict] = []
    for name, r in automl_result["per_model"].items():
        vm = r["val_metrics"]
        recall = r.get("val_per_class_recall", {})
        rows.append({
            "model": name,
            "display": r["display"],
            "val_macro_f1": vm["macro_f1"],
            "val_accuracy": vm["accuracy"],
            "val_weighted_f1": vm["weighted_f1"],
            "worst_class_recall": float(min(recall.values())) if recall else 0.0,
            "mean_class_recall": float(np.mean(list(recall.values()))) if recall else 0.0,
            "complexity": r.get("complexity", 0),
            "inference_ms_per_100": r.get("inference_ms_per_100", 0.0),
            "fit_time_sec": r.get("fit_time_sec", 0.0),
            "opt_time_sec": r.get("opt_time_sec", 0.0),
            "cv_macro_f1": r.get("best_cv_macro_f1", float("nan")),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("val_macro_f1", ascending=False).reset_index(drop=True)


def select_best_model(
    automl_result: Dict[str, Any],
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> Dict[str, Any]:
    """검증 Macro F1 기준으로 최적 모델을 선정합니다.

    Returns
    -------
    dict : {
        "best_model_name", "best_display", "best_model",
        "selection_table"(DataFrame), "tie_break_applied", "tie_break_reason",
        "criteria", "scores"
    }
    """
    table = build_selection_table(automl_result)
    if table.empty:
        raise ValueError("선정 가능한 모델 결과가 없습니다.")

    max_f1 = float(table["val_macro_f1"].max())
    # 1차 후보군: 최고점 차이가 tie_tolerance 이내
    candidates = table[table["val_macro_f1"] >= max_f1 - tie_tolerance].copy()

    tie_break_applied = len(candidates) > 1
    reasons: List[str] = []
    if tie_break_applied:
        reasons.append(
            f"검증 Macro F1 차이 ≤ {tie_tolerance} 인 모델 {len(candidates)}개 → 2차 기준 적용")

        # 2차 기준: (1) 최저 클래스 Recall 상위, (2) 복잡도 하위, (3) 추론 시간 하위
        candidates = candidates.sort_values(
            by=["worst_class_recall", "complexity", "inference_ms_per_100"],
            ascending=[False, True, True],
            kind="mergesort").reset_index(drop=True)
        reasons.append("2차 기준 순서: 최저 클래스 Recall ↓ → 모델 복잡도 ↑ → 추론 시간 ↑")
    else:
        candidates = candidates.sort_values(
            by=["val_macro_f1"], ascending=False).reset_index(drop=True)

    winner = candidates.iloc[0]
    best_name = str(winner["model"])

    selection = {
        "best_model_name": best_name,
        "best_display": str(winner["display"]),
        "best_model": automl_result["per_model"][best_name]["model"],
        "selection_table": table,
        "tie_break_applied": bool(tie_break_applied),
        "tie_break_reason": " / ".join(reasons) if reasons else "",
        "tie_tolerance": tie_tolerance,
        "criteria": [
            "1차: 검증(Validation) Macro F1-score 최대",
            "동점(±tolerance): 최저 클래스 Recall 상위",
            "동점: 모델 복잡도 하위",
            "동점: 추론 시간 하위",
        ],
        "scores": {
            "val_macro_f1": float(winner["val_macro_f1"]),
            "val_accuracy": float(winner["val_accuracy"]),
            "worst_class_recall": float(winner["worst_class_recall"]),
            "complexity": int(winner["complexity"]),
            "inference_ms_per_100": float(winner["inference_ms_per_100"]),
        },
        "note": ("선정은 학습·검증 데이터 내부에서만 수행했으며, "
                 "최종 테스트셋은 모델 선택에 사용하지 않았습니다."),
    }
    return selection


def save_best_model(
    selection: Dict[str, Any],
    models_dir: Optional[str | Path] = None,
    results_dir: Optional[str | Path] = None,
) -> Dict[str, Path]:
    """선정 모델을 models/best_model.pkl 로, 선정 근거를 JSON 으로 저장합니다."""
    models_dir = Path(models_dir) if models_dir is not None else Path("models")
    results_dir = Path(results_dir) if results_dir is not None else Path("results/automl")
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "best_model.pkl"
    joblib.dump(selection["best_model"], model_path)

    info = {
        "best_model_name": selection["best_model_name"],
        "best_display": selection["best_display"],
        "criteria": selection["criteria"],
        "scores": selection["scores"],
        "tie_break_applied": selection["tie_break_applied"],
        "tie_break_reason": selection["tie_break_reason"],
        "note": selection["note"],
        "selection_table": selection["selection_table"].round(6).to_dict(orient="records"),
        "saved_model_path": str(model_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_path = results_dir / "model_selection.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    return {"model": model_path, "selection_json": json_path}


def load_best_model(models_dir: Optional[str | Path] = None):
    """선정된 모델을 로드합니다."""
    models_dir = Path(models_dir) if models_dir is not None else Path("models")
    path = models_dir / "best_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"선정 모델 파일이 없습니다: {path}. run_automl.py 를 실행하세요.")
    return joblib.load(path)


if __name__ == "__main__":
    print("model_selector: Streamlit/AutoML 파이프라인에서 사용됩니다. "
          "직접 실행용 모듈입니다.")
