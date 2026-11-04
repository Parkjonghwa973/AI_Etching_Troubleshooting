"""
run_automl.py
=============

전체 자동 분석 파이프라인을 한 번의 명령으로 실행합니다::

    .\\.venv\\Scripts\\python.exe run_automl.py

실행 절차 (10단계)
------------------
1. 데이터 불러오기
2. 데이터 품질 분석
3. 머신러닝 모델 자동 학습 (RF / XGBoost / Extra Trees / HistGradientBoosting)
4. 하이퍼파라미터 자동 최적화 (Optuna, 분석 모드별 횟수)
5. 모델 성능 비교
6. 최적 모델 선정 (검증 Macro F1 기준)
7. SHAP 기반 불량 원인 후보 분석 (미지원 시 permutation importance)
8. 공정 조건 최적화 (Random Search / Differential Evolution)
9. Etching 프로파일 시뮬레이션 (전후 비교)
10. 결과 보고서 생성 (HTML)

산출물은 **results/automl/** 에 저장하며 기존 results/, models/ 결과는
덮어쓰지 않습니다 (단, models/best_model.pkl 는 새로 생성).

Streamlit 의 "AI Auto Troubleshooting" 페이지가 이 스크립트를 별도 프로세스로
실행하며, --status-file 옵션으로 진행 상태 JSON 을 기록합니다.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Windows cp949 콘솔 안전 출력
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

import warnings

# ndarray 로 예측 시 발생하는 sklearn의 "feature names" 경고는 로그를 잠식하므로 숨김
warnings.filterwarnings(
    "ignore", message="X does not have valid feature names")
warnings.filterwarnings(
    "ignore", category=UserWarning, module="sklearn.utils.validation")

import numpy as np
import pandas as pd

from src.data_generator import (  # noqa: E402
    DEFECT_NAMES,
    INPUT_COLUMNS,
    N_CLASSES,
    RANDOM_STATE,
    generate_shifted_dataset,
    save_dataset,
    validate_dataset,
)
from src.etching_simulator import default_process_conditions  # noqa: E402

RESULTS_DIR = BASE_DIR / "results" / "automl"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

STAGES: List[str] = [
    "1. 데이터 불러오기",
    "2. 데이터 품질 분석",
    "3. 머신러닝 모델 자동 학습",
    "4. 하이퍼파라미터 자동 최적화",
    "5. 모델 성능 비교",
    "6. 최적 모델 선정",
    "7. SHAP 기반 불량 원인 후보 분석",
    "8. 공정 조건 최적화",
    "9. Etching 프로파일 시뮬레이션",
    "10. 결과 보고서 생성",
]


# ---------------------------------------------------------------------------
# 진행 상태 기록 (Streamlit 페이지에서 폴링)
# ---------------------------------------------------------------------------
def write_status(status_path: Path, payload: Dict[str, Any]) -> None:
    """상태 JSON 을 원자적으로 씁니다 (쓰기 중 읽기 방지: 임시파일 → rename)."""
    tmp = status_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, status_path)


class JobStatus:
    """10단계 진행 상태를 관리하고 JSON 으로 기록합니다."""

    def __init__(self, status_path: Optional[Path] = None, job_id: str = "",
                 mode: str = "fast"):
        self.path = Path(status_path) if status_path else None
        self.job_id = job_id or f"job_{int(time.time())}"
        self.mode = mode
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.t0 = time.perf_counter()
        self.payload: Dict[str, Any] = {
            "job_id": self.job_id, "mode": mode, "status": "running",
            "stage_index": 0, "stage_name": "시작 대기", "stage_progress": 0.0,
            "overall_progress": 0.0, "message": "",
            "started_at": self.started_at, "finished_at": None,
            "pid": os.getpid(), "error": None, "report_path": None,
        }
        self._flush()

    def _flush(self) -> None:
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                write_status(self.path, self.payload)
            except Exception:
                pass

    def stage(self, index: int, message: str = "", within: float = 0.0) -> None:
        """현재 단계(index: 1~10)와 단계 내 진행률(0~1)을 기록합니다."""
        index = max(1, min(len(STAGES), index))
        self.payload["stage_index"] = index
        self.payload["stage_name"] = STAGES[index - 1]
        self.payload["stage_progress"] = float(max(0.0, min(1.0, within)))
        self.payload["overall_progress"] = round(
            (index - 1 + within) / len(STAGES) * 100.0, 1)
        if message:
            self.payload["message"] = message
        self._flush()

    def log(self, message: str) -> None:
        self.payload["message"] = message
        self._flush()

    def done(self, report_path: Optional[str] = None) -> None:
        self.payload.update(
            status="done", overall_progress=100.0,
            stage_index=len(STAGES), stage_name=STAGES[-1],
            stage_progress=1.0,
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            elapsed_sec=round(time.perf_counter() - self.t0, 1),
            report_path=report_path,
        )
        self._flush()

    def fail(self, error: str) -> None:
        self.payload.update(
            status="error", error=error,
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            elapsed_sec=round(time.perf_counter() - self.t0, 1),
        )
        self._flush()


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------
def banner(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n  {title}\n{line}")


def package_versions() -> Dict[str, str]:
    from importlib import metadata
    versions = {}
    for pkg in ["numpy", "pandas", "scikit-learn", "xgboost", "shap",
                "optuna", "matplotlib", "streamlit", "plotly"]:
        try:
            versions[pkg] = metadata.version(pkg)
        except Exception:
            versions[pkg] = "N/A"
    return versions


def json_safe(obj: Any) -> Any:
    """numpy/Pandas 값을 JSON 직렬화 가능 자료형으로 변환합니다."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()
                if not k.endswith("_index") and k != "model"}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return json_safe(obj.to_dict(orient="records"))
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# SHAP / 대체 설명
# ---------------------------------------------------------------------------
def analyze_feature_effects(
    model,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_names: List[str],
    results_dir: Path,
    seed: int,
    n_repeats: int = 10,
) -> Dict[str, Any]:
    """선정 모델의 전역 중요 변수 + 불량 유형별 중요 변수 분석.

    SHAP(TreeExplainer) 우선, 실패 시 permutation importance 로 대체합니다.
    """
    method = "SHAP (TreeExplainer, mean |SHAP|)"
    try:
        from src.explain_model import compute_shap_values
        sv3, X_used = compute_shap_values(model, X_val, max_samples=300,
                                          random_state=seed)
        imp = np.abs(sv3).mean(axis=(0, 2))
        importance = pd.DataFrame({
            "feature": feature_names, "importance": imp,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        per_class_top = {}
        class_rows = []
        for cls in range(N_CLASSES):
            cls_imp = np.abs(sv3[:, :, cls]).mean(axis=0)
            order = np.argsort(cls_imp)[::-1][:3]
            top_feats = [feature_names[i] for i in order]
            per_class_top[DEFECT_NAMES[cls]] = top_feats
            class_rows.append({
                "defect_class": DEFECT_NAMES[cls],
                "top_features": ", ".join(top_feats),
                "top1_feature": top_feats[0],
                "top1_mean_abs_shap": float(cls_imp[order[0]]),
            })
        per_class_df = pd.DataFrame(class_rows)
    except Exception as exc:
        print(f"    [주의] SHAP 실패({exc}) → permutation importance 로 대체합니다.")
        method = "Permutation importance (SHAP 미지원 모델 대체)"
        from sklearn.inspection import permutation_importance
        res = permutation_importance(model, X_val, y_val, n_repeats=n_repeats,
                                     random_state=seed, n_jobs=-1)
        importance = pd.DataFrame({
            "feature": feature_names, "importance": res.importances_mean,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        per_class_top, per_class_df = {}, None

    importance_path = results_dir / "automl_feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    from src.visualization import plot_feature_importance
    plot_path = plot_feature_importance(
        importance, results_dir / "automl_feature_importance.png",
        value_column="importance",
        title=f"Feature Importance — {method}", top_n=10)

    if per_class_df is not None:
        per_class_df.to_csv(results_dir / "automl_class_top_features.csv",
                            index=False)

    return {
        "method": method,
        "importance": importance,
        "per_class_top": per_class_top,
        "per_class_table": per_class_df,
        "csv_path": importance_path,
        "plot_path": plot_path,
    }


# ---------------------------------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------------------------------
def run(
    mode: str = "fast",
    n_trials: Optional[int] = None,
    n_samples: int = 5000,
    seed: int = RANDOM_STATE,
    method: str = "de",
    n_iter: int = 400,
    current_conditions: Optional[Dict[str, float]] = None,
    status_path: Optional[Path] = None,
    job_id: str = "",
    results_dir: Path = RESULTS_DIR,
    models_dir: Path = MODELS_DIR,
    quiet: bool = False,
) -> Dict[str, Any]:
    """전체 자동 분석을 실행하고 결과 dict 를 반환합니다."""
    from src.automl_engine import assert_no_leakage, run_automl, split_train_val_test
    from src.auto_troubleshooter import auto_troubleshoot
    from src.model_selector import save_best_model, select_best_model
    from src.train_models import load_training_frame
    from src.validation_engine import run_validation

    results_dir = Path(results_dir)
    models_dir = Path(models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    status = JobStatus(status_path, job_id=job_id, mode=mode)
    t_total = time.perf_counter()
    out: Dict[str, Any] = {"mode": mode, "seed": seed}

    def say(msg: str) -> None:
        if not quiet:
            print(msg, flush=True)

    try:
        # --------------------------------------------------------------
        # 1. 데이터 불러오기
        # --------------------------------------------------------------
        status.stage(1, "데이터 불러오기 중...", within=0.1)
        banner(f"[{mode.upper()}] 1단계: 데이터 불러오기")
        from src.data_generator import default_data_path, generate_dataset

        data_path = default_data_path()
        generated = False
        if data_path.exists():
            df = load_training_frame(data_path)
            say(f"    기존 데이터 사용: {data_path} ({len(df):,}건)")
        else:
            df = generate_dataset(n_samples=n_samples, random_state=seed)
            save_dataset(df, data_path)
            generated = True
            say(f"    데이터 생성: {data_path} ({len(df):,}건)")
        out["data_path"] = str(data_path)
        out["n_samples"] = int(len(df))
        status.stage(1, f"데이터 불러오기 완료 ({len(df):,}건)", within=1.0)

        # --------------------------------------------------------------
        # 2. 데이터 품질 분석
        # --------------------------------------------------------------
        status.stage(2, "데이터 품질 분석 중...")
        banner("2단계: 데이터 품질 분석")
        ok, messages = validate_dataset(df)
        for m in messages:
            say(f"    {m}")
        missing = int(df.isna().sum().sum())
        duplicates = int(df.duplicated().sum())
        class_dist = (df["defect_type"].value_counts().sort_index()
                      if "defect_type" in df.columns else pd.Series(dtype=int))
        quality = {
            "valid": bool(ok), "messages": list(messages),
            "n_rows": int(len(df)), "n_missing": missing,
            "n_duplicates": duplicates,
            "class_distribution": {DEFECT_NAMES.get(int(k), str(k)): int(v)
                                   for k, v in class_dist.items()},
            "generated_now": generated,
        }
        if not ok:
            raise ValueError("데이터 검증 실패: " + " / ".join(messages))
        say(f"    결측치 {missing}건 / 중복 {duplicates}건 / 검증 {'통과' if ok else '실패'}")
        status.stage(2, "데이터 품질 분석 완료", within=1.0)

        # --------------------------------------------------------------
        # 3~4. 모델 자동 학습 + 하이퍼파라미터 최적화 (내부 CV: 60/20/20 분할)
        # --------------------------------------------------------------
        banner("3~4단계: 모델 자동 학습 및 Optuna 하이퍼파라미터 최적화")
        say(f"    분석 모드: {mode} (모델별 {n_trials or {'fast': 5, 'normal': 15, 'precise': 30}[mode]}회)")
        splits = split_train_val_test(df, random_state=seed)
        assert_no_leakage(splits)
        say(f"    분할: 학습 {splits['sizes']['train']:,} / "
            f"검증 {splits['sizes']['val']:,} / 테스트 {splits['sizes']['test']:,} "
            f"(테스트셋은 모델 선정에 사용하지 않음)")
        out["split_sizes"] = splits["sizes"]
        out["split_ratio"] = splits["ratio"]

        half = None

        def progress_cb(stage: str, cur: int, total: int, message: str) -> None:
            nonlocal half
            if half is None:
                half = max(total // 2, 1)
            frac = cur / max(total, 1)
            # 학습(3단계)과 최적화(4단계)를 단계 경계에 따라 표시
            stage_index = 3 if cur <= half else 4
            within = (cur / half if cur <= half else (cur - half) / max(total - half, 1))
            status.stage(stage_index, message, within=within)
            say(f"    [{message}] ({cur}/{total})")

        automl_result = run_automl(splits, mode=mode, n_trials=n_trials,
                                   seed=seed, progress_cb=progress_cb)
        status.stage(4, "하이퍼파라미터 최적화 완료", within=1.0)

        # --------------------------------------------------------------
        # 5. 모델 성능 비교
        # --------------------------------------------------------------
        status.stage(5, "모델 성능 비교 중...")
        banner("5단계: 모델 성능 비교 (검증셋 기준)")
        leaderboard = automl_result["leaderboard"]
        say(leaderboard[["model", "cv_macro_f1", "val_accuracy", "val_macro_f1",
                         "worst_class_recall", "fit_time_sec", "opt_time_sec"]]
            .round(4).to_string(index=False))
        cmp_path = results_dir / "automl_model_comparison.csv"
        leaderboard.to_csv(cmp_path, index=False)
        out["model_comparison"] = leaderboard
        status.stage(5, "모델 성능 비교 완료", within=1.0)

        # --------------------------------------------------------------
        # 6. 최적 모델 선정
        # --------------------------------------------------------------
        status.stage(6, "최적 모델 선정 중...")
        banner("6단계: 최적 모델 선정 (검증 Macro F1 기준)")
        selection = select_best_model(automl_result)
        saved = save_best_model(selection, models_dir=models_dir,
                                results_dir=results_dir)
        say(f"    선정 모델: {selection['best_display']} "
            f"(검증 Macro F1={selection['scores']['val_macro_f1']:.4f})")
        if selection["tie_break_applied"]:
            say(f"    동점 처리: {selection['tie_break_reason']}")
        say(f"    저장: {saved['model']}")
        out["best_model_name"] = selection["best_model_name"]
        status.stage(6, f"최적 모델 선정: {selection['best_display']}", within=1.0)

        best_model = selection["best_model"]

        # --------------------------------------------------------------
        # 6b. 최종 테스트셋 평가 (선정 모델 전용, 1회)
        # --------------------------------------------------------------
        from src.automl_engine import evaluate_classifier
        test_eval = evaluate_classifier(best_model, splits["X_test"],
                                        splits["y_test"])
        say(f"    최종 테스트셋 (선정 모델): Accuracy={test_eval['metrics']['accuracy']:.4f}, "
            f"Macro F1={test_eval['metrics']['macro_f1']:.4f}")
        out["test_metrics"] = test_eval["metrics"]
        test_eval["per_class"].to_csv(results_dir / "automl_test_per_class.csv",
                                      index=False)

        # 분포 변화 데이터: 강건성 평가 전용 (모델 선정 미사용)
        try:
            df_shift = generate_shifted_dataset(
                n_samples=min(max(len(df) // 5, 500), 2000),
                random_state=seed + 100)
            X_shift = df_shift[automl_result["splits"]["feature_names"]].astype(float)
            y_shift = df_shift["defect_type"].astype(int).to_numpy()
            shift_eval = evaluate_classifier(best_model, X_shift, y_shift)
            out["robustness_shift"] = shift_eval["metrics"]
            say(f"    분포 변화 평가 (강건성): Accuracy={shift_eval['metrics']['accuracy']:.4f}, "
                f"Macro F1={shift_eval['metrics']['macro_f1']:.4f}")
        except Exception as exc:
            say(f"    [주의] 분포 변화 평가 실패: {exc}")
            out["robustness_shift"] = None

        # --------------------------------------------------------------
        # 7. SHAP 기반 불량 원인 후보 분석
        # --------------------------------------------------------------
        status.stage(7, "SHAP 분석 중...")
        banner("7단계: SHAP 기반 불량 원인 후보 분석 (선정 모델)")
        shap_info = analyze_feature_effects(
            best_model, splits["X_val"], splits["y_val"],
            list(splits["feature_names"]), results_dir, seed)
        say(f"    설명 방법: {shap_info['method']}")
        say("    ※ 모델 예측에 대한 설명이며 물리적 인과관계를 증명하지 않습니다.")
        status.stage(7, f"SHAP 분석 완료 ({shap_info['method']})", within=1.0)

        # --------------------------------------------------------------
        # 8. 공정 조건 최적화 (AI 자동 트러블슈팅)
        # --------------------------------------------------------------
        status.stage(8, "공정 조건 최적화 중...")
        banner("8단계: 공정 조건 최적화 (Random Search / Differential Evolution)")
        current = dict(current_conditions or default_process_conditions())
        train_stats = None
        try:
            from src.process_optimizer import train_distribution_stats
            train_stats = train_distribution_stats(splits["X_train"])
        except Exception as exc:
            say(f"    [주의] 학습 분포 통계 계산 실패: {exc}")

        trouble = auto_troubleshoot(
            best_model, current, method=method, n_iter=n_iter,
            train_stats=train_stats, random_state=seed,
            feature_names=list(splits["feature_names"]))
        say(trouble["comparison"].to_string(index=False))
        say(f"    정상 확률: {trouble['normal_prob_before']:.4f} → "
            f"{trouble['normal_prob_after']:.4f} ({trouble['improvement']['level']})")
        trouble["comparison"].to_csv(results_dir / "automl_recommendation.csv",
                                     index=False)
        status.stage(8, "공정 조건 최적화 완료", within=1.0)

        # --------------------------------------------------------------
        # 9. Etching 프로파일 시뮬레이션 (전후 비교)
        # --------------------------------------------------------------
        status.stage(9, "Etching 프로파일 시뮬레이션 중...")
        banner("9단계: Etching 프로파일 시뮬레이션 (전후 비교)")
        rec = trouble["recommendation"]
        validation = run_validation(
            rec["original"], rec["recommended"],
            before_defect=trouble["recommendation"]["defect_before_id"],
            after_defect=trouble["recommendation"]["defect_after_id"],
            results_dir=results_dir)
        say(validation["delta"].round(2).to_string(index=False))
        say(f"    {validation['verdict']}")
        status.stage(9, "프로파일 시뮬레이션 완료", within=1.0)

        # --------------------------------------------------------------
        # 10. 결과 보고서 생성
        # --------------------------------------------------------------
        status.stage(10, "HTML 보고서 생성 중...")
        banner("10단계: 자동 보고서 생성 (HTML)")
        from src.report_generator import generate_html_report

        opt_summary = pd.DataFrame([
            {
                "model": name,
                "display": r["display"],
                "baseline_cv_macro_f1": r["baseline_cv_macro_f1"],
                "best_cv_macro_f1": r["best_cv_macro_f1"],
                "cv_improvement": r["cv_improvement"],
                "best_params": json.dumps(
                    {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                     for k, v in r["best_params"].items()}),
                "opt_time_sec": r["opt_time_sec"],
                "fit_time_sec": r["fit_time_sec"],
            }
            for name, r in automl_result["per_model"].items()
        ])
        opt_summary.to_csv(results_dir / "automl_optimization_summary.csv",
                           index=False)

        runtime_sec = time.perf_counter() - t_total
        metadata = {
            "실행 시각": time.strftime("%Y-%m-%d %H:%M:%S"),
            "실행 시간(초)": round(runtime_sec, 1),
            "분석 모드": mode,
            "모델별 최적화 횟수": automl_result["trials_per_model"],
            "CV 폴드 수": automl_result["n_splits"],
            "random_state": seed,
            "데이터": f"{out['n_samples']:,}건 ({out['data_path']})",
            "공정 조건 탐색": f"{METHOD_DISPLAY.get(method, method)} (n_iter={n_iter})",
            "Python": platform.python_version(),
            "플랫폼": platform.platform(),
        }
        metadata.update({f"패키지: {k}": v for k, v in package_versions().items()})

        report_data = {
            "metadata": metadata,
            "data": {
                "전체 샘플 수": out["n_samples"],
                "학습": splits["sizes"]["train"],
                "검증": splits["sizes"]["val"],
                "테스트": splits["sizes"]["test"],
                "학습 비율": f"{splits['ratio']['train'] * 100:.1f}%",
                "검증 비율": f"{splits['ratio']['val'] * 100:.1f}%",
                "테스트 비율": f"{splits['ratio']['test'] * 100:.1f}%",
                "검증 통과": "예" if quality["valid"] else "아니오",
                "결측치": missing,
                "중복행": duplicates,
                "클래스 분포": json.dumps(quality["class_distribution"],
                                          ensure_ascii=False),
            },
            "model_comparison": leaderboard.round(6),
            "selection": selection,
            "optimization": opt_summary,
            "shap": shap_info,
            "troubleshooting": {
                "comparison": trouble["comparison"].round(6),
                "summary": trouble["summary"].round(6),
                "improvement": trouble["improvement"],
                "warnings": trouble["warnings"],
            },
            "validation": validation,
        }
        report_path = generate_html_report(
            report_data, results_dir / "automl_report.html")
        say(f"    보고서 저장: {report_path}")

        # 실행 메타데이터 (재현성)
        run_meta = {
            "runtime_sec": round(runtime_sec, 1),
            "mode": mode,
            "trials_per_model": automl_result["trials_per_model"],
            "n_splits": automl_result["n_splits"],
            "random_state": seed,
            "data_path": str(data_path),
            "n_samples": out["n_samples"],
            "split_sizes": splits["sizes"],
            "package_versions": package_versions(),
            "model_settings": {
                "candidates": automl_result["model_names"],
                "objective": "CV Macro F1 (StratifiedKFold)",
                "early_stopping": ["xgboost", "hist_gradient_boosting"],
                "search_method_for_conditions": method,
                "n_iter_for_conditions": n_iter,
            },
            "best_model": selection["best_model_name"],
            "best_params": automl_result["per_model"][
                selection["best_model_name"]]["best_params"],
            "python": platform.python_version(),
            "platform": platform.platform(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(results_dir / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(json_safe(run_meta), f, ensure_ascii=False, indent=2)

        # 전체 결과 JSON (숫자 위주)
        results_json = {
            "mode": mode, "seed": seed,
            "split_sizes": splits["sizes"],
            "model_comparison": json_safe(leaderboard),
            "best_model": selection["best_model_name"],
            "selection_scores": json_safe(selection["scores"]),
            "test_metrics": json_safe(out["test_metrics"]),
            "robustness_shift": json_safe(out.get("robustness_shift")),
            "recommendation": {
                "original": json_safe(rec["original"]),
                "recommended": json_safe(rec["recommended"]),
                "defect_before": rec["defect_before"],
                "defect_after": rec["defect_after"],
                "normal_prob_before": float(trouble["normal_prob_before"]),
                "normal_prob_after": float(trouble["normal_prob_after"]),
            },
            "validation_delta": json_safe(validation["delta"]),
            "elapsed_sec": round(runtime_sec, 1),
        }
        with open(results_dir / "automl_results.json", "w", encoding="utf-8") as f:
            json.dump(results_json, f, ensure_ascii=False, indent=2)

        status.stage(10, "보고서 생성 완료", within=1.0)
        status.done(report_path=str(report_path))

        out.update({
            "automl_result": automl_result, "selection": selection,
            "shap_info": shap_info, "trouble": trouble,
            "validation": validation, "report_path": report_path,
            "results_dir": results_dir, "runtime_sec": runtime_sec,
        })
        banner(f"전체 자동 분석 완료 — {runtime_sec:.1f}초 · 보고서: {report_path}")
        return out

    except Exception as exc:
        status.fail(f"{exc}")
        print("\n[오류] 자동 분석 실패:", exc, file=sys.stderr)
        traceback.print_exc()
        raise


METHOD_DISPLAY = {"random": "Random Search", "de": "Differential Evolution"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI AutoML Etching Troubleshooting 자동 분석 실행")
    p.add_argument("--mode", choices=["fast", "normal", "precise"],
                   default="fast", help="분석 모드 (모델별 최적화 횟수)")
    p.add_argument("--trials", type=int, default=None,
                   help="모델별 Optuna 최적화 횟수 (mode 프리셋 덮어씀)")
    p.add_argument("--samples", type=int, default=5000,
                   help="데이터가 없을 때 생성할 샘플 수")
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--method", choices=["random", "de"], default="de",
                   help="공정 조건 탐색 방법")
    p.add_argument("--n-iter", type=int, default=400,
                   help="공정 조건 탐색 횟수")
    p.add_argument("--gas", type=float, default=None, help="현재 gas_chemistry")
    p.add_argument("--flow", type=float, default=None, help="현재 gas_flow_sccm")
    p.add_argument("--pressure", type=float, default=None, help="현재 pressure_mtorr")
    p.add_argument("--bias", type=float, default=None, help="현재 bias_power_w")
    p.add_argument("--source", type=float, default=None, help="현재 source_power_w")
    p.add_argument("--temp", type=float, default=None, help="현재 temperature_c")
    p.add_argument("--status-file", type=str, default=None,
                   help="진행 상태 JSON 경로 (Streamlit 페이지용)")
    p.add_argument("--job-id", type=str, default="")
    p.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    p.add_argument("--models-dir", type=str, default=str(MODELS_DIR))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    current = default_process_conditions()
    overrides = {
        "gas_chemistry": args.gas, "gas_flow_sccm": args.flow,
        "pressure_mtorr": args.pressure, "bias_power_w": args.bias,
        "source_power_w": args.source, "temperature_c": args.temp,
    }
    for k, v in overrides.items():
        if v is not None:
            current[k] = float(v)

    try:
        run(
            mode=args.mode, n_trials=args.trials, n_samples=args.samples,
            seed=args.seed, method=args.method, n_iter=args.n_iter,
            current_conditions=current,
            status_path=Path(args.status_file) if args.status_file else None,
            job_id=args.job_id,
            results_dir=Path(args.results_dir),
            models_dir=Path(args.models_dir),
        )
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
