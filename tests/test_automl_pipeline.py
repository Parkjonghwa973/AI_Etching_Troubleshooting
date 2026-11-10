"""
tests/test_automl_pipeline.py
=============================

AI 자동화(AutoML / 자동 트러블슈팅 / 자동 보고서 / Streamlit 페이지) 기능에 대한 테스트.

기존 테스트(test_data_generator / test_etching_simulator / test_ml_pipeline)는
건드리지 않고 새 기능만 검증합니다. 작은 데이터와 적은 최적화 횟수를 사용합니다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data_generator import (  # noqa: E402
    INPUT_COLUMNS,
    N_CLASSES,
    PROCESS_BOUNDS,
    generate_dataset,
)
from src.etching_simulator import default_process_conditions  # noqa: E402

SEED = 42


# ---------------------------------------------------------------------------
# 공통 fixture (모듈 범위: AutoML 실행 비용을 1회로 제한)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    return generate_dataset(n_samples=300, random_state=SEED)


@pytest.fixture(scope="module")
def splits(small_df):
    from src.automl_engine import split_train_val_test
    return split_train_val_test(small_df, random_state=SEED)


@pytest.fixture(scope="module")
def automl_result(splits):
    """작은 데이터 + 모델별 최적화 1회로 빠른 AutoML 실행."""
    from src.automl_engine import run_automl
    return run_automl(splits, mode="fast", n_trials=1, n_splits=2, seed=SEED)


# ---------------------------------------------------------------------------
# 1) 데이터 누출 방지 / 분할
# ---------------------------------------------------------------------------
def test_split_sizes_and_ratios(splits):
    sizes = splits["sizes"]
    assert sizes["train"] + sizes["val"] + sizes["test"] == 300
    # train 60% / val 20% / test 20%
    assert sizes["train"] == pytest.approx(180, abs=2)
    assert sizes["val"] == pytest.approx(60, abs=2)
    assert sizes["test"] == pytest.approx(60, abs=2)


def test_no_index_leakage(splits):
    from src.automl_engine import assert_no_leakage
    assert_no_leakage(splits)  # 교집합이 있으면 AssertionError
    tr, va, te = (set(splits[k]) for k in ("train_index", "val_index", "test_index"))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr | va | te) == 300


def test_labels_cover_all_classes(splits):
    for key in ("y_train", "y_val", "y_test"):
        y = np.asarray(splits[key])
        assert len(np.unique(y)) == N_CLASSES, f"{key} 에 일부 클래스가 없음"


# ---------------------------------------------------------------------------
# 2) AutoML 모델 학습 / 최적화
# ---------------------------------------------------------------------------
def test_automl_trains_all_candidate_models(automl_result):
    expected = {"random_forest", "xgboost", "extra_trees",
                "hist_gradient_boosting"}
    assert set(automl_result["per_model"].keys()) == expected
    assert automl_result["trials_per_model"] == 1


def test_automl_per_model_records(automl_result):
    for name, r in automl_result["per_model"].items():
        assert isinstance(r["best_params"], dict)
        assert 0.0 <= r["val_metrics"]["macro_f1"] <= 1.0
        assert 0.0 <= r["val_metrics"]["accuracy"] <= 1.0
        assert r["fit_time_sec"] >= 0.0
        assert r["opt_time_sec"] > 0.0
        assert r["complexity"] > 0
        assert len(r["val_per_class_recall"]) == N_CLASSES
        # 학습된 모델로 예측 가능
        pred = r["model"].predict(automl_result["splits"]["X_val"])
        assert len(pred) == len(automl_result["splits"]["y_val"])


def test_automl_history_and_leaderboard(automl_result):
    for name, hist in automl_result["history"].items():
        assert len(hist) == 1, f"{name} 히스토리 trial 수 오류"
    lb = automl_result["leaderboard"]
    assert len(lb) == 4
    assert list(lb["val_macro_f1"]) == sorted(lb["val_macro_f1"], reverse=True)
    for col in ("fit_time_sec", "opt_time_sec", "worst_class_recall",
                "complexity", "inference_ms_per_100"):
        assert col in lb.columns


def test_automl_progress_callback(splits):
    calls = []
    from src.automl_engine import run_automl
    run_automl(splits, mode="fast", n_trials=1, n_splits=2, seed=SEED,
               model_names=["random_forest"],
               progress_cb=lambda s, c, t, m: calls.append((s, c, t, m)))
    assert len(calls) > 0
    stage, cur, total, msg = calls[0]
    assert stage == "automl"
    assert 0 <= cur <= total and total > 0
    assert isinstance(msg, str) and msg


# ---------------------------------------------------------------------------
# 3) 최적 모델 선정
# ---------------------------------------------------------------------------
def test_select_best_model_picks_top_macro_f1(automl_result):
    from src.model_selector import select_best_model
    sel = select_best_model(automl_result)
    lb = automl_result["leaderboard"]
    # 검증 Macro F1 최고 모델(차이가 허용치보다 클 때)과 일치해야 함
    top = lb.iloc[0]["model"]
    top_f1 = float(lb.iloc[0]["val_macro_f1"])
    second_f1 = float(lb.iloc[1]["val_macro_f1"])
    if top_f1 - second_f1 > 0.01:
        assert sel["best_model_name"] == top
    else:
        assert sel["best_model_name"] in set(lb["model"])
    assert 0.0 <= sel["scores"]["val_macro_f1"] <= 1.0
    assert sel["selection_table"] is not None
    assert len(sel["criteria"]) >= 3
    assert "테스트셋" in sel["note"]


def test_select_best_model_tie_break_by_recall():
    """동점(±0.01)이면 최저 클래스 Recall 이 높은 모델을 선정해야 함."""
    from src.model_selector import select_best_model

    class _Dummy:
        pass

    def _mk(recalls):
        return {
            "display": "x", "model": _Dummy(),
            "val_metrics": {"accuracy": 0.9, "macro_f1": 0.8, "weighted_f1": 0.85},
            "val_per_class_recall": recalls,
            "val_per_class": pd.DataFrame(),
            "complexity": 100, "inference_ms_per_100": 0.1,
            "fit_time_sec": 1.0, "opt_time_sec": 1.0,
            "best_cv_macro_f1": 0.8, "baseline_cv_macro_f1": 0.7,
        }

    fake = {"per_model": {
        "model_a": _mk({i: 0.5 for i in range(N_CLASSES)}),
        "model_b": _mk({i: (0.9 if i != 1 else 0.3) for i in range(N_CLASSES)}),
    }}
    sel = select_best_model(fake, tie_tolerance=0.01)
    assert sel["best_model_name"] == "model_a"   # 최저 Recall 0.5 > 0.3
    assert sel["tie_break_applied"] is True
    assert sel["tie_break_reason"]


def test_save_best_model_files(automl_result, tmp_path):
    from src.model_selector import save_best_model, select_best_model
    sel = select_best_model(automl_result)
    paths = save_best_model(sel, models_dir=tmp_path / "models",
                            results_dir=tmp_path / "results")
    assert paths["model"].exists()
    assert paths["selection_json"].exists()
    info = json.loads(paths["selection_json"].read_text(encoding="utf-8"))
    assert info["best_model_name"] == sel["best_model_name"]
    assert "val_macro_f1" in info["scores"]
    assert info["selection_table"]

    # 저장된 모델 로드 후 예측 가능
    import joblib
    model = joblib.load(paths["model"])
    X = automl_result["splits"]["X_val"]
    assert len(model.predict(X)) == len(X)


# ---------------------------------------------------------------------------
# 4) 공정 조건 탐색 (자동 트러블슈팅)
# ---------------------------------------------------------------------------
def test_intersect_bounds():
    from src.auto_troubleshooter import intersect_bounds
    from src.data_generator import PROCESS_BOUNDS

    safe = {c: PROCESS_BOUNDS[c] for c in INPUT_COLUMNS}
    safe["pressure_mtorr"] = (20.0, 60.0)   # 더 좁게
    out = intersect_bounds(safe)
    assert out["pressure_mtorr"] == (20.0, 60.0)
    assert out["gas_flow_sccm"] == PROCESS_BOUNDS["gas_flow_sccm"]

    # 교집합이 비면 오류
    with pytest.raises(ValueError):
        intersect_bounds({"pressure_mtorr": (200.0, 300.0)})


def test_auto_troubleshoot_runs_and_stays_in_bounds(automl_result):
    from src.auto_troubleshooter import auto_troubleshoot, intersect_bounds

    model = automl_result["per_model"]["random_forest"]["model"]
    current = default_process_conditions()
    safe = {c: PROCESS_BOUNDS[c] for c in INPUT_COLUMNS}
    safe["temperature_c"] = (30.0, 70.0)    # 안전 범위를 좁게 제한

    out = auto_troubleshoot(model, current, safe_bounds=safe,
                            method="random", n_iter=60, random_state=SEED)

    bounds = intersect_bounds(safe)
    for col in INPUT_COLUMNS:
        lo, hi = bounds[col]
        v = out["recommendation"]["recommended"][col]
        assert lo - 1e-6 <= v <= hi + 1e-6, f"{col}={v} 가 범위({lo}~{hi}) 밖"
    assert out["recommendation"]["within_bounds"] is True
    # 결과 구조
    assert out["current_analysis"]["contributions"] is not None
    assert set(out["current_analysis"]["contributions"]["feature"]) == set(INPUT_COLUMNS)
    assert out["improvement"]["verdict"]
    assert out["improvement"]["caveat"]
    assert any("단정" in w or "주의" in w for w in out["warnings"])
    assert len(out["comparison"]) == len(INPUT_COLUMNS)


def test_explain_current_condition_has_direction(automl_result):
    from src.auto_troubleshooter import explain_current_condition
    model = automl_result["per_model"]["random_forest"]["model"]
    out = explain_current_condition(model, default_process_conditions())
    contrib = out["contributions"]
    assert {"feature", "feature_value", "contribution",
            "abs_contribution", "direction"} <= set(contrib.columns)
    assert len(contrib) == len(INPUT_COLUMNS)
    assert out["predicted_name"]


# ---------------------------------------------------------------------------
# 5) 자동 검증 (validation_engine)
# ---------------------------------------------------------------------------
def test_validate_recommendation_metrics():
    from src.validation_engine import validate_recommendation
    base = default_process_conditions()
    after = dict(base)
    after["bias_power_w"] = min(base["bias_power_w"] + 30.0, 300.0)

    val = validate_recommendation(base, after)
    for key in ("etch_depth_nm", "top_cd_nm", "middle_cd_nm", "bottom_cd_nm",
                "sidewall_angle_deg", "defect_severity"):
        assert key in val["before"] and key in val["after"]
    assert len(val["delta"]) == 6
    assert set(val["delta"]["metric"])  # 6개 지표 이름 존재
    assert "독립" in val["caveat"] or "동일" in val["caveat"]


def test_run_validation_writes_files(tmp_path):
    from src.validation_engine import run_validation
    base = default_process_conditions()
    out = run_validation(base, base, results_dir=tmp_path)
    assert Path(out["profile_plot"]).exists()
    assert Path(out["delta_plot"]).exists()
    assert Path(out["metrics_csv"]).exists()


# ---------------------------------------------------------------------------
# 6) 자동 보고서 생성
# ---------------------------------------------------------------------------
def test_report_generator_creates_html(automl_result, tmp_path):
    from src.report_generator import generate_html_report
    from src.model_selector import select_best_model

    sel = select_best_model(automl_result)
    lb = automl_result["leaderboard"]
    importance = pd.DataFrame({
        "feature": list(INPUT_COLUMNS),
        "importance": np.linspace(0.5, 0.05, len(INPUT_COLUMNS)),
    })
    report_data = {
        "metadata": {"분석 모드": "fast", "random_state": SEED},
        "data": {"전체 샘플 수": 300},
        "model_comparison": lb,
        "selection": sel,
        "optimization": pd.DataFrame([{
            "model": "random_forest", "baseline_cv_macro_f1": 0.30,
            "best_cv_macro_f1": 0.35, "opt_time_sec": 1.0,
        }]),
        "shap": {"method": "Permutation importance (테스트)",
                 "importance": importance,
                 "per_class_top": {"Normal": ["a"], "Undercut": ["b"],
                                   "Bowing": ["c"], "Footing": ["d"],
                                   "Etch Stop": ["e"],
                                   "Punch-through": ["f"]}},
        "troubleshooting": {
            "comparison": pd.DataFrame([{"Parameter": "pressure_mtorr",
                                         "Original": 30.0, "Recommended": 45.0}]),
            "summary": pd.DataFrame(),
            "improvement": {"defect_before": "Bowing", "defect_after": "Normal",
                            "normal_prob_delta": 0.1234, "verdict": "테스트",
                            "level": "개선 가능성 있음"},
            "warnings": ["테스트 경고 문구"],
        },
        "validation": {
            "delta": pd.DataFrame([{"metric": "Etch depth (nm)", "before": 100.0,
                                    "after": 110.0, "delta": 10.0,
                                    "delta_pct": 10.0}]),
            "before_defect_name": "Bowing", "after_defect_name": "Normal",
            "severity_delta": -3.2, "verdict": "테스트 검증",
        },
    }
    path = generate_html_report(report_data, tmp_path / "report.html")
    assert path.exists()
    html = path.read_text(encoding="utf-8")
    # 실제 측정값(리더보드 수치)과 필수 섹션이 들어가는지 확인
    assert f"{lb.iloc[0]['val_macro_f1']:.4f}" in html
    assert "연구의 한계" in html
    assert "물리적 인과관계" in html
    assert "동일한 물리 가정" in html or "동일 가정" in html
    assert "테스트 경고 문구" in html


# ---------------------------------------------------------------------------
# 7) 실행 스크립트 / 상태 기록
# ---------------------------------------------------------------------------
def test_job_status_writes_progress(tmp_path):
    from run_automl import STAGES, JobStatus

    status_file = tmp_path / "status.json"
    js = JobStatus(status_file, job_id="test_job", mode="fast")
    js.stage(1, "데이터 중", within=0.5)
    got = json.loads(status_file.read_text(encoding="utf-8"))
    assert got["status"] == "running"
    assert got["stage_index"] == 1
    assert got["stage_name"] == STAGES[0]
    assert got["overall_progress"] > 0

    js.stage(10, "완료 직전", within=0.9)
    js.done(report_path=str(tmp_path / "report.html"))
    got = json.loads(status_file.read_text(encoding="utf-8"))
    assert got["status"] == "done"
    assert got["overall_progress"] == 100.0
    assert got["report_path"]


def test_job_status_records_failure(tmp_path):
    from run_automl import JobStatus
    status_file = tmp_path / "status.json"
    js = JobStatus(status_file, job_id="fail_job", mode="fast")
    js.fail("테스트 오류")
    got = json.loads(status_file.read_text(encoding="utf-8"))
    assert got["status"] == "error"
    assert got["error"] == "테스트 오류"


def test_run_automl_cli_args():
    from run_automl import parse_args
    args = parse_args(["--mode", "normal", "--trials", "3",
                       "--status-file", "x.json", "--job-id", "j1"])
    assert args.mode == "normal"
    assert args.trials == 3
    assert args.status_file == "x.json"
    assert args.job_id == "j1"


# ---------------------------------------------------------------------------
# 8) Streamlit 페이지 (앱 임포트/등록 검증)
# ---------------------------------------------------------------------------
def test_app_registers_automl_page():
    import app
    assert "7. AI Auto Troubleshooting" in app.PAGES
    assert callable(app.PAGES["7. AI Auto Troubleshooting"])


def test_app_automl_helpers(tmp_path, monkeypatch):
    import app
    # 상태 파일이 없으면 빈 dict / 실행 중 아님
    monkeypatch.setattr(app, "AUTOML_STATUS_PATH", tmp_path / "missing.json")
    assert app.read_automl_status() == {}
    assert app.automl_job_running() is False

    # running 이어도 pid 가 죽으면 중복 실행 허용
    status = {"status": "running", "pid": 999999999}
    p = tmp_path / "status.json"
    p.write_text(json.dumps(status), encoding="utf-8")
    monkeypatch.setattr(app, "AUTOML_STATUS_PATH", p)
    assert app.read_automl_status()["status"] == "running"
    assert app.automl_job_running() is False


def test_automl_page_renders_in_bare_mode():
    """Streamlit 페이지 함수가 bare mode 에서 예외 없이 렌더링되는지 확인."""
    from streamlit.errors import StreamlitAPIException
    import app
    assert "7. AI Auto Troubleshooting" in app.PAGES
    # bare mode: 위젯 값은 기본값(False/기본 선택)을 반환하며 예외 없이 실행됨
    try:
        app.PAGES["7. AI Auto Troubleshooting"]()
    except StreamlitAPIException as exc:  # 환경 차이로 인한 API 제약은 허용
        pytest.skip(f"Streamlit 환경 제약: {exc}")

