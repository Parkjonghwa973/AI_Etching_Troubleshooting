"""
app.py
======

AI_Etching_Troubleshooting 웹 대시보드 (Streamlit).

실행 방법 (프로젝트 루트에서)
----------------------------
    python -m streamlit run app.py

페이지 구성
-----------
1. Project Overview      — 프로젝트 목표와 시스템 구조
2. Etching Process Simulator — 6개 공정 변수 슬라이더 + 2D 식각 프로파일
3. Machine Learning Performance — 3개 모델 성능 비교
4. Explainable AI (SHAP) — 원인 후보 분석 결과
5. AI Troubleshooting    — 불량 예측 + 공정 조건 추천 + 전후 비교
6. Research Results      — 발표용 결과 종합
7. AI Auto Troubleshooting — AutoML 기반 자동 분석
8. Literature Validation — 실제 반도체 식각 논문과의 물리적 타당성 비교

.. note::
    모든 데이터/형상 모델은 교육용 합성 모델이며, 실제 반도체 공정 물리를
    검증한 결과가 아닙니다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.data_generator import (  # noqa: E402
    DEFECT_NAMES,
    GENERATION_RULES,
    GROUND_TRUTH_FEATURES,
    INPUT_COLUMNS,
    PROCESS_BOUNDS,
    compute_indicators,
    default_data_path,
)
from src.etching_simulator import (  # noqa: E402
    compute_all_defect_profiles,
    compute_profile,
    default_process_conditions,
    predict_defect_type,
    profile_metrics_table,
)

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

# AI 자동 분석(AutoML) 전용 폴더 — 기존 results/, models/ 결과와 분리
AUTOML_DIR = RESULTS_DIR / "automl"
AUTOML_STATUS_PATH = AUTOML_DIR / "job_status.json"
AUTOML_LOG_PATH = AUTOML_DIR / "job_log.txt"
AUTOML_STAGES = [
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

PARAM_LABELS = {
    "gas_chemistry": "Gas chemistry (가스 화학 조성 지표)",
    "gas_flow_sccm": "Gas flow (sccm)",
    "pressure_mtorr": "Pressure (mTorr)",
    "bias_power_w": "Bias power (W)",
    "source_power_w": "Source power (W)",
    "temperature_c": "Temperature (°C)",
}

CLASS_COLORS = ["#2E86C1", "#E74C3C", "#8E44AD", "#E67E22", "#16A085", "#D35400"]

st.set_page_config(
    page_title="AI Etching Troubleshooting",
    page_icon="🔬",
    layout="wide",
)


# ---------------------------------------------------------------------------
# 캐시 로더
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    path = default_data_path()
    if not path.exists():
        from src.data_generator import generate_dataset
        return generate_dataset()
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_models():
    from src.train_models import load_models as _load
    return _load(MODELS_DIR)


@st.cache_data(show_spinner=False)
def load_csv(name: str):
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(name: str):
    import json
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def image_or_note(filename: str, caption: str = ""):
    """결과 이미지가 있으면 표시, 없으면 실행 안내."""
    path = RESULTS_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"`{filename}` 파일이 없습니다. 먼저 전체 파이프라인을 실행하세요:\n\n"
                "```powershell\npython run_pipeline.py\n```")


def main_model():
    """사용할 주 모델: 최적화 XGBoost → XGBoost → RandomForest 순."""
    models = load_models()
    for key in ["optimized_xgboost", "xgboost", "random_forest"]:
        if key in models:
            return models[key], key
    return None, None


def process_sliders(prefix: str, defaults: dict) -> dict:
    """6개 공정 변수 슬라이더를 만들고 값 dict 를 반환."""
    vals = {}
    for col in INPUT_COLUMNS:
        lo, hi = PROCESS_BOUNDS[col]
        step = 0.01 if col == "gas_chemistry" else 1.0
        vals[col] = st.slider(
            PARAM_LABELS[col],
            min_value=float(lo), max_value=float(hi),
            value=float(defaults[col]), step=step,
            key=f"{prefix}_{col}",
        )
    return vals


def profile_plotly(profiles: dict, title: str = "2D Etching Profile"):
    """프로파일을 Plotly 로 그립니다 (interactive)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for i, (name, p) in enumerate(profiles.items()):
        color = CLASS_COLORS[i % len(CLASS_COLORS)]
        fig.add_trace(go.Scatter(x=p["right_nm"], y=p["y_nm"], mode="lines",
                                 name=f"{name} (right)", line=dict(color=color, width=2),
                                 legendgroup=name, showlegend=True))
        fig.add_trace(go.Scatter(x=p["left_nm"], y=p["y_nm"], mode="lines",
                                 name=f"{name} (left)", line=dict(color=color, width=2,
                                                                  dash="dot"),
                                 legendgroup=name, showlegend=False))
    fig.update_layout(
        title=title,
        xaxis_title="Lateral position (nm)",
        yaxis_title="Depth (nm, negative = down)",
        template="plotly_white", height=620, legend_title="Defect type",
    )
    return fig


# ---------------------------------------------------------------------------
# 페이지
# ---------------------------------------------------------------------------
def page_overview():
    st.title("1. Project Overview — 프로젝트 개요")
    st.markdown(
        """
### 🎯 프로젝트 목표
반도체 **Etching(식각) 공정**에서 발생하는 불량을 머신러닝으로 예측하고,
**머신러닝 알고리즘의 성능 개선이 불량 분류·원인 분석 능력에 미치는 영향을 정량적으로
평가**하는 교육용 트러블슈팅 시뮬레이터입니다.

또한 머신러닝이 도출한 주요 공정 변수를 시뮬레이션하여, 불량 발생 위험을 줄일 수 있는
공정 조건을 탐색합니다.

> ⚠️ **중요**: 본 시스템의 데이터와 식각 형상 모델은 **교육용 합성(가상) 모델**입니다.
> 실제 식각 물리를 검증한 결과가 아니며, 산출 수치는 특정 장비의 실제 공정 레시피를
> 의미하지 않습니다.
        """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📂 시스템 구조")
        st.code(
            "AI_Etching_Troubleshooting/\n"
            "├── app.py                    # 이 대시보드\n"
            "├── run_pipeline.py           # 전체 파이프라인 자동 실행\n"
            "├── requirements.txt\n"
            "├── src/\n"
            "│   ├── data_generator.py     # 합성 데이터 생성\n"
            "│   ├── etching_simulator.py  # 2D 식각 프로파일 시뮬레이터\n"
            "│   ├── train_models.py       # RF / XGBoost 학습\n"
            "│   ├── optimize_model.py     # Optuna 하이퍼파라미터 최적화\n"
            "│   ├── evaluate_models.py    # 성능 평가\n"
            "│   ├── explain_model.py      # SHAP 원인 후보 분석\n"
            "│   ├── process_optimizer.py  # 공정 조건 추천\n"
            "│   └── visualization.py      # 그래프 디자인\n"
            "├── data/      ├── models/      ├── results/      └── tests/",
            language="text",
        )

    with col2:
        st.subheader("🔁 전체 파이프라인")
        st.markdown(
            """
1. **데이터 생성** — 6개 공정 변수 → 5종 불량 라벨 (규칙 기반, 5,000건)
2. **데이터 검증** — 결측값 / 라벨 / 범위 확인
3. **모델 학습** — Random Forest, XGBoost (80:20 Stratified split)
4. **하이퍼파라미터 최적화** — Optuna (CV Macro F1 최대화)
5. **모델 평가** — 테스트셋 + 분포 변화셋
6. **SHAP 분석** — 원인 후보 변수 도출
7. **공정 조건 최적화** — Normal 확률 최대화 조건 추천
8. **결과 저장 / 그래프 생성**
            """)

    st.subheader("🧪 입력 변수와 불량 유형")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**입력 변수 (6개)**")
        ranges = pd.DataFrame([
            {"Variable": PARAM_LABELS[c],
             "Range": f"{PROCESS_BOUNDS[c][0]} ~ {PROCESS_BOUNDS[c][1]}"}
            for c in INPUT_COLUMNS
        ])
        st.table(ranges)
    with col4:
        st.markdown("**불량 유형 (라벨 5종)**")
        rules = pd.DataFrame([
            {"Class": f"{k}: {v}", "생성 규칙(교육용 가정)": GENERATION_RULES[v]}
            for k, v in DEFECT_NAMES.items()
        ])
        st.table(rules)

    st.subheader("📊 합성 데이터 미리보기")
    df = load_data()
    st.dataframe(df.head(10), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("데이터 건수", f"{len(df):,}")
    c2.metric("변수 개수", f"{len(INPUT_COLUMNS)} 입력 + 5 지표")
    c3.metric("불량 클래스", len(DEFECT_NAMES))


def page_simulator():
    st.title("2. Etching Process Simulator")
    st.caption("교육용 2D 식각 단면 시뮬레이터 — 실제 공정 물리 검증 결과가 아닙니다.")

    defaults = default_process_conditions()
    col_left, col_right = st.columns([1, 1.15])

    with col_left:
        st.markdown("#### 공정 변수 조절")
        conds = process_sliders("sim", defaults)
        model, model_name = main_model()
        lab, name = predict_defect_type(conds)
        if model is not None:
            from src.process_optimizer import predict_defect
            lab_ml, name_ml, probs_ml = predict_defect(model, conds)
            st.info(f"**ML 예측 불량**: {name_ml} "
                    f"(확률 {probs_ml[lab_ml]:.2f}) — 모델: `{model_name}`")
        st.warning("규칙 기반 시뮬레이터 예측: "
                   f"**{name}** (가장 가능성 높은 불량 유형)")

        # 지표 표시
        ind = compute_indicators(pd.DataFrame([conds]), add_noise=False).iloc[0]
        st.markdown("#### 파생 지표 (교육용 계산값)")
        st.write({
            "Etch rate (nm/min)": round(float(ind["etch_rate_nm_min"]), 1),
            "Passivation index": round(float(ind["passivation_index"]), 3),
            "Anisotropy index": round(float(ind["anisotropy_index"]), 3),
            "Footing index": round(float(ind["footing_index"]), 3),
        })

    with col_right:
        st.markdown("#### 현재 조건의 예상 식각 프로파일")
        prof = compute_profile(conds, defect_type=name)
        fig = profile_plotly({f"Predicted: {name}": prof},
                             "Current Condition — Predicted Etching Profile")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 5가지 불량 유형 기준 형상 비교 (같은 공정 조건)")
    profiles = compute_all_defect_profiles(conds)
    st.plotly_chart(profile_plotly(profiles,
                                   "Reference Profiles — 5 Defect Types"),
                    use_container_width=True)
    st.markdown("##### 형상 지표 비교표")
    st.dataframe(profile_metrics_table(profiles), use_container_width=True)

    with st.expander("형상 모델의 가정 보기"):
        st.markdown(
            """
- 식각 깊이 = etch rate(nm/min) × 식각 시간(60 s 가정)
- 초기 상단 반폭 100 nm (Top CD ≈ 200 nm) 가정
- **Undercut**: 상단 부근(깊이 0~30%)의 측방향 식각 증가
- **Bowing**: 중간 깊이(50%)의 가우시안 측벽 팽창
- **Etch Stop**: 식각 깊이 감소 + 하단 둥근 형상
- **Footing**: 하단 부근 측벽 확장(하단 CD 증가)
- 모든 변형량은 측벽 보호 효과(passivation), 방향성(anisotropy),
  pressure, bias power, 온도에 연속적으로 의존합니다.
            """)


def page_performance():
    st.title("3. Machine Learning Performance")
    st.caption("세 모델을 동일한 테스트 데이터로 평가한 결과입니다.")

    cmp_df = load_csv("model_comparison.csv")
    shift_df = load_csv("model_comparison_distribution_shift.csv")
    if cmp_df is None:
        st.warning("결과 파일이 없습니다. `python run_pipeline.py` 를 먼저 실행하세요.")
        return

    disp = {"random_forest": "Random Forest",
            "xgboost": "XGBoost (baseline)",
            "optimized_xgboost": "XGBoost (Optuna optimized)"}
    cmp_df = cmp_df.copy()
    cmp_df["Model"] = cmp_df["model"].map(disp).fillna(cmp_df["model"])

    st.markdown("#### 📋 모델 성능 비교 (테스트셋)")
    st.dataframe(
        cmp_df[["Model", "accuracy", "macro_precision", "macro_recall",
                "macro_f1", "weighted_f1"]].round(4),
        use_container_width=True,
    )

    # 기본 vs 최적화 모델 차이
    if {"xgboost", "optimized_xgboost"}.issubset(set(cmp_df["model"])):
        base = cmp_df[cmp_df["model"] == "xgboost"].iloc[0]
        opt = cmp_df[cmp_df["model"] == "optimized_xgboost"].iloc[0]
        st.markdown("#### 🔧 최적화 전후 차이")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy 향상", f"{opt['accuracy'] - base['accuracy']:+.4f}",
                  f"{base['accuracy']:.4f} → {opt['accuracy']:.4f}")
        c2.metric("Macro F1 향상", f"{opt['macro_f1'] - base['macro_f1']:+.4f}",
                  f"{base['macro_f1']:.4f} → {opt['macro_f1']:.4f}")
        delta = opt["macro_f1"] - base["macro_f1"]
        if delta > 0:
            c3.success("최적화 모델이 더 우수합니다")
        else:
            c3.warning("최적화 모델이 더 우수하지 않습니다 (과적합 가능성)")

    st.markdown("#### 📈 그래프")
    c1, c2 = st.columns(2)
    with c1:
        image_or_note("accuracy_comparison.png", "Model Accuracy Comparison")
        image_or_note("confusion_matrix.png", "Confusion Matrix (최적화 모델)")
    with c2:
        image_or_note("macro_f1_comparison.png", "Macro F1-score Comparison")
        image_or_note("per_class_recall.png", "Per-class Recall Comparison")

    image_or_note("model_performance_comparison.png", "Accuracy & Macro F1 비교")

    if shift_df is not None:
        st.markdown("#### 🌪️ 분포 변화(distribution shift) 평가")
        st.caption("일부 입력 변수 분포와 노이즈 수준을 바꿔 생성한 데이터로 평가한 결과입니다. "
                   "단순 무작위 분할만으로는 일반화 성능을 과대평가할 수 있습니다.")
        shift_df = shift_df.copy()
        shift_df["Model"] = shift_df["model"].map(disp).fillna(shift_df["model"])
        st.dataframe(shift_df[["Model", "accuracy", "macro_f1", "weighted_f1"]].round(4),
                     use_container_width=True)
        image_or_note("confusion_matrix_distribution_shift.png",
                      "Confusion Matrix (분포 변화 데이터)")

    opt = load_json("optimization_result.json")
    if opt is not None:
        st.markdown("#### 🧬 하이퍼파라미터 최적화 결과 (Optuna)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline CV Macro F1", f"{opt['baseline_cv_macro_f1']:.4f}")
        c2.metric("Optimized CV Macro F1", f"{opt['best_cv_macro_f1']:.4f}")
        c3.metric("개선 (Δ)", f"{opt['improvement']:+.4f}",
                  "향상됨" if opt["improved"] else "유의미한 향상 없음")
        st.json(opt["best_params"])


def page_xai():
    st.title("4. Explainable AI (SHAP)")
    model, model_name = main_model()
    if model is None:
        st.warning("`python run_pipeline.py` 를 먼저 실행해 모델을 생성하세요.")
        return

    st.warning("⚠️ **SHAP 값은 모델 예측에 대한 기여도이며, 물리적 인과관계의 증거가 아닙니다.** "
               "중요도가 높다고 해당 변수가 불량의 원인이라고 단정할 수 없습니다.")

    st.markdown("#### 🌍 Global Feature Importance (평균 |SHAP|)")
    imp = load_csv("shap_global_importance.csv")
    if imp is not None:
        st.dataframe(imp.round(5), use_container_width=True)
        image_or_note("feature_importance.png", "Global Feature Importance")
        image_or_note("shap_summary.png", "SHAP Summary (전체)")
    else:
        st.info("SHAP 결과가 없습니다. `python run_pipeline.py` 를 실행하세요.")

    st.markdown("#### 🔍 특정 불량 클래스의 주요 기여 변수")
    top_df = load_csv("shap_top_features_by_class.csv")
    if top_df is not None:
        st.dataframe(top_df, use_container_width=True)

    st.markdown("#### 🧪 특정 공정 조건의 변수 기여도 (실시간 계산)")
    defaults = default_process_conditions()
    conds = process_sliders("xai", defaults)
    try:
        from src.explain_model import explain_prediction

        X_df = pd.DataFrame([conds])[list(INPUT_COLUMNS)]
        out = explain_prediction(model, X_df, list(INPUT_COLUMNS))
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**예측 결과**: {out['predicted_name']} "
                     f"(확률 {out['probabilities'][out['predicted_class']]:.3f})")
            st.bar_chart(pd.Series(out["probabilities"],
                                   index=[f"{k}: {v}" for k, v in DEFECT_NAMES.items()]))
        with c2:
            st.markdown(f"**변수별 기여도 (class: {out['explained_class']} — "
                        f"{DEFECT_NAMES[out['explained_class']]})**")
            st.dataframe(out["contributions"].round(4), use_container_width=True)
        st.caption(out["causality_warning"])
    except Exception as exc:
        st.error(f"실시간 SHAP 계산 실패: {exc}")

    st.markdown("#### 🧾 SHAP 결과 vs 데이터 생성 규칙 비교")
    cmp_df = load_csv("shap_vs_generation_rules.csv")
    if cmp_df is not None:
        st.dataframe(cmp_df.round(5), use_container_width=True)
        st.caption("파란색: 데이터 생성 규칙에서 실제로 사용한 변수 / 회색: 명시적으로 사용하지 않은 변수. "
                   "SHAP 이 생성 규칙의 핵심 변수를 잘 포착했는지 확인용입니다.")
        image_or_note("shap_vs_generation_rules.png", "SHAP vs 생성 규칙")
    else:
        st.info("비교 결과가 없습니다. 파이프라인을 실행하세요.")


def page_troubleshooting():
    st.title("5. AI Troubleshooting")
    st.caption("공정 변수를 입력하면 불량 유형을 예측하고, 개선된 공정 조건을 추천합니다.")

    model, model_name = main_model()
    if model is None:
        st.warning("`python run_pipeline.py` 를 먼저 실행해 모델을 생성하세요.")
        return

    data = load_data()
    from src.process_optimizer import (
        compare_profiles,
        optimize_process,
        result_to_dataframe,
        summary_dataframe,
        train_distribution_stats,
    )

    defaults = default_process_conditions()
    conds = process_sliders("ts", defaults)

    col1, col2 = st.columns([1, 1])
    with col1:
        method_label = st.radio("최적화 방법",
                                ["Random Search", "Differential Evolution"],
                                horizontal=True)
        n_iter = st.slider("탐색 횟수", min_value=200, max_value=6000,
                           value=2000, step=200)
        run_btn = st.button("🚀 분석 및 공정 조건 추천", type="primary")

    if run_btn:
        from src.explain_model import explain_prediction
        from src.process_optimizer import predict_defect

        lab, name, probs = predict_defect(model, conds)
        stats = train_distribution_stats(data[list(INPUT_COLUMNS)])
        method = "random" if method_label.startswith("Random") else "de"
        result = optimize_process(model, conds, method=method, n_iter=n_iter,
                                  train_stats=stats, random_state=42)

        st.markdown("### 1) 현재 조건 예측 결과")
        c1, c2, c3 = st.columns(3)
        c1.metric("예측 불량 유형", name)
        c2.metric("Normal 확률", f"{probs[0]:.3f}")
        c3.metric("모델", model_name)

        prob_df = pd.DataFrame({
            "Defect": [f"{k}: {v}" for k, v in DEFECT_NAMES.items()],
            "Probability": probs,
        }).set_index("Defect")
        st.bar_chart(prob_df)

        st.markdown("### 2) 예측에 기여한 주요 변수 (SHAP)")
        try:
            out = explain_prediction(model, pd.DataFrame([conds])[list(INPUT_COLUMNS)],
                                     list(INPUT_COLUMNS))
            st.dataframe(out["contributions"].round(4), use_container_width=True)
            st.caption(out["causality_warning"])
        except Exception as exc:
            st.info(f"SHAP 계산을 건너뜁니다: {exc}")

        st.markdown("### 3) 추천 공정 조건")
        st.write(f"**최적화 방법**: {result['method']}")
        st.dataframe(result_to_dataframe(result), use_container_width=True)
        st.dataframe(summary_dataframe(result), use_container_width=True)

        before_p, after_p, prof_tbl = compare_profiles(result)
        st.markdown("### 4) 전후 식각 프로파일 비교 (교육용 시뮬레이터 입력)")
        st.plotly_chart(profile_plotly(
            {"Before (original)": before_p, "After (recommended)": after_p},
            "Before / After Optimization — Etching Profile"), use_container_width=True)
        st.dataframe(prof_tbl, use_container_width=True)

        st.markdown("### 5) 주의 사항")
        for w in result["warnings"]:
            st.warning(w)
    else:
        st.info("공정 변수를 조절한 뒤 **분석 및 공정 조건 추천** 버튼을 눌러 주세요.")


def page_research():
    st.title("6. Research Results (발표용)")
    st.caption("모델 성능, 최적화, feature importance, 예측 예시, 전후 프로파일, 연구 한계를 한눈에 확인합니다.")

    st.markdown("### 📊 1. Model performance comparison")
    cmp_df = load_csv("model_comparison.csv")
    if cmp_df is not None:
        st.dataframe(cmp_df.round(4), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            image_or_note("accuracy_comparison.png")
        with c2:
            image_or_note("macro_f1_comparison.png")
    else:
        st.info("파이프라인을 실행하세요.")

    st.markdown("### 🧬 2. Hyperparameter optimization results")
    opt = load_json("optimization_result.json")
    if opt is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline CV Macro F1", f"{opt['baseline_cv_macro_f1']:.4f}")
        c2.metric("Optimized CV Macro F1", f"{opt['best_cv_macro_f1']:.4f}")
        c3.metric("개선 여부", "향상됨" if opt["improved"] else "유의미한 향상 없음")
        image_or_note("optimization_history.png", "Optimization History")
        st.json(opt["best_params"])
    else:
        st.info("최적화 결과가 없습니다.")

    st.markdown("### 🏆 3. Feature importance (SHAP)")
    image_or_note("feature_importance.png", "Global Feature Importance")
    image_or_note("shap_summary.png", "SHAP Summary")

    st.markdown("### 🔮 4. Defect prediction examples")
    model, model_name = main_model()
    data = load_data()
    if model is not None:
        sample = data.sample(5, random_state=0)
        from src.process_optimizer import predict_defect

        rows = []
        for _, r in sample.iterrows():
            conds = r[INPUT_COLUMNS].to_dict()
            lab, name, probs = predict_defect(model, conds)
            rows.append({
                "True defect": DEFECT_NAMES[int(r["defect_type"])],
                "Predicted defect": name,
                "P(Normal)": round(float(probs[0]), 3),
                "Max prob": round(float(probs[lab]), 3),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("모델이 없습니다.")

    st.markdown("### 🔄 5. Before / After etching profiles")
    image_or_note("etching_profiles.png", "5가지 불량 유형 기준 프로파일")
    image_or_note("before_after_profiles.png", "공정 조건 최적화 전후 프로파일")
    prof = load_csv("before_after_profile_metrics.csv")
    if prof is not None:
        st.dataframe(prof, use_container_width=True)
    rec = load_csv("process_recommendation.csv")
    if rec is not None:
        st.markdown("#### 추천 공정 조건 예시")
        st.dataframe(rec, use_container_width=True)

    st.markdown("### ⚠️ 6. Research limitations (연구 한계)")
    st.markdown(
        """
1. **합성 데이터 한계** — 학습 데이터는 교육용 수학 규칙으로 생성된 가상 데이터이며,
   실제 플라즈마 식각 물리(이온 주입, 화학 반응, 확산)를 반영하지 않습니다.
2. **인과관계 미검증** — SHAP 기여도는 모델의 예측 근거일 뿐 원인 인과관계가 아닙니다.
3. **분포 변화에 취약** — 실제 공정은 학습 분포와 다른 조건에서 동작하므로,
   분포 변화 평가에서 성능 저하가 관찰됩니다. OOD 경고가 필요합니다.
4. **공정 조건 추천의 한계** — 추천 조건은 모델 예측 확률을 최대화한 값이며,
   실제 공정에서 불량을 감소시킨다고 단정할 수 없습니다. 실장 전 검증 실험이 필요합니다.
5. **형상 모델의 단순화** — 2D 식각 프로파일은 단순화된 수학 함수로 표현한
   교육용 모델로, 실제 형상 계측 결과와 차이가 있을 수 있습니다.
6. **하이퍼파라미터 탐색 범위** — Optuna 탐색 공간과 trial 수(기본 30회)에 따라
   결과가 달라질 수 있습니다.
        """)


# ---------------------------------------------------------------------------
# AI Auto Troubleshooting (자동 분석) 유틸
# ---------------------------------------------------------------------------
def _pid_alive(pid: int) -> bool:
    """Windows 에서 프로세스 생존 여부 확인 (안전한 쿼리 전용)."""
    if not pid:
        return False
    try:
        if os.name == "nt":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, int(pid))  # QUERY_LIMITED
            if not handle:
                return False
            code = ctypes.c_ulong(0)
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            kernel32.CloseHandle(handle)
            return bool(ok) and code.value == 259
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def read_automl_status() -> dict:
    """진행 상태 JSON 을 읽습니다. 없거나 손상되면 {} 반환."""
    try:
        with open(AUTOML_STATUS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def automl_job_running() -> bool:
    """자동 분석이 실제로 진행 중인지 확인 (죽은 프로세스는 stale 로 처리)."""
    st_info = read_automl_status()
    if st_info.get("status") != "running":
        return False
    if not _pid_alive(int(st_info.get("pid", 0))):
        return False
    return True


def launch_automl_job(mode: str, method: str, n_iter: int) -> str:
    """run_automl.py 를 별도 프로세스(데몬)로 실행하고 job id 를 반환합니다.

    - 장기 학습 작업은 Streamlit 리런과 분리되어 실행됩니다.
    - 진행 상태는 AUTOML_STATUS_PATH 파일에 기록되어 페이지가 폴링합니다.
    - 이미 실행 중이면 중복 실행하지 않습니다.
    """
    job_id = f"auto_{uuid.uuid4().hex[:10]}"
    AUTOML_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(BASE_DIR / "run_automl.py"),
        "--mode", mode, "--method", method, "--n-iter", str(n_iter),
        "--status-file", str(AUTOML_STATUS_PATH),
        "--job-id", job_id,
        "--results-dir", str(AUTOML_DIR),
        "--models-dir", str(MODELS_DIR),
    ]
    kwargs = dict(cwd=str(BASE_DIR), stdin=subprocess.DEVNULL)
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)  # type: ignore[attr-defined]
    log_f = open(AUTOML_LOG_PATH, "w", encoding="utf-8")
    try:
        kwargs.update(stdout=log_f, stderr=subprocess.STDOUT)
        subprocess.Popen(cmd, **kwargs)
    finally:
        log_f.close()
    return job_id


def page_automl():
    st.title("7. AI Auto Troubleshooting")
    st.caption(
        "버튼 하나로 데이터 분석 → 모델 자동 학습 → 하이퍼파라미터 최적화 → 최적 모델 선정 "
        "→ SHAP 원인 후보 분석 → 공정 조건 추천 → 식각 프로파일 비교 → HTML 보고서 생성까지 "
        "자동 실행합니다. (장기 작업은 별도 프로세스로 실행됩니다.)")

    # ---------------- 시작 화면 ----------------
    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        mode_labels = {"빠른 분석 (모델별 5회)": "fast",
                       "일반 분석 (모델별 15회)": "normal",
                       "정밀 분석 (모델별 30회)": "precise"}
        mode_label = st.radio("분석 모드", list(mode_labels.keys()),
                              horizontal=False)
        mode = mode_labels[mode_label]
    with col2:
        method_label = st.radio("공정 조건 탐색 방법",
                                ["Differential Evolution", "Random Search"],
                                horizontal=False)
        method = "de" if method_label.startswith("Differential") else "random"
        n_iter = st.slider("조건 탐색 횟수", min_value=100, max_value=2000,
                           value=400, step=100)
    with col3:
        st.info(
            "**실행 정보**\n"
            "- 후보 모델: Random Forest / XGBoost / Extra Trees / HistGradientBoosting\n"
            "- 선정 기준: 검증셋 Macro F1 (동점 시 Recall·복잡도·추론 시간)\n"
            "- 최종 테스트셋은 모델 선정에 사용하지 않음 (과적합 방지)\n"
            "- 산출물: `results/automl/` + `models/best_model.pkl`\n"
            "- 현재 공정 조건은 기본값을 사용합니다 (세부 조건은 "
            "`run_automl.py --gas ...` 로 지정 가능)")

    running = automl_job_running()
    start_btn = st.button("🚀 AI 자동 분석 시작", type="primary",
                          disabled=running,
                          help="이미 실행 중인 분석이 있습니다." if running else "")

    if start_btn and not running:
        job_id = launch_automl_job(mode, method, n_iter)
        st.session_state["automl_job_id"] = job_id
        st.success(f"자동 분석을 시작했습니다. (job: {job_id})")
        st.rerun()

    st.divider()

    # ---------------- 진행 상태 ----------------
    st.markdown("### 📊 진행 상태")
    status = read_automl_status()
    if not status:
        st.info("아직 실행 기록이 없습니다. 'AI 자동 분석 시작' 버튼을 누르세요.")
    else:
        prog = float(status.get("overall_progress", 0.0))
        st.progress(min(prog / 100.0, 1.0),
                    text=f"{prog:.0f}% — {status.get('stage_name', '')}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("상태", {"running": "⏳ 실행 중", "done": "✅ 완료",
                           "error": "❌ 실패"}.get(status.get("status", ""),
                                                  status.get("status", "-")))
        m2.metric("분석 모드", status.get("mode", "-"))
        m3.metric("시작 시각", status.get("started_at", "-"))
        m4.metric("소요 시간(초)", status.get("elapsed_sec", "-"))
        if status.get("message"):
            st.code(status["message"], language=None)
        if status.get("status") == "error":
            st.error(f"분석 실패: {status.get('error')}")

        # 10단계 목록
        cur_idx = int(status.get("stage_index", 0))
        cols = st.columns(2)
        for i, name in enumerate(AUTOML_STAGES, start=1):
            if i < cur_idx:
                mark = "✅"
            elif i == cur_idx:
                mark = "▶️"
            else:
                mark = "⏳"
            with cols[i % 2]:
                st.markdown(f"{mark} {name}")

        if st.button("🔄 상태 새로고침"):
            st.rerun()

    st.divider()

    # ---------------- 결과 표시 ----------------
    results_json_path = AUTOML_DIR / "automl_results.json"
    if results_json_path.exists():
        st.markdown("### 📈 분석 결과")
        if status.get("status") != "done":
            st.caption("※ 대시보드에서 실행한 작업이 아니므로 **최근 실행 결과**를 "
                       "표시합니다. (CLI: `run_automl.py`)")
        with open(results_json_path, encoding="utf-8") as f:
            res = json.load(f)

        r1, r2, r3, r4 = st.columns(4)
        best_name = res.get("best_model", "-")
        r1.metric("최적 모델", best_name)
        r2.metric("검증 Macro F1",
                  f"{res.get('selection_scores', {}).get('val_macro_f1', float('nan')):.4f}")
        tm = res.get("test_metrics", {})
        r3.metric("테스트 Accuracy", f"{tm.get('accuracy', float('nan')):.4f}")
        r4.metric("테스트 Macro F1", f"{tm.get('macro_f1', float('nan')):.4f}")

        cmp_path = AUTOML_DIR / "automl_model_comparison.csv"
        if cmp_path.exists():
            st.markdown("#### 모델별 성능 비교 (검증셋 기준)")
            st.dataframe(pd.read_csv(cmp_path).round(4),
                         use_container_width=True, hide_index=True)

        st.markdown("#### 불량 원인 후보 (전체 중요 변수)")
        imp_png = AUTOML_DIR / "automl_feature_importance.png"
        if imp_png.exists():
            st.image(str(imp_png), use_container_width=True)
        imp_csv = AUTOML_DIR / "automl_feature_importance.csv"
        if imp_csv.exists():
            st.dataframe(pd.read_csv(imp_csv).round(6),
                         use_container_width=True, hide_index=True)
        st.caption("※ 중요도는 모델 예측에 대한 설명이며 물리적 인과관계를 증명하지 않습니다.")

        st.markdown("#### 추천 공정 조건")
        rec = res.get("recommendation", {})
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("기존 예측", rec.get("defect_before", "-"))
        rc2.metric("추천 후 예측", rec.get("defect_after", "-"))
        rc3.metric("정상(Normal) 확률",
                   f"{rec.get('normal_prob_before', float('nan')):.3f} → "
                   f"{rec.get('normal_prob_after', float('nan')):.3f}")
        rec_csv = AUTOML_DIR / "automl_recommendation.csv"
        if rec_csv.exists():
            st.dataframe(pd.read_csv(rec_csv), use_container_width=True,
                         hide_index=True)

        st.markdown("#### 식각 프로파일 전후 비교 (시뮬레이터)")
        prof_png = AUTOML_DIR / "validation_profile_comparison.png"
        delta_png = AUTOML_DIR / "validation_metric_delta.png"
        vc1, vc2 = st.columns(2)
        if prof_png.exists():
            with vc1:
                st.image(str(prof_png), use_container_width=True)
        if delta_png.exists():
            with vc2:
                st.image(str(delta_png), use_container_width=True)
        val_csv = AUTOML_DIR / "validation_metrics.csv"
        if val_csv.exists():
            st.dataframe(pd.read_csv(val_csv).round(2),
                         use_container_width=True, hide_index=True)
        st.caption("시뮬레이터는 합성 데이터 생성과 동일한 가정을 공유하므로 "
                   "동일 가정 내 정합성 확인이며 독립적 물리 검증이 아닙니다.")

        # 다운로드
        st.markdown("#### 📥 결과 다운로드")
        d1, d2, d3 = st.columns(3)
        report_path = AUTOML_DIR / "automl_report.html"
        if report_path.exists():
            with d1:
                st.download_button("HTML 보고서 다운로드",
                                   data=report_path.read_bytes(),
                                   file_name="automl_report.html",
                                   mime="text/html")
        with d2:
            st.download_button("결과 JSON 다운로드",
                               data=json.dumps(res, ensure_ascii=False,
                                               indent=2),
                               file_name="automl_results.json",
                               mime="application/json")
        meta_path = AUTOML_DIR / "run_metadata.json"
        if meta_path.exists():
            with d3:
                st.download_button("실행 메타데이터 다운로드",
                                   data=meta_path.read_bytes(),
                                   file_name="run_metadata.json",
                                   mime="application/json")

        if report_path.exists():
            with st.expander("📄 HTML 보고서 열기 (raw)"):
                st.markdown(report_path.read_text(encoding="utf-8"),
                            unsafe_allow_html=True)

    # ---------------- 실행 로그 ----------------
    if AUTOML_LOG_PATH.exists():
        with st.expander("🖥 실행 로그 (최신 40줄)"):
            try:
                lines = AUTOML_LOG_PATH.read_text(
                    encoding="utf-8", errors="replace").splitlines()
                st.code("\n".join(lines[-40:]), language=None)
            except Exception as exc:
                st.caption(f"로그 읽기 실패: {exc}")


def page_literature_validation():
    """8. Literature Validation — 실제 반도체 식각 논문과 시뮬레이션 결과 비교."""
    st.title("8. Literature Validation")
    st.caption("실제 반도체 식각 연구 논문과 본 프로젝트의 수학적 모델/머신러닝 결과를 비교합니다.")

    lit_dir = RESULTS_DIR / "literature_validation"
    lit_dir.mkdir(parents=True, exist_ok=True)

    # 기존 분석 결과물 경로
    summary_md = lit_dir / "mathematical_model_summary.md"
    comparison_csv = lit_dir / "literature_comparison.csv"
    report_md = lit_dir / "physical_consistency_report.md"
    trends_png = lit_dir / "parameter_trend_comparison.png"
    shap_png = lit_dir / "shap_literature_comparison.png"
    trends_csv = lit_dir / "parameter_trends.csv"

    # 결과물이 없으면 기존 시뮬레이터로 재생성 (ML 재학습 없음)
    if not all(p.exists() for p in [summary_md, comparison_csv, report_md, trends_png, shap_png]):
        st.info("literature_validation 결과물이 일부 없어 기존 시뮬레이터로 분석을 다시 생성합니다. "
                "(머신러닝 모델은 재학습하지 않습니다.)")
        try:
            from src.literature_validation import main as _run_lit
            _run_lit()
            st.success("분석 결과를 생성했습니다.")
            st.rerun()
        except Exception as exc:
            st.error(f"literature_validation 재생성 중 오류 발생: {exc}")
            return

    # ------------------------------------------------------------------
    # 상단: 연구 목적 및 시뮬레이션 한계
    # ------------------------------------------------------------------
    st.markdown("### 🎯 연구 목적")
    st.markdown(
        """
**연구 주제: 머신러닝을 활용한 반도체 Etching 공정의 트러블슈팅 성능 개선**

본 페이지는 반도체 Etching 공정에서 발생하는 Undercut, Bowing, Etch Stop, Footing 등의 불량과
기존 트러블슈팅 방식의 한계를 설명하고,
식각 공정의 물리적 특성을 단순화한 수학적 모델로 생성한 합성 데이터를 바탕으로
Random Forest, XGBoost, HistGradientBoosting 등의 머신러닝 모델을 활용하여 불량을 예측합니다.
AutoML로 모델을 최적화하고 SHAP으로 주요 원인 후보 변수를 분석하며,
머신러닝이 추천한 공정 조건을 식각 시뮬레이터에 적용하여 개선 전후를 비교합니다.

**Guo et al. (2024)** 및 **Seo et al. (2025)** 논문은
시뮬레이션의 물리적 경향과 머신러닝 분석 결과의 정성적 타당성을 검토하기 위한 **참고 문헌**입니다.
논문 비교는 연구의 핵심 목표가 아니라 시뮬레이터의 물리적 개연성을 뒷받침하는 보조 검증입니다.

> ⚠️ 본 연구는 **합성 데이터 기반 실험**입니다.
> 실제로 측정하지 않은 개선율, 시간 단축률, 정확도는 제시하지 않으며,
> 기존 방식과 AI 방식의 정량적 비교는 아직 수행되지 않았습니다.
> 따라서 본 페이지의 비교는 정량적 검증이 아닌 정성적 평가이며,
> 논문과 시뮬레이션의 정성적 일치성을 실제 공정에서의 정량적 검증으로 표현하지 않습니다.
        """
    )

    st.markdown("### ⚠️ 시뮬레이션 및 모델의 한계")
    st.markdown(
        """
1. 본 시스템의 데이터와 형상 모델은 **교육용 합성(가상) 모델**입니다.
2. `gas_chemistry`, `gas_flow_sccm` 등은 실제 가스 비율/종류가 아닌 **단순화된 지표**입니다.
3. SHAP 중요도는 **모델 예측 기여도**이지, 물리적 인과관계의 직접 증거는 아닙니다.
4. 논문과 공정 조건·목표 변수가 다륯므로 **정성적 비교**에 한계가 있습니다.
        """
    )

    # ------------------------------------------------------------------
    # 중앙: 논문 vs 시뮬레이션 그래프 비교
    # ------------------------------------------------------------------
    st.markdown("### 📊 공정 변수별 경향 비교")
    st.caption(
        "아래 그래프는 **기존 시뮬레이터**에서 다른 변수를 고정하고 한 변수만 변화시켜 계산한 결과입니다. "
        "논문에서 동일한 추세를 보고한 경우에만 정성적으로 비교합니다."
    )

    if trends_png.exists():
        st.image(str(trends_png),
                 caption="Simulation trends (default conditions, no noise)",
                 use_container_width=True)
    else:
        st.warning("parameter_trend_comparison.png 파일이 없습니다.")

    if trends_csv.exists():
        with st.expander("📋 추세 데이터 보기"):
            st.dataframe(pd.read_csv(trends_csv), use_container_width=True)

    st.markdown("### 📑 논문 인용 및 판정")
    if comparison_csv.exists():
        cmp_df = pd.read_csv(comparison_csv)
        # 판정에 따른 색상 태그
        badge = {
            "confirmed": "🟢",
            "partial / conflicting": "🟡",
            "partial": "🟡",
            "confirmed as influential": "🔵",
        }
        cmp_df["판정 태그"] = cmp_df["qualitative_match"].map(
            lambda x: badge.get(x, "⚪")
        )
        st.dataframe(cmp_df[["variable", "our_model_role",
                             "paper1_guo2024", "paper2_seo2025",
                             "qualitative_match", "notes"]],
                     use_container_width=True)
    else:
        st.warning("literature_comparison.csv 파일이 없습니다.")

    # PDF 페이지 인덱스
    citations_json = lit_dir / "literature_citations.json"
    if citations_json.exists():
        with open(citations_json, encoding="utf-8") as f:
            citations = json.load(f)
        st.markdown("#### 📄 제공된 PDF에서 키워드별 페이지 번호")
        st.caption("원문 PDF가 제공된 경우, PyMuPDF로 텍스트를 검색한 결과입니다. "
                   "해당 페이지에서 실제 그래프/수치를 직접 확인해야 합니다.")
        for paper, info in citations.items():
            with st.expander(f"{paper}"):
                if "error" in info:
                    st.warning(f"{info['error']}: {info.get('path', '')}")
                    continue
                st.caption(f"전체 페이지 수: {info.get('total_pages', '-')}")
                rows = []
                for var, pages in info.get("index", {}).items():
                    rows.append({
                        "variable": var,
                        "pages": ", ".join(map(str, pages)) if pages else "미확인",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("literature_citations.json 파일이 없습니다. "
                "`python -m src.literature_validation`을 실행하면 PDF 페이지 인덱스를 생성합니다.")

    # 판정 요약 박스
    st.markdown("#### 종합 판정 요약")
    verdict_cols = st.columns(5)
    verdicts = [
        ("정성적으로 일치", "Bias↑→Etch rate↑, Source↑→Etch rate↑, Gas chemistry 영향"),
        ("부분적으로 일치", "Source power 포화 없음, Gas chemistry 추상화, Passivation 정성적 일치"),
        ("불일치", "Temperature↑→Etch rate↑ (Guo et al.는 STI height 감소 보고, 변수/지표 상이)"),
        ("비교 불가능", "Pressure 추세, Gas chemistry→Undercut severity에 대한 직접 보고 부재"),
        ("문헌 확인 필요", "세부 수치·그림 번호는 제공된 PDF 직접 확인 권장"),
    ]
    for col, (title, desc) in zip(verdict_cols, verdicts):
        with col:
            st.markdown(f"**{title}**")
            st.caption(desc)

    # ------------------------------------------------------------------
    # 하단: SHAP 변수 중요도 비교 및 종합 평가
    # ------------------------------------------------------------------
    st.markdown("### 🔍 SHAP 변수 중요도 vs 문헌")
    st.warning(
        "SHAP 중요도는 **모델 예측에 얼마나 많이 사용되었는지**를 나타낼 뿐, "
        "실제 공정 변수의 인과적 중요도 순위를 의미하지 않습니다."
    )

    if shap_png.exists():
        st.image(str(shap_png),
                 caption="HistGradientBoosting SHAP importance and literature support",
                 use_container_width=True)
    else:
        st.warning("shap_literature_comparison.png 파일이 없습니다.")

    st.markdown("### 📝 최종 결론")
    if report_md.exists():
        st.markdown(report_md.read_text(encoding="utf-8"))
    else:
        st.warning("physical_consistency_report.md 파일이 없습니다.")

    with st.expander("📘 수학적 모델 상세 보기"):
        if summary_md.exists():
            st.markdown(summary_md.read_text(encoding="utf-8"))
        else:
            st.warning("mathematical_model_summary.md 파일이 없습니다.")


PAGES = {
    "1. Project Overview": page_overview,
    "2. Etching Process Simulator": page_simulator,
    "3. Machine Learning Performance": page_performance,
    "4. Explainable AI": page_xai,
    "5. AI Troubleshooting": page_troubleshooting,
    "6. Research Results": page_research,
    "7. AI Auto Troubleshooting": page_automl,
    "8. Literature Validation": page_literature_validation,
}


def main():
    with st.sidebar:
        st.title("🔬 AI Etching\nTroubleshooting")
        st.caption("반도체 Etching 공정\nAI 트러블슈팅 시뮬레이터")
        page = st.radio("Menu (메뉴 선택)", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()
        st.caption(
            "⚠️ 본 시스템의 데이터/형상 모델은\n"
            "교육용 합성 모델이며 실제 공정\n"
            "물리 검증 결과가 아닙니다.\n\n"
            "실행: `python run_pipeline.py`\n"
            "대시보드: `python -m streamlit run app.py`"
        )
    PAGES[page]()


if __name__ == "__main__":
    main()
