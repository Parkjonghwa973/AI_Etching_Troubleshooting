"""
report_generator.py
===================

AutoML 자동 분석이 완료되면 결과를 **단일 HTML 보고서**로 생성하는 모듈입니다.

포함 항목
--------
1. 분석 메타데이터 (실행 시간, 데이터, random_state, 패키지 버전, 모델 설정)
2. 데이터 분석 결과 (분할 정보)
3. 머신러닝 모델 성능 비교 (4개 모델, 실제 측정값만)
4. 최적 모델 선정 결과 (기준과 실제 점수)
5. 하이퍼파라미터 최적화 결과 (모델별 best CV / 최적화 시간)
6. SHAP 기반 불량 원인 후보 (중요 변수, 불량 유형별 중요 변수)
7. 추천 공정 조건 + 기존 조건 비교
8. Etching 프로파일 전후 비교 이미지
9. 연구의 한계 (반드시 포함)

.. note::
    모든 수치는 실제 실행에서 측정된 값만 사용하며, 가상의 수치를 임의로
    작성하지 않습니다. 이미지는 base64 로 인라인하여 단일 파일로 이동 가능하게 합니다.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

DEFAULT_LIMITATIONS = [
    "본 연구의 모든 데이터는 실제 장비 측정이 아닌 **교육용 합성 데이터**입니다. "
    "생성 규칙 기반이므로 실제 공정의 물리·화학 현상을 완전히 대표하지 못합니다.",
    "SHAP(또는 대체 설명 방법) 결과는 **모델 예측에 대한 설명**이며, "
    "물리적 인과관계를 증명하지 않습니다.",
    "AI 추천 공정 조건은 학습 데이터 분포와 탐색 범위 내의 후보일 뿐이며, "
    "실제 장비 적용 시 반드시 엔지니어 검증이 필요합니다.",
    "식각 시뮬레이터는 합성 데이터 생성과 **동일한 가정**을 공유하므로, "
    "시뮬레이터 검증은 독립적인 물리 검증이 아닙니다.",
    "최종 테스트셋 성능과 교차검증 성능은 다를 수 있습니다. "
    "하이퍼파라미터 최적화가 항상 테스트셋 성능 향상을 보장하지 않습니다.",
    "이 프로젝트는 실제 반도체 장비를 제어하지 않습니다 (교육용 시뮬레이션 전용).",
]


def _img_to_data_uri(path: Optional[str | Path]) -> str:
    """PNG 이미지를 base64 data URI 로 인코딩합니다."""
    if path is None:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _df_html(df: Optional[pd.DataFrame], floatfmt: str = "{:.4f}") -> str:
    """DataFrame 을 스타일 있는 HTML 표로 변환합니다 (NaN 안전)."""
    if df is None or len(df) == 0:
        return "<p class='muted'>해당 결과 없음</p>"
    disp = df.copy()
    for col in disp.columns:
        if pd.api.types.is_float_dtype(disp[col]):
            disp[col] = disp[col].map(
                lambda v: floatfmt.format(v) if pd.notna(v) else "-")
    return disp.to_html(index=False, border=0, classes="tbl", escape=False)


def _kv_html(d: Dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td class='k'>{k}</td><td class='v'>{v}</td></tr>"
        for k, v in d.items())
    return f"<table class='tbl kv'>{rows}</table>"


def _pct(v: Optional[float]) -> str:
    return f"{v * 100:.2f}%" if v is not None and pd.notna(v) else "-"


def generate_html_report(
    report_data: Dict[str, Any],
    out_path: str | Path,
) -> Path:
    """HTML 보고서를 생성합니다.

    Parameters
    ----------
    report_data : dict, run_automl.py 가 수집한 실제 측정 결과
        - metadata (dict) : 실행 환경/설정
        - data (dict) : 분할 정보
        - model_comparison (DataFrame) : 모델별 성능 비교
        - selection (dict) : model_selector 결과 (scores/criteria 등)
        - optimization (DataFrame or dict) : Optuna 결과 요약
        - shap (dict) : importance DataFrame, per-class top features, method
        - troubleshooting (dict) : comparison/summary/improvement DataFrame 등
        - validation (dict) : before/after 지표 delta DataFrame, 이미지 경로
        - limitations (list[str], optional) : 기본 한계 문구 + 추가
    out_path : 출력 HTML 경로

    Returns
    -------
    Path : 생성된 HTML 파일 경로
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metadata: Dict[str, Any] = report_data.get("metadata", {})
    data_info: Dict[str, Any] = report_data.get("data", {})
    comparison: Optional[pd.DataFrame] = report_data.get("model_comparison")
    selection: Dict[str, Any] = report_data.get("selection", {})
    optimization: Any = report_data.get("optimization")
    shap_info: Dict[str, Any] = report_data.get("shap", {})
    trouble: Dict[str, Any] = report_data.get("troubleshooting", {})
    validation: Dict[str, Any] = report_data.get("validation", {})
    limitations: List[str] = list(report_data.get("limitations", [])) or list(
        DEFAULT_LIMITATIONS)

    # --- 섹션 구성 ---------------------------------------------------------
    sec_meta = _kv_html(metadata)
    sec_data = _kv_html(data_info)

    sec_comparison = _df_html(comparison)

    scores = selection.get("scores", {})
    sec_selection = _kv_html({
        "선정 모델": selection.get("best_display", selection.get("best_model_name", "-")),
        "검증 Accuracy": _pct(scores.get("val_accuracy")),
        "검증 Macro F1": _pct(scores.get("val_macro_f1")),
        "최저 클래스 Recall": _pct(scores.get("worst_class_recall")),
        "모델 복잡도 (trees×depth)": scores.get("complexity", "-"),
        "추론 시간 (100건, ms)": f"{scores.get('inference_ms_per_100', float('nan')):.2f}"
        if pd.notna(scores.get("inference_ms_per_100")) else "-",
        "동점 처리(tie-break) 적용": "예" if selection.get("tie_break_applied") else "아니오",
        "동점 처리 사유": selection.get("tie_break_reason", "-"),
    })
    sec_criteria = "<ul>" + "".join(
        f"<li>{c}</li>" for c in selection.get("criteria", [])) + "</ul>"

    if isinstance(optimization, pd.DataFrame):
        sec_opt = _df_html(optimization)
    elif isinstance(optimization, dict):
        sec_opt = _kv_html(optimization)
    else:
        sec_opt = "<p class='muted'>해당 결과 없음</p>"

    shap_method = shap_info.get("method", "N/A")
    sec_shap = _df_html(shap_info.get("importance"))
    per_class = shap_info.get("per_class_top")
    sec_shap_class = ""
    if isinstance(per_class, dict):
        rows = "".join(
            f"<tr><td class='k'>{cls}</td><td class='v'>{', '.join(feats)}</td></tr>"
            for cls, feats in per_class.items())
        sec_shap_class = f"<table class='tbl kv'>{rows}</table>"
    elif isinstance(per_class, pd.DataFrame):
        sec_shap_class = _df_html(per_class)

    sec_trouble = _df_html(trouble.get("comparison"))
    sec_trouble_summary = _df_html(trouble.get("summary"))
    improvement = trouble.get("improvement", {})
    sec_improvement = _kv_html({
        "기존 예측 불량": improvement.get("defect_before", "-"),
        "추천 후 예측 불량": improvement.get("defect_after", "-"),
        "정상 확률 변화": f"{improvement.get('normal_prob_delta', float('nan')):+.4f}"
        if pd.notna(improvement.get("normal_prob_delta")) else "-",
        "평가": improvement.get("verdict", "-"),
        "개선 가능성 수준": improvement.get("level", "-"),
    })
    warnings_list: List[str] = trouble.get("warnings", [])

    sec_validation = _df_html(validation.get("delta"), floatfmt="{:.2f}")
    sec_validation_verdict = _kv_html({
        "기존 조건 예측": validation.get("before_defect_name", "-"),
        "추천 조건 예측": validation.get("after_defect_name", "-"),
        "심각도 변화": f"{validation.get('severity_delta', float('nan')):+.2f}"
        if pd.notna(validation.get("severity_delta")) else "-",
        "평가": validation.get("verdict", "-"),
    })
    img_profile = _img_to_data_uri(validation.get("profile_plot"))
    img_delta = _img_to_data_uri(validation.get("delta_plot"))

    sec_warnings = ""
    if warnings_list:
        items = "".join(f"<li class='warn'>{w}</li>" for w in warnings_list)
        sec_warnings = f"<div class='box warnbox'><b>경고</b><ul>{items}</ul></div>"

    sec_limitations = "".join(f"<li>{t}</li>" for t in limitations)

    now = time.strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>AI AutoML Etching Troubleshooting Report</title>
<style>
 body {{ font-family: 'Segoe UI', 'Malgun Gothic', sans-serif; margin: 0;
        background: #f5f6fa; color: #2d3436; }}
 .wrap {{ max-width: 1080px; margin: 0 auto; padding: 28px 32px 64px; }}
 header {{ background: #1e3a5f; color: #fff; padding: 26px 32px; }}
 header h1 {{ margin: 0 0 6px; font-size: 24px; }}
 header p {{ margin: 0; opacity: .85; font-size: 13px; }}
 h2 {{ border-left: 5px solid #2E86C1; padding-left: 10px; margin: 36px 0 14px;
       font-size: 19px; }}
 h3 {{ margin: 20px 0 8px; font-size: 15px; color: #1e3a5f; }}
 table.tbl {{ border-collapse: collapse; width: 100%; background: #fff;
              font-size: 13.5px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
 table.tbl th, table.tbl td {{ border-bottom: 1px solid #e3e6ec; padding: 8px 12px;
                               text-align: left; }}
 table.tbl th {{ background: #eef2f8; font-weight: 600; }}
 table.kv td.k {{ width: 260px; font-weight: 600; color: #4a5568; }}
 .box {{ background: #fff; border: 1px solid #e3e6ec; border-radius: 8px;
         padding: 14px 18px; margin: 14px 0; }}
 .warnbox {{ background: #fff8f6; border-color: #f5b7b1; }}
 .warnbox li {{ color: #b03a2e; }}
 .muted {{ color: #8a94a6; font-size: 13px; }}
 img {{ max-width: 100%; border: 1px solid #dfe3ea; border-radius: 8px;
        margin: 8px 0; }}
 .caveat {{ background: #fdf6e3; border-left: 5px solid #d6a10d; padding: 12px 16px;
            margin: 16px 0; font-size: 13.5px; }}
 footer {{ margin-top: 48px; font-size: 12px; color: #8a94a6; text-align: center; }}
</style>
</head>
<body>
<header>
  <h1>AI AutoML Etching Troubleshooting Report</h1>
  <p>반도체 Etching 공정 AI 자동 머신러닝 최적화 · 트러블슈팅 결과 보고서</p>
  <p>생성 시각: {now}</p>
</header>
<div class="wrap">

<div class="caveat">
 <b>연구의 한계 (요약)</b> — 본 보고서의 모든 수치는 실제 실행에서 측정된 값이며,
 데이터는 교육용 합성 데이터입니다. SHAP 결과는 모델 예측 설명이지 물리적
 인과관계가 아니고, 추천 조건이 실제 공정 불량을 감소시킨다고 단정할 수 없습니다.
</div>

<h2>1. 분석 메타데이터</h2>
{sec_meta}

<h2>2. 데이터 분석 결과</h2>
{sec_data}

<h2>3. 머신러닝 모델 성능 비교</h2>
<p class="muted">모든 모델은 동일한 학습/검증 분할에서 학습되었습니다.</p>
{sec_comparison}

<h2>4. 최적 모델 선정 결과</h2>
<h3>선정 기준</h3>{sec_criteria}
<h3>실제 점수</h3>{sec_selection}

<h2>5. 하이퍼파라미터 최적화 결과</h2>
{sec_opt}

<h2>6. AI 불량 원인 후보 분석</h2>
<p class="muted">설명 방법: {shap_method} — 모델 예측에 대한 설명이며,
물리적 인과관계를 증명하지 않습니다.</p>
<h3>전체 중요 변수</h3>
{sec_shap}
<h3>불량 유형별 주요 변수</h3>
{sec_shap_class}

<h2>7. 추천 공정 조건 및 기존 조건 비교</h2>
{sec_trouble_summary}
<h3>변수별 전후 비교</h3>
{sec_trouble}
<h3>개선 가능성 평가</h3>
{sec_improvement}
{sec_warnings}

<h2>8. Etching 프로파일 전후 비교 (시뮬레이터)</h2>
<p class="muted">시뮬레이터와 합성 데이터 생성 규칙은 동일한 가정을 공유하므로
동일 가정 내 정합성 확인입니다 (독립적 물리 검증 아님).</p>
{sec_validation_verdict}
<h3>지표 차이</h3>
{sec_validation}
<h3>프로파일 비교</h3>
{f"<img src='{img_profile}' alt='profile comparison'>" if img_profile else "<p class='muted'>프로파일 이미지 없음</p>"}
<h3>지표 차이 그래프</h3>
{f"<img src='{img_delta}' alt='metric delta'>" if img_delta else "<p class='muted'>차이 그래프 없음</p>"}

<h2>9. 연구의 한계</h2>
<ul>{"".join(f"<li>{t}</li>" for t in limitations)}</ul>

<footer>AI_Etching_Troubleshooting — 교육용 합성 데이터 기반 시뮬레이터 · 보고서 생성: report_generator.py</footer>
</div>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    print("report_generator: run_automl.py 가 호출합니다 (직접 실행 불필요).")
