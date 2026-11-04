"""
literature_validation.py
========================

기존 AI_Etching_Troubleshooting 시뮬레이션/머신러닝 결과를
실제 반도체 식각 연구 논문(Guo et al. 2024, Seo et al. 2025)과 비교하는
분석 보고서를 생성합니다.

- 새로운 AutoML 모델 학습 없이 기존 결과만 활용
- 기존 코드의 수학식을 정리하고 물리적 의미/가정/한계를 기술
- 시뮬레이터로 변수별 경향을 계산하고 그래프로 저장
- SHAP 결과와 문헌의 물리적 중요도를 정성적으로 비교
- 결과물: results/literature_validation/
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_generator import (
    INPUT_COLUMNS,
    PROCESS_BOUNDS,
    compute_indicators,
    defect_probabilities,
)
from src.etching_simulator import default_process_conditions

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "literature_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CONDITIONS = default_process_conditions()

# 사용자 시스템에 제공된 논문 PDF 경로 (존재하면 페이지 인덱싱 시도)
PAPER_PDFS = {
    "Guo et al. (2024)": Path.home() / (
        r"OneDrive\바탕 화면\Optimizing_Plasma_Etching_Integrating_Precise_"
        r"Three-Dimensional_Etching_Simulation_and_Machine_Learning_for_"
        r"Multi-Objective_Optimization.pdf"
    ),
    "Seo et al. (2025)": Path.home() / (
        r"Downloads\Explainable AI 및 Machine Learning 기반 식각 공정 "
        r"Etch rate 예측.pdf.pdf"
    ),
}

PDF_KEYWORDS = {
    "bias_power_w": ["bias power", "bias rf", "rf bias", "바이어스"],
    "source_power_w": ["source power", "source rf", "rf source", "소스"],
    "temperature_c": ["temperature", "substrate temperature", "온도"],
    "pressure_mtorr": ["pressure", "process pressure", "압력"],
    "gas_chemistry": ["gas chemistry", "gas ratio", "gas composition", "가스"],
    "gas_flow_sccm": ["gas flow", "flow rate", "유량"],
    "etch_rate_nm_min": ["etch rate", "etching rate", "식각 속도", "etch depth"],
    "machine_learning": ["machine learning", "random forest", "xgboost", "histgradient", "머신러닝"],
    "shap": ["shap", "explainable ai", "feature importance", "설명가능"],
}


def index_pdf_pages(pdf_path: Path, keywords: dict):
    """PDF에서 키워드가 등장하는 페이지 번호(1-based)를 색인합니다."""
    if not pdf_path.exists():
        return {"error": "PDF file not found", "path": str(pdf_path)}
    try:
        import pymupdf
    except Exception:
        try:
            import fitz as pymupdf  # type: ignore[no-redef]
        except Exception as exc:
            return {"error": f"PyMuPDF not available: {exc}"}

    doc = pymupdf.open(str(pdf_path))
    total_pages = len(doc)
    index = {}
    for var, kws in keywords.items():
        pages = set()
        for page_num in range(total_pages):
            text = doc[page_num].get_text()
            lower = text.lower()
            for kw in kws:
                if kw.lower() in lower:
                    pages.add(page_num + 1)
                    break
        index[var] = sorted(pages)
    doc.close()
    return {"path": str(pdf_path), "total_pages": total_pages, "index": index}


def write_pdf_citations():
    """제공된 PDF에서 변수별 페이지 번호를 추출해 JSON으로 저장합니다."""
    citations = {}
    for paper, pdf_path in PAPER_PDFS.items():
        citations[paper] = index_pdf_pages(pdf_path, PDF_KEYWORDS)
    path = OUT_DIR / "literature_citations.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(citations, f, ensure_ascii=False, indent=2)
    return path


def vary_one_variable(var: str, n: int = 80):
    """한 변수만 범위 내에서 변화시키고 나머지는 기본값으로 고정."""
    lo, hi = PROCESS_BOUNDS[var]
    vals = np.linspace(lo, hi, n)
    rows = []
    for v in vals:
        cond = dict(DEFAULT_CONDITIONS)
        cond[var] = float(v)
        rows.append(cond)
    return pd.DataFrame(rows, columns=INPUT_COLUMNS)


def compute_trends():
    """변수별 경향 데이터를 계산합니다."""
    trend_vars = {
        "bias_power_w": ("Bias power (W)", "etch_rate_nm_min", "Etch rate (nm/min)"),
        "source_power_w": ("Source power (W)", "etch_rate_nm_min", "Etch rate (nm/min)"),
        "temperature_c": ("Temperature (°C)", "etch_rate_nm_min", "Etch rate (nm/min)"),
        "pressure_mtorr": ("Pressure (mTorr)", "etch_rate_nm_min", "Etch rate (nm/min)"),
        "gas_chemistry": ("Gas chemistry", "undercut_prob", "Undercut probability"),
        "gas_chemistry2": ("Gas chemistry", "passivation_index", "Passivation index"),
    }
    results = {}
    for key, (xlabel, ykey, ylabel) in trend_vars.items():
        var = key if key != "gas_chemistry2" else "gas_chemistry"
        df = vary_one_variable(var)
        ind = compute_indicators(df, noise_scale=0.0, add_noise=False)
        lateral = np.asarray(ind.attrs["lateral_factor"])
        bowing = np.asarray(ind.attrs["bowing_driver"])
        probs = defect_probabilities(
            ind, lateral, bowing, df["temperature_c"].to_numpy()
        )
        undercut_prob = probs[:, 1]  # Undercut class = 1

        if ykey == "etch_rate_nm_min":
            y = ind["etch_rate_nm_min"].to_numpy()
        elif ykey == "undercut_prob":
            y = undercut_prob
        else:
            y = ind["passivation_index"].to_numpy()

        results[key] = {
            "x": df[var].to_numpy(),
            "y": y,
            "xlabel": xlabel,
            "ylabel": ylabel,
            "var": var,
        }
    return results


def plot_trends(results: dict):
    """6개 추세를 하나의 PNG로 저장."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()
    order = ["bias_power_w", "source_power_w", "temperature_c",
             "pressure_mtorr", "gas_chemistry", "gas_chemistry2"]
    titles = [
        "(a) Bias power vs Etch rate",
        "(b) Source power vs Etch rate",
        "(c) Temperature vs Etch rate",
        "(d) Pressure vs Etch rate",
        "(e) Gas chemistry vs Undercut prob.",
        "(f) Gas chemistry vs Passivation index",
    ]
    for ax, key, title in zip(axes, order, titles):
        r = results[key]
        ax.plot(r["x"], r["y"], color="#2E86C1", lw=2)
        ax.set_xlabel(r["xlabel"], fontsize=10)
        ax.set_ylabel(r["ylabel"], fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
    fig.suptitle("Parameter trends from the existing simulator (default conditions, no noise)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = OUT_DIR / "parameter_trend_comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def shap_literature_plot():
    """SHAP 중요도 + 문헌 지원 수준 비교 막대 그래프."""
    shap_path = Path(__file__).resolve().parent.parent / "results" / "automl" / "automl_feature_importance.csv"
    if not shap_path.exists():
        return None
    df = pd.read_csv(shap_path)

    # 문헌 지원 수준 (PDF 원문에서 확인한 내용 기준)
    literature_support = {
        "bias_power_w": "confirmed",
        "gas_chemistry": "confirmed",
        "temperature_c": "confirmed",
        "source_power_w": "confirmed",
        "pressure_mtorr": "confirmed",
        "gas_flow_sccm": "confirmed",
    }
    color_map = {
        "confirmed": "#2E86C1",
        "partial": "#E67E22",
        "unconfirmed": "#95A5A6",
    }
    df["support"] = df["feature"].map(literature_support)
    df["color"] = df["support"].map(color_map)

    fig, ax = plt.subplots(figsize=(9, 5))
    df = df.sort_values("importance", ascending=True)
    bars = ax.barh(df["feature"], df["importance"], color=df["color"], edgecolor="white")
    for b, v in zip(bars, df["importance"]):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=9)
    ax.set_xlabel("Mean |SHAP| (HistGradientBoosting)")
    ax.set_title("SHAP importance vs. literature support")
    ax.grid(axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor=color_map["confirmed"], label="Confirmed in literature"),
        Patch(facecolor=color_map["partial"], label="Partial / regime-dependent"),
        Patch(facecolor=color_map["unconfirmed"], label="Not explicitly confirmed"),
    ]
    ax.legend(handles=legend, loc="lower right")
    fig.tight_layout()
    path = OUT_DIR / "shap_literature_comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_mathematical_model_summary():
    md = """# Mathematical Model Summary

## 1. Purpose
This document summarizes the equations used in `src/data_generator.py` and `src/etching_simulator.py`.
All equations are **simplified educational models** for undergraduate demonstration; they are not validated against real plasma physics.

## 2. Input variables
| Variable | Range | Meaning |
|---|---|---|
| gas_chemistry | [0.0, 1.0] | Abstract gas-chemistry indicator (not a specific gas ratio) |
| gas_flow_sccm | [50, 200] | Total gas flow rate |
| pressure_mtorr | [5, 100] | Chamber pressure |
| bias_power_w | [50, 300] | Bias RF power |
| source_power_w | [300, 1500] | Source RF power |
| temperature_c | [20, 100] | Wafer/substrate temperature |

Normalization used throughout:

```
x_n = (x - x_min) / (x_max - x_min)
```

## 3. Core equations in `compute_indicators`

### 3.1 Passivation index
```
P = clip01(sigmoid(3.0*(g-0.5) + 1.0*(p_n-0.5) - 1.6*(b_n-0.4) + 0.5*(s_n-0.5)))
```
- Inputs: gas_chemistry (g), pressure (p_n), bias (b_n), source (s_n)
- Output: P ∈ [0,1], sidewall passivation strength
- Physical meaning: higher gas_chemistry/pressure increase passivation; higher bias reduces it.
- Assumptions: sigmoid saturation; linear weights; no explicit polymer chemistry.
- Empirical coefficients: 3.0, 1.0, -1.6, 0.5 chosen for educational balance.
- Difference from real physics: real passivation depends on halogen/O₂ ratio, wall temperature, polymer deposition kinetics; this is a one-dimensional proxy.

### 3.2 Etch rate
```
ER = 350.0
     * (0.35 + 0.65*s_n)^0.8
     * (0.30 + 0.70*b_n)^0.9
     * (0.40 + 0.60*f_n)^0.5
     * (1.0 - 0.55*P)
     * (1.25 - 0.50*p_n)
     * (0.75 + 0.50*t_n)
```
- Inputs: source (s_n), bias (b_n), flow (f_n), passivation (P), pressure (p_n), temperature (t_n)
- Output: ER (nm/min)
- Physical meaning: source/bias/flow/temperature increase etch rate; passivation and pressure reduce it.
- Assumptions: power-law dependence; no ion-energy distribution; pressure effect is monotonic.
- Empirical coefficients: prefactor 350, exponents 0.8/0.9/0.5, interaction coefficients 0.55/0.50.
- Difference from real physics: real etch rate follows Langmuir adsorption/reaction kinetics, ion-energy thresholds, and pressure-dependent collisionality; our model is a multiplicative polynomial fit.

### 3.3 Anisotropy index
```
A = clip01(sigmoid(1.8*(b_n-0.45) - 1.4*(p_n-0.45) + 2.2*(P-0.5)))
```
- Inputs: bias, pressure, passivation
- Output: A ∈ [0,1], directional etching tendency
- Physical meaning: bias and passivation increase anisotropy; pressure reduces it.
- Assumptions: sigmoid saturation; no angular ion distribution.
- Difference from real physics: real anisotropy is determined by ion angular energy distribution and sidewall scattering.

### 3.4 Lateral etch factor
```
lateral = (1.0 - A) * (0.4 + 0.9*t_n)
```
- Inputs: anisotropy, temperature
- Output: dimensionless lateral-etch tendency
- Physical meaning: low anisotropy and high temperature promote lateral chemical etching.

### 3.5 Footing index
```
F = clip01(sigmoid(2.0*(b_n-0.55) + 1.4*(t_n-0.5) - 1.0*(g-0.5) + 0.8*(s_n-0.5)))
```
- Inputs: bias, temperature, gas_chemistry, source
- Output: F ∈ [0,1], bottom-distortion tendency
- Physical meaning: bias/temperature/source increase footing; gas_chemistry decreases it.
- Difference from real physics: real footing depends on charging, ion deflection, and mask geometry; this is a phenomenological proxy.

### 3.6 Bowing driver
```
B = exp(-((P-0.42)/0.22)^2) * (0.3 + 0.7*p_n) * (0.3 + 0.7*b_n)
```
- Inputs: passivation, pressure, bias
- Output: bowing tendency
- Physical meaning: bowing peaks at intermediate passivation; pressure and bias amplify it.
- Difference from real physics: real bowing is tied to ion reflection/passivation dynamics at the sidewall.

## 4. Defect probability model

Score functions (before softmax):
```
s_undercut = 4.5*lateral + 2.0*(1-P) - 2.3
s_bowing   = 4.0*B + 0.8*b_n - 2.5
s_stop     = 4.2*(0.52 - (ER-40)/260)
s_footing  = 4.0*F - 2.1
s_normal   = clip(3.0 - 1.5*max(s_undercut, s_bowing, s_stop, s_footing), -3, 6)

p_i = exp(s_i) / sum_j exp(s_j)
```
- Output: 5-class probabilities
- Physical meaning: each defect has a handcrafted score driven by a few indicators.
- Assumptions: mutually exclusive classification; score coefficients are manually tuned.
- Difference from real physics: real defect formation is continuous and coupled; no single score determines a defect.

## 5. Defect severity
```
severity = 100 * (0.12 + 0.50*(1-p_normal) + 0.22*lateral + 0.18*(1-ER_norm) + 0.14*F - 0.10*P)
```
- Output: 0–100 severity score
- Physical meaning: higher non-Normal probability, lateral etch, low etch rate, and footing increase severity; passivation decreases it.
- Empirical coefficients: manually chosen weights.

## 6. Profile simulator (`compute_profile`)

Etch depth:
```
D = ER * (etch_time_sec / 60.0)   # default 60 s => D = ER (nm)
D_stop = 0.45 * D   # if defect_type == Etch Stop
```

Base half-width profile:
```
taper = 0.25 * (1 - A)
base_hw(y) = 100.0 * (1.0 - taper * y)
```

Defect-specific deformations:
```
undercut_delta = amp_u * 8.0 * (lateral + 0.35*(1-P)) * exp(-((y-0.06)/0.22)^2)
bowing_delta   = amp_b * 10.0 * B * exp(-((y-0.50)/0.22)^2)
footing_delta  = amp_f * 18.0 * F * (0.5+0.8*b_n)*(0.5+0.6*t_n) * exp(-((y-1.0)/0.16)^2)
stop_rounding  = amp_s * 4.0 * (0.5-y <= 0.1) * exp(-((y-1.0)/0.18)^2)

hw(y) = base_hw(y) + undercut_delta + bowing_delta + footing_delta + stop_rounding
```
- `amp_own = {Undercut:4, Bowing:4, Etch Stop:2, Footing:4}`, residual = 0.15 for non-own defects.
- Profile metrics: Top/Middle/Bottom CD, sidewall angle, max lateral deviation are derived from `hw(y)`.
- Physical meaning: geometric caricature of defect shapes.
- Difference from real physics: real profiles come from solving plasma-surface interactions (transport, reaction, sputtering); our model superposes fixed Gaussian deformation terms.

## 7. SHAP explanation (`explain_model.py`)

Uses `shap.TreeExplainer` on the trained tree-based classifier and reports mean |SHAP|.
- Inputs: 6 raw process variables
- Output: feature importance and local contributions per class
- Caveat: SHAP measures **prediction contribution**, not physical causation.
"""
    path = OUT_DIR / "mathematical_model_summary.md"
    path.write_text(md, encoding="utf-8")
    return path


def write_literature_comparison_csv():
    rows = [
        {
            "variable": "bias_power_w",
            "our_model_role": "Ion bombardment energy / anisotropy driver",
            "paper1_guo2024": "confirmed: Bias RF power strongly affects STI height/etch depth; low-power linear, high-power saturation",
            "paper2_seo2025": "confirmed: bias power is part of CCP RF power scheme",
            "qualitative_match": "confirmed",
            "notes": "Both studies treat RF bias/source power as major knobs. Our model has no high-power saturation.",
        },
        {
            "variable": "gas_chemistry",
            "our_model_role": "Abstract gas composition affecting passivation and chemical etch",
            "paper1_guo2024": "confirmed: gas ratio (Cl₂/HBr/O₂) changes STI height and profile",
            "paper2_seo2025": "confirmed: NF₃/H₂ gas chemistry and flow are key for dielectric etch",
            "qualitative_match": "confirmed",
            "notes": "Our variable is a 0-1 abstraction, not a specific gas ratio.",
        },
        {
            "variable": "temperature_c",
            "our_model_role": "Accelerates chemical etch, affects lateral factor and footing",
            "paper1_guo2024": "confirmed: substrate temperature influences plasma/sidewall reactions; slight STI height decrease observed",
            "paper2_seo2025": "confirmed: substrate temperature among top SHAP features",
            "qualitative_match": "partial / conflicting",
            "notes": "Our etch rate increases with temperature; Guo et al. report slight STI height decrease. Metrics and regimes differ.",
        },
        {
            "variable": "source_power_w",
            "our_model_role": "Plasma density / ion flux driver",
            "paper1_guo2024": "confirmed: Source RF power increases etch depth/rate, saturates at high power",
            "paper2_seo2025": "confirmed: source power is part of CCP configuration",
            "qualitative_match": "confirmed",
            "notes": "Direction matches; saturation behavior is absent in our power-law model.",
        },
        {
            "variable": "pressure_mtorr",
            "our_model_role": "Reduces ion energy via collisions, affects passivation/anisotropy",
            "paper1_guo2024": "confirmed: pressure is a tuned process parameter",
            "paper2_seo2025": "confirmed: process pressure among top SHAP features",
            "qualitative_match": "confirmed as influential",
            "notes": "No explicit pressure-vs-etch-rate trend given in either paper; our model assumes monotonic decrease.",
        },
        {
            "variable": "gas_flow_sccm",
            "our_model_role": "Etchant supply / plasma density modifier",
            "paper1_guo2024": "confirmed: gas flow affects plasma density and etch rate",
            "paper2_seo2025": "confirmed: H₂/NF₃ flow rates are top SHAP features",
            "qualitative_match": "confirmed",
            "notes": "Papers use element-specific flows; our model uses total flow.",
        },
    ]
    df = pd.DataFrame(rows)
    path = OUT_DIR / "literature_comparison.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_physical_consistency_report(trend_results: dict):
    # Read latest automl results if available
    automl_json = Path(__file__).resolve().parent.parent / "results" / "automl" / "automl_results.json"
    best_model = "N/A"
    shap_top = "N/A"
    if automl_json.exists():
        with open(automl_json, encoding="utf-8") as f:
            data = json.load(f)
        best_model = data.get("best_model", "N/A")
        shap_path = Path(__file__).resolve().parent.parent / "results" / "automl" / "automl_feature_importance.csv"
        if shap_path.exists():
            sdf = pd.read_csv(shap_path).sort_values("importance", ascending=False)
            shap_top = ", ".join([f"{r['feature']}({r['importance']:.3f})" for _, r in sdf.iterrows()])

    md = f"""# Physical Consistency Report

## Abstract

**Research Topic: Improving Troubleshooting Performance for Semiconductor Etching Processes Using Machine Learning**

This study focuses on improving troubleshooting performance for semiconductor Etching processes using machine learning.

We describe defects such as Undercut, Bowing, Etch Stop, and Footing that occur in semiconductor Etching processes, along with the limitations of conventional troubleshooting methods. Synthetic data are generated from a simplified mathematical model of Etching physics. Machine-learning models including Random Forest, XGBoost, and HistGradientBoosting are used to predict defects, AutoML is employed to optimize models, and SHAP analysis is applied to identify candidate cause variables. The process conditions recommended by machine learning are applied to an Etching simulator to compare defect severity and Etching profiles before and after improvement. In addition, the physical trends of the simulator and the qualitative validity of the machine-learning analysis are examined with reference to Guo et al. (2024) and Seo et al. (2025).

The literature comparison is not the core objective of this study; it serves as auxiliary validation supporting the physical plausibility of the simulator. This study is limited by its use of synthetic-data-based experiments, and we do not present improvement rates, time reductions, or accuracy values that have not been actually measured. Quantitative comparison between conventional and AI-based approaches has not yet been performed and remains a future validation task using real process data.

## 1. Research purpose
Evaluate whether a synthetic semiconductor etching dataset generated from a simplified mathematical model produces machine-learning feature-importance rankings and simulator trends that are qualitatively consistent with known plasma-etching physics reported in the literature.

The central topic of this study is improving troubleshooting performance for semiconductor Etching processes using machine learning. The study proceeds as follows: (1) describe Etching defects and the limitations of conventional troubleshooting; (2) generate synthetic data from a simplified mathematical model; (3) predict defects using Random Forest, XGBoost, and HistGradientBoosting; (4) optimize models with AutoML and analyze candidate cause variables with SHAP; (5) apply recommended conditions to an Etching simulator and compare before/after profiles; (6) quantitatively compare conventional and AI-based troubleshooting within the possible scope; and (7) review qualitative validity against Guo et al. (2024) and Seo et al. (2025).

## 2. Mathematical models used
See `mathematical_model_summary.md` for full equations.
The simulator uses:
- Normalized 6 input variables.
- Power-law etch rate with passivation/pressure/temperature modifiers.
- Sigmoid-based passivation, anisotropy, and footing indices.
- Hand-crafted defect scores converted to 5-class probabilities.
- Geometric profile construction by superposing Gaussian deformation terms.

## 3. Synthetic data generation
- 5,000 samples drawn uniformly from the 6-D process box.
- Deterministic indicators plus configurable Gaussian noise.
- Labels sampled from the softmax defect probabilities.

## 4. Machine-learning result
- AutoML selected model: **{best_model}**
- Global SHAP importance (mean |SHAP|):
  {shap_top}

## 5. SHAP analysis
SHAP values indicate **which variables the model used to make predictions**, not which variables physically cause defects.

The top SHAP variables (bias_power_w, gas_chemistry, temperature_c, source_power_w, pressure_mtorr, gas_flow_sccm) are the same set of variables that the generation rules explicitly use, either directly or through derived indicators (passivation, anisotropy, etch rate, lateral factor, footing index). Therefore the ML model is not obviously inconsistent with the synthetic physics, but the ranking can be influenced by correlations and by the model's inductive bias.

## 6. Comparison with plasma-etching literature

### Guo et al. (2024), IEEE Access
- Process: STI etching with Cl₂/HBr/O₂ plasma.
- Key variables: gas ratio, Source RF power, Bias RF power, pressure, temperature, time.
- Findings:
  - Increasing Source/Bias RF power increases etch depth/rate; saturation at high RF power.
  - Gas ratio and flow affect plasma density and STI height.
  - Temperature influences plasma/sidewall reactions; slight STI height decrease with temperature.

### Seo et al. (2025), KSME
- Process: CCP dielectric etching (SiO₂/SiN) with NF₃/H₂.
- Key variables: NF₃/H₂ flow rates, process pressure, substrate temperature, etch material.
- Findings:
  - SHAP top features include H₂ flow, NF₃ flow, substrate temperature, pressure.
  - QMS species such as H₂O are linked to oxidation/etch selectivity.

## 7. Qualitative consistency

| Trend | Our simulator | Literature | Assessment |
|---|---|---|---|
| Bias power ↑ → Etch rate ↑ | increases | Guo: Bias RF increases etch depth/rate | **일치** |
| Source power ↑ → Etch rate ↑ | increases | Guo: Source RF increases etch depth/rate, saturates | **부분적으로 일치** (no saturation in our model) |
| Temperature ↑ → Etch rate ↑ | increases | Guo: slight STI height decrease with temperature | **불일치 / 부분적 일치** (metric/regime differ) |
| Pressure ↑ → Etch rate | decreases (monotonic) | No explicit trend reported | **비교 불가능** |
| Gas chemistry → Undercut severity | undercut prob. decreases as chemistry increases | No explicit undercut-severity trend | **비교 불가능** |
| Gas chemistry → Passivation | passivation increases with chemistry | Chemistry affects passivation/chemical balance | **부분적으로 일치** (qualitative only) |

## 8. Model limitations
1. **Synthetic data**: the generator is a hand-tuned mathematical model, not a physical plasma simulation.
2. **Abstract variables**: `gas_chemistry` and `gas_flow_sccm` are simplified proxies for specific gas ratios and species flows.
3. **No plasma species data**: unlike Seo et al., we do not use OES/QMS diagnostics.
4. **No 3D plasma-surface physics**: unlike Guo et al., we do not solve reaction/diffusion/transport equations.
5. **SHAP is not causality**: high SHAP importance does not prove a variable physically causes a defect.
6. **Trend differences**: our temperature and pressure trends are monotonic simplifications that may not match specific experimental regimes.

## 9. Final conclusion
The synthetic model and the AutoML SHAP results are **qualitatively consistent** with the literature in that all six input variables are recognized as physically relevant process knobs in real plasma etching. However, **quantitative and directional agreement is limited**:
- The model correctly captures the positive effect of RF power (bias/source) on etch rate.
- It does not reproduce the saturation of etch rate at high RF power reported by Guo et al.
- The temperature trend in our model (etch rate increases) conflicts with the slight STI-height decrease observed by Guo et al., although the metrics differ.
- Pressure and gas-chemistry trends cannot be directly validated because the cited papers do not report the same monotonic relationships.

Therefore, the model is suitable for **educational demonstration and methodology exercises**, but it should not be used as a quantitative predictor for a real etcher without physical calibration.
"""
    path = OUT_DIR / "physical_consistency_report.md"
    path.write_text(md, encoding="utf-8")
    return path


def main():
    print("[literature_validation] Computing parameter trends...")
    trends = compute_trends()
    trend_path = plot_trends(trends)
    print(f"  saved: {trend_path}")

    # Save numerical trend data
    trend_csv = OUT_DIR / "parameter_trends.csv"
    combined = []
    for key, r in trends.items():
        tag = key if key != "gas_chemistry2" else "gas_chemistry_passivation"
        for xi, yi in zip(r["x"], r["y"]):
            combined.append({"experiment": tag, "x": xi, "y": yi,
                             "xlabel": r["xlabel"], "ylabel": r["ylabel"]})
    pd.DataFrame(combined).to_csv(trend_csv, index=False)
    print(f"  saved: {trend_csv}")

    print("[literature_validation] Creating SHAP vs literature plot...")
    shap_path = shap_literature_plot()
    if shap_path:
        print(f"  saved: {shap_path}")
    else:
        print("  skipped: results/automl/automl_feature_importance.csv not found")

    print("[literature_validation] Writing reports...")
    print(f"  {write_mathematical_model_summary()}")
    print(f"  {write_literature_comparison_csv()}")
    print(f"  {write_physical_consistency_report(trends)}")

    print("[literature_validation] Indexing provided PDFs...")
    print(f"  {write_pdf_citations()}")
    print("[literature_validation] Done.")


if __name__ == "__main__":
    main()
