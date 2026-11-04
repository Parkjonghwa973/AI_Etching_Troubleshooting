# Physical Consistency Report

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
- AutoML selected model: **hist_gradient_boosting**
- Global SHAP importance (mean |SHAP|):
  bias_power_w(0.401), gas_chemistry(0.288), temperature_c(0.229), source_power_w(0.221), pressure_mtorr(0.173), gas_flow_sccm(0.118)

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
