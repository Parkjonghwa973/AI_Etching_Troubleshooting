# Mathematical Model Summary

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
