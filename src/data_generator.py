"""
data_generator.py
=================

반도체 Etching 공정의 **합성(synthetic) 교육용 데이터**를 생성하는 모듈입니다.

.. warning::
    이 모듈의 수학식은 교육용으로 단순화한 가정입니다. 실제 식각 물리
    (플라즈마 반응, 이온 주입, 화학 식각)를 검증한 모델이 아니며,
    산출된 수치는 특정 장비의 실제 공정 레시피를 의미하지 않습니다.

입력 변수 (6개)
---------------
1. gas_chemistry   : 0.0 ~ 1.0   (가스 화학 조성의 단순화 지표)
2. gas_flow_sccm   : 50 ~ 200
3. pressure_mtorr  : 5 ~ 100
4. bias_power_w    : 50 ~ 300
5. source_power_w  : 300 ~ 1500
6. temperature_c   : 20 ~ 100

불량 유형 (라벨)
---------------
0: Normal, 1: Undercut, 2: Bowing, 3: Etch Stop, 4: Footing

생성 규칙 (요약, 코드의 :data:`GENERATION_RULES` 와 일치)
--------------------------------------------------------
- Passivation index(측벽 보호 효과)는 gas_chemistry/pressure 에 비례하고,
  bias_power 에 반비례합니다.
- Etch rate(식각 속도)는 source_power, bias_power, gas_flow 에 비례하고,
  passivation index 에 반비례합니다.
- Anisotropy index(방향성)는 bias_power 와 passivation index 에 비례하고,
  pressure 에 반비례합니다.
- Undercut / Bowing 은 측벽 보호 효과가 부족할 때 발생 확률이 증가합니다.
- Etch Stop 은 유효 식각 속도가 낮을 때 발생 확률이 증가합니다.
- Footing 은 별도의 하단 형상 외곡 지표(footing index)에 의해 발생합니다.
- 변수 간 상호작용과 측정 오차(노이즈)를 포함하며, random_state 로 재현합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_SAMPLES_DEFAULT = 5000

INPUT_COLUMNS: List[str] = [
    "gas_chemistry",
    "gas_flow_sccm",
    "pressure_mtorr",
    "bias_power_w",
    "source_power_w",
    "temperature_c",
]

INDICATOR_COLUMNS: List[str] = [
    "etch_rate_nm_min",
    "passivation_index",
    "anisotropy_index",
    "footing_index",
    "defect_severity",
]

LABEL_COLUMN = "defect_type"

# 교육용 가상 범위 (특정 장비의 실제 레시피가 아님)
PROCESS_BOUNDS: Dict[str, Tuple[float, float]] = {
    "gas_chemistry": (0.0, 1.0),
    "gas_flow_sccm": (50.0, 200.0),
    "pressure_mtorr": (5.0, 100.0),
    "bias_power_w": (50.0, 300.0),
    "source_power_w": (300.0, 1500.0),
    "temperature_c": (20.0, 100.0),
}

DEFECT_NAMES: Dict[int, str] = {
    0: "Normal",
    1: "Undercut",
    2: "Bowing",
    3: "Etch Stop",
    4: "Footing",
}
DEFECT_NAME_TO_ID: Dict[str, int] = {v: k for k, v in DEFECT_NAMES.items()}
N_CLASSES = len(DEFECT_NAMES)

# README / 대시보드 / SHAP 비교에 사용하는 "생성 규칙 문서"
GENERATION_RULES: Dict[str, str] = {
    "Undercut": (
        "측벽 보호(passivation)가 낮고 방향성(anisotropy)이 낮으며 온도가 높을 때 "
        "상단 부근의 측방향 식각이 커져 발생 확률이 증가합니다."
    ),
    "Bowing": (
        "측벽 보호가 중간 수준(과다하지도 부족하지도 않음)이고 pressure·bias_power가 "
        "높을 때 중간 깊이의 측벽 팽창이 발생할 확률이 증가합니다."
    ),
    "Etch Stop": (
        "유효 식각 속도(etch rate)가 낮아질 때 식각이 멈추는 확률이 증가합니다."
    ),
    "Footing": (
        "별도로 계산하는 하단 형상 외곡 지표(footing index)가 클 때 "
        "하단 CD 변화(footing) 확률이 증가합니다."
    ),
    "Normal": (
        "모든 결함 점수가 낮고 식각 속도·측벽 보호가 적절할 때 Normal 확률이 증가합니다."
    ),
}

# SHAP 결과와 실제 생성 규칙을 비교할 때 사용하는 "진짜로 중요한 변수" 목록
GROUND_TRUTH_FEATURES: Dict[int, List[str]] = {
    0: ["etch_rate_nm_min", "passivation_index", "anisotropy_index"],
    1: ["passivation_index", "anisotropy_index", "temperature_c"],
    2: ["passivation_index", "pressure_mtorr", "bias_power_w"],
    3: ["etch_rate_nm_min", "source_power_w", "bias_power_w"],
    4: ["footing_index", "bias_power_w", "temperature_c"],
}


# ---------------------------------------------------------------------------
# 수학 헬퍼
# ---------------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    """수치적으로 안정한 시그모이드 함수."""
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))),
                    np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


def _clip01(x: np.ndarray, lo: float = 0.02, hi: float = 0.98) -> np.ndarray:
    """지표를 [lo, hi] 구간으로 클리핑."""
    return np.clip(x, lo, hi)


def _normalize(value: np.ndarray, col: str) -> np.ndarray:
    """입력 변수를 0~1 범위로 정규화 (상호작용 계산용)."""
    lo, hi = PROCESS_BOUNDS[col]
    return np.clip((np.asarray(value, dtype=float) - lo) / (hi - lo), 0.0, 1.0)


# ---------------------------------------------------------------------------
# 1) 공정 변수 샘플링
# ---------------------------------------------------------------------------
def sample_process_conditions(
    n_samples: int = N_SAMPLES_DEFAULT,
    random_state: int = RANDOM_STATE,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    shift: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """입력 공정 변수 6개를 균등분포(또는 지정 분포)로 샘플링합니다.

    Parameters
    ----------
    n_samples : 생성할 표본 수
    random_state : 재현성을 위한 시드
    bounds : 변수별 (최소, 최대) 범위. 기본은 :data:`PROCESS_BOUNDS`
    shift : 분포 변화 평가용 평균 이동량 (예: {"pressure_mtorr": 25.0})

    Returns
    -------
    pandas.DataFrame : INPUT_COLUMNS 6개 컬럼
    """
    bounds = bounds or PROCESS_BOUNDS
    rng = np.random.default_rng(random_state)
    data = {}
    for col in INPUT_COLUMNS:
        lo, hi = bounds[col]
        vals = rng.uniform(lo, hi, size=n_samples)
        if shift and col in shift:
            vals = np.clip(vals + shift[col], lo, hi)
        data[col] = vals
    return pd.DataFrame(data, columns=INPUT_COLUMNS)


# ---------------------------------------------------------------------------
# 2) 물리적(교육용) 지표 계산
# ---------------------------------------------------------------------------
def compute_indicators(
    conditions: pd.DataFrame,
    noise_scale: float = 1.0,
    random_state: Optional[int] = None,
    add_noise: bool = True,
) -> pd.DataFrame:
    """공정 조건으로부터 교육용 지표(식각 속도, 측벽 보호, 방향성, 하단 외곡)를 계산합니다.

    **수식에 사용된 가정 (교육용 단순화)**
    - Passivation index = sigmoid(3.0*(g-0.5) + 1.0*(p_n-0.5) - 1.6*(b_n-0.4) + 0.5*(s_n-0.5))
    - Etch rate ≈ 350 * source^0.8 * bias^0.9 * flow^0.5 * (1-0.55*P) * pressure계 * temperature계
    - Anisotropy index = sigmoid(1.8*(b_n-0.45) - 1.4*(p_n-0.45) + 2.2*(P-0.5))
    - Lateral etch factor = (1 - anisotropy) * (0.4 + 0.9*t_n)   # 온도 상승 → 화학 식각 증가
    - Footing index = sigmoid(2.0*(b_n-0.55) + 1.4*(t_n-0.5) - 1.0*(g-0.5) + 0.8*(s_n-0.5))
    - Bowing driver = exp(-((P-0.42)/0.22)^2) * pressure계 * bias계   # 중간 보호에서 최대

    Parameters
    ----------
    conditions : INPUT_COLUMNS 를 포함한 DataFrame
    noise_scale : 측정 오차 배율 (분포 변화 평가에서 2.0~3.0 사용)
    random_state : 노이즈 시드 (None 이면 노이즈 없음과 동일하게 재현 불가)
    add_noise : False 면 측정 오차를 추가하지 않음 (테스트/시뮬레이터용)

    Returns
    -------
    pandas.DataFrame : 원본 컬럼 + 지표 컬럼
    """
    for col in INPUT_COLUMNS:
        if col not in conditions.columns:
            raise KeyError(f"입력 컬럼 누락: {col}")

    c = conditions
    g = _normalize(c["gas_chemistry"].to_numpy(), "gas_chemistry")
    fn = _normalize(c["gas_flow_sccm"].to_numpy(), "gas_flow_sccm")
    pn = _normalize(c["pressure_mtorr"].to_numpy(), "pressure_mtorr")
    bn = _normalize(c["bias_power_w"].to_numpy(), "bias_power_w")
    sn = _normalize(c["source_power_w"].to_numpy(), "source_power_w")
    tn = _normalize(c["temperature_c"].to_numpy(), "temperature_c")

    # --- 측벽 보호 효과 (passivation index) -------------------------------
    passivation = _clip01(sigmoid(3.0 * (g - 0.5) + 1.0 * (pn - 0.5)
                                 - 1.6 * (bn - 0.4) + 0.5 * (sn - 0.5)))

    # --- 유효 식각 속도 (etch rate, nm/min) -------------------------------
    etch_rate = (
        350.0
        * (0.35 + 0.65 * sn) ** 0.8      # source power → 플라즈마 밀도
        * (0.30 + 0.70 * bn) ** 0.9      # bias power → 이온 에너지
        * (0.40 + 0.60 * fn) ** 0.5      # gas flow → 식각제 공급
        * (1.0 - 0.55 * passivation)     # 측벽/표면 보호막 → 식각 지연
        * (1.25 - 0.50 * pn)             # 높은 pressure → 이온 에너지 손실
        * (0.75 + 0.50 * tn)             # 온도 → 화학 반응 가속
    )
    etch_rate = np.clip(etch_rate, 5.0, 450.0)

    # --- 방향성 (anisotropy index) ---------------------------------------
    anisotropy = _clip01(
        sigmoid(1.8 * (bn - 0.45) - 1.4 * (pn - 0.45) + 2.2 * (passivation - 0.5))
    )

    # --- 측방향 식각 성향 (질량 이송/화학 식각) ----------------------------
    lateral_factor = (1.0 - anisotropy) * (0.4 + 0.9 * tn)

    # --- 하단 형상 외곡 지표 (footing index) ------------------------------
    footing_index = _clip01(
        sigmoid(2.0 * (bn - 0.55) + 1.4 * (tn - 0.5) - 1.0 * (g - 0.5) + 0.8 * (sn - 0.5))
    )

    # --- Bowing 유도 계수 (중간 보호 수준에서 최대) ------------------------
    bowing_driver = np.exp(-((passivation - 0.42) / 0.22) ** 2) * (0.3 + 0.7 * pn) * (0.3 + 0.7 * bn)

    # --- 측정 오차 (노이즈) ----------------------------------------------
    if add_noise and random_state is not None and noise_scale > 0:
        rng = np.random.default_rng(random_state)
        etch_rate = etch_rate * (1.0 + rng.normal(0.0, 0.05 * noise_scale, size=len(c)))
        etch_rate = np.clip(etch_rate, 5.0, 450.0)
        passivation = _clip01(passivation + rng.normal(0.0, 0.02 * noise_scale, size=len(c)))
        anisotropy = _clip01(anisotropy + rng.normal(0.0, 0.02 * noise_scale, size=len(c)))
        footing_index = _clip01(footing_index + rng.normal(0.0, 0.02 * noise_scale, size=len(c)))

    out = c.copy()
    out["etch_rate_nm_min"] = np.round(etch_rate, 3)
    out["passivation_index"] = np.round(passivation, 4)
    out["anisotropy_index"] = np.round(anisotropy, 4)
    out["footing_index"] = np.round(footing_index, 4)
    out.attrs["lateral_factor"] = np.round(lateral_factor, 4)
    out.attrs["bowing_driver"] = np.round(bowing_driver, 4)
    return out


# ---------------------------------------------------------------------------
# 3) 불량 확률 (명시적 생성 규칙)
# ---------------------------------------------------------------------------
def defect_probabilities(
    indicators: pd.DataFrame,
    lateral_factor: Optional[np.ndarray] = None,
    bowing_driver: Optional[np.ndarray] = None,
    temperature_c: Optional[np.ndarray] = None,
) -> np.ndarray:
    """지표로부터 불량 유형별 발생 확률 (5-class softmax)을 계산합니다.

    명시적 생성 규칙
    ----------------
    - Undercut  : 4.5*lateral + 2.0*(1-passivation) - 2.3
    - Bowing    : 4.0*bowing_driver + 0.8*bias_norm - 2.5
    - Etch Stop : 4.2*(0.52 - etch_rate_norm)
    - Footing   : 4.0*footing_index - 2.1
    - Normal    : 3.0 - 1.5 * max(결함 점수들)
    """
    P = indicators["passivation_index"].to_numpy(dtype=float)
    A = indicators["anisotropy_index"].to_numpy(dtype=float)
    ER = indicators["etch_rate_nm_min"].to_numpy(dtype=float)
    F = indicators["footing_index"].to_numpy(dtype=float)

    if lateral_factor is None:
        tn = _normalize(indicators["temperature_c"].to_numpy(), "temperature_c")
        lateral_factor = (1.0 - A) * (0.4 + 0.9 * tn)
    if bowing_driver is None:
        pn = _normalize(indicators["pressure_mtorr"].to_numpy(), "pressure_mtorr")
        bn = _normalize(indicators["bias_power_w"].to_numpy(), "bias_power_w")
        bowing_driver = np.exp(-((P - 0.42) / 0.22) ** 2) * (0.3 + 0.7 * pn) * (0.3 + 0.7 * bn)
    if temperature_c is None:
        temperature_c = indicators["temperature_c"].to_numpy(dtype=float)
    tn = _normalize(temperature_c, "temperature_c")
    bn = _normalize(indicators["bias_power_w"].to_numpy(), "bias_power_w")

    er_norm = np.clip((ER - 40.0) / 260.0, 0.0, 1.0)

    s_undercut = 4.5 * lateral_factor + 2.0 * (1.0 - P) - 2.3
    s_bowing = 4.0 * bowing_driver + 0.8 * bn - 2.5
    s_stop = 4.2 * (0.52 - er_norm)
    s_footing = 4.0 * F - 2.1

    defect_stack = np.vstack([s_undercut, s_bowing, s_stop, s_footing])
    max_defect = np.max(defect_stack, axis=0)
    s_normal = np.clip(3.0 - 1.5 * max_defect, -3.0, 6.0)

    scores = np.vstack([s_normal, s_undercut, s_bowing, s_stop, s_footing]).T  # (n, 5)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_s = np.exp(scores)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    return probs


# ---------------------------------------------------------------------------
# 4) 라벨 샘플링 + 결함 심각도
# ---------------------------------------------------------------------------
def assign_defect_labels(
    probs: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """확률 분포로부터 불량 라벨을 샘플링합니다 (단순 무작위 라벨 아님)."""
    rng = np.random.default_rng(random_state)
    n = probs.shape[0]
    u = rng.random(n)
    cum = np.cumsum(probs, axis=1)
    labels = (u[:, None] > cum).sum(axis=1)
    return np.clip(labels, 0, probs.shape[1] - 1).astype(int)


def compute_defect_severity(
    indicators: pd.DataFrame,
    probs: np.ndarray,
    noise_scale: float = 1.0,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """결함 심각도(0~100)를 계산합니다. Normal 샘플은 심각도가 낮게 유지됩니다."""
    P = indicators["passivation_index"].to_numpy(dtype=float)
    A = indicators["anisotropy_index"].to_numpy(dtype=float)
    ER = indicators["etch_rate_nm_min"].to_numpy(dtype=float)
    F = indicators["footing_index"].to_numpy(dtype=float)
    tn = _normalize(indicators["temperature_c"].to_numpy(), "temperature_c")
    lateral = (1.0 - A) * (0.4 + 0.9 * tn)
    er_norm = np.clip((ER - 40.0) / 260.0, 0.0, 1.0)

    defect_prob_sum = 1.0 - probs[:, 0]
    severity = (
        100.0
        * (
            0.12
            + 0.50 * defect_prob_sum
            + 0.22 * lateral
            + 0.18 * (1.0 - er_norm)
            + 0.14 * F
            - 0.10 * P
        )
    )
    # Normal 라벨은 심각도를 크게 낮춤
    severity = np.where(probs.argmax(axis=1) == 0, severity * 0.25, severity)
    if random_state is not None and noise_scale > 0:
        rng = np.random.default_rng(random_state)
        severity = severity + rng.normal(0.0, 3.0 * noise_scale, size=len(severity))
    return np.clip(severity, 0.0, 100.0)


# ---------------------------------------------------------------------------
# 5) 데이터셋 생성 (공개 API)
# ---------------------------------------------------------------------------
def generate_dataset(
    n_samples: int = N_SAMPLES_DEFAULT,
    random_state: int = RANDOM_STATE,
    noise_scale: float = 1.0,
    add_noise: bool = True,
) -> pd.DataFrame:
    """합성 Etching 데이터셋을 생성합니다.

    Parameters
    ----------
    n_samples : 표본 수 (기본 5,000)
    random_state : 재현성 시드
    noise_scale : 측정 오차 배율
    add_noise : False 면 오차 없이 결정론적 지표만 생성

    Returns
    -------
    pandas.DataFrame : INPUT_COLUMNS + INDICATOR_COLUMNS + defect_type
    """
    conditions = sample_process_conditions(n_samples, random_state)
    indicators = compute_indicators(
        conditions,
        noise_scale=noise_scale,
        random_state=(random_state + 1) if add_noise else None,
        add_noise=add_noise,
    )

    lateral = np.asarray(indicators.attrs.get("lateral_factor"))
    bowing = np.asarray(indicators.attrs.get("bowing_driver"))
    probs = defect_probabilities(indicators, lateral, bowing, conditions["temperature_c"].to_numpy())
    labels = assign_defect_labels(probs, random_state + 2)
    severity = compute_defect_severity(
        indicators, probs, noise_scale=noise_scale,
        random_state=(random_state + 3) if add_noise else None,
    )

    df = indicators.copy()
    df["defect_severity"] = np.round(severity, 2)
    df[LABEL_COLUMN] = labels
    df.attrs["defect_probabilities"] = probs
    df.attrs["generation_rules"] = GENERATION_RULES
    df.attrs["ground_truth_features"] = GROUND_TRUTH_FEATURES
    return df


def generate_shifted_dataset(
    n_samples: int = 1000,
    random_state: int = 123,
    noise_scale: float = 2.5,
) -> pd.DataFrame:
    """분포 변화(distribution shift) 평가용 데이터셋을 생성합니다.

    일부 입력 변수의 분포(gas_chemistry, pressure, bias, temperature)를 이동시키고
    측정 오차(noise_scale)를 키워, 단순 무작위 분할로 인한 과대평가를 막습니다.
    """
    shift = {
        "gas_chemistry": 0.20,
        "pressure_mtorr": 25.0,
        "bias_power_w": 40.0,
        "temperature_c": 18.0,
    }
    conditions = sample_process_conditions(n_samples, random_state, shift=shift)
    indicators = compute_indicators(
        conditions, noise_scale=noise_scale, random_state=random_state + 1, add_noise=True
    )
    lateral = np.asarray(indicators.attrs["lateral_factor"])
    bowing = np.asarray(indicators.attrs["bowing_driver"])
    probs = defect_probabilities(indicators, lateral, bowing, conditions["temperature_c"].to_numpy())
    labels = assign_defect_labels(probs, random_state + 2)
    severity = compute_defect_severity(
        indicators, probs, noise_scale=noise_scale, random_state=random_state + 3
    )
    df = indicators.copy()
    df["defect_severity"] = np.round(severity, 2)
    df[LABEL_COLUMN] = labels
    return df


# ---------------------------------------------------------------------------
# 6) 저장 / 로드 / 검증
# ---------------------------------------------------------------------------
def save_dataset(df: pd.DataFrame, path: str | Path) -> Path:
    """CSV 파일로 저장합니다 (폴더가 없으면 생성)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def load_dataset(path: str | Path) -> pd.DataFrame:
    """CSV 파일에서 데이터셋을 불러옵니다."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}")
    return pd.read_csv(path)


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """데이터셋 검증 (결측값, 라벨 범위, 값 범위, 행 수 등).

    Returns
    -------
    (ok, messages) : 모든 검증을 통과하면 ok=True, messages 에 상세 결과 포함
    """
    messages: List[str] = []
    ok = True

    required = INPUT_COLUMNS + INDICATOR_COLUMNS + [LABEL_COLUMN]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        return False, [f"필수 컬럼 누락: {missing_cols}"]

    # 행 수
    if len(df) == N_SAMPLES_DEFAULT:
        messages.append(f"[OK] 행 수: {len(df):,} (기본 5,000건)")
    else:
        messages.append(f"[INFO] 행 수: {len(df):,} (기본 5,000건과 다름 - 사용자 지정)")

    # 결측값
    n_missing = int(df[required].isna().sum().sum())
    if n_missing == 0:
        messages.append("[OK] 결측값: 0")
    else:
        ok = False
        messages.append(f"[FAIL] 결측값: {n_missing}건")

    # 라벨 범위
    labels = set(df[LABEL_COLUMN].unique())
    if labels.issubset(set(DEFECT_NAMES.keys())):
        messages.append(f"[OK] 라벨 범위: {sorted(labels)} (정상)")
    else:
        ok = False
        messages.append(f"[FAIL] 잘못된 라벨: {sorted(labels - set(DEFECT_NAMES.keys()))}")

    # 입력 변수 범위
    for col, (lo, hi) in PROCESS_BOUNDS.items():
        mn, mx = float(df[col].min()), float(df[col].max())
        if mn >= lo and mx <= hi:
            messages.append(f"[OK] {col}: {mn:.3f} ~ {mx:.3f} (허용 {lo}~{hi})")
        else:
            ok = False
            messages.append(f"[FAIL] {col}: {mn:.3f} ~ {mx:.3f} 범위 {lo}~{hi} 초과")

    # 지표 유효성
    for col, lo, hi in [("passivation_index", 0, 1),
                        ("anisotropy_index", 0, 1),
                        ("footing_index", 0, 1),
                        ("defect_severity", 0, 100)]:
        mn, mx = float(df[col].min()), float(df[col].max())
        if mn >= lo and mx <= hi:
            messages.append(f"[OK] {col}: {mn:.3f} ~ {mx:.3f}")
        else:
            ok = False
            messages.append(f"[FAIL] {col}: {mn:.3f} ~ {mx:.3f} 범위 {lo}~{hi} 초과")

    # 라벨 분포
    dist = df[LABEL_COLUMN].value_counts().sort_index()
    for lab, cnt in dist.items():
        messages.append(f"[INFO] class {lab} ({DEFECT_NAMES[int(lab)]}): {cnt:,}건 "
                        f"({cnt / len(df) * 100:.1f}%)")

    return ok, messages


def default_data_path() -> Path:
    """프로젝트 루트의 data/synthetic_etching_data.csv 경로."""
    return Path(__file__).resolve().parent.parent / "data" / "synthetic_etching_data.csv"


# ---------------------------------------------------------------------------
# 직접 실행 시: 데이터 생성 + 검증
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = generate_dataset()
    ok, msgs = validate_dataset(data)
    for m in msgs:
        print(m)
    save_dataset(data, default_data_path())
    print(f"저장 완료: {default_data_path()}")
    print(f"검증 결과: {'통과' if ok else '실패'}")
