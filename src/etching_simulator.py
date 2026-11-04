"""
etching_simulator.py
====================

2D 식각 단면(profile)을 그리는 **교육용** Etching 시뮬레이터입니다.

.. warning::
    본 시뮬레이터의 형상 수식은 머신러닝/교육용으로 단순화한 가정입니다.
    실제 반도체 공정의 물리 기반 검증(PVD/CFD/플라즈마 시뮬레이션 등) 결과가
    아니며, 특정 장비의 레시피를 재현하지 않습니다.

형상 모델의 핵심 가정
--------------------
- 식각 깊이 D 는 etch rate(단위: nm/min)에 식각 시간(기본 60 s = 1 min)을 곱한 값.
- 초기 상단 반폭(half width)은 100 nm (Top CD ≈ 200 nm)로 가정.
- 깊이에 따른 좌우 측벽 위치를 수학 함수로 계산해 2D 단면을 만듭니다.
- 불량 유형별 변형:
    * **Normal**   : 방향성에 따른 약간의 테이퍼(taper)
    * **Undercut** : 상단 부근(깊이 0~30%)의 측방향 식각 증가 → 하부로 갈수록 좁아지는 역테이퍼
    * **Bowing**   : 중간 깊이(50%)의 가우시안 측벽 팽창
    * **Etch Stop** : 식각 깊이 감소 + 하단 둥근 형상
    * **Footing**  : 하단 부근의 측벽 확장(하단 CD 증가)
- 모든 변형량은 입력 공정 변수(측벽 보호, 방향성, 온도, pressure, bias 등)에
  연속적으로 의존합니다.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data_generator import (
    DEFECT_NAMES,
    DEFECT_NAME_TO_ID,
    INPUT_COLUMNS,
    N_CLASSES,
    PROCESS_BOUNDS,
    compute_indicators,
    defect_probabilities,
)

# ---------------------------------------------------------------------------
# 시뮬레이터 가정 (교육용)
# ---------------------------------------------------------------------------
ETCH_TIME_SEC = 60.0          # 식각 시간: 60초(=1분) 가정
INITIAL_HALF_WIDTH_NM = 100.0  # 초기 상단 반폭 → Top CD ≈ 200 nm
N_POINTS_DEFAULT = 240         # 프로파일 점 수


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------
def conditions_to_frame(conditions: Dict[str, float] | pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """dict / array / DataFrame 를 공정 변수 DataFrame 으로 정규화합니다."""
    if isinstance(conditions, pd.DataFrame):
        return conditions[INPUT_COLUMNS].astype(float)
    if isinstance(conditions, dict):
        return pd.DataFrame([{c: float(conditions[c]) for c in INPUT_COLUMNS}])
    arr = np.asarray(conditions, dtype=float).reshape(1, -1)
    return pd.DataFrame(arr, columns=INPUT_COLUMNS)


def clip_conditions(conditions: Dict[str, float]) -> Dict[str, float]:
    """공정 변수를 허용 범위로 클리핑합니다."""
    return {c: float(np.clip(conditions[c], *PROCESS_BOUNDS[c])) for c in INPUT_COLUMNS}


def _indicators_for(conditions: Dict[str, float] | pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """단일/복수 조건의 지표를 노이즈 없이 결정론적으로 계산합니다."""
    df = conditions_to_frame(conditions)
    return compute_indicators(df, noise_scale=0.0, add_noise=False)


def predict_probabilities_rule_based(conditions: Dict[str, float]) -> np.ndarray:
    """생성 규칙 기반으로 5개 불량 클래스 확률을 계산합니다 (딥러닝 불필요)."""
    df = conditions_to_frame(conditions)
    ind = compute_indicators(df, noise_scale=0.0, add_noise=False)
    probs = defect_probabilities(
        ind,
        np.asarray(ind.attrs["lateral_factor"]),
        np.asarray(ind.attrs["bowing_driver"]),
        df["temperature_c"].to_numpy(),
    )
    return probs[0]


def predict_defect_type(conditions: Dict[str, float]) -> Tuple[int, str]:
    """규칙 기반으로 가장 가능성 높은 불량 유형을 반환합니다."""
    probs = predict_probabilities_rule_based(conditions)
    lab = int(np.argmax(probs))
    return lab, DEFECT_NAMES[lab]


# ---------------------------------------------------------------------------
# 프로파일 계산
# ---------------------------------------------------------------------------
def compute_profile(
    conditions: Dict[str, float] | pd.DataFrame | np.ndarray,
    defect_type: Optional[int | str] = None,
    n_points: int = N_POINTS_DEFAULT,
    etch_time_sec: float = ETCH_TIME_SEC,
) -> Dict[str, object]:
    """공정 조건과 불량 유형으로부터 2D 식각 단면 좌표와 지표를 계산합니다.

    Parameters
    ----------
    conditions : 6개 공정 변수
    defect_type : 0~4 또는 "Normal"~"Footing". None 이면 규칙 기반 예측 사용
    n_points : 프로파일 점 수
    etch_time_sec : 식각 시간(초)

    Returns
    -------
    dict with keys:
        depth_nm (음수=하부), y_nm, left_nm, right_nm,
        etch_depth_nm, top_cd_nm, middle_cd_nm, bottom_cd_nm,
        max_lateral_deviation_nm, sidewall_angle_deg, defect_type, defect_name
    """
    if isinstance(defect_type, str):
        defect_type = DEFECT_NAME_TO_ID[defect_type]
    if defect_type is None:
        defect_type, _ = predict_defect_type(conditions_to_frame(conditions).iloc[0].to_dict())
    defect_type = int(defect_type)

    df = conditions_to_frame(conditions)
    ind = _indicators_for(df)
    row = ind.iloc[0]

    er = float(row["etch_rate_nm_min"])          # nm/min
    P = float(row["passivation_index"])          # 측벽 보호 효과
    A = float(row["anisotropy_index"])           # 방향성
    F = float(row["footing_index"])              # 하단 외곡 지표

    g = (df["gas_chemistry"].iloc[0] - 0.0) / 1.0
    pn = (df["pressure_mtorr"].iloc[0] - 5.0) / 95.0
    bn = (df["bias_power_w"].iloc[0] - 50.0) / 250.0
    sn = (df["source_power_w"].iloc[0] - 300.0) / 1200.0
    tn = (df["temperature_c"].iloc[0] - 20.0) / 80.0

    # 식각 깊이 (Etch Stop 발생 시 실제 유효 깊이 감소)
    etch_depth = er * (etch_time_sec / 60.0)
    if defect_type == DEFECT_NAME_TO_ID["Etch Stop"]:
        etch_depth *= 0.45
    etch_depth = float(np.clip(etch_depth, 10.0, 500.0))

    # 깊이 분율: 0 = 상단, 1 = 하단
    yn = np.linspace(0.0, 1.0, n_points)

    # 기본 형상: 방향성에 따른 테이퍼 (anisotropy 낮을수록 하단으로 좁아짐)
    taper = 0.25 * (1.0 - A)
    base_half_width = INITIAL_HALF_WIDTH_NM * (1.0 - taper * yn)

    # 연속 공정 변수 기반 변형 계수
    lateral_factor = (1.0 - A) * (0.4 + 0.9 * tn)                       # 측방향 식각 성향
    bowing_driver = float(np.exp(-((P - 0.42) / 0.22) ** 2) * (0.3 + 0.7 * pn) * (0.3 + 0.7 * bn))
    footing_driver = F * (0.5 + 0.8 * bn) * (0.5 + 0.6 * tn)

    # 각 변형항에 "자기 불량 유형의 계수"만 적용합니다.
    # (해당 유형이 아닌 항은 약한 잔여 변형 0.15 배 → 다른 불량의 형상이 새지 않음)
    residual = 0.15
    amp_own = {
        DEFECT_NAME_TO_ID["Undercut"]: 4.0,   # Undercut 유형: 상단 항만 증폭
        DEFECT_NAME_TO_ID["Bowing"]: 4.0,     # Bowing 유형: 중간 항만 증폭
        DEFECT_NAME_TO_ID["Etch Stop"]: 2.0,  # Etch Stop 유형: 하단 정리 항만 증폭
        DEFECT_NAME_TO_ID["Footing"]: 4.0,    # Footing 유형: 하단 항만 증폭
    }

    def _factor(defect_id: int) -> float:
        """자기 유형이면 증폭 계수, 아니면 잔여 계수."""
        return amp_own[defect_id] if defect_type == defect_id else residual

    amp_u = _factor(DEFECT_NAME_TO_ID["Undercut"])
    amp_b = _factor(DEFECT_NAME_TO_ID["Bowing"])
    amp_f = _factor(DEFECT_NAME_TO_ID["Footing"])
    amp_s = _factor(DEFECT_NAME_TO_ID["Etch Stop"])

    # (1) Undercut: 상단 부근의 측방향 식각 증가
    undercut_delta = amp_u * 8.0 * (lateral_factor + 0.35 * (1.0 - P)) * np.exp(-((yn - 0.06) / 0.22) ** 2)

    # (2) Bowing: 중간 깊이의 측벽 팽창
    bowing_delta = amp_b * 10.0 * bowing_driver * np.exp(-((yn - 0.50) / 0.22) ** 2)

    # (3) Footing: 하단 부근의 측벽 확장 (하단 CD 증가)
    footing_delta = amp_f * 18.0 * footing_driver * np.exp(-((yn - 1.0) / 0.16) ** 2)

    # (4) Etch Stop: 하단 둥근 형상 + 테이퍼 증가
    stop_rounding = amp_s * 4.0 * (0.5 - yn <= 0.1) * np.exp(-((yn - 1.0) / 0.18) ** 2)

    half_width = base_half_width + undercut_delta + bowing_delta + footing_delta + stop_rounding
    half_width = np.clip(half_width, 5.0, None)

    y_nm = -yn * etch_depth            # 상단 0, 하단 -depth
    left_nm = -half_width
    right_nm = half_width

    # ------------------------------------------------------------------
    # 형상 지표
    # ------------------------------------------------------------------
    nominal = base_half_width
    max_lateral = float(np.max(half_width - nominal))
    top_cd = float(2.0 * half_width[np.argmin(np.abs(yn - 0.03))])
    middle_cd = float(2.0 * half_width[np.argmin(np.abs(yn - 0.50))])
    bottom_cd = float(2.0 * half_width[np.argmin(np.abs(yn - 0.97))])
    wall_len = float(np.hypot((half_width[0] - half_width[-1]), etch_depth))
    sidewall_angle = float(np.degrees(np.arctan2(etch_depth, max(half_width[0] - half_width[-1], 1e-6))))

    return {
        "y_nm": y_nm,
        "depth_fraction": yn,
        "left_nm": left_nm,
        "right_nm": right_nm,
        "half_width_nm": half_width,
        "etch_depth_nm": etch_depth,
        "top_cd_nm": top_cd,
        "middle_cd_nm": middle_cd,
        "bottom_cd_nm": bottom_cd,
        "max_lateral_deviation_nm": max_lateral,
        "sidewall_angle_deg": sidewall_angle,
        "profile_length_nm": wall_len,
        "defect_type": defect_type,
        "defect_name": DEFECT_NAMES[defect_type],
        "indicators": {
            "etch_rate_nm_min": er,
            "passivation_index": P,
            "anisotropy_index": A,
            "footing_index": F,
        },
    }


def compute_all_defect_profiles(
    conditions: Dict[str, float],
    n_points: int = N_POINTS_DEFAULT,
) -> Dict[str, Dict[str, object]]:
    """같은 공정 조건에서 5가지 불량 유형의 기준 형상을 모두 계산합니다."""
    return {
        name: compute_profile(conditions, defect_type=i, n_points=n_points)
        for i, name in DEFECT_NAMES.items()
    }


def profile_metrics_table(profiles: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    """프로파일 지표를 표로 정리합니다."""
    rows = []
    for name, p in profiles.items():
        rows.append({
            "Defect": name,
            "Etch depth (nm)": round(p["etch_depth_nm"], 1),
            "Top CD (nm)": round(p["top_cd_nm"], 1),
            "Middle CD (nm)": round(p["middle_cd_nm"], 1),
            "Bottom CD (nm)": round(p["bottom_cd_nm"], 1),
            "Max lateral dev. (nm)": round(p["max_lateral_deviation_nm"], 1),
            "Sidewall angle (deg)": round(p["sidewall_angle_deg"], 1),
        })
    return pd.DataFrame(rows)


def default_process_conditions() -> Dict[str, float]:
    """대시보드 기본값 (각 변수 범위의 중간값)."""
    return {c: float(np.mean(PROCESS_BOUNDS[c])) for c in INPUT_COLUMNS}


if __name__ == "__main__":
    conds = default_process_conditions()
    prof = compute_profile(conds, defect_type="Normal")
    print("기본 조건:", conds)
    for k in ["etch_depth_nm", "top_cd_nm", "middle_cd_nm", "bottom_cd_nm",
              "max_lateral_deviation_nm", "sidewall_angle_deg"]:
        print(f"  {k}: {prof[k]:.2f}")
