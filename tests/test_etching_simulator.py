"""
test_etching_simulator.py
=========================

식각 프로파일 시뮬레이터 모듈에 대한 pytest 테스트입니다.

검증 항목
--------
- 식각 형상이 유효한 좌표를 생성하는가? (유한값, 음수 depth, 좌우 대칭)
- 6가지 불량 유형 프로파일이 모두 계산되는가?
- 형상 지표(Top/Middle/Bottom CD, 식각 깊이, 측벽 각도)가 합리적인가?
- 불량 유형별 형상 특성이 올바르게 구분되는가?
  (Undercut: 상단 확장 / Bowing: 중간 팽창 / Footing: 하단 확장 / Etch Stop: 깊이 감소 /
   Punch-through: 목표층 초과 관통)
- 입력 공정 변수에 따라 형상이 연속적으로 변하는가?
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.etching_simulator import (  # noqa: E402
    INITIAL_HALF_WIDTH_NM,
    compute_all_defect_profiles,
    compute_profile,
    conditions_to_frame,
    default_process_conditions,
    predict_defect_type,
    profile_metrics_table,
)
from src.data_generator import DEFECT_NAMES, INPUT_COLUMNS, PROCESS_BOUNDS  # noqa: E402


@pytest.fixture(scope="module")
def base_conditions():
    return default_process_conditions()


@pytest.fixture(scope="module")
def all_profiles(base_conditions):
    return compute_all_defect_profiles(base_conditions)


# ---------------------------------------------------------------------------
# 1) 유효한 좌표
# ---------------------------------------------------------------------------
def test_profile_coordinates_valid(all_profiles):
    for name, p in all_profiles.items():
        y, left, right = p["y_nm"], p["left_nm"], p["right_nm"]
        assert np.all(np.isfinite(y)) and np.all(np.isfinite(left)) and np.all(np.isfinite(right))
        assert (y <= 0).all(), "깊이는 0(표면) 이하여야 합니다"
        assert (y[0] == 0), "프로파일은 표면(y=0)에서 시작해야 합니다"
        assert (right > 0).all() and (left < 0).all(), "측벽은 좌우 대칭(부호 반대)이어야 합니다"
        assert len(y) == len(left) == len(right)


def test_all_defect_types_present(all_profiles):
    assert set(all_profiles.keys()) == set(DEFECT_NAMES.values())


# ---------------------------------------------------------------------------
# 2) 형상 지표
# ---------------------------------------------------------------------------
def test_metrics_reasonable(all_profiles):
    for name, p in all_profiles.items():
        assert 10 <= p["etch_depth_nm"] <= 500, f"{name} 식각 깊이 이상"
        assert 50 <= p["top_cd_nm"] <= 1000, f"{name} Top CD 이상"
        assert p["bottom_cd_nm"] > 0
        assert p["max_lateral_deviation_nm"] >= 0
        assert 0 < p["sidewall_angle_deg"] <= 90


def test_metrics_table():
    base = default_process_conditions()
    table = profile_metrics_table(compute_all_defect_profiles(base))
    assert len(table) == 6
    assert "Etch depth (nm)" in table.columns
    assert "Sidewall angle (deg)" in table.columns


# ---------------------------------------------------------------------------
# 3) 불량 유형별 형상 특성 구분
# ---------------------------------------------------------------------------
def test_undercut_widens_near_top(all_profiles, base_conditions):
    """Undercut 은 상단 부근이 넓어져야 합니다 (역테이퍼)."""
    p = all_profiles["Undercut"]
    normal = compute_profile(base_conditions, defect_type="Normal")
    w = p["half_width_nm"]
    top_w, bot_w = w[: len(w) // 5].mean(), w[-(len(w) // 5):].mean()
    assert top_w > bot_w, "Undercut 은 상단 부근이 넓어져야 합니다 (역테이퍼)"
    assert p["top_cd_nm"] > normal["top_cd_nm"] + 3.0, \
        "Undercut 의 Top CD 는 Normal 보다 커야 합니다"
    assert p["bottom_cd_nm"] <= normal["bottom_cd_nm"] + 3.0, \
        "Undercut 은 하단 CD 를 크게 늘리지 않아야 합니다"


def test_bowing_bulges_at_middle(all_profiles, base_conditions):
    """Bowing 은 중간 깊이에서 측벽이 팽창해야 합니다."""
    p = all_profiles["Bowing"]
    normal = compute_profile(base_conditions, defect_type="Normal")
    w = p["half_width_nm"]
    mid = w[len(w) // 2 - len(w) // 20: len(w) // 2 + len(w) // 20].mean()
    ends = np.concatenate([w[: len(w) // 10], w[-(len(w) // 10):]]).mean()
    assert mid > ends, "Bowing 은 중간 깊이에서 측벽이 팽창해야 합니다"
    assert p["middle_cd_nm"] > normal["middle_cd_nm"] + 3.0, \
        "Bowing 의 Middle CD 는 Normal 보다 커야 합니다"


def test_footing_widens_near_bottom(all_profiles, base_conditions):
    """Footing 은 하단 CD 가 커져야 합니다."""
    p = all_profiles["Footing"]
    normal = compute_profile(base_conditions, defect_type="Normal")
    w = p["half_width_nm"]
    bot_w, top_w = w[-(len(w) // 5):].mean(), w[: len(w) // 5].mean()
    assert bot_w > top_w, "Footing 은 하단 CD 가 커져야 합니다"
    assert p["bottom_cd_nm"] > normal["bottom_cd_nm"] + 3.0, \
        "Footing 의 Bottom CD 는 Normal 보다 커야 합니다"


def test_etch_stop_is_shallower(all_profiles):
    assert all_profiles["Etch Stop"]["etch_depth_nm"] < all_profiles["Normal"]["etch_depth_nm"]


def test_punch_through_exceeds_target_depth(all_profiles):
    """Punch-through 는 목표층 두께를 초과해야 합니다."""
    p = all_profiles["Punch-through"]
    assert p["etch_depth_nm"] > p["target_depth_nm"] * 1.05


def test_punch_through_narrows_at_bottom_when_over_etched(base_conditions):
    """Punch-through 는 과식각 시 하단에서 좁아집니다."""
    high_etch = dict(base_conditions)
    high_etch["source_power_w"] = 1500.0
    high_etch["bias_power_w"] = 300.0
    high_etch["temperature_c"] = 100.0
    p = compute_profile(high_etch, defect_type="Punch-through")
    normal = compute_profile(base_conditions, defect_type="Normal")
    assert p["etch_depth_nm"] > p["target_depth_nm"]
    assert p["bottom_cd_nm"] < normal["bottom_cd_nm"], \
        "Punch-through 는 하단 CD 가 Normal 보다 작아야 합니다"


def test_punch_through_not_occurs_at_low_etch_rate():
    """식각 속도가 낮으면 규칙 기반 예측이 Punch-through 를 선택하지 않아야 합니다."""
    low_etch = default_process_conditions()
    low_etch["source_power_w"] = 300.0
    low_etch["bias_power_w"] = 50.0
    low_etch["temperature_c"] = 20.0
    lab, name = predict_defect_type(low_etch)
    assert name != "Punch-through", \
        "저속 식각 조건에서 Punch-through 가 예측되면 안 됩니다"


def test_normal_is_most_tapered_shape():
    """Normal 은 Undercut/Bowing 대비 측방향 최대 편차가 작아야 합니다."""
    base = default_process_conditions()
    normal = compute_profile(base, defect_type="Normal")
    for name in ["Undercut", "Bowing"]:
        other = compute_profile(base, defect_type=name)
        assert other["max_lateral_deviation_nm"] > normal["max_lateral_deviation_nm"], name


# ---------------------------------------------------------------------------
# 4) 공정 변수에 따른 연속 변화
# ---------------------------------------------------------------------------
def test_profile_changes_continuously_with_temperature():
    """온도를 조금 바꾸면 형상 지표도 조금 변해야 합니다 (불연속 없음)."""
    base = default_process_conditions()
    lo = dict(base, temperature_c=base["temperature_c"] - 2.0)
    hi = dict(base, temperature_c=base["temperature_c"] + 2.0)
    p_lo = compute_profile(lo, defect_type="Undercut")
    p_hi = compute_profile(hi, defect_type="Undercut")
    diff = abs(p_lo["max_lateral_deviation_nm"] - p_hi["max_lateral_deviation_nm"])
    assert diff < 5.0, "작은 입력 변화에 형상이 급격히 변하면 안 됩니다"


def test_higher_source_power_increases_depth():
    base = default_process_conditions()
    low_p = compute_profile(dict(base, source_power_w=500.0), defect_type="Normal")
    high_p = compute_profile(dict(base, source_power_w=1300.0), defect_type="Normal")
    assert high_p["etch_depth_nm"] > low_p["etch_depth_nm"]


# ---------------------------------------------------------------------------
# 5) 규칙 기반 예측 / 입력 정규화
# ---------------------------------------------------------------------------
def test_predict_defect_type_returns_valid_label(base_conditions):
    lab, name = predict_defect_type(base_conditions)
    assert lab in range(6)
    assert name == DEFECT_NAMES[lab]


def test_conditions_to_frame_columns(base_conditions):
    df = conditions_to_frame(base_conditions)
    assert list(df.columns) == INPUT_COLUMNS
    assert len(df) == 1


def test_profile_accepts_defect_name_string(base_conditions):
    p = compute_profile(base_conditions, defect_type="Etch Stop")
    assert p["defect_name"] == "Etch Stop"


def test_profile_clips_out_of_range_conditions(base_conditions):
    bad = dict(base_conditions, bias_power_w=9999.0)
    p = compute_profile(bad, defect_type="Normal")
    assert np.isfinite(p["etch_depth_nm"])
