"""
test_data_generator.py
======================

데이터 생성 모듈에 대한 pytest 테스트입니다.

검증 항목
--------
- 데이터가 5,000개 생성되는가?
- 입력 변수에 결측값이 없는가?
- 불량 라벨이 올바른가? (0~4)
- 같은 random_state 에서 동일한 데이터가 생성되는가?
- 입력 변수가 지정 범위를 벗어나지 않는가?
- 검증 함수(validate_dataset)가 정상 작동하는가?
- 분포 변화 데이터가 별도로 생성되는가?
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_generator import (  # noqa: E402
    DEFECT_NAMES,
    INPUT_COLUMNS,
    LABEL_COLUMN,
    N_CLASSES,
    N_SAMPLES_DEFAULT,
    PROCESS_BOUNDS,
    generate_dataset,
    generate_shifted_dataset,
    validate_dataset,
)

# 작은 표본으로 빠르게 테스트 (기본값 검증 테스트는 별도로 수행)
SMALL_N = 600


@pytest.fixture(scope="module")
def df_small() -> pd.DataFrame:
    return generate_dataset(n_samples=SMALL_N, random_state=42)


# ---------------------------------------------------------------------------
# 1) 행 수
# ---------------------------------------------------------------------------
def test_default_sample_count_is_5000():
    """기본 옵션으로 실행하면 정확히 5,000개가 생성되어야 합니다."""
    df = generate_dataset()   # 기본 n_samples=5000
    assert len(df) == N_SAMPLES_DEFAULT == 5000


def test_custom_sample_count(df_small):
    assert len(df_small) == SMALL_N


# ---------------------------------------------------------------------------
# 2) 결측값
# ---------------------------------------------------------------------------
def test_no_missing_values(df_small):
    assert df_small[INPUT_COLUMNS].isna().sum().sum() == 0
    assert df_small.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# 3) 라벨 유효성
# ---------------------------------------------------------------------------
def test_labels_are_valid(df_small):
    labels = set(df_small[LABEL_COLUMN].unique())
    assert labels.issubset(set(range(N_CLASSES)))
    assert labels == set(range(N_CLASSES)), "6개 불량 클래스가 모두 나타나야 합니다"


def test_defect_names_mapping():
    assert DEFECT_NAMES == {0: "Normal", 1: "Undercut", 2: "Bowing",
                            3: "Etch Stop", 4: "Footing", 5: "Punch-through"}


# ---------------------------------------------------------------------------
# 4) 재현성 (같은 seed → 같은 데이터)
# ---------------------------------------------------------------------------
def test_reproducibility_same_seed():
    a = generate_dataset(n_samples=SMALL_N, random_state=123)
    b = generate_dataset(n_samples=SMALL_N, random_state=123)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_differs():
    a = generate_dataset(n_samples=SMALL_N, random_state=1)
    b = generate_dataset(n_samples=SMALL_N, random_state=2)
    assert not np.allclose(a[INPUT_COLUMNS].to_numpy(), b[INPUT_COLUMNS].to_numpy())


# ---------------------------------------------------------------------------
# 5) 입력 변수 범위
# ---------------------------------------------------------------------------
def test_input_ranges(df_small):
    for col, (lo, hi) in PROCESS_BOUNDS.items():
        assert df_small[col].min() >= lo, f"{col} 최소값이 범위 초과"
        assert df_small[col].max() <= hi, f"{col} 최대값이 범위 초과"


# ---------------------------------------------------------------------------
# 6) 지표 유효성
# ---------------------------------------------------------------------------
def test_indicator_ranges(df_small):
    for col in ["passivation_index", "anisotropy_index", "footing_index"]:
        assert df_small[col].between(0, 1).all(), f"{col} 는 0~1 이어야 합니다"
    assert df_small["defect_severity"].between(0, 100).all()
    assert (df_small["etch_rate_nm_min"] > 0).all()


# ---------------------------------------------------------------------------
# 7) 규칙 기반 정상성 (단순 무작위 라벨이 아님을 확인)
# ---------------------------------------------------------------------------
def test_labels_not_random(df_small):
    """라벨과 파생 지표 사이에 통계적 연관이 있어야 합니다 (단순 무작위 아님)."""
    from scipy.stats import chi2_contingency

    er_norm = (df_small["etch_rate_nm_min"] - df_small["etch_rate_nm_min"].min()) / (
        df_small["etch_rate_nm_min"].max() - df_small["etch_rate_nm_min"].min() + 1e-9)
    low_er = er_norm < er_norm.quantile(0.25)
    ct = pd.crosstab(low_er, df_small[LABEL_COLUMN])
    chi2, p, *_ = chi2_contingency(ct)
    assert p < 0.05, "식각 속도와 불량 라벨이 독립이면 규칙 기반 생성이 아닙니다"


def test_severity_higher_for_defects(df_small):
    normal_sev = df_small.loc[df_small[LABEL_COLUMN] == 0, "defect_severity"].mean()
    defect_sev = df_small.loc[df_small[LABEL_COLUMN] != 0, "defect_severity"].mean()
    assert normal_sev < defect_sev


# ---------------------------------------------------------------------------
# 8) 검증 함수 / 분포 변화 데이터
# ---------------------------------------------------------------------------
def test_validate_dataset_passes():
    df = generate_dataset(n_samples=SMALL_N, random_state=7)
    ok, messages = validate_dataset(df)
    assert ok, "\n".join(messages)
    assert len(messages) > 0


def test_shifted_dataset_generated():
    df = generate_shifted_dataset(n_samples=300, random_state=9)
    assert len(df) == 300
    assert df[LABEL_COLUMN].between(0, 5).all()
    ok, _ = validate_dataset(df)
    assert ok
    # 분포 변화 데이터의 평균 pressure 는 기본 범위보다 높아야 함
    assert df["pressure_mtorr"].mean() > 55.0
