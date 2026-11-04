# 🔬 AI_Etching_Troubleshooting

## 머신러닝을 활용한 반도체 Etching 공정의 트러블슈팅 성능 개선

### 초록 (Abstract in Korean)

본 연구는 **머신러닝을 활용한 반도체 Etching 공정의 트러블슈팅 성능 개선**을 중심 주제로 합니다.

반도체 Etching 공정에서 발생하는 Undercut, Bowing, Etch Stop, Footing 등의 불량과
기존 트러블슈팅 방식의 한계를 설명하고,
식각 공정의 물리적 특성을 단순화한 수학적 모델로 합성 데이터를 생성합니다.
Random Forest, XGBoost, HistGradientBoosting 등의 머신러닝 모델을 활용하여 불량을 예측하고,
AutoML로 모델을 최적화하며 SHAP으로 주요 원인 후보 변수를 분석합니다.
머신러닝이 추천한 공정 조건을 식각 시뮬레이터에 적용하여
개선 전후의 불량 정도와 식각 형상을 비교합니다.
또한 Guo et al.(2024), Seo et al.(2025) 논문을 참고하여
시뮬레이션의 물리적 경향과 머신러닝 분석 결과의 정성적 타당성을 검토합니다.

논문 비교는 연구의 핵심 목표가 아니라 시뮬레이터의 물리적 개연성을 뒷받침하는 보조 검증입니다.
본 연구는 **합성 데이터 기반 실험**이라는 한계를 가지며,
실제로 측정하지 않은 개선율, 시간 단축률, 정확도는 제시하지 않습니다.
기존 방식과 AI 방식의 정량적 비교는 아직 수행되지 않았으며,
향후 실제 공정 데이터를 활용한 검증 과제로 남아 있습니다.

### Abstract

This study focuses on **improving troubleshooting performance for semiconductor Etching processes using machine learning**.

We describe defects such as Undercut, Bowing, Etch Stop, and Footing that occur in semiconductor Etching processes,
along with the limitations of conventional troubleshooting methods.
Synthetic data are generated from a simplified mathematical model of Etching physics.
Machine-learning models including Random Forest, XGBoost, and HistGradientBoosting are used to predict defects,
AutoML is employed to optimize models,
and SHAP analysis is applied to identify candidate cause variables.
The process conditions recommended by machine learning are applied to an Etching simulator
to compare defect severity and Etching profiles before and after improvement.
In addition, the physical trends of the simulator and the qualitative validity of the machine-learning analysis
are examined with reference to Guo et al. (2024) and Seo et al. (2025).

The literature comparison is not the core objective of this study;
it serves as auxiliary validation supporting the physical plausibility of the simulator.
This study is limited by its use of **synthetic-data-based experiments**,
and we do not present improvement rates, time reductions, or accuracy values that have not been actually measured.
Quantitative comparison between conventional and AI-based approaches has not yet been performed
and remains a future validation task using real process data.

> ### ⚠️ 반드시 확인하세요 (교육용 프로젝트)
> - 본 프로젝트의 데이터는 **교육용으로 단순화한 수학 규칙**으로 생성한 **합성(가상) 데이터**입니다.
> - 식각 형상 시뮬레이터도 **교육용 단순 모델**이며, 실제 식각 물리(플라즈마 반응, 이온 주입, 화학 식각)를 검증한 결과가 **아닙니다**.
> - 산출된 수치는 **특정 장비의 실제 공정 레시피를 의미하지 않습니다**.
> - SHAP 결과는 **모델 예측에 대한 기여도**일 뿐, **물리적 인과관계의 증거가 아닙니다**.
> - 공정 조건 추천 결과가 실제 공정에서 불량을 감소시킨다고 **단정할 수 없습니다**.

---

## 🔄 전체 연구 흐름

```
Etching 공정 불량
    ↓
기존 트러블슈팅의 한계
    ↓
머신러닝 도입
    ↓
불량 예측 및 SHAP 분석
    ↓
공정 조건 최적화
    ↓
기존 방식과 비교
    ↓
논문 기반 물리적 타당성 검토
    ↓
결론
```

---

## 📁 프로젝트 구성

```
AI_Etching_Troubleshooting/
├── app.py                     # Streamlit 웹 대시보드 (8개 페이지)
├── run_pipeline.py            # 전체 파이프라인 자동 실행 스크립트
├── run_automl.py              # AI 자동 분석(AutoML) 전체 실행 스크립트
├── requirements.txt           # 필요 패키지 목록
├── README.md                  # 이 문서
├── src/
│   ├── __init__.py
│   ├── data_generator.py      # 합성 Etching 데이터 생성 (5,000건)
│   ├── etching_simulator.py   # 2D 식각 프로파일 시뮬레이터
│   ├── train_models.py        # Random Forest / XGBoost 학습
│   ├── optimize_model.py      # Optuna 하이퍼파라미터 최적화
│   ├── evaluate_models.py     # 모델 성능 평가 (지표 + 그래프)
│   ├── explain_model.py       # SHAP 원인 후보 분석
│   ├── process_optimizer.py   # 머신러닝 기반 공정 조건 추천
│   ├── automl_engine.py       # AutoML: 4개 모델 자동 학습/최적화 (과적합 방지)
│   ├── model_selector.py      # 검증 성능 기준 최적 모델 자동 선정
│   ├── auto_troubleshooter.py # 자동 불량 분석 → 조건 탐색 → 추천
│   ├── validation_engine.py   # 추천 조건의 식각 프로파일 전후 검증
│   ├── report_generator.py    # HTML 자동 보고서 생성
│   └── visualization.py       # 그래프 디자인/저장 공통 모듈
├── data/                      # 생성되는 데이터 CSV (실행 시 자동 생성)
├── models/                    # 저장되는 모델 .pkl (실행 시 자동 생성)
├── results/                   # 결과 CSV / 그래프 PNG (실행 시 자동 생성)
│   └── automl/                # AI 자동 분석 전용 결과 폴더
└── tests/                     # pytest 테스트
    ├── test_data_generator.py
    ├── test_etching_simulator.py
    ├── test_ml_pipeline.py
    └── test_automl_pipeline.py
```

---

## 🛠️ 1. 설치 방법 (Windows PowerShell, 초보자용)

### 1-1. Python 3.11 이상 설치 (3.11 / 3.12 / 3.13 모두 지원)

1. [Python 공식 사이트](https://www.python.org/downloads/)에서 **Python 3.11 이상** 버전을
   설치합니다. (개발 기준 버전은 **Python 3.11** 이며, 3.12 / 3.13 에서도 설치되도록
   `requirements.txt` 에 버전별 호환 조건이 포함되어 있습니다.)
2. 설치 마법사에서 **`Add Python to PATH` 체크박스를 반드시 선택**합니다.
3. 설치 확인 (PowerShell 열고 입력):

```powershell
python --version
```

`Python 3.11.x` (또는 3.12/3.13) 가 표시되면 성공입니다.

> **Windows 스토어 별칭 문제(중요)**
> PowerShell 에서 `python` 을 입력했을 때 아무 출력 없이 Microsoft Store 창이
> 열리거나 `Python` 이라고만 표시된다면, 아직 Python 을 설치하지 않았거나
> **스토어 별칭(alias)** 이 잡힌 상태입니다. 이때는 `py` 런처를 사용하세요.
>
> ```powershell
> py --version        # 설치된 Python 버전 확인
> py -0p              # 설치된 모든 Python 목록 확인
> ```
>
> 아래 설치/실행 명령어는 `python` 대신 `py` 를 앞에 붙여도 똑같이 동작합니다.
> 예) `py -m venv .venv` / `py -m pip install -r requirements.txt` /
> `py run_pipeline.py` / `py -m streamlit run app.py` / `py -m pytest`

### 1-2. 프로젝트 폴더로 이동

PowerShell에서 프로젝트 폴더로 이동합니다. 예:

```powershell
cd "C:\Users\이름\Documents\AI_Etching_Troubleshooting"
```

### 1-3. 가상환경 생성

프로젝트 전용 가상환경을 만듭니다. (다른 프로젝트와 패키지가 섞이지 않습니다.)

```powershell
python -m venv .venv
```

`.venv` 폴더가 새로 만들어지면 성공입니다.

### 1-4. 가상환경 활성화

```powershell
.\.venv\Scripts\Activate.ps1
```

활성화되면 PowerShell 프롬프트 앞에 `(.venv)` 가 표시됩니다.

> **⚠️ "이 스크립트를 실행할 수 없습니다" 오류가 나는 경우**
> PowerShell 실행 정책(execution policy)이 제한되어 있는 것입니다.
> 다음 명령어로 현재 사용자에 대해서만 허용할 수 있습니다.
>
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```
>
> 그래도 되지 않으면 **가상환경을 활성화하지 않고** 바로 실행해도 됩니다.
> 가상환경의 파이썬을 직접 지정하면 됩니다.
>
> ```powershell
> # 활성화 없이 아래처럼 경로를 지정해 실행
> .\.venv\Scripts\python.exe -m pip install -r requirements.txt
> .\.venv\Scripts\python.exe run_pipeline.py
> .\.venv\Scripts\python.exe -m streamlit run app.py
> .\.venv\Scripts\python.exe -m pytest
> ```

### 1-5. 필요한 패키지 설치

```powershell
python -m pip install -r requirements.txt
```

> 오류가 나면 아래 **[문제 해결](#-문제-해결-faq)** 항목을 참고하세요.

---

## 🚀 2. 실행 방법

### 2-1. 전체 시뮬레이션 실행 (권장, 처음 한 번)

```powershell
python run_pipeline.py
```

실행하면 다음 과정이 순서대로 진행되고, 콘솔에 진행 상황이 출력됩니다.

| 단계 | 내용 |
|------|------|
| 1단계 | 합성 Etching 데이터 생성 (5,000건 + 분포 변화 평가 데이터) |
| 2단계 | 데이터 검증 (결측값, 라벨, 값 범위) |
| 3단계 | 머신러닝 학습 (Random Forest / XGBoost, 80:20 Stratified split) |
| 4단계 | Optuna 하이퍼파라미터 최적화 (기본 30회) |
| 5단계 | 모델 평가 (테스트셋 + 분포 변화셋) |
| 6단계 | SHAP 원인 후보 분석 |
| 7~8단계 | 결과 CSV / 그래프 PNG 저장 |

**실행 옵션 (선택 사항):**

```powershell
python run_pipeline.py --trials 50        # Optuna 최적화 횟수 변경 (기본 30)
python run_pipeline.py --samples 8000     # 데이터 샘플 수 변경 (기본 5000)
python run_pipeline.py --skip-shap        # SHAP 분석 건너뛰기 (실험 속도 향상)
python run_pipeline.py --seed 7           # random seed 변경 (기본 42)
```

> 처음 실행 시 라이브러리 설치와 학습으로 **1~5분 정도** 걸릴 수 있습니다.
> (PC 사양과 `--trials` 값에 따라 달라집니다)

### 2-1b. AI 자동 분석 실행 (AutoML, 권장)

```powershell
.\.venv\Scripts\python.exe run_automl.py
```

버튼 하나(또는 이 명령 하나)로 다음 **10단계**를 자동 실행합니다.

| 단계 | 내용 |
|------|------|
| 1단계 | 데이터 불러오기 (없으면 자동 생성) |
| 2단계 | 데이터 품질 분석 (결측/중복/클래스 분포/검증) |
| 3단계 | 머신러닝 모델 자동 학습 (RF / XGBoost / Extra Trees / HistGradientBoosting) |
| 4단계 | Optuna 하이퍼파라미터 자동 최적화 (모델별, 분석 모드별 횟수) |
| 5단계 | 모델 성능 비교 (학습 내부 CV + 검증셋) |
| 6단계 | 최적 모델 선정 → `models/best_model.pkl` 저장 |
| 7단계 | SHAP 기반 불량 원인 후보 분석 (미지원 시 permutation importance) |
| 8단계 | 공정 조건 최적화 (Random Search / Differential Evolution) |
| 9단계 | Etching 프로파일 시뮬레이션 전후 비교 |
| 10단계 | HTML 보고서 자동 생성 (`results/automl/automl_report.html`) |

**분석 모드 (모델별 Optuna 최적화 횟수)**

| 모드 | 횟수 | 예상 소요 시간 |
|------|------|----------------|
| `--mode fast` (기본) | 모델별 5회 | 약 3~5분 |
| `--mode normal` | 모델별 15회 | 약 8~15분 |
| `--mode precise` | 모델별 30회 | 약 15~30분 |

```powershell
.\.venv\Scripts\python.exe run_automl.py --mode normal      # 일반 분석
.\.venv\Scripts\python.exe run_automl.py --mode precise     # 정밀 분석
.\.venv\Scripts\python.exe run_automl.py --trials 8         # 모델별 횟수 직접 지정
.\.venv\Scripts\python.exe run_automl.py --method random    # 조건 탐색을 Random Search 로
.\.venv\Scripts\python.exe run_automl.py --bias 180 --temp 65   # 현재 조건 직접 지정
```

> **과적합 방지**: 전체 데이터를 학습 60% / 검증 20% / 테스트 20% 로 나눈 뒤,
> 하이퍼파라미터 탐색과 모델 선정은 **학습·검증 데이터 내부**(Stratified K-Fold CV)에서만
> 수행합니다. 최종 테스트셋과 분포 변화 데이터는 모델 선정에 사용하지 않고
> 선정된 모델의 최종 평가·강건성 평가에만 1회 사용합니다.
> XGBoost / HistGradientBoosting 은 내부 홀드아웃으로 **Early Stopping** 합니다.

### 2-2. 웹 대시보드 실행

```powershell
python -m streamlit run app.py
```

실행 후 브라우저가 자동으로 열립니다.
(자동으로 열리지 않으면 주소창에 `http://localhost:8501` 을 입력하세요.)

**대시보드 페이지**

| 페이지 | 내용 |
|--------|------|
| 1. Project Overview | 프로젝트 목표와 전체 시스템 구조 |
| 2. Etching Process Simulator | 6개 공정 변수 슬라이더 + 2D 식각 프로파일, 5가지 불량 기준 형상 비교 |
| 3. Machine Learning Performance | Accuracy, Macro F1, Confusion Matrix, 최적화 전후 비교 |
| 4. Explainable AI | SHAP 중요도 / 특정 조건의 변수 기여도 / 생성 규칙 비교 |
| 5. AI Troubleshooting | 불량 예측 + 확률 + 기여 변수 + 추천 공정 조건 + 전후 비교 |
| 6. Research Results | 발표용 결과 종합 (성능, 최적화, 예측 예시, 전후 프로파일, 한계) |
| 7. AI Auto Troubleshooting | 자동 분석 시작 버튼 → 진행 상태(10단계) → 모델 비교/최적 모델/SHAP/추천 조건/프로파일 비교/결과 다운로드 (별도 프로세스 실행) |
| 8. Literature Validation | Guo et al.(2024), Seo et al.(2025) 논문과 시뮬레이션/SHAP 결과를 비교하여 물리적 타당성 및 머신러닝 분석 방법의 타당성 검토 |

> 대시보드를 보기 전에 **먼저 `python run_pipeline.py` 를 한 번 실행**해야
> 모델(`models/*.pkl`)과 결과(`results/*.png`)가 생성됩니다.

### 2-3. 테스트 실행

```powershell
python -m pytest
```

verbose 버전으로 상세히 보려면:

```powershell
python -m pytest -v
```

---

## 🧪 3. 데이터와 모델에 대한 설명

### 3-1. 입력 변수 (6개)

| 변수 | 설명 | 범위 (교육용 가상) |
|------|------|--------------------|
| Gas chemistry | 가스 화학 조성의 단순화 지표 | 0.0 ~ 1.0 |
| Gas flow | 가스 유량 | 50 ~ 200 sccm |
| Pressure | 챔버 압력 | 5 ~ 100 mTorr |
| Bias power | 바이어스 전력 | 50 ~ 300 W |
| Source power | 소스 전력 | 300 ~ 1500 W |
| Temperature | 웨이퍼 온도 | 20 ~ 100 °C |

### 3-2. 불량 유형 (라벨 5종)

| 라벨 | 불량 | 발생 규칙 (교육용 가정) |
|------|------|--------------------------|
| 0 | Normal | 결함 점수가 낮고 식각 속도·측벽 보호가 적절할 때 |
| 1 | Undercut | 측벽 보호(passivation)가 부족하고 온도가 높을 때 → 상단 부근 측방향 식각 증가 |
| 2 | Bowing | 측벽 보호가 중간 수준이고 pressure·bias가 높을 때 → 중간 깊이 측벽 팽창 |
| 3 | Etch Stop | 유효 식각 속도가 낮을 때 → 식각 중단 |
| 4 | Footing | 하단 형상 외곡 지표(footing index)가 클 때 → 하단 CD 변화 |

> **단순 무작위 라벨이 아닙니다.** 측벽 보호 효과·식각 속도·방향성 등 파생 지표를
> 수학적으로 계산한 뒤, 그 점수를 소프트맥스 확률로 변환해 라벨을 샘플링합니다.
> 변수 간 **상호작용**(예: 온도 ↑ → 화학 식각 ↑, bias ↑ → 측벽 보호막 sputtering)과
> **측정 오차(노이즈)**가 포함되어 있습니다.
> `random_state` 를 고정해 항상 같은 데이터가 재현됩니다.

데이터에는 불량 유형 외에 다음 지표도 포함됩니다.

- `etch_rate_nm_min` : 식각 속도 (nm/min)
- `passivation_index` : 측벽 보호 효과 지표 (0~1)
- `anisotropy_index` : 식각 방향성 지표 (0~1)
- `footing_index` : 하단 형상 외곡 지표 (0~1)
- `defect_severity` : 결함 심각도 (0~100)

### 3-3. 머신러닝 모델

| 모델 | 설명 |
|------|------|
| Model A | Random Forest |
| Model B | XGBoost (기본 하이퍼파라미터) |
| Model C | Hyperparameter Optimized XGBoost (Optuna) |

- 데이터 분할: **훈련 80% / 테스트 20%**, **Stratified split**
- 최적화(하이퍼파라미터 탐색)는 **훈련 데이터 내부의 교차검증(CV Macro F1)** 으로만 수행하며,
  **테스트 데이터는 최종 평가 전에는 사용하지 않습니다.**
- 합성 데이터의 특성상 단순 무작위 분할은 일반화 성능을 과대평가할 수 있어,
  **분포 변화 평가용 데이터**(일부 변수 분포·노이즈 수준 변경)를 별도 생성해 평가합니다.

### 3-4. Optuna 최적화

- 최적화 변수: `n_estimators`, `max_depth`, `learning_rate`, `subsample`,
  `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda`
- 목표: 교차검증 **Macro F1-score 최대화** (기본 30회, `--trials` 로 변경 가능)
- 최적화 이력은 `results/optimization_history.csv/.png` 에 저장됩니다.
- **최적화 후 성능이 실제로 향상되지 않으면 "향상됨"으로 표시하지 않습니다.**
  이 경우 기본 파라미터를 유지합니다.

### 3-5. 공정 조건 추천 (process_optimizer)

현재 공정 조건을 입력하면 모델이 불량 유형과 확률을 예측한 뒤,
**Random Search** 또는 **Differential Evolution** 으로 조건을 탐색해
**Normal 확률이 높은 조건**을 추천합니다.

- 추천 조건은 반드시 지정된 공정 변수 **범위 내부**에 있습니다.
- 초기 조건에서 지나치게 멀어지지 않도록 **변경량 패널티**를 적용합니다.
- 학습 데이터 분포에서 벗어난 조건에는 **분포 이탈(OOD) 경고와 패널티**를 적용합니다.
- 추천 조건은 **교육용 식각 시뮬레이터**에 입력되어 전후 형상 지표를 비교합니다.

---

## 📈 4. 결과 파일

`python run_pipeline.py` 실행 후 `results/` 폴더에 생성됩니다.

| 파일 | 내용 |
|------|------|
| `model_comparison.csv` | 세 모델의 Accuracy / Macro Precision·Recall·F1 / Weighted F1 |
| `model_comparison_distribution_shift.csv` | 분포 변화 데이터 평가 결과 |
| `per_class_metrics.csv` | 클래스별 Precision / Recall / F1 |
| `accuracy_comparison.png` | Model Accuracy Comparison 그래프 |
| `macro_f1_comparison.png` | Macro F1-score Comparison 그래프 |
| `confusion_matrix.png` | Confusion Matrix (최적화 모델, 테스트셋) |
| `confusion_matrix_distribution_shift.png` | Confusion Matrix (분포 변화 데이터) |
| `per_class_recall.png` | Per-class Recall Comparison 그래프 |
| `optimization_history.csv/.png` | Optuna 최적화 이력 |
| `optimization_result.json` | 최적 파라미터 / 전후 성능 / 향상 여부 |
| `feature_importance.png` | SHAP Global Feature Importance |
| `shap_summary.png`, `shap_summary_class*.png` | SHAP summary plot |
| `shap_global_importance.csv`, `shap_local_contributions.csv` | SHAP 수치 결과 |
| `shap_vs_generation_rules.png/.csv` | SHAP 결과 vs 실제 생성 규칙 비교 |
| `etching_profiles.png` | 5가지 불량 유형 기준 식각 프로파일 |
| `before_after_profiles.png` | 공정 조건 최적화 전후 프로파일 |
| `process_recommendation.csv` | 추천 공정 조건 예시 |

**그래프 디자인 규칙**: 흰색 배경 / 파란색 계열 강조 / 명확한 제목 / 축 단위 표기 /
200 dpi 고해상도 PNG / 색상만으로 구분하지 않도록 범례·값 라벨 추가.

`models/` 폴더에는 `random_forest.pkl`, `xgboost.pkl`, `optimized_xgboost.pkl` 이
자동 생성됩니다. `data/` 폴더에는 `synthetic_etching_data.csv` 가 생성됩니다.

### 4-bis. AI 자동 분석 결과 (`results/automl/`)

`run_automl.py` 또는 대시보드 7페이지에서 실행할 때 **기존 `results/` 와는 분리된**
`results/automl/` 폴더에 저장되므로 기존 결과를 덮어쓰지 않습니다.

| 파일 | 내용 |
|------|------|
| `automl_model_comparison.csv` | 4개 모델의 CV/검증 성능, 학습·최적화 시간 |
| `automl_optimization_summary.csv` | 모델별 기준선 CV vs 최적 CV, 최적 파라미터 |
| `automl_test_per_class.csv` | 선정 모델의 최종 테스트셋 클래스별 지표 |
| `model_selection.json` | 선정 기준과 실제 점수 |
| `automl_feature_importance.csv/.png` | 선정 모델의 중요 변수 (SHAP 또는 permutation) |
| `automl_class_top_features.csv` | 불량 유형별 주요 변수 (SHAP) |
| `automl_recommendation.csv` | 기존/추천 공정 조건 비교표 |
| `validation_metrics.csv/.png` | 추천 조건의 지표 전후 차이 + 프로파일 비교 그래프 |
| `automl_report.html` | 전체 결과 통합 HTML 보고서 |
| `automl_results.json`, `run_metadata.json` | 결과 요약 / 재현 정보 (패키지 버전·seed 등) |
| `job_status.json`, `job_log.txt` | 대시보드 실행 시 진행 상태/로그 |

선정된 모델은 `models/best_model.pkl` 에 저장됩니다 (기존 `models/*.pkl` 은 그대로 둠).

---

## 🧾 5. 테스트 항목

`tests/` 폴더의 pytest 는 다음을 실제 구현 함수에 연결해 검증합니다.

- 데이터가 5,000개 생성되는가?
- 입력 변수에 결측값이 없는가?
- 불량 라벨이 올바른가? (0~4, 5개 클래스 모두 존재)
- 같은 random_state 에서 동일한 데이터가 생성되는가?
- 입력 변수가 지정 범위 내에 있는가?
- 라벨이 단순 무작위가 아닌가? (지표와의 통계적 연관 검정)
- 모델 학습이 정상적으로 수행되는가? (RF / XGBoost)
- 예측 확률이 유효한가? (5클래스, 합 = 1)
- Stratified split이 클래스 분포를 보존하는가?
- 식각 형상이 유효한 좌표를 생성하는가? (유한값, 좌우 대칭, 표면 시작)
- 불량 유형별 형상 특성이 구분되는가? (Undercut/Bowing/Footing/Etch Stop)
- 공정 최적화 결과가 지정된 변수 범위를 벗어나지 않는가?
- SHAP 출력 형상 정규화가 다양한 버전의 입력을 처리하는가?
- AutoML 이 4개 모델을 모두 학습하고 지표/시간/히스토리를 기록하는가?
- 최적 모델 선정과 동점(tie-break) 처리가 올바른가?
- 데이터 분할에 누출(인덱스 중복)이 없는가?
- 자동 트러블슈팅의 추천 조건이 (합성 범위 ∩ 안전 범위) 내부에 있는가?
- 자동 보고서(HTML)가 실제 측정값과 한계 문구를 포함하는가?
- 진행 상태 기록과 CLI 인자가 정상 동작하는가?
- Streamlit 자동 분석 페이지가 등록되고 렌더링되는가?

> `test_data_generator.py` 와 `test_ml_pipeline.py` 는 비교적 빠르지만,
> 모델 학습과 Optuna 테스트가 포함되어 **1~3분** 정도 걸릴 수 있습니다.

---

## ❓ 문제 해결 (FAQ)

**Q1. `python` 명령이 인식되지 않습니다.**
→ Python 설치 시 `Add Python to PATH` 를 선택했는지 확인하거나,
PowerShell 을 재시작하세요. `py --version` 으로 확인할 수도 있습니다.

**Q2. `Activate.ps1` 을 실행할 수 없다는 오류가 납니다.**
→ 실행 정책 문제입니다. `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 를
실행하거나, 가상환경을 활성화하지 않고 `.\.venv\Scripts\python.exe` 를 직접 사용하세요.
(1-4절 참고)

**Q3. 패키지 설치 중 `Microsoft Visual C++ Build Tools` 관련 오류가 납니다.**
→ `numpy`, `pandas`, `scikit-learn`, `xgboost` 등은 Windows용 사전 빌드 바이너리 wheel 을
제공하므로 보통 빌드 도구가 필요 없습니다. 오류가 나면 먼저 `python -m pip install --upgrade pip`
를 실행한 뒤 재시도하세요. (Python 버전이 3.11이 맞는지 확인 — 일부 패키지는 최신 Python을
아직 지원하지 않을 수 있습니다)

**Q4. `run_pipeline.py` 실행 중 `ModuleNotFoundError: No module named 'src'` 가 납니다.**
→ 반드시 **프로젝트 루트 폴더**(`app.py` 가 있는 폴더)에서 실행해야 합니다.

**Q5. Streamlit 대시보드에서 이미지가 비어 있습니다.**
→ 먼저 `python run_pipeline.py` 를 실행해 `results/` 에 그래프를 생성해야 합니다.

**Q6. Optuna / SHAP 단계가 너무 느립니다.**
→ `python run_pipeline.py --skip-shap --trials 10` 처럼 횟수를 줄이거나 SHAP 을
건너뛸 수 있습니다.

**Q7. `ModuleNotFoundError: No module named 'matplotlib'` 등이 납니다.**
→ 가상환경이 **활성화된 상태**에서 설치했는지, 같은 가상환경의 python 으로
실행했는지 확인하세요. (`where python` 으로 경로 확인 가능)

**Q8. 그래프에 한글이 깨져 보입니다.**
→ 이 프로젝트는 Windows 기본 글꼴(`Malgun Gothic`)을 우선 사용하도록 설정되어
있습니다. 그래도 깨지면 그래프 라벨을 영문으로 확인하거나, 한글 글꼴을 설치하세요.

---

## 🔧 6. 주요 라이브러리

| 라이브러리 | 용도 |
|------------|------|
| NumPy / Pandas | 수치 계산, 데이터 처리 |
| Scikit-learn | Random Forest, 지표 계산, 데이터 분할 |
| XGBoost | Gradient Boosting 모델 |
| Optuna | 하이퍼파라미터 최적화 |
| SHAP | 모델 예측 기여도(원인 후보) 분석 |
| Matplotlib / Seaborn / Plotly | 정적/인터랙티브 그래프 |
| Streamlit | 웹 대시보드 |
| Joblib | 모델 저장/로드 |
| pytest | 자동 테스트 |

---

## 📌 참고

- 데이터 생성 규칙, 식각 형상 가정의 상세 수식은 각 소스코드의 docstring 에
  주석으로 명시되어 있습니다 (`src/data_generator.py`, `src/etching_simulator.py`).
- 이 프로젝트는 **교육·학습 목적**이며, 산출 결과를 공정 제어에 직접 사용하지 마세요.
