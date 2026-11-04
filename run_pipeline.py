"""
run_pipeline.py
===============

전체 시뮬레이션 파이프라인을 순서대로 자동 실행하는 스크립트입니다.

실행 순서
--------
1. 데이터 생성
2. 데이터 검증
3. 머신러닝 학습 (Random Forest / XGBoost)
4. 하이퍼파라미터 최적화 (Optuna → Optimized XGBoost)
5. 모델 평가 (테스트셋 + 분포 변화셋)
6. SHAP 원인 후보 분석
7. 결과 저장
8. 그래프 생성

사용법 (Windows PowerShell)
--------------------------
    python run_pipeline.py
    python run_pipeline.py --trials 50        # Optuna 반복 횟수 변경
    python run_pipeline.py --samples 8000     # 데이터 샘플 수 변경
    python run_pipeline.py --skip-shap        # SHAP 분석 건너뛰기 (속도 우선)
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Windows 콘솔(cp949 등)에서 인코딩할 수 없는 문자가 나와도
# 프로그램이 중단되지 않도록 안전하게 출력 (예: 특수 기호)
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.data_generator import (  # noqa: E402
    N_SAMPLES_DEFAULT,
    RANDOM_STATE,
    default_data_path,
    generate_dataset,
    generate_shifted_dataset,
    save_dataset,
    validate_dataset,
)


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def run(n_samples: int = N_SAMPLES_DEFAULT,
        n_trials: int = 30,
        skip_shap: bool = False,
        random_state: int = RANDOM_STATE) -> int:
    """전체 파이프라인을 실행합니다. 성공 시 0, 실패 시 1 반환."""
    t_total = time.time()

    try:
        # ------------------------------------------------------------------
        # 1. 데이터 생성
        # ------------------------------------------------------------------
        banner("1단계: 합성 Etching 데이터 생성")
        df = generate_dataset(n_samples=n_samples, random_state=random_state)
        data_path = default_data_path()
        save_dataset(df, data_path)
        print(f"    생성 완료: {len(df):,}건 → {data_path}")
        print("    ※ 교육용 합성 데이터이며 실제 공정 레시피가 아닙니다.")

        # 분포 변화 평가용 데이터 (단순 무작위 분할 과대평가 방지)
        df_shift = generate_shifted_dataset(n_samples=max(n_samples // 5, 500),
                                            random_state=random_state + 100)
        shift_path = data_path.parent / "synthetic_etching_data_shift.csv"
        save_dataset(df_shift, shift_path)
        print(f"    분포 변화 평가 데이터: {len(df_shift):,}건 → {shift_path}")

        # ------------------------------------------------------------------
        # 2. 데이터 검증
        # ------------------------------------------------------------------
        banner("2단계: 데이터 검증")
        ok, messages = validate_dataset(df)
        for m in messages:
            print(f"    {m}")
        if not ok:
            print("    [오류] 데이터 검증에 실패했습니다. 위 메시지를 확인하세요.")
            return 1

        # ------------------------------------------------------------------
        # 3. 머신러닝 학습
        # ------------------------------------------------------------------
        banner("3단계: 머신러닝 모델 학습 (80:20 Stratified split)")
        from src.train_models import (
            FEATURE_COLUMNS,
            load_training_frame,
            save_models,
            split_stratified,
            train_baseline_models,
            train_optimized_model,
        )

        data = load_training_frame(data_path)
        X_train, X_test, y_train, y_test = split_stratified(data, random_state=random_state)
        print(f"    훈련 데이터: {len(X_train):,}건 / 테스트 데이터: {len(X_test):,}건")
        models = train_baseline_models(X_train, y_train, random_state=random_state)

        # ------------------------------------------------------------------
        # 4. 하이퍼파라미터 최적화
        # ------------------------------------------------------------------
        banner("4단계: Optuna 하이퍼파라미터 최적화")
        from src.optimize_model import (
            compare_models_before_after,
            run_optimization,
            save_optimization_results,
        )

        opt_result = run_optimization(X_train, y_train, n_trials=n_trials,
                                      random_state=random_state, show_progress=False)
        save_optimization_results(opt_result)

        # 최적화가 실제로 향상되었을 때만 최적 파라미터를 사용 (아니면 기본 유지)
        if opt_result["improved"]:
            print("    최적 파라미터를 Model C 에 적용합니다.")
            best_params = opt_result["best_params"]
        else:
            print("    [주의] 최적화가 기본 성능을 넘지 못해 기본 파라미터를 유지합니다.")
            best_params = {}
        models["optimized_xgboost"] = train_optimized_model(
            X_train, y_train, best_params, random_state=random_state)

        cmp_tbl = compare_models_before_after(opt_result["baseline_score"],
                                              opt_result["best_score"])
        print(cmp_tbl.to_string(index=False))

        save_models(models)
        # 최적 파라미터도 저장 (재현용)
        import json
        params_path = BASE_DIR / "models" / "optimized_xgboost_params.json"
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump({"improved": opt_result["improved"],
                       "best_params": best_params}, f, indent=2, ensure_ascii=False)
        print(f"    [저장] {params_path}")

        # ------------------------------------------------------------------
        # 5. 모델 평가
        # ------------------------------------------------------------------
        banner("5단계: 모델 성능 평가 (동일 테스트 데이터 + 분포 변화 데이터)")
        from src.evaluate_models import evaluate_all_models, save_evaluation_results

        eval_results = evaluate_all_models(models, X_test, y_test, df_shift[X_test.columns],
                                           df_shift["defect_type"].to_numpy())
        saved_paths = save_evaluation_results(eval_results)

        # ------------------------------------------------------------------
        # 6. SHAP 분석
        # ------------------------------------------------------------------
        banner("6단계: SHAP 원인 후보 분석")
        shap_results = None
        if skip_shap:
            print("    (--skip-shap 옵션) SHAP 분석을 건너뜁니다.")
        else:
            try:
                from src.explain_model import run_shap_analysis

                shap_results = run_shap_analysis(
                    models["optimized_xgboost"], X_train, X_test, max_samples=400)
            except Exception as exc:
                print(f"    [경고] SHAP 분석 실패: {exc}")
                traceback.print_exc()

        # ------------------------------------------------------------------
        # 7 + 8. 결과 저장 / 그래프 생성 (공정 최적화 데모 포함)
        # ------------------------------------------------------------------
        banner("7~8단계: 공정 조건 최적화 데모 + 결과 요약")
        from src.etching_simulator import (
            compute_all_defect_profiles,
            default_process_conditions,
            profile_metrics_table,
        )
        from src.process_optimizer import (
            optimize_process,
            result_to_dataframe,
            train_distribution_stats,
        )
        from src.visualization import plot_before_after_profiles, plot_etching_profiles

        results_dir = BASE_DIR / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # (a) 5가지 불량 유형 기준 프로파일 그래프
        conds = default_process_conditions()
        profiles = compute_all_defect_profiles(conds)
        plot_etching_profiles(profiles, results_dir / "etching_profiles.png")
        profile_metrics_table(profiles).to_csv(
            results_dir / "etching_profile_metrics.csv",
            index=False, encoding="utf-8-sig")

        # (b) 공정 조건 최적화 예시
        stats = train_distribution_stats(X_train)
        rec = optimize_process(models["optimized_xgboost"], conds,
                               method="random", train_stats=stats,
                               random_state=random_state)
        result_to_dataframe(rec).to_csv(results_dir / "process_recommendation.csv",
                                        index=False, encoding="utf-8-sig")
        print("\n    [공정 조건 추천 예시]")
        print(result_to_dataframe(rec).to_string(index=False))
        print(f"    예측 불량: {rec['defect_before']} → {rec['defect_after']}")
        print(f"    P(Normal): {rec['normal_prob_before']:.3f} → {rec['normal_prob_after']:.3f}")
        for w in rec["warnings"]:
            print(f"    {w}")

        # (c) 전후 프로파일 그래프
        from src.process_optimizer import compare_profiles

        before_p, after_p, prof_tbl = compare_profiles(rec)
        plot_before_after_profiles(before_p, after_p,
                                   results_dir / "before_after_profiles.png")
        prof_tbl.to_csv(results_dir / "before_after_profile_metrics.csv",
                        index=False, encoding="utf-8-sig")

        # ------------------------------------------------------------------
        # 완료 요약
        # ------------------------------------------------------------------
        banner("파이프라인 완료 — 생성된 결과 파일 요약")
        for p in sorted(results_dir.glob("*")):
            print(f"    - {p.name}")
        print(f"\n    총 소요 시간: {time.time() - t_total:.1f}초")
        print("    웹 대시보드 실행: python -m streamlit run app.py")
        return 0

    except Exception as exc:
        print("\n" + "!" * 78)
        print(f"  [오류] 파이프라인 실행 중 예외가 발생했습니다: {exc}")
        print("!" * 78)
        traceback.print_exc()
        print("\n  해결 방법:")
        print("    1) requirements.txt 의 패키지가 모두 설치되었는지 확인")
        print("       python -m pip install -r requirements.txt")
        print("    2) 프로젝트 루트 폴더에서 실행했는지 확인")
        print("    3) python --version 이 3.11 이상인지 확인")
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Etching Troubleshooting 파이프라인 실행")
    parser.add_argument("--samples", type=int, default=N_SAMPLES_DEFAULT,
                        help="생성할 합성 데이터 샘플 수 (기본 5000)")
    parser.add_argument("--trials", type=int, default=30,
                        help="Optuna 최적화 반복 횟수 (기본 30)")
    parser.add_argument("--skip-shap", action="store_true",
                        help="SHAP 분석 건너뛰기 (실험 속도 향상)")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE,
                        help="random seed (기본 42)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("AI_Etching_Troubleshooting 파이프라인을 시작합니다.")
    print(f"  - 샘플 수: {args.samples:,} / Optuna trials: {args.trials} / seed: {args.seed}")
    code = run(n_samples=args.samples, n_trials=args.trials,
               skip_shap=args.skip_shap, random_state=args.seed)
    sys.exit(code)
