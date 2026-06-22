# Artifact Index

Các artifact đã được gom khỏi root vào `artifacts/`.

## Production

| Path | Trạng thái | Nội dung chính |
| --- | --- | --- |
| `artifacts/production/full_selected_after_review/` | Source of truth mới nhất | Full 100k normal path sau review. |
| `artifacts/production/full_final_selected/` | Tương đương source of truth | Full 100k selected config trước review; metric khớp. |
| `artifacts/production/full_baseline/` | So sánh cũ | Full 100k trước final selected configs. |

Metric chính của `full_selected_after_review`:

| Task | Metric |
| --- | --- |
| Before-game ROC-AUC | 0.5788 |
| After-3 ROC-AUC | 0.5667 |
| After-10 ROC-AUC | 0.6107 |
| Elo White MAE | 91.05 |
| Elo Black MAE | 91.97 |

## Experiments

| Path | Dùng để làm gì |
| --- | --- |
| `artifacts/experiments/10k_experiments/` | Grid nhỏ history/identity. |
| `artifacts/experiments/10k_clock_experiments/` | Clock feature experiment. |
| `artifacts/experiments/10k_history/` | Causal history thử riêng. |
| `artifacts/experiments/10k_final_selected/` | Verification 10k cho selected config. |
| `artifacts/experiments/10k_refined_lightweight/` | Enhanced-board lightweight trial, không chọn. |
| `artifacts/experiments/10k_selected_after_review/` | Verification 10k sau review. |
| `artifacts/experiments/report_best_models_10k/` | Heavy/Stockfish report run 10k. |
| `artifacts/experiments/report_best_models_100k/` | Heavy/Stockfish report run 100k. |
| `artifacts/experiments/boosting_no_stockfish_10k/` | LightGBM/XGBoost không Stockfish 10k, optional dependency. |
| `artifacts/experiments/boosting_no_stockfish_100k/` | LightGBM/XGBoost không Stockfish 100k, optional dependency. |
| `artifacts/experiments/ab_testing_10k/` | Kết quả chạy A/B testing 10k (Có vs Không Stockfish) cho mọi model. |
| `artifacts/experiments/ab_testing_100k/` | Kết quả chạy A/B testing 100k (Có vs Không Stockfish) cho mọi model. |

Heavy/Stockfish 100k tốt nhất:

| Task | Best metric |
| --- | --- |
| T1 LogReg + history | ROC-AUC 0.5792 |
| T2 GradientBoosting + Stockfish | ROC-AUC 0.5827 |
| T3 HistGradientBoosting + Stockfish | ROC-AUC 0.6483 |
| T4 RandomForest + Stockfish | White/Black MAE 27.97 / 28.07 |

Các kết quả heavy rất mạnh nhưng phụ thuộc cache/Stockfish và không nên nhập vào pipeline final lightweight nếu ràng buộc nộp vẫn là reproducible, nhỏ, không dependency nặng.

Boosting no-Stockfish tốt nhất (10k & 100k):

- **Thử nghiệm 10k**:
  | Task | Best config | Metric |
  | --- | --- | ---: |
  | T1 Before | `lightgbm_conservative_before_history` | ROC-AUC 0.5486 |
  | T2 After-3 | `xgboost_conservative_after3_enhanced` | ROC-AUC 0.5556 |
  | T3 After-10 | `xgboost_conservative_after10_enhanced_clock` | ROC-AUC 0.5917 |
  | T4 Elo | `lightgbm_conservative_elo_enhanced_history` | Avg MAE 91.12 |

- **Thử nghiệm 100k (Dữ liệu tháng 2023-11)**:
  | Task | Production Baseline (100k) | Best Boosting (100k) | Cấu hình tốt nhất |
  | --- | --- | --- | --- |
  | T1 Before | **ROC-AUC: 0.5788** | ROC-AUC: 0.5778 | `lightgbm_conservative_before_history` |
  | T2 After-3 | ROC-AUC: 0.5667 | **ROC-AUC: 0.5787** | `xgboost_conservative_after3_enhanced` |
  | T3 After-10 | ROC-AUC: 0.6107 | **ROC-AUC: 0.6219** | `xgboost_balanced_after10_enhanced_clock` |
  | T4 Elo | Avg MAE: 91.51 ELO | **Avg MAE: 29.31 ELO** | `lightgbm_balanced_elo_enhanced_history` |

**Kết luận**: Kết quả 100k xác nhận Boosting mang lại cải tiến vượt bậc ở T2, T3 và đặc biệt là T4 (Elo MAE giảm mạnh từ 91.52 xuống 29.27). Tuy nhiên, do yêu cầu cài đặt binary phức tạp của LightGBM/XGBoost, chúng tôi quyết định giữ chúng làm **Experiment Appendix** (chạy bằng flag `--run-boosting-experiments`) thay vì đưa vào Production Path mặc định để đảm bảo tính gọn nhẹ và chạy an toàn 100% của chương trình chính.

## Audits, Robustness, XAI

| Path | Vai trò |
| --- | --- |
| `artifacts/audits/current/` | Leakage audit, feature audit, repeat/unseen diagnostics. |
| `artifacts/robustness/current/` | Monthly robustness, calibration, lift, bootstrap CI. |
| `artifacts/xai/current/` | Feature importance, explainability snippets, checklist. |

## Archive

| Path | Vai trò |
| --- | --- |
| `artifacts/archive/smoke/outputs_100/` | Smoke test 100 games. |
| `artifacts/archive/smoke/outputs_1k/` | Scale test 1k. |
| `artifacts/archive/smoke/lightweight_profile_100/` | Smoke test 100 games cho `--model-profile lightweight`. |
| `artifacts/archive/smoke/boosting_profile_100/` | Smoke test 100 games cho `--model-profile boosting`. |
| `artifacts/archive/legacy_10k/outputs_10k/` | Baseline 10k cũ. |

## Quy Ước Khi Có Output Mới

1. Chạy vào output dir có tên rõ ràng.
2. Nếu output đáng giữ, chuyển vào `artifacts/<group>/<name>/`.
3. Cập nhật bảng này với mục đích và metric chính.
4. Không commit raw `.pgn`, `.zst`, `.pgn.zst`, cache Stockfish lớn, hoặc `.venv/`.
