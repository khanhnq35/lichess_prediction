# Tổng Hợp Toàn Bộ Thí Nghiệm Lichess

Report này tổng hợp các thí nghiệm chính đã chạy và kết luận cấu hình nào đáng dùng trong `solution.py`.

## 1. Production Lightweight Hiện Tại

Nguồn: `artifacts/production/full_selected_after_review/metrics.json`.

| Task | Model / Feature | Metric chính |
| --- | --- | ---: |
| T1 Before-game | LogisticRegression, Elo/time-control | ROC-AUC 0.5788 |
| T2 After-3 | LogisticRegression + move/board + hashed identity | ROC-AUC 0.5667 |
| T3 After-10 | LogisticRegression + move/board + identity + `clk10_*` | ROC-AUC 0.6107 |
| T4 Elo | Ridge + move/board + causal history + identity | White/Black MAE 91.05 / 91.97 |

Ưu điểm: nhẹ, chỉ dùng dependency chính trong `requirements.txt`, phù hợp yêu cầu reproducible và package gọn.

## 2. History / Identity / Clock Experiments

Nguồn:

- `artifacts/experiments/10k_experiments/`
- `artifacts/experiments/10k_clock_experiments/`
- `artifacts/experiments/10k_selected_after_review/`

Kết luận:

- Causal history giúp Elo regression mạnh, nhưng không cải thiện White-win classification ổn định.
- Hashed player identity giúp after-3 nhẹ trong một số run, nhưng không phải yếu tố mạnh nhất sau khi có boosting.
- Clock features hữu ích nhất cho after-10 classification.
- Cấu hình lightweight selected 10k đạt:
  - Before ROC-AUC 0.5464
  - After-3 ROC-AUC 0.5453
  - After-10 ROC-AUC 0.5772
  - Elo White/Black MAE 147.23 / 148.99

## 3. Refined Lightweight Trial

Nguồn: `artifacts/experiments/10k_refined_lightweight/`.

Enhanced-board lightweight variant không được chọn vì giảm chất lượng so với selected lightweight:

| Task | Selected 10k | Refined lightweight 10k |
| --- | ---: | ---: |
| After-3 ROC-AUC | 0.5453 | 0.5421 |
| After-10 ROC-AUC | 0.5772 | 0.5700 |
| Elo Avg MAE | 148.11 | 149.94 |

## 4. Boosting No-Stockfish Experiments

Nguồn:

- `artifacts/experiments/boosting_no_stockfish_10k/`
- `artifacts/experiments/boosting_no_stockfish_100k/`

Boosting dùng LightGBM/XGBoost, không dùng Stockfish, không dùng deep learning. Dependency nằm riêng trong `requirements-experiments.txt`.

### 4.1 Kết quả 10k

| Task | Production 10k | Best boosting 10k |
| --- | ---: | ---: |
| Before ROC-AUC | 0.5464 | 0.5486 |
| After-3 ROC-AUC | 0.5453 | 0.5556 |
| After-10 ROC-AUC | 0.5772 | 0.5917 |
| Elo Avg MAE | 148.11 | 91.12 |

### 4.2 Kết quả 100k

| Task | Production 100k | Best boosting 100k | Best config |
| --- | ---: | ---: | --- |
| Before ROC-AUC | **0.5788** | 0.5778 | `production_logreg_C1.0` vẫn tốt nhất |
| After-3 ROC-AUC | 0.5667 | **0.5787** | `xgboost_conservative_after3_enhanced` |
| After-10 ROC-AUC | 0.6107 | **0.6219** | `xgboost_balanced_after10_enhanced_clock` |
| Elo Avg MAE | 91.51 | **29.31** | `lightgbm_balanced_elo_enhanced_history` |

Kết luận: nếu chấp nhận optional dependencies, boosting là hướng tốt nhất không dùng Stockfish cho T2/T3/T4. T1 nên giữ LogisticRegression.

## 5. Heavy / Stockfish Experiments

Nguồn: `artifacts/experiments/report_best_models_100k/`.

| Task | Best heavy config | Metric |
| --- | --- | ---: |
| T1 Before | LogReg + history | ROC-AUC 0.5792 |
| T2 After-3 | GradientBoosting + Stockfish | ROC-AUC 0.5827 |
| T3 After-10 | HistGradientBoosting + Stockfish | ROC-AUC 0.6483 |
| T4 Elo | RandomForest + Stockfish | White/Black MAE 27.97 / 28.07 |

Stockfish cải thiện after-10 rất rõ, nhưng phụ thuộc external engine/cache. Không nên đưa vào default pipeline nếu mục tiêu là submission gọn và tái lập dễ.

## 6. A/B Stockfish

Nguồn:

- `artifacts/experiments/ab_testing_10k/`
- `artifacts/experiments/ab_testing_100k/`

Kết luận từ A/B 100k:

- After-3: Stockfish gần như không cải thiện nhiều, thường chỉ khoảng `+0.001` đến `+0.002` AUC.
- After-10: Stockfish cải thiện lớn, khoảng `+0.026` đến `+0.027` AUC.
- Elo: Stockfish gần như không giúp nhiều; các feature history/enhanced/player pattern đã giải thích Elo rất mạnh.

## 7. Cấu Hình Tốt Nhất Để Đưa Vào `solution.py`

Đã chọn cách đưa vào `solution.py` bằng `--model-profile`:

| Profile | Mục đích |
| --- | --- |
| `lightweight` | Default, giữ dependency chính gọn và phù hợp assessment ban đầu. |
| `boosting` | Optional no-Stockfish best models, cần `requirements-experiments.txt`. |

Best `boosting` profile:

| Task | Model |
| --- | --- |
| T1 Before | Production LogisticRegression `C=1.0` |
| T2 After-3 | XGBoost conservative + enhanced after-3 features |
| T3 After-10 | XGBoost balanced + enhanced after-10 + `clk10_*` |
| T4 Elo | LightGBM balanced + enhanced after-10 + causal history |

Không đưa Stockfish vào normal profile vì dependency nặng và external engine.

## 8. Khuyến Nghị Tiếp Theo

1. Chạy smoke cho `--model-profile boosting` để xác nhận code path mới.
2. Chạy 10k verification cho boosting profile và lưu artifact.
3. Thêm repeat/unseen diagnostics cho Elo boosting vì MAE khoảng 29 rất mạnh, cần giải thích generalization.
4. Khi viết final report, trình bày rõ:
   - Default lightweight là submission-safe.
   - Boosting là optional improved profile.
   - Stockfish là exploratory appendix, không phải pipeline mặc định.
