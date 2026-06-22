# Refined Selection Report

Report này ghi lại lần chọn lại cấu hình dựa trên kết quả hiện có và một vòng verification mới. Đây là working report, không phải final submission report.

## Mục tiêu

Chọn các thành phần phù hợp nhất cho `solution.py` trong phạm vi lightweight và leakage-safe:

- Không dùng Stockfish.
- Không dùng deep learning.
- Không dùng LightGBM, XGBoost, CatBoost.
- Không thêm dependency mới.
- Không dùng validation rows để fit.
- Không dùng post-game/future features.
- Elo regression vẫn không dùng current `WhiteElo`, `BlackElo`, `elo_diff`, `mean_elo`.

## Những thứ đã xem xét

| Nhóm | Kết luận |
|---|---|
| Causal player history | Giữ cho Elo regression, không dùng cho classification |
| Hashed player identity | Giữ cho after-3, after-10, Elo regression |
| Clock features | Giữ cho after-10 classification, không dùng cho after-3/Elo production |
| Lightweight enhanced board features | Implement và test, nhưng không chọn cho default production vì 10k verification giảm metric |
| Stockfish/deep learning/tree-heavy results trong `experiment/` | Không chọn vì ngoài lightweight constraints |

## Refined lightweight test bị loại

Command:

```bash
python solution.py --target-games 10000 --selected-month 2023-01 --output-dir outputs_10k_refined_lightweight
```

Config đã thử:

- after-3: enhanced board + `clk3_*`, no identity, no text, `C=0.5`.
- after-10: enhanced board + identity + `clk10_*`, `C=0.25`.
- Elo: enhanced board + history + identity + `clk10_*`.

Kết quả chính:

| Task | Metric |
|---|---:|
| Before ROC-AUC | 0.5464 |
| After-3 ROC-AUC | 0.5421 |
| After-10 ROC-AUC | 0.5700 |
| White Elo MAE | 148.75 |
| Black Elo MAE | 151.13 |

So với selected 10k cũ, after-10 giảm từ 0.5772 xuống 0.5700 và Elo MAE xấu hơn. Vì vậy không chạy full 100k với config này.

## Cấu hình được chọn lại

Command verification 10k:

```bash
python solution.py --target-games 10000 --selected-month 2023-01 --output-dir outputs_10k_selected_after_review
```

Command full 100k:

```bash
python solution.py --target-games 100000 --output-dir outputs_full_selected_after_review
```

Final selected config hiện tại:

| Task | Config |
|---|---|
| White win before | baseline no-history, no-identity, no-clock, `C=1.0` |
| White win after 3 | player identity, no-history, no-clock, `C=0.25` |
| White win after 10 | player identity + `clk10_*`, no-history, `C=0.25` |
| Elo after 10 | causal history + player identity, no-clock, current Elo excluded |

`metrics.json` có thêm notes:

- `lightweight_enhanced_board_features_available = true`
- `lightweight_enhanced_board_features_selected = false`
- `enhanced_board_10k_verification = not_selected_metric_regression`
- `stockfish_or_deep_learning_used = false`
- `heavy_dependencies_added = false`

## 10k selected-after-review metrics

Nguồn: `outputs_10k_selected_after_review/metrics.json`.

| Model | ROC-AUC | Log loss | Brier | Accuracy |
|---|---:|---:|---:|---:|
| White win before | 0.5464 | 0.6856 | 0.2466 | 0.5220 |
| White win after 3 | 0.5453 | 0.6892 | 0.2483 | 0.5235 |
| White win after 10 | 0.5772 | 0.6863 | 0.2467 | 0.5500 |
| Elo expected baseline | 0.5513 | 0.6908 | 0.2485 | n/a |

| Model | White MAE | White RMSE | White R2 | Black MAE | Black RMSE | Black R2 |
|---|---:|---:|---:|---:|---:|---:|
| Elo after 10 | 147.23 | 195.40 | 0.7192 | 148.99 | 197.29 | 0.7138 |
| Elo mean baseline | 302.05 | 368.75 | -0.0000 | 303.51 | 368.76 | -0.0000 |

## Full 100k selected-after-review metrics

Nguồn: `outputs_full_selected_after_review/metrics.json`.

- Runtime: 483.67 giây.
- Selected month: `2023-11`.
- Parsed games: 213,463.
- Header-eligible games: 104,005.
- Final eligible games: 100,000.
- Train/validation: 80,000 / 20,000.
- Train positive rate: 0.49395.
- Validation positive rate: 0.49640.

### Classification

| Model | ROC-AUC | Log loss | Brier | Accuracy |
|---|---:|---:|---:|---:|
| White win before | 0.5788 | 0.6788 | 0.2433 | 0.5525 |
| White win after 3 | 0.5667 | 0.6837 | 0.2455 | 0.5427 |
| White win after 10 | 0.6107 | 0.6698 | 0.2391 | 0.5718 |
| Elo expected baseline | 0.5785 | 0.6808 | 0.2440 | n/a |
| Majority baseline | n/a | n/a | n/a | 0.5036 |

### Regression

| Model | White MAE | White RMSE | White R2 | Black MAE | Black RMSE | Black R2 |
|---|---:|---:|---:|---:|---:|---:|
| Elo after 10 | 91.05 | 132.21 | 0.8709 | 91.97 | 133.54 | 0.8687 |
| Elo mean baseline | 300.22 | 368.03 | -0.0002 | 300.59 | 368.53 | -0.0002 |

## Probability diagnostics full 100k

| Model | Min | Max | Mean | Std | P05 | P50 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Before | 0.0068 | 0.9857 | 0.4943 | 0.0823 | 0.3816 | 0.4933 | 0.6109 |
| After 3 | 0.0024 | 0.9977 | 0.4948 | 0.0960 | 0.3533 | 0.4949 | 0.6382 |
| After 10 | 0.0044 | 0.9881 | 0.4945 | 0.1244 | 0.2945 | 0.4945 | 0.6994 |
| Elo expected baseline | 0.0009 | 0.9977 | 0.5000 | 0.1069 | 0.3415 | 0.5000 | 0.6610 |

## Files and size

| Path | Size |
|---|---:|
| `outputs_10k_refined_lightweight/metrics.json` | 4.4K |
| `outputs_10k_refined_lightweight/validation_predictions.csv` | 329K |
| `outputs_10k_selected_after_review/metrics.json` | 4.4K |
| `outputs_10k_selected_after_review/validation_predictions.csv` | 329K |
| `outputs_full_selected_after_review/metrics.json` | 4.5K |
| `outputs_full_selected_after_review/validation_predictions.csv` | 3.2M |

Workspace size vẫn khoảng 1.0G do `.venv/` và exploratory `experiment/`. Không có raw `.pgn`, `.zst`, hoặc `.pgn.zst` trong lần kiểm tra này.

## Kết luận

Cấu hình phù hợp nhất hiện tại vẫn là selected config trước đó, không phải enhanced-board variant mới. Enhanced board code được giữ như optional feature implementation, nhưng default production không chọn vì 10k verification cho thấy metric regression. Không có dependency nặng mới và full 100k selected-after-review khớp kết quả tốt nhất đã verify trước đó.
