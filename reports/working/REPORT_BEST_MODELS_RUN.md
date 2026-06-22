# Report-Selected Best Models Run

Report này ghi lại lần cài đặt và chạy các model tốt nhất theo bảng thí nghiệm:

| Task | Phương pháp |
|---|---|
| T1 Before-game | `LogReg(C=1.0)+Hist` |
| T2 After-3 | `GradientBoosting + Stockfish` |
| T3 After-10 | `HistGradientBoosting + Stockfish` |
| T4 Elo | `RandomForest + Stockfish` và `RandomForest` |

Đây là experiment mode nặng, không phải default production path của `solution.py`.

## Thay đổi đã implement trong `solution.py`

- Thêm CLI mode:

```bash
--run-report-best-models
```

- Thêm Stockfish options:

```bash
--stockfish-depth 10
--stockfish-cache-path experiment/stockfish_cache.json
```

- Thêm cache input option để chạy lại từ dataset cache khi network Lichess bị gãy:

```bash
--input-cache-csv experiment/outputs/cache/games_2023-11_100000.csv.gz
```

- Thêm các feature nhẹ từ `experiment/features.py`:
  - Piece-square table score
  - Pawn structure
  - King safety
  - Mobility
  - Development

- Thêm Stockfish features:
  - `sf3_cp`
  - `sf3_mate`
  - `sf10_cp`
  - `sf10_mate`
  - `sf10_cp_diff`

## Network note

Run trực tiếp từ Lichess bị lỗi stream:

- 10k run bị `BrokenPipeError` sau khoảng hơn 5,000 eligible games.
- 5k retry cũng bị `BrokenPipeError`.

Vì vậy kết quả chính được chạy từ cache có sẵn:

```bash
experiment/outputs/cache/games_2023-11_100000.csv.gz
```

và Stockfish cache:

```bash
experiment/stockfish_cache.json
```

Điều này giúp kiểm tra model implementation mà không phụ thuộc network. Cache này không phải artifact nên đưa vào submission vì file lớn hơn 10MB.

## Commands

10k run:

```bash
python solution.py \
  --target-games 10000 \
  --selected-month 2023-11 \
  --output-dir outputs_report_best_models_10k \
  --run-report-best-models \
  --stockfish-depth 10 \
  --input-cache-csv experiment/outputs/cache/games_2023-11_100000.csv.gz \
  --stockfish-cache-path experiment/stockfish_cache.json
```

100k run:

```bash
python solution.py \
  --target-games 100000 \
  --selected-month 2023-11 \
  --output-dir outputs_report_best_models_100k \
  --run-report-best-models \
  --stockfish-depth 10 \
  --input-cache-csv experiment/outputs/cache/games_2023-11_100000.csv.gz \
  --stockfish-cache-path experiment/stockfish_cache.json
```

## 10k results

Nguồn: `outputs_report_best_models_10k/metrics.json`.

- Runtime: 38.02s.
- Eligible games: 10,000.
- Train/validation: 8,000 / 2,000.
- Month: `2023-11`.

| Task | Model | ROC-AUC | Log loss | Brier | Accuracy |
|---|---|---:|---:|---:|---:|
| T1 | LogReg + history | 0.5655 | 0.6807 | 0.2442 | 0.5345 |
| T2 | GradientBoosting + Stockfish | 0.5739 | 0.6828 | 0.2449 | 0.5430 |
| T3 | HistGradientBoosting + Stockfish | 0.6248 | 0.6694 | 0.2388 | 0.5745 |

| Task | Model | White MAE | Black MAE | Avg MAE | White R2 | Black R2 |
|---|---|---:|---:|---:|---:|---:|
| T4 | RandomForest + Stockfish | 83.49 | 74.37 | 78.93 | 0.8363 | 0.8474 |

## 100k results

Nguồn: `outputs_report_best_models_100k/metrics.json`.

- Runtime: 458.55s.
- Eligible games: 100,000.
- Train/validation: 80,000 / 20,000.
- Parsed games from original cache metadata: 213,463.
- Stockfish cache entries after run: 119,352.
- Month: `2023-11`.

### Classification

| Task | Model | ROC-AUC | Log loss | Brier | Accuracy |
|---|---|---:|---:|---:|---:|
| T1 Before | `LogReg(C=1.0)+Hist` | 0.5792 | 0.6787 | 0.2432 | 0.5522 |
| T2 After-3 | `GradientBoosting + Stockfish` | 0.5827 | 0.6780 | 0.2429 | 0.5523 |
| T3 After-10 | `HistGradientBoosting + Stockfish` | 0.6483 | 0.6532 | 0.2314 | 0.6028 |
| Elo expected baseline | Elo expected score | 0.5785 | 0.6808 | 0.2440 | n/a |
| Majority baseline | Majority class | n/a | n/a | n/a | 0.5036 |

### Elo regression

| Model | White MAE | Black MAE | Avg MAE | White RMSE | Black RMSE | White R2 | Black R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `RandomForest + Stockfish` | 27.97 | 28.07 | 28.02 | 80.34 | 80.88 | 0.9523 | 0.9518 |
| `RandomForest` no Stockfish | 28.02 | 28.08 | 28.05 | 81.07 | 81.40 | 0.9515 | 0.9512 |
| Elo mean baseline | 300.22 | 300.59 | 300.41 | 368.03 | 368.53 | -0.0002 | -0.0002 |

## Comparison với default lightweight production

Default lightweight full 100k hiện tại:

| Task | Default lightweight | Report-selected heavy |
|---|---:|---:|
| T1 ROC-AUC | 0.5788 | 0.5792 |
| T2 ROC-AUC | 0.5667 | 0.5827 |
| T3 ROC-AUC | 0.6107 | 0.6483 |
| T4 Avg MAE | 91.51 | 28.02 |

Heavy models cải thiện rõ, đặc biệt T3 và T4. Tuy nhiên chúng không phù hợp nếu giữ strict lightweight submission constraints vì dùng Stockfish và tạo cache/artifact lớn.

## Output files

| Path | Size |
|---|---:|
| `outputs_report_best_models_10k/metrics.json` | 2.4K |
| `outputs_report_best_models_10k/experiment_results.csv` | 760B |
| `outputs_report_best_models_100k/metrics.json` | 2.4K |
| `outputs_report_best_models_100k/experiment_results.csv` | 789B |

Workspace vẫn khoảng 1.0G vì có `.venv/`, Stockfish cache, và cached dataset. Không có raw `.pgn`, `.zst`, hoặc `.pgn.zst` file trong lần kiểm tra sau run.

## Kết luận

Các model tốt nhất theo report đã được cài trong `solution.py` dưới một optional mode và đã chạy thành công. Kết quả 100k khớp rất gần với bảng thí nghiệm người dùng đưa ra:

- T1: 0.5792 ROC-AUC.
- T2: 0.5827 ROC-AUC.
- T3: 0.6483 ROC-AUC.
- T4: Avg MAE 28.0 với RandomForest + Stockfish.

Nên giữ mode này như experiment/heavy mode. Chỉ nên đưa vào final submission nếu chấp nhận phụ thuộc Stockfish và cache/model artifacts ngoài yêu cầu lightweight ban đầu.
