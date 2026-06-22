# Current Pipeline State

Tài liệu này mô tả trạng thái production path hiện tại của `solution.py`. Đây là tài liệu nội bộ để tiếp tục tinh chỉnh, không phải final submission report.

## Entry point

Chạy production path hiện tại bằng:

```bash
python solution.py --target-games 100000 --output-dir outputs_full_final_selected
```

Không dùng:

```bash
--run-experiments
--run-clock-experiments
```

## Production configs hiện tại

| Task | Config |
|---|---|
| White win before-game | baseline no-history, no-identity, no-clock, `LogisticRegression(C=1.0)` |
| White win after 3 full moves | player identity, no-history, no-clock, `LogisticRegression(C=0.25)` |
| White win after 10 full moves | player identity, no-history, `clk10_*` clock features, `LogisticRegression(C=0.25)` |
| Elo after 10 full moves | causal history + player identity, no-clock, `Ridge(alpha=10.0)` |

Nguồn xác nhận: `outputs_full_final_selected/metrics.json` phần `feature_notes`.
Sau review mới, `outputs_full_selected_after_review/metrics.json` xác nhận cùng production config và thêm note rằng lightweight enhanced board features có sẵn nhưng không được chọn.

## Feature matrix

| Feature group | Before win | After-3 win | After-10 win | Elo after-10 |
|---|---:|---:|---:|---:|
| Current `white_elo`, `black_elo`, `elo_diff`, `mean_elo` | yes | yes | yes | no |
| Time-control features | yes | yes | yes | yes |
| First 3 move text | no | yes | no | no |
| First 10 move text | no | no | yes | yes |
| Board features after 6 plies | no | yes | no | no |
| Board features after 20 plies | no | no | yes | yes |
| Move-behavior `m3_*` | no | yes | no | no |
| Move-behavior `m10_*` | no | no | yes | yes |
| Hashed player identity | no | yes | yes | yes |
| Causal player-history features | no | no | no | yes |
| Clock features `clk3_*` | no | no | no | no |
| Clock features `clk10_*` | no | no | yes | no |
| Optional enhanced board `m3_enh_*` / `m10_enh_*` | no | no | no | no |

## Leakage controls đang có

- Chronological split: first 80% eligible games cho train, last 20% cho validation.
- `Pipeline` và `ColumnTransformer` đảm bảo imputer/scaler/vectorizer/model chỉ fit trên train.
- Không dùng validation rows để fit preprocessing hoặc model.
- Không dùng post-game leakage features:
  - `Result`
  - `Termination`
  - `WhiteRatingDiff`
  - `BlackRatingDiff`
  - total game length
  - future moves beyond prediction point
- After-3 model chỉ dùng first 6 plies và board sau 6 plies.
- After-10 model chỉ dùng first 20 plies và board sau 20 plies.
- Elo model không dùng current `WhiteElo`, `BlackElo`, `elo_diff`, `mean_elo`.
- Causal history features được compute trước khi update history bằng game hiện tại.
- Clock features được lấy từ clock comments trong allowed plies, không dùng clock sau prediction point.

## Features vẫn có trong code nhưng không dùng ở production path

- `clk3_*`: có thể dùng cho experiments nhưng final selected config không dùng after-3 clock.
- Classification history features: có trong code nhưng final selected classification models không dùng.
- Elo clock features: có thể test trong clock experiments nhưng final selected Elo model không dùng.
- `--run-experiments`: dùng để test history/identity configs.
- `--run-clock-experiments`: dùng để test clock configs.
- Lightweight enhanced board features: đã implement và test, nhưng default production không include vì 10k verification giảm metric.

## Baselines trong production output

- Majority class baseline:
  - Dùng train positive rate để chọn majority class.
  - Full 100k validation accuracy: 0.5036.
- Elo expected-score baseline:
  - `p = 1 / (1 + 10 ** (-(white_elo - black_elo) / 400))`.
  - Full 100k ROC-AUC: 0.5785.
- Elo mean baseline:
  - Predict validation Elo bằng train mean Elo.
  - Full 100k White MAE: 300.22; Black MAE: 300.59.

## Điều cần cẩn trọng khi tinh chỉnh tiếp

- Nếu thêm feature cho Elo regression, phải kiểm tra lại không có current Elo hoặc biến suy ra trực tiếp từ current Elo.
- Nếu thêm calibration, calibrator cũng phải fit chỉ trên train hoặc train-internal split, không fit trên validation cuối.
- Nếu test thêm tháng khác, nên giữ command/output folder riêng để không overwrite canonical artifacts.
- Nếu dùng folder `experiment/`, cần tách rõ exploratory results khỏi lightweight assessment pipeline.
