# Next Tuning Ideas

Tài liệu này liệt kê các hướng tinh chỉnh tiếp theo. Mục tiêu là cải thiện metric hợp lệ mà vẫn giữ pipeline lightweight, reproducible, leakage-safe.

## Ưu tiên 1: Ridge alpha grid cho Elo regression

Hiện Elo after-10 dùng history + identity + Ridge. Full 100k MAE đã giảm còn khoảng 91, tốt hơn rất nhiều so với baseline.

Đề xuất:

- Test `alpha` nhỏ quanh hiện tại:
  - `1.0`
  - `3.0`
  - `10.0`
  - `30.0`
  - `100.0`
- Chỉ chạy 10k trước.
- Chọn bằng avg MAE = `(White MAE + Black MAE) / 2`.

Rủi ro leakage: thấp nếu giữ nguyên feature set hiện tại.

Tiêu chí dừng:

- Nếu avg MAE cải thiện dưới 1 Elo ở 10k, không đáng đổi.
- Nếu cải thiện 3-5 Elo trở lên, verify bằng 100k sau.

## Ưu tiên 2: Regularization grid nhỏ cho after-10 classifier

After-10 full 100k đang là classifier tốt nhất: ROC-AUC 0.6107.

Đề xuất:

- Giữ feature set hiện tại: identity + `clk10_*`, no history.
- Test `C`:
  - `0.1`
  - `0.15`
  - `0.25`
  - `0.4`
  - `0.5`
- Chọn bằng ROC-AUC, tie-break bằng log loss rồi Brier.

Rủi ro leakage: thấp nếu chỉ đổi hyperparameter.

Tiêu chí dừng:

- Nếu ROC-AUC tăng dưới 0.002 ở 10k, không đổi production.
- Nếu log loss/Brier xấu đi nhiều, không đổi dù ROC-AUC tăng nhẹ.

## Ưu tiên 3: Recheck after-3 selected config

After-3 final selected full 100k thấp hơn previous full 100k:

- Previous full 100k after-3 ROC-AUC: 0.5728.
- Final selected full 100k after-3 ROC-AUC: 0.5667.

Đề xuất:

- Test lại after-3 trên 10k và 100k với:
  - no identity, no clock, `C=0.25`
  - identity, no clock, `C=0.25`
  - no identity, no clock, `C=0.5`
  - identity, no clock, `C=0.5`
- Nếu no-identity tốt hơn trên 100k, revert after-3 về baseline no-identity.

Rủi ro leakage: thấp.

Tiêu chí dừng:

- Ưu tiên full 100k result nếu 10k và 100k mâu thuẫn.
- Không thêm clock cho after-3 nếu metric không ổn định.

## Ưu tiên 4: Lightweight interaction features

Có thể thêm vài feature tương tác không leakage:

- `elo_diff_abs`
- `elo_diff_squared`
- `initial_time_seconds * increment_seconds`
- interaction giữa `elo_diff` và material_diff cho classification sau move 3/10.
- clock pressure flags sau 10 moves:
  - `low_clock_white`
  - `low_clock_black`
  - `both_low_clock`

Rủi ro leakage: thấp nếu chỉ dùng allowed info.

Tiêu chí dừng:

- Chỉ thêm nếu feature có tác dụng trên 10k và không làm code rối.
- Tránh tạo feature quá nhiều khiến report khó giải thích.

## Ưu tiên 5: Calibration diagnostics hoặc calibration model

Hiện đã có probability diagnostics. Nếu cần cải thiện log loss/Brier:

- Có thể thử `CalibratedClassifierCV` nhưng phải fit chỉ trên train.
- Nên dùng train-internal split hoặc CV trong train, không dùng validation cuối.

Rủi ro leakage: trung bình nếu implement sai split.

Tiêu chí dừng:

- Chỉ giữ nếu log loss/Brier cải thiện mà ROC-AUC không giảm đáng kể.

## Ưu tiên 6: Robustness test theo tháng khác

Hiện default full run chọn `2023-11`, còn 10k experiments dùng `2023-01`.

Đề xuất:

```bash
python solution.py --target-games 10000 --selected-month 2023-03 --output-dir outputs_10k_2023_03_check
python solution.py --target-games 10000 --selected-month 2023-07 --output-dir outputs_10k_2023_07_check
```

Rủi ro leakage: thấp.

Tiêu chí dừng:

- Nếu config chỉ tốt ở một tháng nhưng giảm mạnh ở tháng khác, không nên claim quá mạnh.

## Ưu tiên 7: Packaging cleanup

Trước khi nộp:

- Không nén toàn bộ workspace.
- Không include `.venv/`.
- Không include `experiment/stockfish_cache.json`.
- Không include raw `.pgn`, `.zst`, `.pgn.zst`.
- Chỉ include code/report/output cần thiết.

Rủi ro: rất quan trọng vì workspace hiện khoảng 1.0G, vượt xa yêu cầu dù production files nhỏ.

## Không nên làm

- Không thêm Stockfish vào final production pipeline.
- Không thêm deep learning.
- Không thêm XGBoost, LightGBM, CatBoost.
- Không dùng validation rows cho fitting hoặc model selection final mà không ghi rõ.
- Không dùng current Elo trong Elo regression.
- Không dùng total game length, termination, rating diff, future moves.
