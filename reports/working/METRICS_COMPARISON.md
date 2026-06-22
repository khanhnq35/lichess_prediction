# Metrics Comparison Notes

Tài liệu này so sánh metric giữa các lần chạy để biết hướng cải thiện nào đã có tác dụng. Đây là report nội bộ, chưa phải final submission.

## 10k runs

| Run | Before ROC-AUC | After-3 ROC-AUC | After-10 ROC-AUC | White Elo MAE | Black Elo MAE | Ghi chú |
|---|---:|---:|---:|---:|---:|---|
| `outputs_10k` | 0.5464 | 0.5409 | 0.5569 | 243.82 | 246.14 | Baseline 10k trước history/identity/clock |
| `outputs_10k_history` | 0.5417 | 0.5371 | 0.5543 | 147.80 | 149.24 | Causal history giúp Elo mạnh, classification giảm nhẹ |
| `outputs_10k_final_selected` | 0.5464 | 0.5453 | 0.5772 | 147.23 | 148.99 | Normal path với selected configs |
| `outputs_10k_refined_lightweight` | 0.5464 | 0.5421 | 0.5700 | 148.75 | 151.13 | Enhanced-board variant, không được chọn |
| `outputs_10k_selected_after_review` | 0.5464 | 0.5453 | 0.5772 | 147.23 | 148.99 | Config được chọn lại sau review |

## 10k experiment best configs

Nguồn: `outputs_10k_experiments/best_config.json` và `outputs_10k_clock_experiments/best_config.json`.

| Task | Best config | ROC-AUC / MAE | Ghi chú |
|---|---|---:|---|
| White win before | no history, no identity, no clock, `C=1.0` | ROC-AUC 0.5464 | History/identity không cải thiện before-game ở 10k |
| White win after 3 | player identity, no history, no clock, `C=0.25` | ROC-AUC 0.5453 | Identity giúp nhẹ, clock không giúp |
| White win after 10 | player identity + clock, no history, `C=0.25` | ROC-AUC 0.5772 | Clock + identity là cải thiện tốt nhất cho after-10 |
| Elo after 10 | history + identity, no clock | Avg MAE 148.11 | History + identity cải thiện lớn nhất |

## Full 100k comparison

| Run | Before ROC-AUC | After-3 ROC-AUC | After-10 ROC-AUC | White Elo MAE | Black Elo MAE |
|---|---:|---:|---:|---:|---:|
| `outputs_full` | 0.5788 | 0.5728 | 0.5956 | 235.95 | 236.34 |
| `outputs_full_final_selected` | 0.5788 | 0.5667 | 0.6107 | 91.05 | 91.97 |
| `outputs_full_selected_after_review` | 0.5788 | 0.5667 | 0.6107 | 91.05 | 91.97 |
| Delta | +0.0000 | -0.0061 | +0.0151 | -144.90 | -144.38 |

## Full 100k final selected metrics

Nguồn: `outputs_full_final_selected/metrics.json`.

### Classification

| Model | ROC-AUC | Log loss | Brier | Accuracy |
|---|---:|---:|---:|---:|
| White win before | 0.5788 | 0.6788 | 0.2433 | 0.5525 |
| White win after 3 | 0.5667 | 0.6837 | 0.2455 | 0.5427 |
| White win after 10 | 0.6107 | 0.6698 | 0.2391 | 0.5718 |
| Elo expected baseline | 0.5785 | 0.6808 | 0.2440 | n/a |
| Majority baseline | n/a | n/a | n/a | 0.5036 |

### Elo regression

| Model | White MAE | White RMSE | White R2 | Black MAE | Black RMSE | Black R2 |
|---|---:|---:|---:|---:|---:|---:|
| Elo after 10 | 91.05 | 132.21 | 0.8709 | 91.97 | 133.54 | 0.8687 |
| Elo mean baseline | 300.22 | 368.03 | -0.0002 | 300.59 | 368.53 | -0.0002 |

## Nhận xét chính

- After-10 classification là horizon tốt nhất: full 100k ROC-AUC tăng từ 0.5956 lên 0.6107 sau khi thêm selected identity + clock config.
- After-3 classification giảm so với previous full 100k dù selected 10k config từng tốt hơn baseline 10k. Điều này cho thấy after-3 identity signal chưa ổn định khi scale sang tháng mặc định `2023-11`.
- Elo regression cải thiện rất mạnh nhờ causal history + identity: MAE giảm từ khoảng 236 xuống khoảng 91.
- Enhanced-board variant mới được implement và test ở 10k nhưng không được chọn vì after-10 ROC-AUC và Elo MAE đều xấu hơn selected config.
- Kết quả Elo MAE 91 không tự động là leakage, vì model không dùng current Elo; tuy nhiên cần diễn giải rõ là identity/history giúp nhiều cho repeat-player prediction trong cùng stream tháng.
- Accuracy 0.8 hoặc Elo MAE dưới 50 không nên đặt làm mục tiêu cho lightweight leakage-safe pipeline này.

## Probability diagnostics full 100k

| Model | Min | Max | Mean | Std | P05 | P50 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Before | 0.0068 | 0.9857 | 0.4943 | 0.0823 | 0.3816 | 0.4933 | 0.6109 |
| After 3 | 0.0024 | 0.9977 | 0.4948 | 0.0960 | 0.3533 | 0.4949 | 0.6382 |
| After 10 | 0.0044 | 0.9881 | 0.4945 | 0.1244 | 0.2945 | 0.4945 | 0.6994 |
| Elo expected baseline | 0.0009 | 0.9977 | 0.5000 | 0.1069 | 0.3415 | 0.5000 | 0.6610 |

After-10 có probability spread lớn hơn, phù hợp với việc nó có nhiều thông tin quan sát hơn sau 20 plies. Chưa thấy dấu hiệu overconfidence bất thường theo diagnostics này.
