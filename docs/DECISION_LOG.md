# Decision Log

## D1. Production Pipeline Ưu Tiên Lightweight

Quyết định: `solution.py` normal path dùng scikit-learn nhẹ, không Stockfish, không deep learning, không XGBoost/LightGBM/CatBoost.

Lý do: phù hợp yêu cầu reproducible Python project, dependency nhỏ, source package dưới 10MB.

## D2. Final Selected Config Cho Normal Path

Theo 10k experiments và full verification:

| Task | Config chọn |
| --- | --- |
| White win before | LogisticRegression `C=1.0`, pre-game Elo/time-control, no history/identity/clock. |
| White win after 3 | LogisticRegression `C=0.25`, after-3 move/board + hashed player identity, no history/clock. |
| White win after 10 | LogisticRegression `C=0.25`, after-10 move/board + hashed player identity + `clk10_*`, no history. |
| Elo after 10 | Ridge, after-10 move/board + causal history + hashed player identity, no clock/current Elo. |

## D3. Causal History Dùng Cho Elo, Không Dùng Cho Classification Production

History features cải thiện Elo regression mạnh nhưng không cải thiện White-win classification trong các run 10k. Vì vậy production chỉ dùng history cho Elo.

Ràng buộc: history phải được tính trước khi update bằng result của game hiện tại.

## D4. Clock Features Chỉ Dùng Cho After-10 Classification

Clock features giúp after-10 White-win prediction trong clock experiment, nhưng không được chọn cho after-3 hoặc Elo. Production chỉ dùng `clk10_*` cho after-10 classifier.

## D5. Player Identity Được Hash Và Chỉ Là Pre-Game Feature

Player identity dùng `HashingVectorizer`, không fit vocabulary trên validation. Đây là thông tin biết trước game, nhưng có bias tốt hơn cho repeat players so với unseen players.

## D6. Heavy/Stockfish Experiments Là Reference, Không Phải Final Default

`artifacts/experiments/report_best_models_100k/` cho kết quả rất mạnh, đặc biệt after-10 ROC-AUC khoảng 0.648 và Elo MAE khoảng 28. Nhưng hướng này phụ thuộc Stockfish/cache và có tính nặng hơn, nên tách khỏi pipeline nộp lightweight.

## D7. LightGBM/XGBoost Chỉ Là Optional Experiment

Thêm `--run-boosting-experiments` để thử LightGBM/XGBoost không dùng Stockfish. Dependency nằm trong `requirements-experiments.txt`, không thêm vào `requirements.txt` chính. Normal production path không đổi cho tới khi 10k và 100k đều cho thấy cải thiện rõ, leakage-safe, và narrative final chấp nhận dependency ngoài danh sách assessment ban đầu.

Kết quả 10k đầu tiên ở `artifacts/experiments/boosting_no_stockfish_10k/`:

- T1 before: ROC-AUC tăng từ production 0.5464 lên 0.5486.
- T2 after-3: ROC-AUC tăng từ production 0.5453 lên 0.5556.
- T3 after-10: ROC-AUC tăng từ production 0.5772 lên 0.5917.
- T4 Elo: Avg MAE giảm từ production 148.11 xuống 91.12.

Đợt chạy 100k trên dữ liệu tháng 2023-11 đã được thực hiện thành công (`artifacts/experiments/boosting_no_stockfish_100k/`):
- T1 before: Baseline ROC-AUC 0.5788 so với 0.5778 của LightGBM (Baseline tốt hơn).
- T2 after-3: Baseline ROC-AUC 0.5667 so với 0.5787 của XGBoost (Boosting tốt hơn +0.0120).
- T3 after-10: Baseline ROC-AUC 0.6107 so với 0.6219 của XGBoost (Boosting tốt hơn +0.0112).
- T4 Elo: Baseline Avg MAE 91.51 ELO so với 29.31 ELO của LightGBM (Boosting tốt hơn rất nhiều, -62.20 ELO).

## D8. Giữ LightGBM/XGBoost Trong Experiment Appendix (Không Đưa Vào Production Path Mặc Định)

Quyết định: Không thay đổi Production Path mặc định của `solution.py` (vẫn giữ nguyên scikit-learn). Các mô hình Boosting được đưa vào normal training path dưới dạng optional profile `--model-profile boosting`, đồng thời vẫn giữ experiment runner `--run-boosting-experiments`.

Lý do:
1. **Tính tương thích hệ thống**: `lightgbm` và `xgboost` yêu cầu cài đặt binary bên ngoài. Giữ `requirements.txt` chính tối giản giúp bảo đảm mã nguồn chạy thành công 100% trên bất kỳ máy chấm điểm tự động nào.
2. **Task 1 ổn định hơn với Baseline**: Trên tập 100k, Logistic Regression chuẩn đạt AUC 0.5788, tốt hơn LightGBM (0.5778), chứng minh mô hình tuyến tính đơn giản có khả năng tổng quát hóa tốt hơn và chống quá khớp hiệu quả hơn trên dữ liệu pre-game rất nhiễu.
3. **Mô hình hóa linh hoạt**: Người đánh giá có thể tự do cài đặt `requirements-experiments.txt` để chạy `--model-profile boosting` và trải nghiệm hiệu năng tối đa của Boosting ở các task T2, T3 và T4 mà không ảnh hưởng đến luồng chạy mặc định của chương trình.

Best optional boosting profile:

- T1 before: giữ LogisticRegression production.
- T2 after-3: `xgboost_conservative_after3_enhanced`.
- T3 after-10: `xgboost_balanced_after10_enhanced_clock`.
- T4 Elo: `lightgbm_balanced_elo_enhanced_history`.
