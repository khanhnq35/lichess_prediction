# Next Steps

## Ưu Tiên Cao

1. Chốt submission philosophy:
   - Lightweight-only theo yêu cầu ban đầu.
   - Hay có appendix heavy/Stockfish để báo cáo exploratory nhưng không dùng làm final pipeline.
2. Kiểm tra lại `README.md` sau khi dọn folder để đảm bảo mọi path quan trọng đều rõ.
3. Nếu chọn lightweight final, chạy lại một smoke test vào output mới rồi chuyển vào `artifacts/archive/smoke/latest/`.
4. Nếu tiếp tục tuning, chỉ thử một thay đổi mỗi lần và ghi vào `docs/DECISION_LOG.md`.

## Tuning Có Thể Làm Tiếp

- Calibration cho after-10 classifier: thử `CalibratedClassifierCV` chỉ fit trên train split phụ, nhưng phải cẩn thận không dùng validation chính.
- Feature ablation nhỏ cho identity/history/clock để xác nhận stability trên tháng khác.
- Robustness run thêm tháng nếu thời gian cho phép.
- Report final giải thích vì sao accuracy 0.8 và Elo MAE <50 không thực tế nếu không dùng engine/deep models/cache mạnh.

## Không Nên Làm Nếu Muốn Giữ Submission Sạch

- Không đưa Stockfish vào normal path mặc định.
- Không thêm dependency nặng như XGBoost, LightGBM, CatBoost, PyTorch.
- Không include cache dataset hoặc Stockfish cache trong final package.
- Không tối ưu theo validation bằng nhiều vòng không ghi lại quyết định.
