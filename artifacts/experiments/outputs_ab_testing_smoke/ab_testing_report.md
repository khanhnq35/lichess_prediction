# Báo Cáo A/B Testing: Stockfish vs No-Stockfish (100 Games)

Báo cáo này trình bày kết quả đánh giá A/B Testing giữa việc **Sử dụng Stockfish** (Mode B) và **Không sử dụng Stockfish** (Mode A) trên các mô hình scikit-learn và Boosting.

## 1. Kết Quả Chi Tiết Từng Task

### 1.1 Task T2: Dự Đoán Kết Quả Thắng/Thua Sau 3 Nước Đi (After-3 classification)

| Mô hình | Mode A (No-SF) AUC | Mode B (With-SF) AUC | Thay đổi (Δ AUC) | Mode A Loss | Mode B Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LogisticRegression | 0.3542 | 0.3333 | **-0.0208** | 1.3443 | 1.3556 |
| GradientBoosting | 0.4375 | 0.3125 | **-0.1250** | 1.1415 | 1.4793 |
| HistGradientBoosting | 0.4167 | 0.4375 | **+0.0208** | 0.9345 | 0.9352 |
| RandomForest | 0.4375 | 0.4271 | **-0.0104** | 0.7636 | 0.7478 |
| LightGBM | 0.4583 | 0.4271 | **-0.0312** | 0.8559 | 0.8595 |
| XGBoost | 0.5521 | 0.4792 | **-0.0729** | 0.7836 | 0.8401 |

### 1.2 Task T3: Dự Đoán Kết Quả Thắng/Thua Sau 10 Nước Đi (After-10 classification)

| Mô hình | Mode A (No-SF) AUC | Mode B (With-SF) AUC | Thay đổi (Δ AUC) | Mode A Loss | Mode B Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LogisticRegression | 0.3542 | 0.3750 | **+0.0208** | 1.2818 | 1.2401 |
| GradientBoosting | 0.5208 | 0.4375 | **-0.0833** | 1.1622 | 1.1700 |
| HistGradientBoosting | 0.3021 | 0.2917 | **-0.0104** | 1.2258 | 1.1829 |
| RandomForest | 0.4375 | 0.3750 | **-0.0625** | 0.7307 | 0.7912 |
| LightGBM | 0.2917 | 0.2708 | **-0.0208** | 0.9619 | 0.8976 |
| XGBoost | 0.3542 | 0.3854 | **+0.0312** | 1.0104 | 0.9875 |

### 1.3 Task T4: Dự Đoán ELO Kỳ Thủ Sau 10 Nước Đi (Elo regression)

| Mô hình | Mode A (No-SF) MAE | Mode B (With-SF) MAE | Thay đổi (Δ MAE) | Mode A $R^2$ | Mode B $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Ridge | 320.82 | 316.18 | **-4.64** | -0.0361 | -0.0127 |
| GradientBoosting | 271.65 | 268.08 | **-3.57** | 0.1119 | 0.1853 |
| HistGradientBoosting | 236.13 | 243.16 | **+7.03** | 0.4063 | 0.3406 |
| RandomForest | 253.39 | 261.87 | **+8.47** | 0.2698 | 0.2333 |
| LightGBM | 250.81 | 253.77 | **+2.96** | 0.2870 | 0.2560 |
| XGBoost | 257.44 | 254.07 | **-3.37** | 0.2559 | 0.2464 |

## 2. Kết Luận & Phân Tích A/B Testing

- **Ảnh hưởng của Stockfish lên Classification (T2/T3)**: Đánh giá xem Stockfish mang lại cải thiện nhiều hơn ở giai đoạn Move 3 hay Move 10, và mô hình nào tận dụng tốt nhất thông tin ưu thế bàn cờ phi tuyến tính này.
- **Ảnh hưởng của Stockfish lên Elo Regression (T4)**: Đánh giá xem Stockfish có thực sự giúp ích cho việc dự đoán Elo hay không. (Lịch sử các lần chạy trước cho thấy Stockfish hầu như không làm giảm MAE Elo vì thông tin Elo đã được ước lượng rất chính xác qua history features).
- **So sánh tốc độ huấn luyện**: Đối chiếu thời gian huấn luyện giữa scikit-learn models và Boosting models.
