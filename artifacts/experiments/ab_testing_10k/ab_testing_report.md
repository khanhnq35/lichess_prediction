# Báo Cáo A/B Testing: Stockfish vs No-Stockfish (10,000 Games)

Báo cáo này trình bày kết quả đánh giá A/B Testing giữa việc **Sử dụng Stockfish** (Mode B) và **Không sử dụng Stockfish** (Mode A) trên các mô hình scikit-learn và Boosting.

## 1. Kết Quả Chi Tiết Từng Task

### 1.1 Task T2: Dự Đoán Kết Quả Thắng/Thua Sau 3 Nước Đi (After-3 classification)

| Mô hình | Mode A (No-SF) AUC | Mode B (With-SF) AUC | Thay đổi (Δ AUC) | Mode A Loss | Mode B Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LogisticRegression | 0.5685 | 0.5682 | **-0.0003** | 0.6806 | 0.6806 |
| GradientBoosting | 0.5788 | 0.5743 | **-0.0045** | 0.6805 | 0.6828 |
| HistGradientBoosting | 0.5663 | 0.5616 | **-0.0047** | 0.7000 | 0.7061 |
| RandomForest | 0.5618 | 0.5626 | **+0.0007** | 0.6847 | 0.6847 |
| LightGBM | 0.5737 | 0.5713 | **-0.0024** | 0.6791 | 0.6802 |
| XGBoost | 0.5795 | 0.5814 | **+0.0019** | 0.6776 | 0.6777 |

### 1.2 Task T3: Dự Đoán Kết Quả Thắng/Thua Sau 10 Nước Đi (After-10 classification)

| Mô hình | Mode A (No-SF) AUC | Mode B (With-SF) AUC | Thay đổi (Δ AUC) | Mode A Loss | Mode B Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LogisticRegression | 0.6195 | 0.6420 | **+0.0225** | 0.6651 | 0.6587 |
| GradientBoosting | 0.6085 | 0.6549 | **+0.0464** | 0.6698 | 0.6503 |
| HistGradientBoosting | 0.5979 | 0.6320 | **+0.0341** | 0.6830 | 0.6669 |
| RandomForest | 0.6018 | 0.6403 | **+0.0385** | 0.6760 | 0.6641 |
| LightGBM | 0.6123 | 0.6505 | **+0.0382** | 0.6667 | 0.6524 |
| XGBoost | 0.6128 | 0.6511 | **+0.0384** | 0.6686 | 0.6520 |

### 1.3 Task T4: Dự Đoán ELO Kỳ Thủ Sau 10 Nước Đi (Elo regression)

| Mô hình | Mode A (No-SF) MAE | Mode B (With-SF) MAE | Thay đổi (Δ MAE) | Mode A $R^2$ | Mode B $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Ridge | 146.14 | 146.15 | **+0.00** | 0.7198 | 0.7199 |
| GradientBoosting | 97.12 | 100.49 | **+3.37** | 0.8256 | 0.8224 |
| HistGradientBoosting | 91.11 | 91.77 | **+0.66** | 0.8326 | 0.8356 |
| RandomForest | 83.33 | 84.21 | **+0.88** | 0.8333 | 0.8330 |
| LightGBM | 87.98 | 89.97 | **+1.99** | 0.8364 | 0.8362 |
| XGBoost | 97.80 | 100.64 | **+2.83** | 0.8219 | 0.8198 |

## 2. Kết Luận & Phân Tích A/B Testing

- **Ảnh hưởng của Stockfish lên Classification (T2/T3)**: Đánh giá xem Stockfish mang lại cải thiện nhiều hơn ở giai đoạn Move 3 hay Move 10, và mô hình nào tận dụng tốt nhất thông tin ưu thế bàn cờ phi tuyến tính này.
- **Ảnh hưởng của Stockfish lên Elo Regression (T4)**: Đánh giá xem Stockfish có thực sự giúp ích cho việc dự đoán Elo hay không. (Lịch sử các lần chạy trước cho thấy Stockfish hầu như không làm giảm MAE Elo vì thông tin Elo đã được ước lượng rất chính xác qua history features).
- **So sánh tốc độ huấn luyện**: Đối chiếu thời gian huấn luyện giữa scikit-learn models và Boosting models.
