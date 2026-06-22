# Báo Cáo A/B Testing: Stockfish vs No-Stockfish (100,000 Games)

Báo cáo này trình bày kết quả đánh giá A/B Testing giữa việc **Sử dụng Stockfish** (Mode B) và **Không sử dụng Stockfish** (Mode A) trên các mô hình scikit-learn và Boosting.

## 1. Kết Quả Chi Tiết Từng Task

### 1.1 Task T2: Dự Đoán Kết Quả Thắng/Thua Sau 3 Nước Đi (After-3 classification)

| Mô hình | Mode A (No-SF) AUC | Mode B (With-SF) AUC | Thay đổi (Δ AUC) | Mode A Loss | Mode B Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LogisticRegression | 0.5817 | 0.5818 | **+0.0002** | 0.6776 | 0.6777 |
| GradientBoosting | 0.5811 | 0.5827 | **+0.0015** | 0.6786 | 0.6780 |
| HistGradientBoosting | 0.5807 | 0.5826 | **+0.0019** | 0.6789 | 0.6785 |
| RandomForest | 0.5723 | 0.5731 | **+0.0009** | 0.6826 | 0.6826 |
| LightGBM | 0.5818 | 0.5831 | **+0.0013** | 0.6783 | 0.6781 |
| XGBoost | 0.5819 | 0.5831 | **+0.0013** | 0.6784 | 0.6782 |

### 1.2 Task T3: Dự Đoán Kết Quả Thắng/Thua Sau 10 Nước Đi (After-10 classification)

| Mô hình | Mode A (No-SF) AUC | Mode B (With-SF) AUC | Thay đổi (Δ AUC) | Mode A Loss | Mode B Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LogisticRegression | 0.6188 | 0.6447 | **+0.0258** | 0.6655 | 0.6557 |
| GradientBoosting | 0.6219 | 0.6486 | **+0.0267** | 0.6649 | 0.6524 |
| HistGradientBoosting | 0.6225 | 0.6495 | **+0.0270** | 0.6645 | 0.6525 |
| RandomForest | 0.6130 | 0.6404 | **+0.0274** | 0.6732 | 0.6631 |
| LightGBM | 0.6220 | 0.6485 | **+0.0264** | 0.6649 | 0.6525 |
| XGBoost | 0.6205 | 0.6478 | **+0.0273** | 0.6660 | 0.6532 |

### 1.3 Task T4: Dự Đoán ELO Kỳ Thủ Sau 10 Nước Đi (Elo regression)

| Mô hình | Mode A (No-SF) MAE | Mode B (With-SF) MAE | Thay đổi (Δ MAE) | Mode A $R^2$ | Mode B $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Ridge | 89.39 | 89.38 | **-0.01** | 0.8699 | 0.8699 |
| GradientBoosting | 37.54 | 37.44 | **-0.10** | 0.9448 | 0.9454 |
| HistGradientBoosting | 30.58 | 30.31 | **-0.27** | 0.9518 | 0.9529 |
| RandomForest | 38.85 | 38.85 | **+0.00** | 0.9402 | 0.9402 |
| LightGBM | 33.55 | 33.34 | **-0.21** | 0.9469 | 0.9477 |
| XGBoost | 38.40 | 38.96 | **+0.55** | 0.9411 | 0.9410 |

## 2. Kết Luận & Phân Tích A/B Testing

- **Ảnh hưởng của Stockfish lên Classification (T2/T3)**: Đánh giá xem Stockfish mang lại cải thiện nhiều hơn ở giai đoạn Move 3 hay Move 10, và mô hình nào tận dụng tốt nhất thông tin ưu thế bàn cờ phi tuyến tính này.
- **Ảnh hưởng của Stockfish lên Elo Regression (T4)**: Đánh giá xem Stockfish có thực sự giúp ích cho việc dự đoán Elo hay không. (Lịch sử các lần chạy trước cho thấy Stockfish hầu như không làm giảm MAE Elo vì thông tin Elo đã được ước lượng rất chính xác qua history features).
- **So sánh tốc độ huấn luyện**: Đối chiếu thời gian huấn luyện giữa scikit-learn models và Boosting models.
