# Output Inventory

Tài liệu này liệt kê các output folders hiện có và vai trò của chúng. Đây là report nội bộ để tránh nhầm artifact khi tinh chỉnh.

## Canonical hiện tại

| Folder | Vai trò |
|---|---|
| `outputs_full_final_selected/` | Full 100k production path mới nhất, source of truth hiện tại |
| `outputs_10k_final_selected/` | 10k verification của normal production path sau khi chọn config |
| `outputs_full_selected_after_review/` | Full 100k sau lần review lựa chọn mới; cùng metric với canonical hiện tại |
| `outputs_10k_selected_after_review/` | 10k verification sau review; xác nhận selected config vẫn tốt hơn refined enhanced variant |

## Các output folders khác

| Folder | Vai trò | Có nên dùng làm source of truth? |
|---|---|---|
| `outputs/` | Smoke test 100 games | Không, chỉ xác nhận pipeline chạy |
| `outputs_1k/` | Scale test 1k | Không, sample nhỏ còn nhiễu |
| `outputs_10k/` | Baseline 10k trước history/identity/clock | Dùng để so sánh baseline |
| `outputs_10k_history/` | 10k với causal history | Dùng để hiểu tác dụng history |
| `outputs_10k_experiments/` | Grid nhỏ history/identity | Dùng để xem model selection 10k |
| `outputs_10k_clock_experiments/` | Clock feature experiments 10k | Dùng để xem tác dụng clock |
| `outputs_full/` | Previous full 100k trước final selected configs | Dùng để so sánh trước/sau |
| `outputs_full_final_selected/` | Full 100k final selected configs | Có, canonical hiện tại |
| `outputs_10k_refined_lightweight/` | 10k thử enhanced-board lightweight variant | Không, bị loại do metric giảm |
| `outputs_10k_selected_after_review/` | 10k confirm config được chọn lại | Có, dùng làm bằng chứng sau review |
| `outputs_full_selected_after_review/` | Full 100k confirm config được chọn lại | Có, source of truth mới nhất |
| `experiment/` | Exploratory framework riêng | Không, chỉ tham khảo |

## Kích thước đáng chú ý

Theo lần kiểm tra gần nhất:

| Path | Size |
|---|---:|
| Workspace `.` | khoảng 1.0G |
| `outputs_full_final_selected/` | khoảng 3.2M |
| `outputs_full_selected_after_review/` | khoảng 3.2M |
| `outputs_10k_final_selected/` | khoảng 340K |
| `outputs_10k_selected_after_review/` | khoảng 340K |
| `outputs_10k_refined_lightweight/` | khoảng 340K |
| `outputs_10k_experiments/` | khoảng 12K |
| `outputs_10k_clock_experiments/` | khoảng 8K |
| `experiment/` | khoảng 33M |
| `experiment/stockfish_cache.json` | khoảng 13M |
| `outputs_full_final_selected/metrics.json` | khoảng 4.2K |
| `outputs_full_final_selected/validation_predictions.csv` | khoảng 3.2M |

Workspace lớn chủ yếu do `.venv/` và exploratory files. Production output hiện tại vẫn nhỏ.

## File nên giữ để tham chiếu trong quá trình tinh chỉnh

- `outputs_full_final_selected/metrics.json`
- `outputs_full_final_selected/validation_predictions.csv`
- `outputs_10k_final_selected/metrics.json`
- `outputs_10k_experiments/experiment_results.csv`
- `outputs_10k_experiments/best_config.json`
- `outputs_10k_clock_experiments/experiment_results.csv`
- `outputs_10k_clock_experiments/best_config.json`
- `outputs_full/metrics.json`

## File/folder không nên đưa vào final submission

- `.venv/`
- `__pycache__/`
- `experiment/__pycache__/`
- `experiment/stockfish_cache.json`
- raw `.pgn`, `.zst`, `.pgn.zst` nếu xuất hiện
- các output folders cũ nếu không cần chứng minh thí nghiệm
- toàn bộ `experiment/` nếu cần submission gọn dưới 10MB

## Raw data check

Lần kiểm tra gần nhất bằng:

```bash
find . -type f \( -name '*.pgn' -o -name '*.zst' -o -name '*.pgn.zst' \) -print
```

Kết quả: không in ra file nào.

## Khuyến nghị khi chuẩn bị nộp sau này

Không zip toàn bộ `/Users/khanhnq35/Documents/Chess`. Nên tạo một staging folder riêng chỉ gồm các file cần nộp, ví dụ:

- `solution.py`
- `requirements.txt`
- `README.md`
- `outputs_full_final_selected/metrics.json`
- `outputs_full_final_selected/validation_predictions.csv`
- report cuối đã viết sau khi chốt tuning

Bộ report trong `reports/working/` là internal notes; có thể dùng để viết final report sau, nhưng không nhất thiết nộp nguyên bộ.
