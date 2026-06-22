# Experiment Folder Review

Tài liệu này review riêng folder `experiment/`. Đây là nhánh exploratory, không phải production path hiện tại và không nên trộn trực tiếp vào final lightweight submission.

## Tổng quan

Folder `experiment/` chứa một framework thí nghiệm rộng hơn `solution.py`, gồm:

- Baseline recap.
- Enhanced features.
- Tree-based models.
- Stockfish evaluation.
- Deep learning models bằng PyTorch.
- Ensemble/stacking.
- Report so sánh tự sinh trong `experiment/outputs/comparison_report.md`.

Các artifact chính:

| File/folder | Vai trò |
|---|---|
| `experiment/run_experiments.py` | Runner nhiều phase thí nghiệm |
| `experiment/features.py` | Enhanced chess/tabular features |
| `experiment/models.py` | Tree/ensemble/baseline model helpers |
| `experiment/deep_learning.py` | PyTorch MLP/CNN models |
| `experiment/stockfish_eval.py` | Stockfish evaluator + local cache |
| `experiment/stockfish_cache.json` | Cache Stockfish, khoảng 13M |
| `experiment/outputs/experiment_results.csv` | Bảng kết quả thí nghiệm |
| `experiment/outputs/best_models.json` | Best models theo experiment framework |
| `experiment/outputs/comparison_report.md` | Report tự sinh từ experiment framework |
| `experiment/requirements.txt` | Extra dependencies cho experiment folder |

## Vì sao không phải production path

Final assessment constraints yêu cầu lightweight dependencies:

- `requests`
- `zstandard`
- `python-chess`
- `pandas`
- `numpy`
- `scikit-learn`

Trong khi folder `experiment/` có hoặc tham chiếu:

- `lightgbm`
- XGBoost/ensemble-style models trong report kết quả
- PyTorch
- Stockfish
- local Stockfish cache

Những thành phần này không phù hợp với production path đã chọn vì:

- Heavy dependencies ngoài yêu cầu.
- Stockfish có thể không có trong assessment environment.
- Deep learning làm submission nặng và khó reproducible hơn trong 24-hour take-home.
- `stockfish_cache.json` riêng nó đã lớn hơn giới hạn 10MB nếu nộp kèm.

## Kết quả tham khảo trong `experiment/outputs/comparison_report.md`

Report exploratory ghi nhận:

- Task T1 best: `LogReg(C=1.0)` với ROC-AUC khoảng 0.5786.
- Task T2 best exploratory: GradientBoosting + Stockfish với ROC-AUC khoảng 0.5793.
- Task T3 best exploratory: GradientBoosting + Stockfish với ROC-AUC khoảng 0.6505.
- Task T4 best exploratory: RandomForest với Avg MAE khoảng 73.0.

Các kết quả này có thể dùng để hiểu ceiling khi dùng mô hình/feature nặng hơn, nhưng không nên so sánh trực tiếp như final answer vì production constraints hiện tại không cho phép Stockfish/deep learning/heavy dependencies.

## Những phần có thể tham khảo sau

- Ý tưởng enhanced board features trong `experiment/features.py`.
- Cách tổ chức bảng kết quả trong `experiment/outputs/comparison_report.md`.
- Một số insight về feature importance nếu sau này muốn phân tích kỹ hơn.

Chỉ nên port ý tưởng sang `solution.py` nếu:

- Không thêm dependency nặng.
- Không cần Stockfish.
- Không dùng post-game/future information.
- Không làm file submission vượt 10MB.
- Có experiment 10k riêng để chứng minh cải thiện.

## Những phần không nên đưa vào final submission

- Toàn bộ `.venv/`.
- `experiment/stockfish_cache.json`.
- `experiment/__pycache__/`.
- Bất kỳ file/model/cache nào sinh ra từ Stockfish hoặc deep learning.
- Extra requirements ngoài package list chính.

Nếu cần nộp code cuối, nên coi `solution.py`, `requirements.txt`, `README.md`, selected outputs và report final gọn là canonical. Folder `experiment/` chỉ là exploratory notebook/script equivalent.

## Rủi ro nếu dùng nhầm kết quả `experiment/`

- Có thể vô tình claim final pipeline dùng Stockfish hoặc LightGBM dù `solution.py` không dùng.
- Có thể làm submission vượt 10MB.
- Có thể gây nghi ngờ reproducibility vì assessment environment không chắc có Stockfish/PyTorch/LightGBM.
- Có thể trộn metric từ sample/config khác với full production run.

## Kết luận

`experiment/` hữu ích như khu vực nghiên cứu phụ, nhưng không phải source of truth cho final lightweight pipeline. Khi tinh chỉnh tiếp, nên ưu tiên `solution.py` và các output folders `outputs_10k_*` / `outputs_full_final_selected`. Nếu lấy ý tưởng từ `experiment/`, cần port lại theo cách lightweight và leakage-safe.
