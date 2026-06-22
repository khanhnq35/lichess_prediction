# Runbook

Các lệnh dưới đây chạy từ root project.

## Cài Đặt

```bash
pip install -r requirements.txt
```

## Pipeline Chính

Smoke test:

```bash
python solution.py --target-games 100 --selected-month 2023-01 --output-dir outputs
```

10k verification:

```bash
python solution.py --target-games 10000 --selected-month 2023-01 --output-dir outputs_10k_check
```

Full assessment-scale:

```bash
python solution.py --target-games 100000 --output-dir outputs_full_check
```

Optional boosting profile, không dùng Stockfish:

```bash
pip install -r requirements-experiments.txt
python solution.py --target-games 100000 --output-dir outputs_full_boosting --model-profile boosting
```

Sau khi chạy xong, nếu output đáng giữ, chuyển vào `artifacts/` và cập nhật `docs/ARTIFACT_INDEX.md`.

## Experiment Modes Trong `solution.py`

History/identity experiment:

```bash
python solution.py --target-games 10000 --selected-month 2023-01 --output-dir outputs_10k_experiments_new --run-experiments
```

Clock experiment:

```bash
python solution.py --target-games 10000 --selected-month 2023-01 --output-dir outputs_10k_clock_new --run-clock-experiments
```

LightGBM/XGBoost experiment không dùng Stockfish:

10k run:
```bash
pip install -r requirements-experiments.txt
python solution.py --target-games 10000 --selected-month 2023-01 --output-dir outputs_boosting_experiments_10k --run-boosting-experiments
```

100k run (khuyên dùng cache CSV cục bộ để tránh đứt kết nối mạng):
```bash
python solution.py --target-games 100000 --output-dir outputs_boosting_experiments_100k --run-boosting-experiments --input-cache-csv experiment/outputs/cache/games_2023-11_100000.csv.gz
```

Mode này chỉ dùng dependency experiment riêng, không thay đổi `requirements.txt` chính và không dùng Stockfish.


Heavy report best models:

```bash
python solution.py \
  --target-games 100000 \
  --selected-month 2023-11 \
  --output-dir outputs_report_best_models_new \
  --run-report-best-models \
  --input-cache-csv experiment/outputs/cache/games_2023-11_100000.csv.gz \
  --stockfish-cache-path experiment/stockfish_cache.json
```

Heavy mode là research/reference, không phải default submission path.

## Script Phân Tích Phụ

```bash
python scripts/analysis/run_audit.py
python scripts/analysis/run_robustness_analysis.py
python scripts/analysis/generate_xai.py
```

Các script này ghi vào `artifacts/audits/current/`, `artifacts/robustness/current/`, và `artifacts/xai/current/`.

## Sanity Checks Trước Khi Nộp

```bash
python -m py_compile solution.py
find . -type f \( -name '*.pgn' -o -name '*.zst' -o -name '*.pgn.zst' \) -print
find . -type f -size +10M -print
du -sh .
```

Raw PGN/ZST không được xuất hiện. File lớn trong `.venv/`, `experiment/outputs/cache/`, hoặc `experiment/stockfish_cache.json` không được đưa vào final package.
