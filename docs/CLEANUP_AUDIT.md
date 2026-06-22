# Cleanup Audit

Audit sau khi dọn workspace.

## Kết Quả Sanity Check

- `python -m py_compile solution.py scripts/analysis/generate_xai.py scripts/analysis/run_audit.py scripts/analysis/run_robustness_analysis.py`: pass.
- Không tìm thấy raw `.pgn`, `.zst`, hoặc `.pgn.zst`.
- Workspace size hiện khoảng `1.0G`, chủ yếu do `.venv/`, cache experiment, và git objects.
- `artifacts/` khoảng `13M`.
- `artifacts/production/` khoảng `9.7M`.
- `artifacts/experiments/` khoảng `1.4M`.
- `experiment/` khoảng `31M`.

## File Lớn Hơn 10MB

Các file lớn hiện có:

- `.venv/lib/python3.14/site-packages/torch/lib/libtorch_python.dylib`
- `.venv/lib/python3.14/site-packages/torch/lib/libtorch_cpu.dylib`
- `experiment/outputs/cache/games_2023-11_100000.csv.gz`
- `experiment/stockfish_cache.json`
- `.git/objects/f4/4e91061031446725f5aaeada7c3f637a03c3e6`

Các file này không thuộc final source submission. `.venv/`, dataset cache, và Stockfish cache đã được ignore hoặc được ghi rõ trong `docs/SUBMISSION_SCOPE.md`.

## Ghi Chú

- `.agents/` hiện read-only trong sandbox, nên tài liệu agent được đặt ở `docs/agents/`.
- Các report lịch sử trong `reports/working/` vẫn có thể nhắc tên output cũ dạng `outputs_*`; artifact tương ứng đã được gom vào `artifacts/` và index mới nằm ở `docs/ARTIFACT_INDEX.md`.
