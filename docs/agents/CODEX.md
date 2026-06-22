# Codex Handoff

## Trạng Thái Hiện Tại

- Root đã được dọn: output chuyển vào `artifacts/`, scripts phụ vào `scripts/analysis/`.
- `.agents/` bị sandbox read-only, nên tài liệu agent nằm trong `docs/agents/`.
- Pipeline chính vẫn là `solution.py`; chưa đổi logic model trong lần dọn này.

## Khi Codex Làm Tiếp

1. Kiểm tra `git status --short`.
2. Nếu sửa model/feature, cập nhật `docs/DECISION_LOG.md`.
3. Nếu chạy output mới, chuyển vào `artifacts/` và cập nhật `docs/ARTIFACT_INDEX.md`.
4. Trước khi trả kết quả, chạy tối thiểu `python -m py_compile solution.py`.

## Cẩn Trọng

- Không dùng validation để fit hoặc chọn preprocessing.
- Không đưa current Elo vào Elo regression features.
- Không để Stockfish/heavy mode trở thành default production path nếu chưa có quyết định rõ.
