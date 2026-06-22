# Project Map

Tài liệu này là bản đồ nhanh để bạn, Codex, và agy không phải lần lại toàn bộ workspace.

## Root Cốt Lõi

| Path | Vai trò |
| --- | --- |
| `problem.md` | Đề bài gốc và ràng buộc cần đạt. Đây là contract cao nhất. |
| `solution.py` | Pipeline chính: download/stream PGN, parse, feature engineering, train, evaluate, output. |
| `requirements.txt` | Dependency lightweight cho bài nộp. |
| `README.md` | Hướng dẫn chạy và mô tả pipeline chính. |
| `.gitignore` | Chặn raw PGN, cache lớn, output ad-hoc. |

## Thư Mục Chính

| Path | Vai trò |
| --- | --- |
| `docs/` | Tài liệu điều phối, traceability, runbook, quyết định kỹ thuật. |
| `docs/agents/` | Handoff và quy tắc riêng cho Codex/agy. `.agents/` hiện read-only trong sandbox nên dùng thư mục này. |
| `reports/working/` | Report làm việc trước đây, giữ nguyên để tham khảo lịch sử. Một số path trong đó dùng tên output cũ trước khi dọn. |
| `scripts/analysis/` | Script phân tích phụ: audit, robustness, XAI. Không phải pipeline nộp chính. |
| `artifacts/` | Tất cả output đã chạy, được gom theo nhóm production/experiments/audits/robustness/XAI/archive. |
| `experiment/` | Framework thí nghiệm phụ, gồm các hướng nặng hơn như Stockfish/deep-learning style. Không phải source of truth cho final lightweight pipeline. |

## Source Of Truth Hiện Tại

- Đề bài: `problem.md`
- Code chính: `solution.py`
- Production artifact mới nhất: `artifacts/production/full_selected_after_review/`
- Full selected artifact tương đương trước đó: `artifacts/production/full_final_selected/`
- Report chọn model gần nhất: `reports/working/REFINED_SELECTION_REPORT.md`
- Best heavy/Stockfish experiment artifact: `artifacts/experiments/report_best_models_100k/`

## Lưu Ý

- Không zip toàn bộ workspace. `.venv/`, `experiment/outputs/cache/`, `experiment/stockfish_cache.json`, và các artifact lớn không thuộc final source package.
- Nếu chạy mới bằng `solution.py`, output mặc định vẫn là `outputs/`. Sau khi chạy xong nên đổi tên hoặc chuyển vào `artifacts/` và cập nhật `docs/ARTIFACT_INDEX.md`.
