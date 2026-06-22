# Submission Scope

Đây là phạm vi gợi ý cho final package khi đến bước nộp. Hiện tại chưa package.

## Nên Include

| File/Folder | Lý do |
| --- | --- |
| `solution.py` | Pipeline chính reproducible. |
| `requirements.txt` | Dependency lightweight. |
| `README.md` | Hướng dẫn chạy, assumptions, leakage prevention, LLM usage. |
| `problem.md` | Có thể include để reviewer thấy mapping, nếu cần. |
| `docs/PROBLEM_TRACEABILITY.md` | Giải thích đáp ứng yêu cầu. |
| `docs/DECISION_LOG.md` | Tóm tắt quyết định model/feature. |
| `docs/ARTIFACT_INDEX.md` | Nếu include outputs nhỏ, giải thích nguồn. |

## Có Thể Include Nếu Muốn Minh Chứng

| File | Ghi chú |
| --- | --- |
| `artifacts/production/full_selected_after_review/metrics.json` | Nhỏ, useful. |
| `artifacts/production/full_selected_after_review/validation_predictions.csv` | Khoảng vài MB, vẫn dưới 10MB riêng lẻ; cân nhắc theo yêu cầu final size. |

## Không Include

| Path | Lý do |
| --- | --- |
| `.venv/` | Rất lớn, tái tạo bằng `pip install`. |
| `experiment/outputs/cache/` | Dataset cache lớn, không cần cho reproducible streaming path. |
| `experiment/stockfish_cache.json` | Cache lớn và thuộc heavy experiment. |
| `artifacts/experiments/` | Không cần cho final source package, trừ khi muốn nộp appendix riêng. |
| `artifacts/archive/` | Smoke/legacy outputs. |
| Raw `.pgn`, `.zst`, `.pgn.zst` | Bị loại theo yêu cầu. |

## Cách Stage Khi Gần Nộp

Tạo folder staging riêng, ví dụ `submission_package/`, rồi copy đúng file cần nộp. Không zip toàn bộ workspace.

Checklist:

- Source package dưới 10MB.
- Không raw Lichess data.
- `python solution.py --target-games 100 --selected-month 2023-01` chạy được.
- README nêu rõ LLM usage, leakage prevention, validation split, và cách reproduce full run.
