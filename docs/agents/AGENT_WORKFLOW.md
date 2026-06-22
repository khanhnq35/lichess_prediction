# Agent Workflow

Workspace này đang có hai agent chính: `codex` và `agy`. Mục tiêu là tránh sửa chồng chéo và giữ được lịch sử quyết định.

## Source Of Truth

1. `problem.md`: yêu cầu gốc.
2. `solution.py`: pipeline chính.
3. `docs/DECISION_LOG.md`: quyết định đã chốt.
4. `docs/ARTIFACT_INDEX.md`: output nào là quan trọng.
5. `reports/working/`: lịch sử report chi tiết.

## Quy Tắc Khi Một Agent Làm Việc

- Trước khi sửa model/feature, đọc `docs/PROBLEM_TRACEABILITY.md` và `docs/DECISION_LOG.md`.
- Sau khi tạo output mới, cập nhật `docs/ARTIFACT_INDEX.md`.
- Sau khi đổi lựa chọn model/feature, cập nhật `docs/DECISION_LOG.md`.
- Nếu sửa path hoặc dọn file, cập nhật `docs/PROJECT_MAP.md`.
- Không xoá output cũ nếu chưa có xác nhận; chuyển vào `artifacts/archive/` nếu cần.

## Phân Vai Gợi Ý

| Agent | Nên làm |
| --- | --- |
| Codex | Code changes, leakage audit, reproducibility checks, run smoke/full pipeline, package planning. |
| agy | Review report, so sánh metric, viết narrative, kiểm tra README/final report, đề xuất experiment tiếp. |

Phân vai này chỉ là mặc định. Nếu một agent làm việc của agent kia, phải ghi rõ trong handoff.

## Handoff Bắt Buộc

Khi dừng giữa chừng, tạo hoặc cập nhật một note theo mẫu `docs/agents/HANDOFF_TEMPLATE.md` trong report hoặc docs phù hợp.
