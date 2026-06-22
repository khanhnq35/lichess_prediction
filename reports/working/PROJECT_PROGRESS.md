# Project Progress Notes

Tài liệu này ghi lại các mốc đã làm để tiện tiếp tục tinh chỉnh. Đây không phải final submission report.

## Trạng thái hiện tại

- Pipeline chính nằm ở `solution.py`.
- Dependency submission nằm ở `requirements.txt`: `requests`, `zstandard`, `python-chess`, `pandas`, `numpy`, `scikit-learn`.
- Production artifact mới nhất là `outputs_full_final_selected/`.
- Full production run mới nhất dùng 100,000 eligible Blitz games, selected month `2023-11`, random seed `42`.
- Không thấy raw `.pgn`, `.zst`, hoặc `.pgn.zst` trong workspace tại lần kiểm tra gần nhất.

## Timeline chính

| Giai đoạn | Artifact chính | Mục đích | Kết luận |
|---|---|---|---|
| Smoke test 100 games | `outputs/` | Kiểm tra code path end-to-end | Pipeline chạy được, metric không dùng để đánh giá chất lượng model |
| Scale test 1k | `outputs_1k/` | Kiểm tra thời gian chạy và output | Chạy ổn, sample nhỏ còn nhiễu |
| Baseline 10k | `outputs_10k/` | Đo baseline không history/identity/clock | After-10 tốt hơn after-3 nhẹ; Elo regression còn yếu |
| History 10k | `outputs_10k_history/` | Thử causal player-history | Elo regression cải thiện mạnh; classification không cải thiện |
| History/identity experiments 10k | `outputs_10k_experiments/` | Tìm cấu hình tốt với history và hashed player identity | Identity giúp after-3 nhẹ; history + identity tốt nhất cho Elo |
| Clock experiments 10k | `outputs_10k_clock_experiments/` | Thử clock features từ PGN comments | Clock giúp after-10 khi kết hợp identity; không dùng cho Elo |
| Final selected 10k | `outputs_10k_final_selected/` | Verify normal production path sau khi chọn config | Khớp best 10k experiment results |
| Previous full 100k | `outputs_full/` | Full run trước khi dùng final selected configs | After-10 ROC-AUC 0.5956; Elo MAE khoảng 236 |
| Final selected full 100k | `outputs_full_final_selected/` | Full production path hiện tại | After-10 ROC-AUC 0.6107; Elo MAE khoảng 91 |
| Refined lightweight trial 10k | `outputs_10k_refined_lightweight/` | Thử enhanced-board lightweight variant | Không chọn vì after-10/Elo giảm |
| Selected after review 10k/full | `outputs_10k_selected_after_review/`, `outputs_full_selected_after_review/` | Xác nhận lại config phù hợp nhất | Khớp best verified config trước đó |

## Command quan trọng đã dùng

```bash
python solution.py --target-games 100 --selected-month 2023-01 --output-dir outputs
python solution.py --target-games 1000 --selected-month 2023-01 --output-dir outputs_1k
python solution.py --target-games 10000 --selected-month 2023-01 --output-dir outputs_10k
python solution.py --target-games 10000 --selected-month 2023-01 --output-dir outputs_10k_history
python solution.py --target-games 10000 --selected-month 2023-01 --output-dir outputs_10k_experiments --run-experiments
python solution.py --target-games 10000 --selected-month 2023-01 --output-dir outputs_10k_clock_experiments --run-clock-experiments
python solution.py --target-games 10000 --selected-month 2023-01 --output-dir outputs_10k_final_selected
python solution.py --target-games 100000 --output-dir outputs_full
python solution.py --target-games 100000 --output-dir outputs_full_final_selected
python solution.py --target-games 10000 --selected-month 2023-01 --output-dir outputs_10k_refined_lightweight
python solution.py --target-games 10000 --selected-month 2023-01 --output-dir outputs_10k_selected_after_review
python solution.py --target-games 100000 --output-dir outputs_full_selected_after_review
```

## Full run hiện tại

Nguồn: `outputs_full_final_selected/metrics.json`.
Kết quả mới nhất sau review nằm ở `outputs_full_selected_after_review/metrics.json` và khớp với full selected config trước đó.

- Runtime: 558.45 giây.
- Selected month: `2023-11`.
- Time-control: `Blitz`.
- Parsed games: 213,463.
- Header-eligible games: 104,005.
- Final eligible games: 100,000.
- Train/validation: 80,000 / 20,000.
- Train positive rate: 0.49395.
- Validation positive rate: 0.49640.
- Result distribution: `1-0`: 49,444; `0-1`: 46,373; `1/2-1/2`: 4,183.

## Việc đã cải thiện được

- Thêm leakage-safe baselines:
  - Elo expected-score baseline cho White-win classification.
  - Elo mean baseline cho Elo regression.
- Thêm probability diagnostics cho classifier outputs.
- Tăng `LogisticRegression(max_iter=5000)` và dùng solver ổn định cho sparse hashed text để xử lý convergence warnings.
- Thêm UCI/SAN move text, board features, move-behavior features.
- Thêm causal player-history features, computed before current-game history update.
- Thêm hashed player identity features bằng `HashingVectorizer`.
- Thêm clock features từ Lichess PGN comments, giới hạn trong allowed plies.
- Chọn final configs từ 10k experiments rồi verify bằng normal path.

## Lưu ý hiện tại

- `experiment/` là nhánh exploratory riêng, có Stockfish, PyTorch, LightGBM/XGBoost/ensemble artifacts. Không phải production path của assessment.
- Workspace hiện khoảng 1.0G vì có `.venv/` và `experiment/`; không được nén toàn bộ workspace để nộp.
- Final output files nhỏ: `outputs_full_final_selected/metrics.json` khoảng 4.2K, `validation_predictions.csv` khoảng 3.2M.
