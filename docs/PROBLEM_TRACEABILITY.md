# Problem Traceability

Mapping này nối từng yêu cầu trong `problem.md` với phần hiện có trong project.

| Yêu cầu | Hiện trạng |
| --- | --- |
| Dùng Lichess open database | `solution.py` build URL `https://database.lichess.org/standard/lichess_db_standard_rated_{YYYY-MM}.pgn.zst`. |
| Pick random month reproducibly | `Config.random_seed=42`, candidate months 2023-01..2023-12, có `--selected-month` để override. |
| Chọn một time-control | Default `Blitz`. Filter bằng `Event` header. |
| First 100,000 eligible games | Pipeline stream PGN, dừng sau `target_games` eligible. |
| Không decompress full PGN ra disk | Dùng `requests` stream + `zstandard` stream reader + `python-chess`. |
| Predict White win before game | Model `white_win_before_game` trong normal path. |
| Predict White win after 3 full moves | Model `white_win_after_3_moves`, dùng 6 plies đầu. |
| Predict White win after 10 full moves | Model `white_win_after_10_moves`, dùng 20 plies đầu. |
| Predict both Elo after 10 moves | Model `elo_after_10_moves`. |
| Không dùng validation để fit | Split chronological 80/20; sklearn `Pipeline` fit trên train. |
| Không dùng post-game leakage | Feature guards loại `Result`, rating diff, termination, total length, future moves. |
| Elo regression không dùng current Elo | `elo_after_10_moves` loại `white_elo`, `black_elo`, `elo_diff`, `mean_elo`. |
| Metrics và predictions | `metrics.json`, `validation_predictions.csv` trong output dir. |
| LLM usage clarified | README có section LLM usage. |
| Final solution <=10MB | Source package nên chỉ gồm code/docs nhỏ; không include raw data/cache/artifacts lớn. |

## Cấu Hình Production Hiện Tại

Theo artifact mới nhất `artifacts/production/full_selected_after_review/metrics.json`:

- Month: `2023-11`
- Time-control: `Blitz`
- Eligible games: `100000`
- Train/validation: `80000 / 20000`
- Before-game: LogisticRegression `C=1.0`, pre-game Elo/time-control only.
- After-3: LogisticRegression `C=0.25`, move/board + hashed player identity, no history/clock.
- After-10: LogisticRegression `C=0.25`, move/board + hashed player identity + `clk10_*`.
- Elo after-10: Ridge, move/board + causal history + hashed player identity, no current Elo, no clock.

## Điểm Cần Giữ Khi Tinh Chỉnh

- Nếu thử model nặng hoặc Stockfish, đặt trong experiment và report rõ là không thuộc lightweight production.
- Nếu thêm feature mới, ghi rõ thời điểm quan sát được feature đó: before game, sau 6 plies, hay sau 20 plies.
- Nếu output mới quan trọng, cập nhật `docs/ARTIFACT_INDEX.md` và `docs/DECISION_LOG.md`.
