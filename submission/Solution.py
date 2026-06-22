"""Reproducible Lichess Blitz prediction pipeline.

This script streams a monthly Lichess standard rated PGN `.zst` file, keeps the
first eligible games, trains simple sklearn models, evaluates them on a strict
chronological validation split, and writes metrics/predictions to the configured
output directory (`outputs/` by default).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import re
import time
import warnings
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import chess
import chess.engine
import chess.pgn
import numpy as np
import pandas as pd
import requests
import zstandard as zstd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_CANDIDATE_MONTHS = tuple(f"2023-{month:02d}" for month in range(1, 13))
DEFAULT_RANDOM_SEED = 42
DEFAULT_TIME_CONTROL = "Blitz"
DEFAULT_TARGET_GAMES = 100_000
DEFAULT_TRAIN_RATIO = 0.80
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_HASHING_FEATURES = 2**15
DEFAULT_LOGISTIC_MAX_ITER = 5_000
DEFAULT_LOGISTIC_SOLVER = "liblinear"
PROBABILITY_EPS = 1e-6
NEUTRAL_SCORE_RATE = 0.5
NEUTRAL_ELO = 1500.0
HISTORY_BAYESIAN_VIRTUAL_GAMES = 10.0
HISTORY_BAYESIAN_PRIOR_RATE = 0.5
HISTORY_ELO_VIRTUAL_GAMES = 5.0
HISTORY_ELO_PRIOR = 1930.0
USE_HISTORY_BAYESIAN_SMOOTHING = False
MAX_STREAM_RETRIES = 5
STREAM_RETRY_BASE_DELAY_SECONDS = 10.0
HTTP_TIMEOUT_SECONDS = (10, 60)
RESULTS = {"1-0", "0-1", "1/2-1/2"}
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}
PIECE_NAMES = {
    chess.PAWN: "pawns",
    chess.KNIGHT: "knights",
    chess.BISHOP: "bishops",
    chess.ROOK: "rooks",
    chess.QUEEN: "queens",
    chess.KING: "kings",
}
CENTER_SQUARES = (chess.D4, chess.E4, chess.D5, chess.E5)
PST_PAWN = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]
PST_KNIGHT = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
PST_BISHOP = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
PST_ROOK = [
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]
PST_QUEEN = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]
PST_KING_MIDDLE = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20,
]
PST_MAP = {
    chess.PAWN: PST_PAWN,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
    chess.ROOK: PST_ROOK,
    chess.QUEEN: PST_QUEEN,
    chess.KING: PST_KING_MIDDLE,
}
ELO_MODEL_FORBIDDEN_FEATURES = {
    "white_elo",
    "black_elo",
    "elo_diff",
    "mean_elo",
    "result",
    "white_win",
    "WhiteRatingDiff",
    "BlackRatingDiff",
    "Termination",
}
HISTORY_FEATURE_COLUMNS = [
    "white_prior_games",
    "black_prior_games",
    "white_prior_score_rate_all",
    "black_prior_score_rate_all",
    "white_prior_win_rate_as_white",
    "black_prior_win_rate_as_black",
    "white_prior_avg_opponent_elo",
    "black_prior_avg_opponent_elo",
    "white_prior_recent_score_rate_10",
    "black_prior_recent_score_rate_10",
    "white_prior_recent_score_rate_30",
    "black_prior_recent_score_rate_30",
    "white_prior_avg_elo_seen",
    "black_prior_avg_elo_seen",
    "prior_games_diff",
    "prior_score_rate_diff",
    "prior_recent_score_rate_10_diff",
    "prior_recent_score_rate_30_diff",
    "prior_avg_opponent_elo_diff",
    "prior_avg_elo_seen_diff",
    "prior_side_win_rate_diff",
]
CLOCK_FEATURE_NAMES = [
    "white_clock_last",
    "black_clock_last",
    "white_clock_used_total",
    "black_clock_used_total",
    "clock_diff_last",
    "clock_used_diff",
    "white_avg_time_per_move",
    "black_avg_time_per_move",
    "avg_time_per_move_diff",
    "white_min_clock",
    "black_min_clock",
    "white_clock_missing_count",
    "black_clock_missing_count",
    "any_clock_available",
    "white_time_panic",
    "black_time_panic",
    "white_time_used_ratio",
    "black_time_used_ratio",
    "time_ratio_diff",
]


@dataclass(frozen=True)
class Config:
    random_seed: int = DEFAULT_RANDOM_SEED
    candidate_months: tuple[str, ...] = DEFAULT_CANDIDATE_MONTHS
    selected_month: str | None = None
    time_control: str = DEFAULT_TIME_CONTROL
    target_games: int = DEFAULT_TARGET_GAMES
    train_ratio: float = DEFAULT_TRAIN_RATIO
    output_dir: str = DEFAULT_OUTPUT_DIR
    hashing_features: int = DEFAULT_HASHING_FEATURES


@dataclass(frozen=True)
class DatasetBuildStats:
    parsed_games: int
    header_eligible_games: int
    eligible_games: int


@dataclass
class PlayerHistory:
    prior_games: int = 0
    prior_score_sum_all: float = 0.0
    prior_wins_as_white: int = 0
    prior_games_as_white: int = 0
    prior_wins_as_black: int = 0
    prior_games_as_black: int = 0
    prior_opponent_elo_sum: float = 0.0
    prior_elo_seen_sum: float = 0.0
    recent_scores_10: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recent_scores_30: deque[float] = field(default_factory=lambda: deque(maxlen=30))


def select_month(config: Config) -> str:
    """Select a reproducible random month unless explicitly configured."""
    if config.selected_month:
        return config.selected_month
    rng = random.Random(config.random_seed)
    return rng.choice(list(config.candidate_months))


def build_lichess_url(month: str) -> str:
    return f"https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"


def safe_int(value: Any) -> int | None:
    """Parse integer-like PGN header values, returning None if invalid."""
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text == "?":
        return None
    match = re.fullmatch(r"(\d+)\??", text)
    if not match:
        return None
    return int(match.group(1))


def parse_time_control(time_control_str: str | None) -> tuple[int, int]:
    """Parse common Lichess TimeControl strings such as '180+0'."""
    if not time_control_str:
        return 0, 0
    text = str(time_control_str).strip()
    if text in {"-", "?", ""}:
        return 0, 0
    match = re.fullmatch(r"(\d+)\+(\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    if text.isdigit():
        return int(text), 0
    return 0, 0


def parse_clk_comment(comment: str) -> float | None:
    """Parse Lichess [%clk ...] comments into remaining seconds."""
    match = re.search(r"\[%clk\s+([^\]]+)\]", comment or "")
    if not match:
        return None
    text = match.group(1).strip()
    try:
        if ":" not in text:
            return float(text)
        parts = [float(part) for part in text.split(":")]
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return hours * 3600.0 + minutes * 60.0 + seconds
        if len(parts) == 2:
            minutes, seconds = parts
            return minutes * 60.0 + seconds
    except ValueError:
        return None
    return None


class _StreamInterrupted(Exception):
    """Raised when a Lichess stream interruption can be retried."""


def stream_pgn_games(url: str, skip_games: int = 0) -> Iterable[chess.pgn.Game]:
    """Stream and decompress PGN games without writing the decompressed file.

    `skip_games` supports coarse resume after reconnect: the caller restarts
    the compressed stream and fast-forwards past games already parsed.
    """
    try:
        with requests.get(url, stream=True, timeout=HTTP_TIMEOUT_SECONDS) as response:
            response.raise_for_status()
            response.raw.decode_content = False
            dctx = zstd.ZstdDecompressor(max_window_size=2**31)
            with dctx.stream_reader(response.raw) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
                consecutive_parse_errors = 0
                games_seen = 0
                while True:
                    try:
                        game = chess.pgn.read_game(text_stream)
                    except Exception as exc:  # PGN corruption should not kill the run.
                        exc_text = repr(exc)
                        if any(
                            marker in exc_text
                            for marker in ("Connection broken", "IncompleteRead", "BrokenPipe", "ConnectionReset")
                        ):
                            raise _StreamInterrupted(
                                f"Stream interrupted at game {games_seen:,} while reading {url}: {exc}"
                            ) from exc
                        consecutive_parse_errors += 1
                        if consecutive_parse_errors > 20:
                            raise RuntimeError(
                                f"Too many consecutive PGN parse failures while reading {url}: {exc}"
                            ) from exc
                        print(f"Warning: failed to parse a PGN game: {exc}")
                        continue
                    if game is None:
                        break
                    consecutive_parse_errors = 0
                    games_seen += 1
                    if games_seen <= skip_games:
                        continue
                    yield game
    except _StreamInterrupted:
        raise
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download or stream {url}: {exc}") from exc


def is_eligible_game(game: chess.pgn.Game, time_control_name: str) -> bool:
    """Check cheap header-level eligibility before replaying moves."""
    headers = game.headers
    event = headers.get("Event", "")
    if time_control_name not in event:
        return False
    if headers.get("Result") not in RESULTS:
        return False
    if safe_int(headers.get("WhiteElo")) is None or safe_int(headers.get("BlackElo")) is None:
        return False
    if getattr(game, "errors", None):
        return False
    return True


def safe_rate(numerator: float, denominator: int, default: float = NEUTRAL_SCORE_RATE) -> float:
    return float(numerator / denominator) if denominator else float(default)


def recent_score_rate(scores: deque[float]) -> float:
    return safe_rate(float(sum(scores)), len(scores))


def player_snapshot_features(history: PlayerHistory, prefix: str, side: str) -> dict[str, float | int]:
    """Return causal features based only on games already processed.

    Rates use a neutral Bayesian prior to avoid brittle 0%/100% values for
    players with very little same-month history.
    """
    if side == "white":
        side_count = history.prior_games_as_white
        side_wins = history.prior_wins_as_white
        side_key = "prior_win_rate_as_white"
    elif side == "black":
        side_count = history.prior_games_as_black
        side_wins = history.prior_wins_as_black
        side_key = "prior_win_rate_as_black"
    else:
        raise ValueError(f"Unexpected side: {side}")

    if USE_HISTORY_BAYESIAN_SMOOTHING:
        side_win_rate = (
            side_wins + HISTORY_BAYESIAN_PRIOR_RATE * HISTORY_BAYESIAN_VIRTUAL_GAMES
        ) / (side_count + HISTORY_BAYESIAN_VIRTUAL_GAMES)
        score_rate = (
            history.prior_score_sum_all + HISTORY_BAYESIAN_PRIOR_RATE * HISTORY_BAYESIAN_VIRTUAL_GAMES
        ) / (history.prior_games + HISTORY_BAYESIAN_VIRTUAL_GAMES)
        recent_rate_10 = (
            sum(history.recent_scores_10) + HISTORY_BAYESIAN_PRIOR_RATE * HISTORY_BAYESIAN_VIRTUAL_GAMES
        ) / (len(history.recent_scores_10) + HISTORY_BAYESIAN_VIRTUAL_GAMES)
        recent_rate_30 = (
            sum(history.recent_scores_30) + HISTORY_BAYESIAN_PRIOR_RATE * 30.0
        ) / (len(history.recent_scores_30) + 30.0)
        avg_opponent_elo = (
            history.prior_opponent_elo_sum + HISTORY_ELO_VIRTUAL_GAMES * HISTORY_ELO_PRIOR
        ) / (history.prior_games + HISTORY_ELO_VIRTUAL_GAMES)
        avg_elo_seen = (
            history.prior_elo_seen_sum + HISTORY_ELO_VIRTUAL_GAMES * HISTORY_ELO_PRIOR
        ) / (history.prior_games + HISTORY_ELO_VIRTUAL_GAMES)
    else:
        side_win_rate = safe_rate(side_wins, side_count)
        score_rate = safe_rate(history.prior_score_sum_all, history.prior_games)
        recent_rate_10 = recent_score_rate(history.recent_scores_10)
        recent_rate_30 = recent_score_rate(history.recent_scores_30)
        avg_opponent_elo = safe_rate(history.prior_opponent_elo_sum, history.prior_games, default=NEUTRAL_ELO)
        avg_elo_seen = safe_rate(history.prior_elo_seen_sum, history.prior_games, default=NEUTRAL_ELO)

    return {
        f"{prefix}_prior_games": history.prior_games,
        f"{prefix}_prior_score_rate_all": score_rate,
        f"{prefix}_{side_key}": side_win_rate,
        f"{prefix}_prior_avg_opponent_elo": avg_opponent_elo,
        f"{prefix}_prior_recent_score_rate_10": recent_rate_10,
        f"{prefix}_prior_recent_score_rate_30": recent_rate_30,
        f"{prefix}_prior_avg_elo_seen": avg_elo_seen,
    }


def player_history_features(
    histories: dict[str, PlayerHistory],
    white_player: str,
    black_player: str,
) -> dict[str, float | int]:
    """Build current-game history features before updating with current result."""
    white_history = histories[white_player]
    black_history = histories[black_player]
    features: dict[str, float | int] = {}
    features.update(player_snapshot_features(white_history, "white", "white"))
    features.update(player_snapshot_features(black_history, "black", "black"))
    features["prior_games_diff"] = features["white_prior_games"] - features["black_prior_games"]
    features["prior_score_rate_diff"] = (
        features["white_prior_score_rate_all"] - features["black_prior_score_rate_all"]
    )
    features["prior_recent_score_rate_10_diff"] = (
        features["white_prior_recent_score_rate_10"] - features["black_prior_recent_score_rate_10"]
    )
    features["prior_recent_score_rate_30_diff"] = (
        features["white_prior_recent_score_rate_30"] - features["black_prior_recent_score_rate_30"]
    )
    features["prior_avg_opponent_elo_diff"] = (
        features["white_prior_avg_opponent_elo"] - features["black_prior_avg_opponent_elo"]
    )
    features["prior_avg_elo_seen_diff"] = features["white_prior_avg_elo_seen"] - features["black_prior_avg_elo_seen"]
    features["prior_side_win_rate_diff"] = (
        features["white_prior_win_rate_as_white"] - features["black_prior_win_rate_as_black"]
    )
    return features


def result_scores(result: str) -> tuple[float, float]:
    if result == "1-0":
        return 1.0, 0.0
    if result == "0-1":
        return 0.0, 1.0
    if result == "1/2-1/2":
        return 0.5, 0.5
    raise ValueError(f"Unexpected result: {result}")


def update_single_player_history(
    history: PlayerHistory,
    score: float,
    own_elo: int,
    opponent_elo: int,
    played_white: bool,
) -> None:
    history.prior_games += 1
    history.prior_score_sum_all += score
    history.prior_opponent_elo_sum += opponent_elo
    history.prior_elo_seen_sum += own_elo
    history.recent_scores_10.append(score)
    history.recent_scores_30.append(score)
    if played_white:
        history.prior_games_as_white += 1
        history.prior_wins_as_white += int(score == 1.0)
    else:
        history.prior_games_as_black += 1
        history.prior_wins_as_black += int(score == 1.0)


def update_player_histories(histories: dict[str, PlayerHistory], record: dict[str, Any]) -> None:
    """Update histories after feature extraction for the current eligible game."""
    white_score, black_score = result_scores(record["result"])
    update_single_player_history(
        histories[record["white_player"]],
        score=white_score,
        own_elo=int(record["white_elo"]),
        opponent_elo=int(record["black_elo"]),
        played_white=True,
    )
    update_single_player_history(
        histories[record["black_player"]],
        score=black_score,
        own_elo=int(record["black_elo"]),
        opponent_elo=int(record["white_elo"]),
        played_white=False,
    )


def empty_move_behavior_features(prefix: str) -> dict[str, int]:
    return {
        f"{prefix}capture_count": 0,
        f"{prefix}check_count": 0,
        f"{prefix}castle_count_white": 0,
        f"{prefix}castle_count_black": 0,
        f"{prefix}queen_move_count": 0,
        f"{prefix}king_move_count": 0,
        f"{prefix}knight_move_count": 0,
        f"{prefix}bishop_move_count": 0,
        f"{prefix}rook_move_count": 0,
        f"{prefix}pawn_move_count": 0,
    }


def add_move_behavior(features: dict[str, int], prefix: str, board: chess.Board, move: chess.Move) -> None:
    """Update move-behavior counters using only the current legal move."""
    piece = board.piece_at(move.from_square)
    if piece is None:
        return
    if board.is_capture(move):
        features[f"{prefix}capture_count"] += 1
    if board.is_castling(move):
        color_name = "white" if piece.color == chess.WHITE else "black"
        features[f"{prefix}castle_count_{color_name}"] += 1

    piece_feature = {
        chess.QUEEN: "queen_move_count",
        chess.KING: "king_move_count",
        chess.KNIGHT: "knight_move_count",
        chess.BISHOP: "bishop_move_count",
        chess.ROOK: "rook_move_count",
        chess.PAWN: "pawn_move_count",
    }[piece.piece_type]
    features[f"{prefix}{piece_feature}"] += 1


def extract_clock_features(
    ply_clocks: list[float | None],
    plies: int,
    prefix: str,
    initial_time_seconds: int,
    increment_seconds: int,
) -> dict[str, float | int]:
    """Extract clock features using only clock comments through the given ply."""
    neutral_clock = float(initial_time_seconds)
    side_values = {"white": [], "black": []}
    missing = {"white": 0, "black": 0}
    used_total = {"white": 0.0, "black": 0.0}
    observed_moves = {"white": 0, "black": 0}
    previous_remaining = {"white": neutral_clock, "black": neutral_clock}

    for ply_index in range(plies):
        side = "white" if ply_index % 2 == 0 else "black"
        current = ply_clocks[ply_index] if ply_index < len(ply_clocks) else None
        if current is None:
            missing[side] += 1
            continue
        current = float(current)
        side_values[side].append(current)
        used_total[side] += max(previous_remaining[side] + float(increment_seconds) - current, 0.0)
        previous_remaining[side] = current
        observed_moves[side] += 1

    white_last = side_values["white"][-1] if side_values["white"] else neutral_clock
    black_last = side_values["black"][-1] if side_values["black"] else neutral_clock
    white_min = min(side_values["white"]) if side_values["white"] else neutral_clock
    black_min = min(side_values["black"]) if side_values["black"] else neutral_clock
    white_avg = safe_rate(used_total["white"], observed_moves["white"], default=0.0)
    black_avg = safe_rate(used_total["black"], observed_moves["black"], default=0.0)
    any_clock_available = int(bool(side_values["white"] or side_values["black"]))
    denom = float(initial_time_seconds) + 1.0
    white_time_used_ratio = used_total["white"] / denom
    black_time_used_ratio = used_total["black"] / denom

    return {
        f"{prefix}white_clock_last": white_last,
        f"{prefix}black_clock_last": black_last,
        f"{prefix}white_clock_used_total": used_total["white"],
        f"{prefix}black_clock_used_total": used_total["black"],
        f"{prefix}clock_diff_last": white_last - black_last,
        f"{prefix}clock_used_diff": used_total["white"] - used_total["black"],
        f"{prefix}white_avg_time_per_move": white_avg,
        f"{prefix}black_avg_time_per_move": black_avg,
        f"{prefix}avg_time_per_move_diff": white_avg - black_avg,
        f"{prefix}white_min_clock": white_min,
        f"{prefix}black_min_clock": black_min,
        f"{prefix}white_clock_missing_count": missing["white"],
        f"{prefix}black_clock_missing_count": missing["black"],
        f"{prefix}any_clock_available": any_clock_available,
        f"{prefix}white_time_panic": int(white_last < 15.0),
        f"{prefix}black_time_panic": int(black_last < 15.0),
        f"{prefix}white_time_used_ratio": white_time_used_ratio,
        f"{prefix}black_time_used_ratio": black_time_used_ratio,
        f"{prefix}time_ratio_diff": white_time_used_ratio - black_time_used_ratio,
    }


def extract_moves_and_boards(
    game: chess.pgn.Game,
    initial_time_seconds: int,
    increment_seconds: int,
) -> tuple[list[str], chess.Board, chess.Board, dict[str, int], dict[str, int], dict[str, float | int], dict[str, float | int]] | None:
    """Return move text, board snapshots, and behavior features at 3/10 moves."""
    board = game.board()
    move_tokens: list[str] = []
    ply_clocks: list[float | None] = []
    m3_behavior = empty_move_behavior_features("m3_")
    m10_behavior = empty_move_behavior_features("m10_")
    board_after_3: chess.Board | None = None
    board_after_10: chess.Board | None = None
    try:
        node = game
        for ply in range(1, 21):
            if not node.variations:
                break
            next_node = node.variations[0]
            move = next_node.move
            if move not in board.legal_moves:
                return None
            san = board.san(move)
            uci = move.uci()
            move_tokens.extend([san, uci])
            ply_clocks.append(parse_clk_comment(next_node.comment))
            if ply <= 6:
                add_move_behavior(m3_behavior, "m3_", board, move)
            if ply <= 20:
                add_move_behavior(m10_behavior, "m10_", board, move)
            board.push(move)
            if board.is_check():
                if ply <= 6:
                    m3_behavior["m3_check_count"] += 1
                if ply <= 20:
                    m10_behavior["m10_check_count"] += 1
            if ply == 6:
                board_after_3 = board.copy(stack=False)
            if ply == 20:
                board_after_10 = board.copy(stack=False)
                break
            node = next_node
    except Exception:
        return None
    if len(move_tokens) < 40 or board_after_3 is None or board_after_10 is None:
        return None
    clk3 = extract_clock_features(ply_clocks, 6, "clk3_", initial_time_seconds, increment_seconds)
    clk10 = extract_clock_features(ply_clocks, 20, "clk10_", initial_time_seconds, increment_seconds)
    return move_tokens, board_after_3, board_after_10, m3_behavior, m10_behavior, clk3, clk10


def extract_board_features(board: chess.Board, prefix: str) -> dict[str, int]:
    """Extract non-engine board features observable at the prediction point."""
    features: dict[str, int] = {}
    material: dict[chess.Color, int] = {chess.WHITE: 0, chess.BLACK: 0}

    for color, color_name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        for piece_type, piece_name in PIECE_NAMES.items():
            count = len(board.pieces(piece_type, color))
            features[f"{prefix}{color_name}_{piece_name}"] = count
            material[color] += count * PIECE_VALUES[piece_type]

    features[f"{prefix}material_white"] = material[chess.WHITE]
    features[f"{prefix}material_black"] = material[chess.BLACK]
    features[f"{prefix}material_diff"] = material[chess.WHITE] - material[chess.BLACK]
    features[f"{prefix}legal_moves_count"] = board.legal_moves.count()
    features[f"{prefix}is_check"] = int(board.is_check())
    features[f"{prefix}turn_white"] = int(board.turn == chess.WHITE)
    features[f"{prefix}white_can_castle_kingside"] = int(board.has_kingside_castling_rights(chess.WHITE))
    features[f"{prefix}white_can_castle_queenside"] = int(board.has_queenside_castling_rights(chess.WHITE))
    features[f"{prefix}black_can_castle_kingside"] = int(board.has_kingside_castling_rights(chess.BLACK))
    features[f"{prefix}black_can_castle_queenside"] = int(board.has_queenside_castling_rights(chess.BLACK))
    features[f"{prefix}fullmove_number"] = board.fullmove_number

    white_center = black_center = white_center_attacks = black_center_attacks = 0
    for square in CENTER_SQUARES:
        piece = board.piece_at(square)
        if piece:
            if piece.color == chess.WHITE:
                white_center += 1
            else:
                black_center += 1
        white_center_attacks += len(board.attackers(chess.WHITE, square))
        black_center_attacks += len(board.attackers(chess.BLACK, square))
    features[f"{prefix}white_center_occupancy"] = white_center
    features[f"{prefix}black_center_occupancy"] = black_center
    features[f"{prefix}center_occupancy_diff"] = white_center - black_center
    features[f"{prefix}white_center_attacks"] = white_center_attacks
    features[f"{prefix}black_center_attacks"] = black_center_attacks
    features[f"{prefix}center_attack_diff"] = white_center_attacks - black_center_attacks
    return features


def pst_score(board: chess.Board, color: chess.Color) -> int:
    """Piece-square table score from one side's perspective at the current board."""
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == color:
            index = square if color == chess.WHITE else chess.square_mirror(square)
            score += PST_MAP[piece.piece_type][index]
    return score


def pawn_structure_features(board: chess.Board, color: chess.Color) -> dict[str, int]:
    pawns = board.pieces(chess.PAWN, color)
    pawn_files = [chess.square_file(square) for square in pawns]
    pawn_file_counts = {file_index: pawn_files.count(file_index) for file_index in range(8)}
    doubled = sum(count - 1 for count in pawn_file_counts.values() if count > 1)
    isolated = 0
    passed = 0
    backward = 0

    for square in pawns:
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        adjacent_files = [file_index - 1, file_index + 1]
        has_adjacent_pawn = any(
            pawn_file_counts.get(adjacent_file, 0) > 0
            for adjacent_file in adjacent_files
            if 0 <= adjacent_file < 8
        )
        isolated += int(not has_adjacent_pawn)

        is_passed = True
        for enemy_file in [file_index - 1, file_index, file_index + 1]:
            if not 0 <= enemy_file < 8:
                continue
            for enemy_rank in range(8):
                in_front = enemy_rank > rank_index if color == chess.WHITE else enemy_rank < rank_index
                enemy_piece = board.piece_at(chess.square(enemy_file, enemy_rank))
                if in_front and enemy_piece == chess.Piece(chess.PAWN, not color):
                    is_passed = False
                    break
            if not is_passed:
                break
        passed += int(is_passed)

        is_backward = bool(has_adjacent_pawn)
        for adjacent_file in adjacent_files:
            if not 0 <= adjacent_file < 8:
                continue
            for adjacent_rank in range(8):
                adjacent_piece = board.piece_at(chess.square(adjacent_file, adjacent_rank))
                if adjacent_piece == chess.Piece(chess.PAWN, color):
                    behind_or_beside = adjacent_rank <= rank_index if color == chess.WHITE else adjacent_rank >= rank_index
                    if behind_or_beside:
                        is_backward = False
                        break
            if not is_backward:
                break
        backward += int(is_backward)

    pawn_islands = 0
    in_island = False
    for file_index in range(8):
        if pawn_file_counts[file_index] > 0:
            if not in_island:
                pawn_islands += 1
                in_island = True
        else:
            in_island = False

    return {
        "pawns_doubled": doubled,
        "pawns_isolated": isolated,
        "pawns_passed": passed,
        "pawns_backward": backward,
        "pawns_islands": pawn_islands,
    }


def king_safety_features(board: chess.Board, color: chess.Color) -> dict[str, int]:
    king_square = board.king(color)
    if king_square is None:
        return {"king_pawn_shield": 0, "king_attackers_near": 0, "king_open_files_near": 0}
    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)

    pawn_shield = 0
    shield_rank = king_rank + 1 if color == chess.WHITE else king_rank - 1
    if 0 <= shield_rank < 8:
        for file_index in [king_file - 1, king_file, king_file + 1]:
            if 0 <= file_index < 8:
                piece = board.piece_at(chess.square(file_index, shield_rank))
                pawn_shield += int(piece == chess.Piece(chess.PAWN, color))

    adjacent_squares = []
    for file_delta in [-1, 0, 1]:
        for rank_delta in [-1, 0, 1]:
            if file_delta == 0 and rank_delta == 0:
                continue
            target_file = king_file + file_delta
            target_rank = king_rank + rank_delta
            if 0 <= target_file < 8 and 0 <= target_rank < 8:
                adjacent_squares.append(chess.square(target_file, target_rank))
    enemy_attackers = set()
    for square in adjacent_squares:
        enemy_attackers.update(board.attackers(not color, square))

    open_files_near = 0
    for file_index in [king_file - 1, king_file, king_file + 1]:
        if not 0 <= file_index < 8:
            continue
        has_friendly_pawn = any(
            board.piece_at(chess.square(file_index, rank_index)) == chess.Piece(chess.PAWN, color)
            for rank_index in range(8)
        )
        open_files_near += int(not has_friendly_pawn)

    return {
        "king_pawn_shield": pawn_shield,
        "king_attackers_near": len(enemy_attackers),
        "king_open_files_near": open_files_near,
    }


def mobility_features(board: chess.Board, color: chess.Color) -> dict[str, int]:
    attacks_by_piece = {
        chess.KNIGHT: set(),
        chess.BISHOP: set(),
        chess.ROOK: set(),
        chess.QUEEN: set(),
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == color and piece.piece_type in attacks_by_piece:
            attacks_by_piece[piece.piece_type].update(board.attacks(square))
    total_attacks = set().union(*attacks_by_piece.values())
    return {
        "knight_mobility": len(attacks_by_piece[chess.KNIGHT]),
        "bishop_mobility": len(attacks_by_piece[chess.BISHOP]),
        "rook_mobility": len(attacks_by_piece[chess.ROOK]),
        "queen_mobility": len(attacks_by_piece[chess.QUEEN]),
        "total_attack_coverage": len(total_attacks),
    }


def development_features(board: chess.Board, color: chess.Color) -> dict[str, int]:
    starting_squares = (
        (chess.B1, chess.C1, chess.F1, chess.G1)
        if color == chess.WHITE
        else (chess.B8, chess.C8, chess.F8, chess.G8)
    )
    minor_developed = 0
    for square in starting_squares:
        piece = board.piece_at(square)
        if piece is None or piece.color != color or piece.piece_type not in {chess.KNIGHT, chess.BISHOP}:
            minor_developed += 1

    back_rank = 0 if color == chess.WHITE else 7
    back_rank_count = sum(
        1
        for file_index in range(8)
        if (piece := board.piece_at(chess.square(file_index, back_rank))) is not None and piece.color == color
    )
    return {"dev_minor_developed": minor_developed, "dev_back_rank_count": back_rank_count}


def color_enhanced_features(board: chess.Board, color: chess.Color, prefix: str) -> dict[str, int]:
    features = {f"{prefix}pst_score": pst_score(board, color)}
    for source in (
        pawn_structure_features(board, color),
        king_safety_features(board, color),
        mobility_features(board, color),
        development_features(board, color),
    ):
        for key, value in source.items():
            features[f"{prefix}{key}"] = value
    return features


def extract_enhanced_board_features(board: chess.Board, prefix: str) -> dict[str, int]:
    """Extract lightweight positional features from the current board only."""
    white = color_enhanced_features(board, chess.WHITE, f"{prefix}white_")
    black = color_enhanced_features(board, chess.BLACK, f"{prefix}black_")
    features = {**white, **black}
    features[f"{prefix}pst_diff"] = white[f"{prefix}white_pst_score"] - black[f"{prefix}black_pst_score"]
    features[f"{prefix}mobility_diff"] = (
        white[f"{prefix}white_total_attack_coverage"] - black[f"{prefix}black_total_attack_coverage"]
    )
    return features


def extract_game_record(game: chess.pgn.Game, game_index: int) -> dict[str, Any] | None:
    """Convert one eligible PGN game into a model-ready row."""
    headers = game.headers
    white_elo = safe_int(headers.get("WhiteElo"))
    black_elo = safe_int(headers.get("BlackElo"))
    result = headers.get("Result")
    if white_elo is None or black_elo is None or result not in RESULTS:
        return None

    initial_seconds, increment_seconds = parse_time_control(headers.get("TimeControl"))
    moves_and_boards = extract_moves_and_boards(game, initial_seconds, increment_seconds)
    if moves_and_boards is None:
        return None
    move_tokens, board_after_3, board_after_10, m3_behavior, m10_behavior, clk3_features, clk10_features = moves_and_boards
    record: dict[str, Any] = {
        "game_index": game_index,
        "white_player": headers.get("White", ""),
        "black_player": headers.get("Black", ""),
        "result": result,
        "event": headers.get("Event", ""),
        "time_control": headers.get("TimeControl", ""),
        "utc_date": headers.get("UTCDate", ""),
        "utc_time": headers.get("UTCTime", ""),
        "white_elo": white_elo,
        "black_elo": black_elo,
        "elo_diff": white_elo - black_elo,
        "mean_elo": (white_elo + black_elo) / 2.0,
        "initial_time_seconds": initial_seconds,
        "increment_seconds": increment_seconds,
        "log_initial_time_seconds": math.log1p(initial_seconds),
        "player_pair_text": f"white={headers.get('White', '')} black={headers.get('Black', '')}",
        "first_3_moves_text": " ".join(move_tokens[:12]),
        "first_10_moves_text": " ".join(move_tokens[:40]),
        "white_win": int(result == "1-0"),
    }
    record.update(extract_board_features(board_after_3, "m3_"))
    record.update(extract_enhanced_board_features(board_after_3, "m3_enh_"))
    record.update(m3_behavior)
    record.update(clk3_features)
    record.update(extract_board_features(board_after_10, "m10_"))
    record.update(extract_enhanced_board_features(board_after_10, "m10_enh_"))
    record.update(m10_behavior)
    record.update(clk10_features)
    return record


def build_dataset(config: Config, selected_month: str) -> tuple[pd.DataFrame, DatasetBuildStats]:
    """Stream games and return the first configured number of eligible rows."""
    url = build_lichess_url(selected_month)
    print(f"Streaming {url}")
    records: list[dict[str, Any]] = []
    player_histories: dict[str, PlayerHistory] = defaultdict(PlayerHistory)
    parsed_games = 0
    header_eligible = 0

    for attempt in range(MAX_STREAM_RETRIES):
        if attempt > 0:
            delay = STREAM_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            print(
                f"Retry {attempt}/{MAX_STREAM_RETRIES - 1}: reconnecting in {delay:.0f}s "
                f"(skipping first {parsed_games:,} parsed games)"
            )
            time.sleep(delay)
        try:
            for game in stream_pgn_games(url, skip_games=parsed_games):
                parsed_games += 1
                if not is_eligible_game(game, config.time_control):
                    continue
                header_eligible += 1
                record = extract_game_record(game, game_index=parsed_games)
                if record is None:
                    continue
                record.update(
                    player_history_features(
                        player_histories,
                        white_player=str(record["white_player"]),
                        black_player=str(record["black_player"]),
                    )
                )
                update_player_histories(player_histories, record)
                records.append(record)
                if len(records) % 5_000 == 0:
                    print(f"Collected {len(records):,} eligible games from {parsed_games:,} parsed games")
                if len(records) >= config.target_games:
                    break
            break
        except _StreamInterrupted as exc:
            print(f"Warning: stream interrupted after {len(records):,} eligible records: {exc}")
            if attempt == MAX_STREAM_RETRIES - 1:
                raise RuntimeError(
                    f"Stream failed after {MAX_STREAM_RETRIES} attempts. "
                    f"Collected {len(records):,}/{config.target_games:,} eligible games."
                ) from exc
            continue
        if len(records) >= config.target_games:
            break

    if not records:
        raise RuntimeError("No eligible games were collected; check the selected month/time-control.")

    print(f"Parsed games: {parsed_games:,}")
    print(f"Header-eligible games before move-length validation: {header_eligible:,}")
    print(f"Eligible games collected: {len(records):,}")
    if len(records) < config.target_games:
        print(f"Warning: requested {config.target_games:,} games but only found {len(records):,}.")
    stats = DatasetBuildStats(
        parsed_games=parsed_games,
        header_eligible_games=header_eligible,
        eligible_games=len(records),
    )
    return pd.DataFrame.from_records(records), stats


def split_train_validation(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological/order split: train first, validate last."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    split_index = int(len(df) * train_ratio)
    if split_index == 0 or split_index == len(df):
        raise ValueError("Dataset is too small for the configured train/validation split.")
    train_df = df.iloc[:split_index].copy()
    val_df = df.iloc[split_index:].copy()
    train_df["split"] = "train"
    val_df["split"] = "validation"
    return train_df, val_df


def numeric_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        ],
        remainder="drop",
    )


def hashing_vectorizer(hashing_features: int) -> HashingVectorizer:
    return HashingVectorizer(
        n_features=hashing_features,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\S+",
    )


def text_numeric_preprocessor(numeric_cols: list[str], text_cols: list[str], hashing_features: int) -> ColumnTransformer:
    transformers: list[tuple[str, Any, Any]] = [
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            ),
            numeric_cols,
        )
    ]
    for text_col in text_cols:
        transformers.append((f"text_{text_col}", hashing_vectorizer(hashing_features), text_col))
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0)


def build_classifier_pipeline(
    numeric_cols: list[str],
    text_cols: list[str],
    hashing_features: int,
    random_seed: int,
    c_value: float = 1.0,
) -> Pipeline:
    features = (
        numeric_preprocessor(numeric_cols)
        if not text_cols
        else text_numeric_preprocessor(numeric_cols, text_cols, hashing_features)
    )
    return Pipeline(
        steps=[
            ("features", features),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    max_iter=DEFAULT_LOGISTIC_MAX_ITER,
                    solver=DEFAULT_LOGISTIC_SOLVER,
                    random_state=random_seed,
                ),
            ),
        ]
    )


def build_numeric_classifier_pipeline(numeric_cols: list[str], random_seed: int) -> Pipeline:
    return build_classifier_pipeline(
        numeric_cols=numeric_cols,
        text_cols=[],
        hashing_features=DEFAULT_HASHING_FEATURES,
        random_seed=random_seed,
    )


def build_text_numeric_classifier_pipeline(
    numeric_cols: list[str],
    text_col: str,
    hashing_features: int,
    random_seed: int,
) -> Pipeline:
    return build_classifier_pipeline(
        numeric_cols=numeric_cols,
        text_cols=[text_col],
        hashing_features=hashing_features,
        random_seed=random_seed,
    )


def build_elo_regression_pipeline(numeric_cols: list[str], text_cols: str | list[str], hashing_features: int) -> Pipeline:
    normalized_text_cols = [text_cols] if isinstance(text_cols, str) else text_cols
    return Pipeline(
        steps=[
            ("features", text_numeric_preprocessor(numeric_cols, normalized_text_cols, hashing_features)),
            ("model", Ridge(alpha=10.0)),
        ]
    )


def clipped_probabilities(probabilities: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)


def probability_diagnostics(probabilities: np.ndarray | pd.Series) -> dict[str, float]:
    values = clipped_probabilities(probabilities)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
    }


def evaluate_probability_predictions(
    name: str,
    y_val: pd.Series,
    probabilities: np.ndarray | pd.Series,
    include_accuracy: bool,
) -> dict[str, float | None]:
    clipped = clipped_probabilities(probabilities)
    metrics: dict[str, float | None] = {
        "log_loss": float(log_loss(y_val, clipped, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_val, clipped)),
    }
    if include_accuracy:
        metrics["accuracy"] = float(accuracy_score(y_val, (clipped >= 0.5).astype(int)))
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_val, clipped))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def evaluate_classifier(name: str, model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, float | None]:
    """Evaluate a probability classifier, tolerating tiny one-class smoke tests."""
    probabilities = model.predict_proba(X_val)[:, 1]
    return evaluate_probability_predictions(name, y_val, probabilities, include_accuracy=True)


def evaluate_regression_predictions(name: str, y_val: pd.DataFrame, predictions: np.ndarray) -> dict[str, float]:
    white_true = y_val.iloc[:, 0].to_numpy()
    black_true = y_val.iloc[:, 1].to_numpy()
    white_pred = predictions[:, 0]
    black_pred = predictions[:, 1]
    metrics = {
        "white_elo_mae": float(mean_absolute_error(white_true, white_pred)),
        "black_elo_mae": float(mean_absolute_error(black_true, black_pred)),
        "white_elo_rmse": float(np.sqrt(mean_squared_error(white_true, white_pred))),
        "black_elo_rmse": float(np.sqrt(mean_squared_error(black_true, black_pred))),
        "white_elo_r2": float(r2_score(white_true, white_pred)),
        "black_elo_r2": float(r2_score(black_true, black_pred)),
    }
    return metrics


def evaluate_regressor(name: str, model: Pipeline, X_val: pd.DataFrame, y_val: pd.DataFrame) -> dict[str, float]:
    predictions = model.predict(X_val)
    return evaluate_regression_predictions(name, y_val, predictions)


def elo_expected_score_probability(df: pd.DataFrame) -> np.ndarray:
    return 1.0 / (1.0 + np.power(10.0, -((df["white_elo"] - df["black_elo"]) / 400.0)))


def elo_mean_baseline_predictions(train_df: pd.DataFrame, val_df: pd.DataFrame) -> np.ndarray:
    white_mean = float(train_df["white_elo"].mean())
    black_mean = float(train_df["black_elo"].mean())
    return np.column_stack(
        [
            np.full(len(val_df), white_mean, dtype=float),
            np.full(len(val_df), black_mean, dtype=float),
        ]
    )


def majority_class_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict[str, float | int]:
    train_positive_rate = float(train_df["white_win"].mean())
    validation_positive_rate = float(val_df["white_win"].mean())
    majority_class = int(train_positive_rate >= 0.5)
    majority_predictions = np.full(len(val_df), majority_class, dtype=int)
    return {
        "train_positive_rate": train_positive_rate,
        "validation_positive_rate": validation_positive_rate,
        "train_majority_class": majority_class,
        "majority_class_validation_accuracy": float(accuracy_score(val_df["white_win"], majority_predictions)),
    }


def require_two_classes(y_train: pd.Series, model_name: str) -> None:
    if y_train.nunique() < 2:
        raise ValueError(f"{model_name} needs both target classes in training; increase TARGET_GAMES.")


def model_feature_columns(
    df: pd.DataFrame,
    use_history: bool = True,
    use_clock: bool = False,
    use_enhanced_board: bool = False,
) -> dict[str, list[str]]:
    history_numeric = [col for col in HISTORY_FEATURE_COLUMNS if use_history and col in df.columns]
    before_numeric = [
        "white_elo",
        "black_elo",
        "elo_diff",
        "mean_elo",
        "initial_time_seconds",
        "increment_seconds",
        "log_initial_time_seconds",
    ] + history_numeric
    m3_board = sorted(
        col for col in df.columns if col.startswith("m3_") and (use_enhanced_board or not col.startswith("m3_enh_"))
    )
    m10_board = sorted(
        col for col in df.columns if col.startswith("m10_") and (use_enhanced_board or not col.startswith("m10_enh_"))
    )
    clk3 = sorted(col for col in df.columns if use_clock and col.startswith("clk3_"))
    clk10 = sorted(col for col in df.columns if use_clock and col.startswith("clk10_"))
    after3_numeric = before_numeric + m3_board + clk3
    after10_numeric = before_numeric + m10_board + clk10

    # Leakage prevention: Elo regression excludes WhiteElo/BlackElo and all
    # Elo-derived columns, using only time control plus board/move features.
    elo_after10_numeric = [
        "initial_time_seconds",
        "increment_seconds",
        "log_initial_time_seconds",
    ] + history_numeric + m10_board + clk10
    leaked_elo_features = sorted(ELO_MODEL_FORBIDDEN_FEATURES.intersection(elo_after10_numeric))
    if leaked_elo_features:
        raise ValueError(f"Elo model feature leakage detected: {leaked_elo_features}")
    return {
        "before_numeric": before_numeric,
        "after3_numeric": after3_numeric,
        "after10_numeric": after10_numeric,
        "elo_after10_numeric": elo_after10_numeric,
    }


def print_dataset_summary(df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, config: Config, month: str) -> None:
    print("\nDataset summary")
    print(f"Selected month: {month}")
    print(f"Selected time-control: {config.time_control}")
    print(f"Eligible games: {len(df):,}")
    print(f"Train games: {len(train_df):,}")
    print(f"Validation games: {len(val_df):,}")
    print("Result distribution:")
    print(df["result"].value_counts(dropna=False).to_string())
    print("WhiteElo summary:")
    print(df["white_elo"].describe().to_string())
    print("BlackElo summary:")
    print(df["black_elo"].describe().to_string())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def print_metrics_report(metrics: dict[str, Any]) -> None:
    """Print a compact, separated report for classification and regression metrics."""
    print("\nClassification metrics")
    classification_names = [
        ("white_win_before_game", metrics["models"]["white_win_before_game"]),
        ("white_win_after_3_moves", metrics["models"]["white_win_after_3_moves"]),
        ("white_win_after_10_moves", metrics["models"]["white_win_after_10_moves"]),
        ("elo_expected_score_baseline", metrics["baselines"]["elo_expected_score_baseline"]),
    ]
    for name, values in classification_names:
        print(
            f"{name}: "
            f"roc_auc={values.get('roc_auc')}, "
            f"log_loss={values['log_loss']:.6f}, "
            f"brier_score={values['brier_score']:.6f}"
            + (f", accuracy={values['accuracy']:.6f}" if "accuracy" in values else "")
        )

    majority = metrics["baselines"]["majority_class_baseline"]
    print(
        "majority_class_baseline: "
        f"train_positive_rate={majority['train_positive_rate']:.6f}, "
        f"validation_positive_rate={majority['validation_positive_rate']:.6f}, "
        f"train_majority_class={majority['train_majority_class']}, "
        f"validation_accuracy={majority['validation_accuracy']:.6f}"
    )

    print("\nRegression metrics")
    regression_names = [
        ("elo_after_10_moves", metrics["models"]["elo_after_10_moves"]),
        ("elo_mean_baseline", metrics["baselines"]["elo_mean_baseline"]),
    ]
    for name, values in regression_names:
        print(
            f"{name}: "
            f"white_mae={values['white_elo_mae']:.6f}, "
            f"white_rmse={values['white_elo_rmse']:.6f}, "
            f"white_r2={values['white_elo_r2']:.6f}, "
            f"black_mae={values['black_elo_mae']:.6f}, "
            f"black_rmse={values['black_elo_rmse']:.6f}, "
            f"black_r2={values['black_elo_r2']:.6f}"
        )

    if "probability_diagnostics" in metrics:
        print("\nProbability diagnostics")
        for name, values in metrics["probability_diagnostics"].items():
            print(
                f"{name}: "
                f"min={values['min']:.6f}, max={values['max']:.6f}, "
                f"mean={values['mean']:.6f}, std={values['std']:.6f}, "
                f"p05={values['p05']:.6f}, p50={values['p50']:.6f}, p95={values['p95']:.6f}"
            )


def classifier_text_columns(move_text_col: str | None, use_player_identity: bool) -> list[str]:
    text_cols: list[str] = []
    if move_text_col:
        text_cols.append(move_text_col)
    if use_player_identity:
        text_cols.append("player_pair_text")
    return text_cols


def classifier_row(
    model_name: str,
    config_name: str,
    use_history: bool,
    use_player_identity: bool,
    use_clock: bool,
    c_value: float,
    metrics: dict[str, float | None],
) -> dict[str, Any]:
    return {
        "task": "classification",
        "horizon/model_name": model_name,
        "config_name": config_name,
        "use_history": use_history,
        "use_player_identity": use_player_identity,
        "use_clock": use_clock,
        "C": c_value,
        "roc_auc": metrics.get("roc_auc"),
        "log_loss": metrics.get("log_loss"),
        "brier": metrics.get("brier_score"),
        "accuracy": metrics.get("accuracy"),
        "white_mae": None,
        "black_mae": None,
        "avg_mae": None,
        "white_rmse": None,
        "black_rmse": None,
        "white_r2": None,
        "black_r2": None,
    }


def regression_row(
    model_name: str,
    config_name: str,
    use_history: bool,
    use_player_identity: bool,
    use_clock: bool,
    metrics: dict[str, float],
) -> dict[str, Any]:
    avg_mae = (metrics["white_elo_mae"] + metrics["black_elo_mae"]) / 2.0
    return {
        "task": "regression",
        "horizon/model_name": model_name,
        "config_name": config_name,
        "use_history": use_history,
        "use_player_identity": use_player_identity,
        "use_clock": use_clock,
        "C": None,
        "roc_auc": None,
        "log_loss": None,
        "brier": None,
        "accuracy": None,
        "white_mae": metrics["white_elo_mae"],
        "black_mae": metrics["black_elo_mae"],
        "avg_mae": avg_mae,
        "white_rmse": metrics["white_elo_rmse"],
        "black_rmse": metrics["black_elo_rmse"],
        "white_r2": metrics["white_elo_r2"],
        "black_r2": metrics["black_elo_r2"],
    }


def select_best_configs(results_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for model_name in ["white_win_before", "white_win_after_3", "white_win_after_10"]:
        subset = results_df[results_df["horizon/model_name"] == model_name].copy()
        subset = subset.sort_values(
            by=["roc_auc", "log_loss", "brier"],
            ascending=[False, True, True],
            na_position="last",
        )
        best[model_name] = subset.iloc[0].where(pd.notna(subset.iloc[0]), None).to_dict()

    elo_subset = results_df[results_df["horizon/model_name"] == "elo_after_10"].copy()
    elo_subset["avg_rmse"] = (elo_subset["white_rmse"] + elo_subset["black_rmse"]) / 2.0
    elo_subset["avg_r2"] = (elo_subset["white_r2"] + elo_subset["black_r2"]) / 2.0
    elo_subset = elo_subset.sort_values(
        by=["avg_mae", "avg_rmse", "avg_r2"],
        ascending=[True, True, False],
        na_position="last",
    )
    best["elo_after_10"] = elo_subset.iloc[0].where(pd.notna(elo_subset.iloc[0]), None).to_dict()
    return best


def clock_availability_summary(df: pd.DataFrame) -> dict[str, float]:
    return {
        "first_3_moves_any_clock_rate": float(df["clk3_any_clock_available"].mean()),
        "first_10_moves_any_clock_rate": float(df["clk10_any_clock_available"].mean()),
    }


def replay_uci_text_to_board(moves_text: str, ply_limit: int) -> chess.Board:
    """Replay UCI tokens from the SAN+UCI move text up to a ply limit."""
    board = chess.Board()
    tokens = str(moves_text).split()
    uci_tokens = tokens[1::2]
    for uci in uci_tokens[:ply_limit]:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if move not in board.legal_moves:
            break
        board.push(move)
    return board


class StockfishEvaluator:
    """Small Stockfish wrapper with JSON FEN cache for optional heavy experiments."""

    def __init__(self, cache_path: Path, depth: int) -> None:
        self.cache_path = cache_path
        self.depth = depth
        self.cache: dict[str, dict[str, float]] = {}
        self.engine: chess.engine.SimpleEngine | None = None
        if cache_path.exists():
            try:
                self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"Warning: could not read Stockfish cache {cache_path}: {exc}")
                self.cache = {}
        for path in ("/opt/homebrew/bin/stockfish", "/usr/local/bin/stockfish", "stockfish"):
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(path)
                print(f"Using Stockfish at {path}")
                break
            except Exception:
                continue
        if self.engine is None:
            print("Warning: Stockfish unavailable; SF features will use neutral zeros.")

    def evaluate(self, board: chess.Board) -> tuple[float, float]:
        fen = board.fen()
        if fen in self.cache:
            cached = self.cache[fen]
            return float(cached.get("cp", 0.0)), float(cached.get("mate", 0.0))
        if self.engine is None:
            return 0.0, 0.0
        try:
            info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
            score = info["score"].pov(chess.WHITE)
            if score.is_mate():
                mate = float(score.mate() or 0)
                cp = 10000.0 if mate > 0 else -10000.0
            else:
                cp = float(score.score() or 0)
                mate = 0.0
        except Exception:
            cp, mate = 0.0, 0.0
        self.cache[fen] = {"cp": cp, "mate": mate}
        return cp, mate

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")

    def close(self) -> None:
        self.save()
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                pass


def add_stockfish_features(df: pd.DataFrame, cache_path: Path, depth: int) -> pd.DataFrame:
    """Add Stockfish features for boards after 3 and 10 moves."""
    evaluator = StockfishEvaluator(cache_path=cache_path, depth=depth)
    rows: list[dict[str, float]] = []
    try:
        for idx, row in df.iterrows():
            board3 = replay_uci_text_to_board(str(row["first_3_moves_text"]), 6)
            board10 = replay_uci_text_to_board(str(row["first_10_moves_text"]), 20)
            sf3_cp, sf3_mate = evaluator.evaluate(board3)
            sf10_cp, sf10_mate = evaluator.evaluate(board10)
            rows.append(
                {
                    "sf3_cp": sf3_cp,
                    "sf3_mate": sf3_mate,
                    "sf10_cp": sf10_cp,
                    "sf10_mate": sf10_mate,
                    "sf10_cp_diff": sf10_cp - sf3_cp,
                }
            )
            if (len(rows) % 1000) == 0:
                print(f"Stockfish evaluated/cached {len(rows):,}/{len(df):,} games")
                evaluator.save()
    finally:
        evaluator.close()
    sf_df = pd.DataFrame(rows, index=df.index)
    return pd.concat([df, sf_df], axis=1)


def ensure_enhanced_board_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute enhanced board features from stored move text when not already present."""
    if "m3_enh_pst_diff" in df.columns and "m10_enh_pst_diff" in df.columns:
        return df
    print("Computing lightweight enhanced board features from cached move text")
    rows: list[dict[str, int]] = []
    for idx, row in df.iterrows():
        board3 = replay_uci_text_to_board(str(row["first_3_moves_text"]), 6)
        board10 = replay_uci_text_to_board(str(row["first_10_moves_text"]), 20)
        features = {}
        features.update(extract_enhanced_board_features(board3, "m3_enh_"))
        features.update(extract_enhanced_board_features(board10, "m10_enh_"))
        rows.append(features)
        if (len(rows) % 5000) == 0:
            print(f"Enhanced features computed for {len(rows):,}/{len(df):,} games")
    return pd.concat([df, pd.DataFrame(rows, index=df.index)], axis=1)


def numeric_model_pipeline(numeric_cols: list[str], model: Any) -> Pipeline:
    return Pipeline(steps=[("features", numeric_preprocessor(numeric_cols)), ("model", model)])


def run_report_best_models(
    config: Config,
    selected_month: str,
    output_dir: Path,
    run_started_at: float,
    stockfish_depth: int,
    input_cache_csv: str | None,
    stockfish_cache_path: str | None,
) -> None:
    """Run the heavier report-selected models as an explicit optional experiment."""
    loaded_from_cache = bool(input_cache_csv)
    if input_cache_csv:
        print(f"Loading cached dataset from {input_cache_csv}")
        df = pd.read_csv(input_cache_csv).head(config.target_games).copy()
        dataset_build_stats = DatasetBuildStats(
            parsed_games=int(df["game_index"].max()) if "game_index" in df.columns else len(df),
            header_eligible_games=len(df),
            eligible_games=len(df),
        )
    else:
        df, dataset_build_stats = build_dataset(config, selected_month)
    df = ensure_enhanced_board_features(df)
    stockfish_cache = Path(stockfish_cache_path) if stockfish_cache_path else output_dir / "stockfish_cache.json"
    df = add_stockfish_features(df, cache_path=stockfish_cache, depth=stockfish_depth)
    train_df, val_df = split_train_validation(df, config.train_ratio)
    print_dataset_summary(df, train_df, val_df, config, selected_month)
    y_train = train_df["white_win"]
    y_val = val_df["white_win"]
    require_two_classes(y_train, "Report-selected classifiers")

    feature_cols = model_feature_columns(df, use_history=True, use_clock=True, use_enhanced_board=True)
    before_hist_cols = feature_cols["before_numeric"]
    after3_sf_cols = feature_cols["after3_numeric"] + ["sf3_cp", "sf3_mate"]
    after10_sf_cols = feature_cols["after10_numeric"] + ["sf3_cp", "sf3_mate", "sf10_cp", "sf10_mate", "sf10_cp_diff"]
    elo_sf_cols = feature_cols["elo_after10_numeric"] + ["sf3_cp", "sf3_mate", "sf10_cp", "sf10_mate", "sf10_cp_diff"]

    t1_model = build_classifier_pipeline(
        numeric_cols=before_hist_cols,
        text_cols=[],
        hashing_features=config.hashing_features,
        random_seed=config.random_seed,
        c_value=1.0,
    )
    t2_model = numeric_model_pipeline(
        after3_sf_cols,
        GradientBoostingClassifier(random_state=config.random_seed),
    )
    t3_model = numeric_model_pipeline(
        after10_sf_cols,
        HistGradientBoostingClassifier(random_state=config.random_seed),
    )
    t4_rf_sf_model = numeric_model_pipeline(
        elo_sf_cols,
        RandomForestRegressor(n_estimators=100, random_state=config.random_seed, n_jobs=-1),
    )
    t4_rf_no_sf_model = numeric_model_pipeline(
        feature_cols["elo_after10_numeric"],
        RandomForestRegressor(n_estimators=100, random_state=config.random_seed, n_jobs=-1),
    )

    t1_model.fit(train_df[before_hist_cols], y_train)
    t2_model.fit(train_df[after3_sf_cols], y_train)
    t3_model.fit(train_df[after10_sf_cols], y_train)
    t4_rf_sf_model.fit(train_df[elo_sf_cols], train_df[["white_elo", "black_elo"]])
    t4_rf_no_sf_model.fit(train_df[feature_cols["elo_after10_numeric"]], train_df[["white_elo", "black_elo"]])

    t4_sf_metrics = evaluate_regressor("t4_random_forest_stockfish", t4_rf_sf_model, val_df[elo_sf_cols], val_df[["white_elo", "black_elo"]])
    t4_no_sf_metrics = evaluate_regressor(
        "t4_random_forest_no_stockfish",
        t4_rf_no_sf_model,
        val_df[feature_cols["elo_after10_numeric"]],
        val_df[["white_elo", "black_elo"]],
    )
    metrics = {
        "run_config": {
            **asdict(config),
            "selected_month": selected_month,
            "stockfish_depth": stockfish_depth,
            "loaded_from_cache_csv": input_cache_csv,
            "stockfish_cache_path": str(stockfish_cache),
            "output_dir": str(output_dir),
        },
        "feature_notes": {
            "report_selected_heavy_models": True,
            "uses_stockfish": True,
            "uses_sklearn_tree_models": True,
            "not_default_production_path": True,
            "current_elo_excluded_from_elo_features": True,
            "dataset_loaded_from_cache": loaded_from_cache,
        },
        "dataset_summary": {
            "parsed_games": int(dataset_build_stats.parsed_games),
            "header_eligible_games": int(dataset_build_stats.header_eligible_games),
            "eligible_games": int(len(df)),
            "train_games": int(len(train_df)),
            "validation_games": int(len(val_df)),
            "train_positive_rate": float(train_df["white_win"].mean()),
            "validation_positive_rate": float(val_df["white_win"].mean()),
            "stockfish_cache_entries": len(json.loads(stockfish_cache.read_text(encoding="utf-8"))) if stockfish_cache.exists() else 0,
        },
        "models": {
            "t1_before_logreg_history": evaluate_classifier("t1_before_logreg_history", t1_model, val_df[before_hist_cols], y_val),
            "t2_after3_gradient_boosting_stockfish": evaluate_classifier(
                "t2_after3_gradient_boosting_stockfish",
                t2_model,
                val_df[after3_sf_cols],
                y_val,
            ),
            "t3_after10_hist_gradient_boosting_stockfish": evaluate_classifier(
                "t3_after10_hist_gradient_boosting_stockfish",
                t3_model,
                val_df[after10_sf_cols],
                y_val,
            ),
            "t4_elo_random_forest_stockfish": t4_sf_metrics,
            "t4_elo_random_forest_no_stockfish": t4_no_sf_metrics,
        },
    }
    metrics["run_config"]["runtime_seconds"] = float(time.perf_counter() - run_started_at)
    rows = []
    for name, values in metrics["models"].items():
        row = {"model_name": name}
        row.update(values)
        if "white_elo_mae" in values:
            row["avg_mae"] = (values["white_elo_mae"] + values["black_elo_mae"]) / 2.0
        rows.append(row)
    write_json(output_dir / "metrics.json", metrics)
    pd.DataFrame(rows).to_csv(output_dir / "experiment_results.csv", index=False)
    print_metrics_report(
        {
            "models": {
                "white_win_before_game": metrics["models"]["t1_before_logreg_history"],
                "white_win_after_3_moves": metrics["models"]["t2_after3_gradient_boosting_stockfish"],
                "white_win_after_10_moves": metrics["models"]["t3_after10_hist_gradient_boosting_stockfish"],
                "elo_after_10_moves": metrics["models"]["t4_elo_random_forest_stockfish"],
            },
            "baselines": {
                "elo_expected_score_baseline": evaluate_probability_predictions(
                    "elo_expected_score_baseline",
                    y_val,
                    elo_expected_score_probability(val_df),
                    include_accuracy=False,
                ),
                "majority_class_baseline": {
                    "train_positive_rate": float(train_df["white_win"].mean()),
                    "validation_positive_rate": float(val_df["white_win"].mean()),
                    "train_majority_class": int(float(train_df["white_win"].mean()) >= 0.5),
                    "validation_accuracy": float(
                        accuracy_score(
                            y_val,
                            np.full(len(y_val), int(float(train_df["white_win"].mean()) >= 0.5), dtype=int),
                        )
                    ),
                },
                "elo_mean_baseline": evaluate_regression_predictions(
                    "elo_mean_baseline",
                    val_df[["white_elo", "black_elo"]],
                    elo_mean_baseline_predictions(train_df, val_df),
                ),
            },
        }
    )
    print(f"\nWrote report-selected model metrics to {output_dir / 'metrics.json'}")
    print(f"Wrote report-selected model results to {output_dir / 'experiment_results.csv'}")


def load_boosting_classes() -> tuple[Any, Any, Any, Any]:
    """Load optional boosting dependencies only for --run-boosting-experiments."""
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "LightGBM/XGBoost experiment dependencies are missing. "
            "Install them with: pip install -r requirements-experiments.txt"
        ) from exc
    return LGBMClassifier, LGBMRegressor, XGBClassifier, XGBRegressor


def boosting_classifier_configs(random_seed: int) -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (
            "lightgbm",
            "conservative",
            {
                "n_estimators": 200,
                "learning_rate": 0.03,
                "num_leaves": 15,
                "max_depth": 4,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 5.0,
                "random_state": random_seed,
                "n_jobs": 1,
                "verbose": -1,
            },
        ),
        (
            "lightgbm",
            "balanced",
            {
                "n_estimators": 400,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": 6,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 2.0,
                "random_state": random_seed,
                "n_jobs": 1,
                "verbose": -1,
            },
        ),
        (
            "xgboost",
            "conservative",
            {
                "n_estimators": 200,
                "learning_rate": 0.03,
                "max_depth": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 5.0,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "random_state": random_seed,
                "n_jobs": 1,
                "verbosity": 0,
            },
        ),
        (
            "xgboost",
            "balanced",
            {
                "n_estimators": 400,
                "learning_rate": 0.05,
                "max_depth": 4,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 2.0,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "random_state": random_seed,
                "n_jobs": 1,
                "verbosity": 0,
            },
        ),
    ]


def boosting_regressor_configs(random_seed: int) -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (
            "lightgbm",
            "conservative",
            {
                "n_estimators": 200,
                "learning_rate": 0.03,
                "num_leaves": 15,
                "max_depth": 4,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 5.0,
                "random_state": random_seed,
                "n_jobs": 1,
                "verbose": -1,
            },
        ),
        (
            "lightgbm",
            "balanced",
            {
                "n_estimators": 400,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": 6,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 2.0,
                "random_state": random_seed,
                "n_jobs": 1,
                "verbose": -1,
            },
        ),
        (
            "xgboost",
            "conservative",
            {
                "n_estimators": 200,
                "learning_rate": 0.03,
                "max_depth": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 5.0,
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "random_state": random_seed,
                "n_jobs": 1,
                "verbosity": 0,
            },
        ),
        (
            "xgboost",
            "balanced",
            {
                "n_estimators": 400,
                "learning_rate": 0.05,
                "max_depth": 4,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 2.0,
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "random_state": random_seed,
                "n_jobs": 1,
                "verbosity": 0,
            },
        ),
    ]


def build_boosting_classifier(
    library: str,
    params: dict[str, Any],
    numeric_cols: list[str],
    classes: tuple[Any, Any, Any, Any],
) -> Pipeline:
    LGBMClassifier, _, XGBClassifier, _ = classes
    model = LGBMClassifier(**params) if library == "lightgbm" else XGBClassifier(**params)
    return numeric_model_pipeline(numeric_cols, model)


def build_boosting_regressor(
    library: str,
    params: dict[str, Any],
    numeric_cols: list[str],
    classes: tuple[Any, Any, Any, Any],
) -> MultiOutputRegressor:
    _, LGBMRegressor, _, XGBRegressor = classes
    model = LGBMRegressor(**params) if library == "lightgbm" else XGBRegressor(**params)
    return MultiOutputRegressor(numeric_model_pipeline(numeric_cols, model))


def run_boosting_experiments(
    config: Config,
    selected_month: str,
    output_dir: Path,
    run_started_at: float,
    input_cache_csv: str | None = None,
) -> None:
    """Run no-Stockfish LightGBM/XGBoost experiments against production baselines."""
    classes = load_boosting_classes()
    warnings.filterwarnings(
        "ignore",
        message=r"X does not have valid feature names, but LGBM.*",
        category=UserWarning,
    )
    if input_cache_csv and Path(input_cache_csv).exists():
        print(f"Loading cached dataset from {input_cache_csv}")
        df = pd.read_csv(input_cache_csv).head(config.target_games).copy()
        dataset_build_stats = DatasetBuildStats(
            parsed_games=len(df),
            header_eligible_games=len(df),
            eligible_games=len(df),
        )
    else:
        df, dataset_build_stats = build_dataset(config, selected_month)
    df = ensure_enhanced_board_features(df)
    train_df, val_df = split_train_validation(df, config.train_ratio)
    print_dataset_summary(df, train_df, val_df, config, selected_month)

    y_train = train_df["white_win"]
    y_val = val_df["white_win"]
    require_two_classes(y_train, "Boosting classifiers")
    y_train_elo = train_df[["white_elo", "black_elo"]]
    y_val_elo = val_df[["white_elo", "black_elo"]]

    base_no_clock = model_feature_columns(df, use_history=False, use_clock=False, use_enhanced_board=False)
    base_clock = model_feature_columns(df, use_history=False, use_clock=True, use_enhanced_board=False)
    enhanced_no_history = model_feature_columns(df, use_history=False, use_clock=False, use_enhanced_board=True)
    enhanced_history = model_feature_columns(df, use_history=True, use_clock=False, use_enhanced_board=True)
    enhanced_clock = model_feature_columns(df, use_history=False, use_clock=True, use_enhanced_board=True)
    elo_base = model_feature_columns(df, use_history=True, use_clock=False, use_enhanced_board=False)["elo_after10_numeric"]

    before_base = base_no_clock["before_numeric"]
    before_history = enhanced_history["before_numeric"]
    after3_base = base_no_clock["after3_numeric"]
    after3_enhanced = enhanced_no_history["after3_numeric"]
    after10_base_clock = base_clock["after10_numeric"]
    after10_enhanced_clock = enhanced_clock["after10_numeric"]
    elo_enhanced = enhanced_history["elo_after10_numeric"]

    after3_text_cols = ["first_3_moves_text", "player_pair_text"]
    after10_text_cols = ["first_10_moves_text", "player_pair_text"]
    elo_text_cols = ["first_10_moves_text", "player_pair_text"]
    feature_sets = {
        "before_base": before_base,
        "before_history": before_history,
        "after3_base": after3_base,
        "after3_enhanced": after3_enhanced,
        "after10_base_clock": after10_base_clock,
        "after10_enhanced_clock": after10_enhanced_clock,
        "elo_base": elo_base,
        "elo_enhanced": elo_enhanced,
    }

    rows: list[dict[str, Any]] = []

    print("\nFitting production baselines for comparison")
    production_before = build_classifier_pipeline(before_base, [], config.hashing_features, config.random_seed, c_value=1.0)
    production_after3 = build_classifier_pipeline(after3_base, after3_text_cols, config.hashing_features, config.random_seed, c_value=0.25)
    production_after10 = build_classifier_pipeline(after10_base_clock, after10_text_cols, config.hashing_features, config.random_seed, c_value=0.25)
    production_elo = build_elo_regression_pipeline(elo_base, elo_text_cols, config.hashing_features)
    production_before.fit(train_df[before_base], y_train)
    production_after3.fit(train_df[after3_base + after3_text_cols], y_train)
    production_after10.fit(train_df[after10_base_clock + after10_text_cols], y_train)
    production_elo.fit(train_df[elo_base + elo_text_cols], y_train_elo)

    baseline_rows = [
        classifier_row(
            "white_win_before",
            "production_logreg_C1.0",
            False,
            False,
            False,
            1.0,
            evaluate_classifier("production_before", production_before, val_df[before_base], y_val),
        ),
        classifier_row(
            "white_win_after_3",
            "production_logreg_identity_C0.25",
            False,
            True,
            False,
            0.25,
            evaluate_classifier("production_after3", production_after3, val_df[after3_base + after3_text_cols], y_val),
        ),
        classifier_row(
            "white_win_after_10",
            "production_logreg_identity_clock_C0.25",
            False,
            True,
            True,
            0.25,
            evaluate_classifier("production_after10", production_after10, val_df[after10_base_clock + after10_text_cols], y_val),
        ),
        regression_row(
            "elo_after_10",
            "production_ridge_history_identity",
            True,
            True,
            False,
            evaluate_regressor("production_elo", production_elo, val_df[elo_base + elo_text_cols], y_val_elo),
        ),
    ]
    for row in baseline_rows:
        row.update({"algorithm": "production", "params_json": "{}", "feature_set": row["config_name"]})
        rows.append(row)

    print("\nRunning LightGBM/XGBoost classifiers")
    for model_name, numeric_cols, feature_set_name, use_history, use_clock in [
        ("white_win_before", before_history, "before_history", True, False),
        ("white_win_after_3", after3_enhanced, "after3_enhanced", False, False),
        ("white_win_after_10", after10_enhanced_clock, "after10_enhanced_clock", False, True),
    ]:
        for library, profile, params in boosting_classifier_configs(config.random_seed):
            config_name = f"{library}_{profile}_{feature_set_name}"
            print(f"Fitting {model_name}: {config_name}")
            model = build_boosting_classifier(library, params, numeric_cols, classes)
            model.fit(train_df[numeric_cols], y_train)
            metrics = evaluate_classifier(config_name, model, val_df[numeric_cols], y_val)
            row = classifier_row(model_name, config_name, use_history, False, use_clock, None, metrics)
            row.update(
                {
                    "algorithm": library,
                    "params_json": json.dumps(params, sort_keys=True),
                    "feature_set": feature_set_name,
                }
            )
            rows.append(row)

    print("\nRunning LightGBM/XGBoost Elo regressors")
    for library, profile, params in boosting_regressor_configs(config.random_seed):
        config_name = f"{library}_{profile}_elo_enhanced_history"
        print(f"Fitting elo_after_10: {config_name}")
        model = build_boosting_regressor(library, params, elo_enhanced, classes)
        model.fit(train_df[elo_enhanced], y_train_elo)
        metrics = evaluate_regressor(config_name, model, val_df[elo_enhanced], y_val_elo)
        row = regression_row("elo_after_10", config_name, True, False, False, metrics)
        row.update(
            {
                "algorithm": library,
                "params_json": json.dumps(params, sort_keys=True),
                "feature_set": "elo_enhanced_history",
            }
        )
        rows.append(row)

    results_df = pd.DataFrame(rows)
    best_configs = select_best_configs(results_df)
    results_path = output_dir / "experiment_results.csv"
    best_path = output_dir / "best_config.json"
    results_df.to_csv(results_path, index=False)
    write_json(best_path, best_configs)

    metrics = {
        "run_config": {
            **asdict(config),
            "selected_month": selected_month,
            "output_dir": str(output_dir),
            "runtime_seconds": float(time.perf_counter() - run_started_at),
            "experiment_dependency_file": "requirements-experiments.txt",
        },
        "feature_notes": {
            "boosting_no_stockfish_experiment": True,
            "uses_lightgbm": True,
            "uses_xgboost": True,
            "uses_stockfish": False,
            "uses_deep_learning": False,
            "normal_production_path_changed": False,
            "current_elo_excluded_from_elo_features": True,
            "validation_rows_used_for_fitting": False,
        },
        "dataset_summary": {
            "parsed_games": int(dataset_build_stats.parsed_games),
            "header_eligible_games": int(dataset_build_stats.header_eligible_games),
            "eligible_games": int(len(df)),
            "train_games": int(len(train_df)),
            "validation_games": int(len(val_df)),
            "train_positive_rate": float(train_df["white_win"].mean()),
            "validation_positive_rate": float(val_df["white_win"].mean()),
            "result_distribution": df["result"].value_counts().to_dict(),
        },
        "feature_columns": feature_sets,
        "best_configs": best_configs,
        "all_results": json.loads(results_df.where(pd.notna(results_df), None).to_json(orient="records")),
    }
    write_json(output_dir / "metrics.json", metrics)
    report_lines = [
        "# Boosting No-Stockfish Experiment",
        "",
        "This experiment compares optional LightGBM/XGBoost candidates against the current production solution path.",
        "",
        "## Scope",
        "",
        "- Stockfish: not used.",
        "- Deep learning: not used.",
        "- LightGBM/XGBoost are optional experiment dependencies, not part of `requirements.txt`.",
        "- Current Elo and Elo-derived columns remain excluded from Elo regression features.",
        "",
        "## Best Configs",
        "",
    ]
    for key, value in best_configs.items():
        report_lines.append(f"- `{key}`: `{value.get('config_name')}`")
    report_lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- `{results_path}`",
            f"- `{best_path}`",
            f"- `{output_dir / 'metrics.json'}`",
        ]
    )
    (output_dir / "BOOSTING_NO_STOCKFISH_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nSaved boosting experiment results to {results_path}")
    print(f"Saved best configs to {best_path}")
    for key, value in best_configs.items():
        print(f"Best {key}: {value.get('config_name')}")


def run_experiments(config: Config, selected_month: str, output_dir: Path, run_started_at: float) -> None:
    """Run compact 10k model configuration experiments without changing production defaults."""
    df, dataset_build_stats = build_dataset(config, selected_month)
    train_df, val_df = split_train_validation(df, config.train_ratio)
    print_dataset_summary(df, train_df, val_df, config, selected_month)
    y_train = train_df["white_win"]
    y_val = val_df["white_win"]
    require_two_classes(y_train, "White win experiment classifiers")

    results: list[dict[str, Any]] = []
    classification_configs = [
        ("current_baseline_no_history", False, False),
        ("with_history", True, False),
        ("with_player_identity", False, True),
        ("with_history_and_player_identity", True, True),
    ]
    classifier_specs = [
        ("white_win_before", "before_numeric", None, [1.0]),
        ("white_win_after_3", "after3_numeric", "first_3_moves_text", [1.0, 0.5, 0.25]),
        ("white_win_after_10", "after10_numeric", "first_10_moves_text", [1.0, 0.5, 0.25]),
    ]

    for config_name, use_history, use_player_identity in classification_configs:
        feature_cols = model_feature_columns(df, use_history=use_history)
        for model_name, numeric_key, move_text_col, c_values in classifier_specs:
            text_cols = classifier_text_columns(move_text_col, use_player_identity)
            x_train_cols = feature_cols[numeric_key] + text_cols
            x_val_cols = feature_cols[numeric_key] + text_cols
            for c_value in c_values:
                model = build_classifier_pipeline(
                    numeric_cols=feature_cols[numeric_key],
                    text_cols=text_cols,
                    hashing_features=config.hashing_features,
                    random_seed=config.random_seed,
                    c_value=c_value,
                )
                model.fit(train_df[x_train_cols], y_train)
                metrics = evaluate_classifier(model_name, model, val_df[x_val_cols], y_val)
                results.append(
                    classifier_row(
                        model_name=model_name,
                        config_name=config_name,
                        use_history=use_history,
                        use_player_identity=use_player_identity,
                        use_clock=False,
                        c_value=c_value,
                        metrics=metrics,
                    )
                )
                print(f"Experiment {model_name} {config_name} C={c_value}: {metrics}")

    regression_configs = [
        ("no_history_no_identity", False, False),
        ("history_only", True, False),
        ("identity_only", False, True),
        ("history_and_identity", True, True),
    ]
    for config_name, use_history, use_player_identity in regression_configs:
        feature_cols = model_feature_columns(df, use_history=use_history)
        text_cols = ["first_10_moves_text"]
        if use_player_identity:
            text_cols.append("player_pair_text")
        x_train_cols = feature_cols["elo_after10_numeric"] + text_cols
        x_val_cols = feature_cols["elo_after10_numeric"] + text_cols
        model = build_elo_regression_pipeline(
            numeric_cols=feature_cols["elo_after10_numeric"],
            text_cols=text_cols,
            hashing_features=config.hashing_features,
        )
        model.fit(train_df[x_train_cols], train_df[["white_elo", "black_elo"]])
        metrics = evaluate_regressor("elo_after_10", model, val_df[x_val_cols], val_df[["white_elo", "black_elo"]])
        results.append(
            regression_row(
                model_name="elo_after_10",
                config_name=config_name,
                use_history=use_history,
                use_player_identity=use_player_identity,
                use_clock=False,
                metrics=metrics,
            )
        )
        print(f"Experiment elo_after_10 {config_name}: {metrics}")

    results_df = pd.DataFrame(results)
    best_config = {
        "run_config": {**asdict(config), "selected_month": selected_month},
        "feature_notes": {
            "causal_player_history_features_available": True,
            "history_features_computed_before_current_game_update": True,
            "player_identity_hashed_features_available": True,
        },
        "dataset_summary": {
            "parsed_games": int(dataset_build_stats.parsed_games),
            "header_eligible_games": int(dataset_build_stats.header_eligible_games),
            "eligible_games": int(len(df)),
            "train_games": int(len(train_df)),
            "validation_games": int(len(val_df)),
            "train_positive_rate": float(train_df["white_win"].mean()),
            "validation_positive_rate": float(val_df["white_win"].mean()),
        },
        "best_configs": select_best_configs(results_df),
    }
    best_config["run_config"]["runtime_seconds"] = float(time.perf_counter() - run_started_at)

    results_path = output_dir / "experiment_results.csv"
    best_path = output_dir / "best_config.json"
    results_df.to_csv(results_path, index=False)
    write_json(best_path, best_config)
    print(f"\nWrote experiment results to {results_path}")
    print(f"Wrote best config to {best_path}")


def run_clock_experiments(config: Config, selected_month: str, output_dir: Path, run_started_at: float) -> None:
    """Run narrow clock-feature experiments using the selected 10k best configs."""
    df, dataset_build_stats = build_dataset(config, selected_month)
    train_df, val_df = split_train_validation(df, config.train_ratio)
    print_dataset_summary(df, train_df, val_df, config, selected_month)
    y_train = train_df["white_win"]
    y_val = val_df["white_win"]
    require_two_classes(y_train, "White win clock experiment classifiers")

    results: list[dict[str, Any]] = []

    classifier_configs = [
        ("white_win_before", "before_numeric", None, "selected_before_no_clock", False, False, False, 1.0),
        ("white_win_after_3", "after3_numeric", "first_3_moves_text", "selected_after3_no_clock", False, True, False, 0.25),
        ("white_win_after_3", "after3_numeric", "first_3_moves_text", "selected_after3_with_clock", False, True, True, 0.25),
        ("white_win_after_10", "after10_numeric", "first_10_moves_text", "selected_after10_no_clock", False, False, False, 0.25),
        ("white_win_after_10", "after10_numeric", "first_10_moves_text", "selected_after10_with_clock", False, False, True, 0.25),
        ("white_win_after_10", "after10_numeric", "first_10_moves_text", "after10_identity_with_clock", False, True, True, 0.25),
    ]
    for model_name, numeric_key, move_text_col, config_name, use_history, use_player_identity, use_clock, c_value in classifier_configs:
        feature_cols = model_feature_columns(df, use_history=use_history, use_clock=use_clock)
        text_cols = classifier_text_columns(move_text_col, use_player_identity)
        x_train_cols = feature_cols[numeric_key] + text_cols
        x_val_cols = feature_cols[numeric_key] + text_cols
        model = build_classifier_pipeline(
            numeric_cols=feature_cols[numeric_key],
            text_cols=text_cols,
            hashing_features=config.hashing_features,
            random_seed=config.random_seed,
            c_value=c_value,
        )
        model.fit(train_df[x_train_cols], y_train)
        metrics = evaluate_classifier(model_name, model, val_df[x_val_cols], y_val)
        results.append(
            classifier_row(
                model_name=model_name,
                config_name=config_name,
                use_history=use_history,
                use_player_identity=use_player_identity,
                use_clock=use_clock,
                c_value=c_value,
                metrics=metrics,
            )
        )
        print(f"Clock experiment {model_name} {config_name}: {metrics}")

    regression_configs = [
        ("selected_elo_no_clock", True, True, False),
        ("selected_elo_with_clock", True, True, True),
    ]
    for config_name, use_history, use_player_identity, use_clock in regression_configs:
        feature_cols = model_feature_columns(df, use_history=use_history, use_clock=use_clock)
        text_cols = ["first_10_moves_text", "player_pair_text"] if use_player_identity else ["first_10_moves_text"]
        x_train_cols = feature_cols["elo_after10_numeric"] + text_cols
        x_val_cols = feature_cols["elo_after10_numeric"] + text_cols
        model = build_elo_regression_pipeline(
            numeric_cols=feature_cols["elo_after10_numeric"],
            text_cols=text_cols,
            hashing_features=config.hashing_features,
        )
        model.fit(train_df[x_train_cols], train_df[["white_elo", "black_elo"]])
        metrics = evaluate_regressor("elo_after_10", model, val_df[x_val_cols], val_df[["white_elo", "black_elo"]])
        results.append(
            regression_row(
                model_name="elo_after_10",
                config_name=config_name,
                use_history=use_history,
                use_player_identity=use_player_identity,
                use_clock=use_clock,
                metrics=metrics,
            )
        )
        print(f"Clock experiment elo_after_10 {config_name}: {metrics}")

    results_df = pd.DataFrame(results)
    best_config = {
        "run_config": {**asdict(config), "selected_month": selected_month},
        "feature_notes": {
            "clock_features_available": True,
            "clock_features_limited_to_allowed_plies": True,
            "causal_player_history_features_available": True,
            "history_features_computed_before_current_game_update": True,
            "player_identity_hashed_features_available": True,
        },
        "dataset_summary": {
            "parsed_games": int(dataset_build_stats.parsed_games),
            "header_eligible_games": int(dataset_build_stats.header_eligible_games),
            "eligible_games": int(len(df)),
            "train_games": int(len(train_df)),
            "validation_games": int(len(val_df)),
            "train_positive_rate": float(train_df["white_win"].mean()),
            "validation_positive_rate": float(val_df["white_win"].mean()),
            "clock_availability": clock_availability_summary(df),
        },
        "best_configs": select_best_configs(results_df),
    }
    best_config["run_config"]["runtime_seconds"] = float(time.perf_counter() - run_started_at)

    results_path = output_dir / "experiment_results.csv"
    best_path = output_dir / "best_config.json"
    results_df.to_csv(results_path, index=False)
    write_json(best_path, best_config)
    print(f"\nWrote clock experiment results to {results_path}")
    print(f"Wrote clock best config to {best_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Lichess Blitz prediction models.")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--candidate-months", default=",".join(DEFAULT_CANDIDATE_MONTHS))
    parser.add_argument("--selected-month", default=None, help="YYYY-MM; overrides reproducible random selection.")
    parser.add_argument("--time-control", default=DEFAULT_TIME_CONTROL)
    parser.add_argument("--target-games", type=int, default=DEFAULT_TARGET_GAMES)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hashing-features", type=int, default=DEFAULT_HASHING_FEATURES)
    parser.add_argument(
        "--model-profile",
        choices=["lightweight", "boosting", "report_best"],
        default="report_best",
        help=(
            "Normal training path profile. 'report_best' uses the best portable sklearn "
            "configuration selected from the 100k experiments without Stockfish; "
            "'boosting' keeps the optional LightGBM/XGBoost no-Stockfish profile."
        ),
    )
    parser.add_argument("--run-experiments", action="store_true", help="Run compact 10k config experiments.")
    parser.add_argument("--run-clock-experiments", action="store_true", help="Run compact 10k clock feature experiments.")
    parser.add_argument("--run-report-best-models", action="store_true", help="Run report-selected Stockfish/tree model experiment.")
    parser.add_argument("--run-boosting-experiments", action="store_true", help="Run optional LightGBM/XGBoost experiments without Stockfish.")
    parser.add_argument("--stockfish-depth", type=int, default=10, help="Stockfish search depth for --run-report-best-models.")
    parser.add_argument("--input-cache-csv", default=None, help="Optional cached dataset CSV/CSV.GZ for --run-report-best-models.")
    parser.add_argument("--stockfish-cache-path", default=None, help="Optional Stockfish JSON cache path for --run-report-best-models.")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    months = tuple(month.strip() for month in args.candidate_months.split(",") if month.strip())
    if not months:
        raise ValueError("candidate_months must contain at least one YYYY-MM value.")
    return Config(
        random_seed=args.random_seed,
        candidate_months=months,
        selected_month=args.selected_month,
        time_control=args.time_control,
        target_games=args.target_games,
        train_ratio=args.train_ratio,
        output_dir=args.output_dir,
        hashing_features=args.hashing_features,
    )


def main() -> None:
    run_started_at = time.perf_counter()
    args = parse_args()
    config = config_from_args(args)
    selected_month = select_month(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.run_experiments:
        run_experiments(config, selected_month, output_dir, run_started_at)
        return
    if args.run_clock_experiments:
        run_clock_experiments(config, selected_month, output_dir, run_started_at)
        return
    if args.run_report_best_models:
        run_report_best_models(
            config,
            selected_month,
            output_dir,
            run_started_at,
            args.stockfish_depth,
            args.input_cache_csv,
            args.stockfish_cache_path,
        )
        return
    if args.run_boosting_experiments:
        run_boosting_experiments(config, selected_month, output_dir, run_started_at, args.input_cache_csv)
        return

    df, dataset_build_stats = build_dataset(config, selected_month)
    if args.model_profile in {"boosting", "report_best"}:
        df = ensure_enhanced_board_features(df)
    train_df, val_df = split_train_validation(df, config.train_ratio)
    print_dataset_summary(df, train_df, val_df, config, selected_month)

    optional_classes: tuple[Any, Any, Any, Any] | None = None
    if args.model_profile == "boosting":
        optional_classes = load_boosting_classes()
        warnings.filterwarnings(
            "ignore",
            message=r"X does not have valid feature names, but LGBM.*",
            category=UserWarning,
        )
        before_feature_cols = model_feature_columns(df, use_history=False, use_clock=False, use_enhanced_board=False)["before_numeric"]
        after3_feature_cols = model_feature_columns(df, use_history=False, use_clock=False, use_enhanced_board=True)["after3_numeric"]
        after10_feature_cols = model_feature_columns(df, use_history=False, use_clock=True, use_enhanced_board=True)["after10_numeric"]
        elo_feature_cols = model_feature_columns(df, use_history=True, use_clock=False, use_enhanced_board=True)["elo_after10_numeric"]
        after3_text_cols: list[str] = []
        after10_text_cols: list[str] = []
        elo_text_cols: list[str] = []
        xgb_after3_params = dict(boosting_classifier_configs(config.random_seed)[2][2])
        xgb_after10_params = dict(boosting_classifier_configs(config.random_seed)[3][2])
        lgbm_elo_params = dict(boosting_regressor_configs(config.random_seed)[1][2])
    elif args.model_profile == "report_best":
        before_feature_cols = model_feature_columns(df, use_history=True, use_clock=False, use_enhanced_board=True)["before_numeric"]
        after3_feature_cols = model_feature_columns(df, use_history=False, use_clock=True, use_enhanced_board=True)["after3_numeric"]
        after10_feature_cols = model_feature_columns(df, use_history=False, use_clock=True, use_enhanced_board=True)["after10_numeric"]
        elo_feature_cols = model_feature_columns(df, use_history=True, use_clock=False, use_enhanced_board=True)["elo_after10_numeric"]
        after3_text_cols = []
        after10_text_cols = []
        elo_text_cols = []
    else:
        before_feature_cols = model_feature_columns(df, use_history=False, use_clock=False)["before_numeric"]
        after3_feature_cols = model_feature_columns(df, use_history=False, use_clock=False)["after3_numeric"]
        after10_feature_cols = model_feature_columns(df, use_history=False, use_clock=True)["after10_numeric"]
        elo_feature_cols = model_feature_columns(df, use_history=True, use_clock=False)["elo_after10_numeric"]
        after3_text_cols = ["first_3_moves_text", "player_pair_text"]
        after10_text_cols = ["first_10_moves_text", "player_pair_text"]
        elo_text_cols = ["first_10_moves_text", "player_pair_text"]

    print("\nFinal selected feature columns used")
    selected_feature_sets = {
        "white_win_before_numeric": before_feature_cols,
        "white_win_after_3_numeric": after3_feature_cols,
        "white_win_after_10_numeric": after10_feature_cols,
        "elo_after_10_numeric": elo_feature_cols,
        "white_win_after_3_text": after3_text_cols,
        "white_win_after_10_text": after10_text_cols,
        "elo_after_10_text": elo_text_cols,
    }
    for key, cols in selected_feature_sets.items():
        print(f"{key}: {len(cols)} columns")
        print(", ".join(cols))

    y_train = train_df["white_win"]
    y_val = val_df["white_win"]
    require_two_classes(y_train, "White win classifiers")

    before_model = build_classifier_pipeline(
        numeric_cols=before_feature_cols,
        text_cols=[],
        hashing_features=config.hashing_features,
        random_seed=config.random_seed,
        c_value=1.0,
    )
    if args.model_profile == "boosting":
        assert optional_classes is not None
        after3_model = build_boosting_classifier("xgboost", xgb_after3_params, after3_feature_cols, optional_classes)
        after10_model = build_boosting_classifier("xgboost", xgb_after10_params, after10_feature_cols, optional_classes)
        elo_model = build_boosting_regressor("lightgbm", lgbm_elo_params, elo_feature_cols, optional_classes)
    elif args.model_profile == "report_best":
        after3_model = build_classifier_pipeline(
            numeric_cols=after3_feature_cols,
            text_cols=[],
            hashing_features=config.hashing_features,
            random_seed=config.random_seed,
            c_value=0.5,
        )
        after10_model = numeric_model_pipeline(
            after10_feature_cols,
            HistGradientBoostingClassifier(random_state=config.random_seed),
        )
        elo_model = numeric_model_pipeline(
            elo_feature_cols,
            RandomForestRegressor(n_estimators=100, random_state=config.random_seed, n_jobs=-1),
        )
    else:
        after3_model = build_classifier_pipeline(
            numeric_cols=after3_feature_cols,
            text_cols=after3_text_cols,
            hashing_features=config.hashing_features,
            random_seed=config.random_seed,
            c_value=0.25,
        )
        after10_model = build_classifier_pipeline(
            numeric_cols=after10_feature_cols,
            text_cols=after10_text_cols,
            hashing_features=config.hashing_features,
            random_seed=config.random_seed,
            c_value=0.25,
        )
        elo_model = build_elo_regression_pipeline(
            numeric_cols=elo_feature_cols,
            text_cols=elo_text_cols,
            hashing_features=config.hashing_features,
        )

    # All preprocessing is fit on train_df through sklearn Pipelines. Validation
    # rows are only transformed/predicted after fitting, preventing leakage.
    before_model.fit(train_df[before_feature_cols], y_train)
    after3_model.fit(train_df[after3_feature_cols + after3_text_cols], y_train)
    after10_model.fit(train_df[after10_feature_cols + after10_text_cols], y_train)
    elo_model.fit(
        train_df[elo_feature_cols + elo_text_cols],
        train_df[["white_elo", "black_elo"]],
    )

    elo_baseline_probs = elo_expected_score_probability(val_df)
    elo_mean_predictions = elo_mean_baseline_predictions(train_df, val_df)
    majority_metrics = majority_class_baseline(train_df, val_df)

    if args.model_profile == "boosting":
        feature_notes = {
            "model_profile": "boosting",
            "selected_from_boosting_no_stockfish_100k": True,
            "white_win_before": "production_logreg_C1.0",
            "white_win_after_3": "xgboost_conservative_after3_enhanced",
            "white_win_after_10": "xgboost_balanced_after10_enhanced_clock",
            "elo_after_10": "lightgbm_balanced_elo_enhanced_history",
            "boosting_dependencies_in": "requirements-experiments.txt",
            "stockfish_or_deep_learning_used": False,
            "normal_lightweight_requirements_changed": False,
            "lightweight_enhanced_board_features_selected": True,
            "clock_features_used_for": "white_win_after_10_only",
            "history_features_used_for": "elo_regression_only",
            "identity_features_used_for": [],
            "causal_player_history_features": True,
            "history_features_computed_before_current_game_update": True,
            "clock_features_limited_to_allowed_plies": True,
            "current_elo_excluded_from_elo_features": True,
            "bayesian_history_smoothing": USE_HISTORY_BAYESIAN_SMOOTHING,
            "history_virtual_games": HISTORY_BAYESIAN_VIRTUAL_GAMES if USE_HISTORY_BAYESIAN_SMOOTHING else 0.0,
            "time_pressure_features": True,
            "stream_retry_resume": True,
        }
    elif args.model_profile == "report_best":
        feature_notes = {
            "model_profile": "report_best",
            "selected_from_experiment_outputs_and_report_best_100k": True,
            "t2_selected_from_experiment_outputs": "F2_LogReg_C0.5_after3_enhanced_clock_no_stockfish",
            "portable_best_without_stockfish": True,
            "overall_stockfish_best_kept_as_research_reference": True,
            "white_win_before": "logistic_regression_C1.0_with_causal_history",
            "white_win_after_3": "logistic_regression_C0.5_after3_enhanced_clock_no_stockfish",
            "white_win_after_10": "sklearn_hist_gradient_boosting_after10_enhanced_clock_no_stockfish",
            "elo_after_10": "sklearn_random_forest_elo_enhanced_history_no_stockfish",
            "stockfish_or_deep_learning_used": False,
            "heavy_dependencies_added": False,
            "lightweight_enhanced_board_features_selected": True,
            "clock_features_used_for": ["white_win_after_3", "white_win_after_10"],
            "history_features_used_for": ["white_win_before", "elo_regression"],
            "identity_features_used_for": [],
            "causal_player_history_features": True,
            "history_features_computed_before_current_game_update": True,
            "clock_features_limited_to_allowed_plies": True,
            "current_elo_excluded_from_elo_features": True,
            "bayesian_history_smoothing": USE_HISTORY_BAYESIAN_SMOOTHING,
            "history_virtual_games": HISTORY_BAYESIAN_VIRTUAL_GAMES if USE_HISTORY_BAYESIAN_SMOOTHING else 0.0,
            "time_pressure_features": True,
            "stream_retry_resume": True,
        }
    else:
        feature_notes = {
            "model_profile": "lightweight",
            "selected_configs_from_10k_experiments": True,
            "lightweight_enhanced_board_features_available": True,
            "lightweight_enhanced_board_features_selected": False,
            "enhanced_board_10k_verification": "not_selected_metric_regression",
            "white_win_before": "baseline_no_history_no_identity_no_clock_C1.0",
            "white_win_after_3": "player_identity_no_history_no_clock_C0.25",
            "white_win_after_10": "player_identity_no_history_clock_C0.25",
            "elo_after_10": "history_and_identity_no_clock",
            "clock_features_used_for": "white_win_after_10_only",
            "history_features_used_for": "elo_regression_only",
            "identity_features_used_for": ["white_win_after_3", "white_win_after_10", "elo_regression"],
            "causal_player_history_features": True,
            "history_features_computed_before_current_game_update": True,
            "clock_features_limited_to_allowed_plies": True,
            "player_identity_hashed_features": True,
            "stockfish_or_deep_learning_used": False,
            "heavy_dependencies_added": False,
            "bayesian_history_smoothing": USE_HISTORY_BAYESIAN_SMOOTHING,
            "history_virtual_games": HISTORY_BAYESIAN_VIRTUAL_GAMES if USE_HISTORY_BAYESIAN_SMOOTHING else 0.0,
            "time_pressure_features": True,
            "stream_retry_resume": True,
        }

    metrics = {
        "run_config": {**asdict(config), "selected_month": selected_month, "model_profile": args.model_profile},
        "feature_notes": feature_notes,
        "dataset_summary": {
            "parsed_games": int(dataset_build_stats.parsed_games),
            "header_eligible_games": int(dataset_build_stats.header_eligible_games),
            "eligible_games": int(len(df)),
            "train_games": int(len(train_df)),
            "validation_games": int(len(val_df)),
            "result_distribution": df["result"].value_counts().to_dict(),
            "train_positive_rate": majority_metrics["train_positive_rate"],
            "validation_positive_rate": majority_metrics["validation_positive_rate"],
        },
        "baselines": {
            "majority_class_baseline": {
                "train_majority_class": majority_metrics["train_majority_class"],
                "validation_accuracy": majority_metrics["majority_class_validation_accuracy"],
                "train_positive_rate": majority_metrics["train_positive_rate"],
                "validation_positive_rate": majority_metrics["validation_positive_rate"],
            },
            "elo_expected_score_baseline": evaluate_probability_predictions(
                "elo_expected_score_baseline",
                y_val,
                elo_baseline_probs,
                include_accuracy=False,
            ),
            "elo_mean_baseline": evaluate_regression_predictions(
                "elo_mean_baseline",
                val_df[["white_elo", "black_elo"]],
                elo_mean_predictions,
            ),
        },
        "models": {
            "white_win_before_game": evaluate_classifier(
                "white_win_before_game",
                before_model,
                val_df[before_feature_cols],
                y_val,
            ),
            "white_win_after_3_moves": evaluate_classifier(
                "white_win_after_3_moves",
                after3_model,
                val_df[after3_feature_cols + after3_text_cols],
                y_val,
            ),
            "white_win_after_10_moves": evaluate_classifier(
                "white_win_after_10_moves",
                after10_model,
                val_df[after10_feature_cols + after10_text_cols],
                y_val,
            ),
            "elo_after_10_moves": evaluate_regressor(
                "elo_after_10_moves",
                elo_model,
                val_df[elo_feature_cols + elo_text_cols],
                val_df[["white_elo", "black_elo"]],
            ),
        },
    }

    before_probs = before_model.predict_proba(val_df[before_feature_cols])[:, 1]
    after3_probs = after3_model.predict_proba(val_df[after3_feature_cols + after3_text_cols])[:, 1]
    after10_probs = after10_model.predict_proba(val_df[after10_feature_cols + after10_text_cols])[:, 1]
    elo_predictions = elo_model.predict(val_df[elo_feature_cols + elo_text_cols])
    metrics["probability_diagnostics"] = {
        "white_win_before": probability_diagnostics(before_probs),
        "white_win_after_3": probability_diagnostics(after3_probs),
        "white_win_after_10": probability_diagnostics(after10_probs),
        "elo_expected_score_baseline": probability_diagnostics(elo_baseline_probs),
    }

    predictions = val_df[
        [
            "game_index",
            "white_player",
            "black_player",
            "result",
            "white_win",
            "white_elo",
            "black_elo",
        ]
    ].copy()
    predictions = predictions.rename(columns={"white_win": "white_win_true"})
    predictions["p_white_win_elo_baseline"] = clipped_probabilities(elo_baseline_probs)
    predictions["p_white_win_before"] = before_probs
    predictions["p_white_win_after_3"] = after3_probs
    predictions["p_white_win_after_10"] = after10_probs
    predictions["white_elo_pred_after_10"] = elo_predictions[:, 0]
    predictions["black_elo_pred_after_10"] = elo_predictions[:, 1]
    predictions["split"] = "validation"

    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "validation_predictions.csv"
    metrics["run_config"]["runtime_seconds"] = float(time.perf_counter() - run_started_at)
    print_metrics_report(metrics)
    write_json(metrics_path, metrics)
    predictions.to_csv(predictions_path, index=False)
    print(f"\nWrote metrics to {metrics_path}")
    print(f"Wrote validation predictions to {predictions_path}")


if __name__ == "__main__":
    main()
