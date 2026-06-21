"""Stockfish evaluation module with local JSON caching.

Provides class StockfishEvaluator to evaluate positions at depth 10
and cache evaluations to stockfish_cache.json to speed up repeat runs.
"""

import os
import json
from pathlib import Path
import chess
import chess.engine

class StockfishEvaluator:
    """Evaluates chess positions using the Stockfish engine with local caching."""

    def __init__(self, cache_path: Path, depth: int = 10):
        self.cache_path = cache_path
        self.depth = depth
        self.cache = {}
        self.engine = None
        self._find_and_init_engine()
        self._load_cache()

    def _find_and_init_engine(self):
        """Find stockfish executable and initialize it."""
        possible_paths = [
            "/opt/homebrew/bin/stockfish",       # macOS Homebrew Apple Silicon
            "/usr/local/bin/stockfish",          # macOS Homebrew Intel
            "stockfish",                         # system path
        ]
        
        # Check if any path works
        for path in possible_paths:
            try:
                # If path contains '/', check if file exists
                if "/" in path and not os.path.exists(path):
                    continue
                # Try to initialize
                print(f"Attempting to start Stockfish at: {path}")
                self.engine = chess.engine.SimpleEngine.popen_uci(path)
                try:
                    self.engine.configure({"Threads": 4, "Hash": 256})
                    print("Configured Stockfish with 4 threads and 256MB Hash")
                except Exception as ce:
                    print(f"Warning: could not configure engine options: {ce}")
                print(f"Successfully started Stockfish from {path}")
                return
            except Exception as e:
                print(f"Could not start engine at {path}: {e}")
                pass
                
        print("WARNING: Stockfish engine could not be started. Evaluations will fall back to neutral values (0.0).")

    def _load_cache(self):
        """Load FEN evaluations from cache file."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                print(f"Loaded {len(self.cache)} evaluations from cache at {self.cache_path}")
            except Exception as e:
                print(f"Error loading Stockfish cache: {e}. Starting fresh.")
                self.cache = {}
        else:
            self.cache = {}

    def save_cache(self):
        """Save current cache to disk."""
        if not self.cache:
            return
        try:
            # Create parents if needed
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
            print(f"Saved {len(self.cache)} evaluations to cache at {self.cache_path}")
        except Exception as e:
            print(f"Error saving Stockfish cache: {e}")

    def evaluate(self, board: chess.Board) -> tuple[float, float]:
        """Return (cp_score, mate_score) from White's perspective.

        Checks cache first. If not found and engine is available, runs stockfish.
        Otherwise returns default (0.0, 0.0).
        """
        # Key cache by FEN
        fen = board.fen()
        if fen in self.cache:
            res = self.cache[fen]
            return float(res["cp"]), float(res["mate"])
            
        if self.engine is None:
            return 0.0, 0.0
            
        try:
            info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
            score = info["score"].pov(chess.WHITE)
            
            if score.is_mate():
                mate_plies = score.mate()
                cp_score = 10000.0 if mate_plies > 0 else -10000.0
                mate_score = float(mate_plies)
            else:
                cp_score = float(score.score())
                mate_score = 0.0
                
            # Update cache
            self.cache[fen] = {"cp": cp_score, "mate": mate_score}
            return cp_score, mate_score
            
        except Exception as e:
            # print(f"Error evaluating FEN {fen}: {e}")
            return 0.0, 0.0

    def close(self):
        """Clean up engine and save cache."""
        self.save_cache()
        if self.engine:
            try:
                self.engine.quit()
            except Exception:
                pass
            self.engine = None
