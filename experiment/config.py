"""Configuration parameters and paths for the experiment framework."""

from pathlib import Path

# Paths
EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parent
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

# Create output directories
for phase in [
    "phase1_baseline",
    "phase2_enhanced",
    "phase3_trees",
    "phase4_stockfish",
    "phase5_deep_learning",
    "phase6_ensemble",
    "final_100k",
    "plots",
]:
    (OUTPUT_DIR / phase).mkdir(parents=True, exist_ok=True)

# Default Settings
DEFAULT_RANDOM_SEED = 42
DEFAULT_TIME_CONTROL = "Blitz"
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_MONTH = "2023-11"  # Target month

# Stockfish Config
STOCKFISH_PATH = "stockfish"  # Will assume stockfish is in PATH
STOCKFISH_DEPTH = 10
STOCKFISH_CACHE_FILE = EXPERIMENT_DIR / "stockfish_cache.json"

# Models and features
HASHING_FEATURES = 2**15
