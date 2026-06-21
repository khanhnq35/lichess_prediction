"""Data loader module for loading Lichess games and splitting train/val.

Reuses the data streaming and preprocessing logic from solution.py.
Caches the dataset locally to avoid repeatedly downloading from Lichess.
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path to import solution.py
sys.path.append(str(Path(__file__).resolve().parent.parent))

import solution
from experiment.config import OUTPUT_DIR, DEFAULT_RANDOM_SEED, DEFAULT_MONTH

CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_dataset(target_games: int, month: str = DEFAULT_MONTH) -> pd.DataFrame:
    """Load games from Lichess or retrieve from local cache if available."""
    cache_file = CACHE_DIR / f"games_{month}_{target_games}.csv.gz"
    
    if cache_file.exists():
        print(f"Loading cached dataset from {cache_file}")
        # Read zipped CSV, parse dates if necessary, but keep most columns as is
        df = pd.read_csv(cache_file)
        return df
        
    print(f"Cache not found at {cache_file}. Streaming from Lichess...")
    
    # Create configuration matching solution.py's Config
    config = solution.Config(
        target_games=target_games,
        selected_month=month,
        random_seed=DEFAULT_RANDOM_SEED
    )
    
    df, stats = solution.build_dataset(config, month)
    
    # Save to local cache
    print(f"Caching dataset to {cache_file}...")
    df.to_csv(cache_file, index=False, compression="gzip")
    
    return df

def get_train_val_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset chronologically into train and validation sets."""
    return solution.split_train_validation(df, train_ratio)
