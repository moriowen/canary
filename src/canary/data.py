"""Dataset access and local data paths."""

import os
from pathlib import Path

from huggingface_hub import HfFileSystem

REPO = "nebius/SWE-agent-trajectories"

DATA = Path(os.environ.get("CANARY_DATA", "data"))
METADATA = DATA / "interim" / "metadata.parquet"
SAMPLE = DATA / "raw" / "sample.parquet"
RESULTS = Path(os.environ.get("CANARY_RESULTS", "results"))   # raw outputs, not in git
REPORTS = Path("reports")                                     # figures + tables, in git


def processed(scaffold):
    return DATA / "processed" / scaffold


def shards():
    """hf:// URLs of the dataset's parquet shards, in order."""
    fs = HfFileSystem()
    return [f"hf://{s}" for s in sorted(fs.glob(f"datasets/{REPO}/data/train-*.parquet"))]
