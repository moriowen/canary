"""Dataset access and local data paths."""

from pathlib import Path

from huggingface_hub import HfFileSystem

REPO = "nebius/SWE-agent-trajectories"

DATA = Path("data")
METADATA = DATA / "interim" / "metadata.parquet"
SAMPLE = DATA / "raw" / "sample.parquet"


def shards():
    """hf:// URLs of the dataset's parquet shards, in order."""
    fs = HfFileSystem()
    return [f"hf://{s}" for s in sorted(fs.glob(f"datasets/{REPO}/data/train-*.parquet"))]
