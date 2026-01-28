"""Project-wide configuration."""

from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()  # ensure environment variables in .env are available to the repo

# Project root directory
ROOT_DIR = Path(__file__).parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"

# API keys (populated from the environment via dotenv)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
