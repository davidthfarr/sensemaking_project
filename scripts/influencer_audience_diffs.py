from pathlib import Path
import pandas as pd
from datetime import timedelta

from sensemaking.clustering.hdbscan import HDBSCANClusterer
from sensemaking.data.schemas import Post

from scripts_environment_wrapper import environment

# -------------------------
# Configuration
# -------------------------
PROCESSED_PATH = Path(environment.PROCESSED_FILE_PATH())
OUTPUT_DIR = Path(environment.EVALUATED_DIR())