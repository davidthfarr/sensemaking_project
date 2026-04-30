"""
Core data model for the sensemaking pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np


@dataclass
class Post:
    post_id: str
    text: str = ""
    timestamp: Optional[datetime] = None
    user_id: Optional[str] = None
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    cluster_id: Optional[int] = None
    is_noise: bool = False
    stance: Optional[str] = None
