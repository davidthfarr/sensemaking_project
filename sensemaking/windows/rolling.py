"""
Rolling time window utilities.

This module is responsible ONLY for:
- Partitioning posts into rolling temporal windows

No embeddings.
No stance.
No clustering.
"""

from typing import Iterable, Iterator, List, Tuple
from datetime import datetime, timedelta

from sensemaking.data.schemas import Post


TimeWindow = Tuple[datetime, datetime]


def generate_rolling_windows(
    posts: Iterable[Post],
    window_size: timedelta,
    step_size: timedelta,
) -> Iterator[Tuple[TimeWindow, List[Post]]]:
    """
    Generate rolling time windows over a collection of posts.

    Parameters
    ----------
    posts : Iterable[Post]
        Collection of Post objects.
    window_size : timedelta
        Length of each rolling window.
    step_size : timedelta
        Step size between successive windows.

    Yields
    ------
    ((start_time, end_time), posts_in_window)
        A tuple containing the window interval and the list of posts
        whose timestamps fall within that interval.
    """
    posts = sorted(posts, key=lambda p: p.timestamp)

    if not posts:
        return

    start_time = posts[0].timestamp
    end_time = posts[-1].timestamp

    current_start = start_time

    while current_start <= end_time:
        current_end = current_start + window_size

        window_posts = [
            p for p in posts
            if current_start <= p.timestamp < current_end
        ]

        yield (current_start, current_end), window_posts

        current_start += step_size
