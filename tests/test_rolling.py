from datetime import datetime, timedelta

from sensemaking.data.schemas import Post
from sensemaking.windows.rolling import generate_rolling_windows


def make_post(post_id: str, ts: datetime) -> Post:
    return Post(
        post_id=post_id,
        text=f"post {post_id}",
        timestamp=ts,
    )


def test_single_window():
    t0 = datetime(2024, 1, 1)

    posts = [
        make_post("1", t0),
        make_post("2", t0 + timedelta(hours=1)),
    ]

    windows = list(
        generate_rolling_windows(
            posts,
            window_size=timedelta(days=1),
            step_size=timedelta(days=1),
        )
    )

    assert len(windows) == 1
    (_, window_posts) = windows[0]
    assert len(window_posts) == 2


def test_rolling_overlap():
    t0 = datetime(2024, 1, 1)

    posts = [
        make_post("1", t0),
        make_post("2", t0 + timedelta(days=1)),
        make_post("3", t0 + timedelta(days=2)),
    ]

    windows = list(
        generate_rolling_windows(
            posts,
            window_size=timedelta(days=2),
            step_size=timedelta(days=1),
        )
    )

    # Expected windows:
    # [day 0,2): posts 1,2
    # [day 1,3): posts 2,3
    # [day 2,4): posts 3
    assert len(windows) == 3

    assert [p.post_id for p in windows[0][1]] == ["1", "2"]
    assert [p.post_id for p in windows[1][1]] == ["2", "3"]
    assert [p.post_id for p in windows[2][1]] == ["3"]


def test_empty_posts():
    windows = list(
        generate_rolling_windows(
            [],
            window_size=timedelta(days=1),
            step_size=timedelta(days=1),
        )
    )

    assert windows == []


def test_posts_on_boundary():
    t0 = datetime(2024, 1, 1)

    posts = [
        make_post("1", t0),
        make_post("2", t0 + timedelta(days=1)),
    ]

    windows = list(
        generate_rolling_windows(
            posts,
            window_size=timedelta(days=1),
            step_size=timedelta(days=1),
        )
    )

    # First window includes post 1 only
    assert [p.post_id for p in windows[0][1]] == ["1"]
