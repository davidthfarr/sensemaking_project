import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sensemaking.data.schemas import Post
from sensemaking.themes.stationary_labeler import StationaryThemeLabeler, ThemeStore


def make_post(post_id, text="sample text", embedding=None):
    return Post(
        post_id=post_id,
        text=text,
        timestamp=datetime.now(),
        embedding=embedding,
        cluster_id=0,
    )


def make_mock_response(theme: str) -> MagicMock:
    msg = MagicMock()
    msg.content = theme
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# ThemeStore
# ---------------------------------------------------------------------------

def test_theme_store_is_new():
    store = ThemeStore()
    assert store.is_new(0)
    store.set(0, "Some narrative theme")
    assert not store.is_new(0)


def test_theme_store_get_returns_none_for_missing():
    store = ThemeStore()
    assert store.get(99) is None


def test_theme_store_roundtrip_dict():
    store = ThemeStore({1: "Theme A", 2: "Theme B"})
    restored = ThemeStore.from_dict(store.to_dict())
    assert restored.get(1) == "Theme A"
    assert restored.get(2) == "Theme B"


def test_theme_store_save_and_load():
    store = ThemeStore({0: "Ceasefire is being violated", 1: "Aid convoys are blocked"})
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "themes.json"
        store.save(path)
        loaded = ThemeStore.load(path)
    assert loaded.get(0) == "Ceasefire is being violated"
    assert loaded.get(1) == "Aid convoys are blocked"


def test_theme_store_save_creates_parent_dirs():
    store = ThemeStore({5: "Theme"})
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nested" / "dir" / "themes.json"
        store.save(path)
        assert path.exists()


def test_theme_store_len():
    store = ThemeStore({0: "A", 1: "B", 2: "C"})
    assert len(store) == 3


# ---------------------------------------------------------------------------
# StationaryThemeLabeler — representative selection
# ---------------------------------------------------------------------------

@pytest.fixture
def labeler():
    with patch("sensemaking.themes.stationary_labeler.OpenAI"):
        l = StationaryThemeLabeler(n_representative=3)
    return l


def test_select_representative_respects_n(labeler):
    posts = [make_post(str(i), embedding=np.random.rand(8)) for i in range(10)]
    selected = labeler._select_representative(posts)
    assert len(selected) == 3


def test_select_representative_fewer_than_n(labeler):
    posts = [make_post(str(i), embedding=np.random.rand(8)) for i in range(2)]
    selected = labeler._select_representative(posts)
    assert len(selected) == 2


def test_select_representative_no_embeddings_fallback(labeler):
    posts = [make_post(str(i)) for i in range(5)]
    selected = labeler._select_representative(posts)
    assert len(selected) == 3
    assert selected == posts[:3]


def test_select_representative_prefers_centroid_proximity():
    with patch("sensemaking.themes.stationary_labeler.OpenAI"):
        labeler = StationaryThemeLabeler(n_representative=2)

    # post "A" is near [1,0,0], post "B" is far, post "C" is near [1,0,0]
    centroid_dir = np.array([1.0, 0.0, 0.0])
    posts = [
        make_post("A", embedding=centroid_dir + np.array([0.0, 0.01, 0.0])),
        make_post("B", embedding=np.array([0.0, 0.0, 1.0])),  # orthogonal — far
        make_post("C", embedding=centroid_dir + np.array([0.0, -0.01, 0.0])),
    ]
    selected_ids = {p.post_id for p in labeler._select_representative(posts)}
    assert "A" in selected_ids
    assert "C" in selected_ids
    assert "B" not in selected_ids


# ---------------------------------------------------------------------------
# StationaryThemeLabeler — label_cluster
# ---------------------------------------------------------------------------

def test_label_cluster_calls_api_and_returns_theme():
    theme = "Foreign mercenaries are fighting alongside government troops"
    mock_response = make_mock_response(theme)

    with patch("sensemaking.themes.stationary_labeler.OpenAI") as MockOpenAI:
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value = mock_response

        labeler = StationaryThemeLabeler(n_representative=5)
        posts = [make_post(str(i), f"post text {i}", np.ones(4)) for i in range(5)]
        result = labeler.label_cluster(posts)

    assert result == theme
    assert mock_client.chat.completions.create.call_count == 1


def test_label_cluster_empty_posts_returns_fallback():
    with patch("sensemaking.themes.stationary_labeler.OpenAI"):
        labeler = StationaryThemeLabeler()
    assert labeler.label_cluster([]) == "Unknown narrative"


# ---------------------------------------------------------------------------
# assign_new_themes
# ---------------------------------------------------------------------------

def test_assign_new_themes_only_labels_new_clusters():
    theme_a = "Cluster zero theme"
    theme_b = "Cluster one theme"

    with patch("sensemaking.themes.stationary_labeler.OpenAI") as MockOpenAI:
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.side_effect = [
            make_mock_response(theme_b),  # only global_id=1 is new
        ]

        store = ThemeStore({0: theme_a})  # global_id=0 already labelled
        labeler = StationaryThemeLabeler()

        posts = [
            make_post("1", embedding=np.ones(4)),
            make_post("2", embedding=np.ones(4)),
        ]
        posts[0].cluster_id = 0
        posts[1].cluster_id = 1

        global_ids = {0: 0, 1: 1}
        updated = labeler.assign_new_themes(posts, global_ids, store)

    assert updated.get(0) == theme_a  # unchanged
    assert updated.get(1) == theme_b  # newly assigned
    assert mock_client.chat.completions.create.call_count == 1


def test_assign_new_themes_skips_noise_posts():
    with patch("sensemaking.themes.stationary_labeler.OpenAI") as MockOpenAI:
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value = make_mock_response("Theme")

        store = ThemeStore()
        labeler = StationaryThemeLabeler()

        noise_post = make_post("noise")
        noise_post.cluster_id = None  # noise

        cluster_post = make_post("real", embedding=np.ones(4))
        cluster_post.cluster_id = 0

        labeler.assign_new_themes([noise_post, cluster_post], {0: 0}, store)

    # API should only have seen the non-noise post
    call_args = mock_client.chat.completions.create.call_args
    user_message = call_args.kwargs["messages"][1]["content"]
    assert "noise" not in user_message or "real" in user_message
