import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from sensemaking.data.schemas import Post
from sensemaking.stance.posthoc_gpt import PosthocGPTStanceClassifier, STANCE_LABELS


def make_post(post_id, text="sample text"):
    return Post(
        post_id=post_id,
        text=text,
        timestamp=datetime.now(),
    )


def make_mock_response(labels: list) -> MagicMock:
    content = json.dumps({"stances": labels})
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture
def classifier():
    with patch("sensemaking.stance.posthoc_gpt.OpenAI"):
        c = PosthocGPTStanceClassifier(model="gpt-4o-mini", batch_size=5)
    return c


# ------------------------------------------------------------------
# _parse_labels
# ------------------------------------------------------------------

def test_parse_labels_valid(classifier):
    content = json.dumps({"stances": ["support", "oppose", "neutral"]})
    labels = classifier._parse_labels(content, 3)
    assert labels == ["support", "oppose", "neutral"]


def test_parse_labels_bare_list(classifier):
    content = json.dumps(["neutral", "support"])
    labels = classifier._parse_labels(content, 2)
    assert labels == ["neutral", "support"]


def test_parse_labels_wrong_count_returns_none(classifier):
    content = json.dumps({"stances": ["support", "oppose"]})
    assert classifier._parse_labels(content, 3) is None


def test_parse_labels_invalid_label_returns_none(classifier):
    content = json.dumps({"stances": ["support", "UNKNOWN"]})
    assert classifier._parse_labels(content, 2) is None


def test_parse_labels_bad_json_returns_none(classifier):
    assert classifier._parse_labels("not json at all", 2) is None


# ------------------------------------------------------------------
# classify_posts
# ------------------------------------------------------------------

def test_classify_posts_attaches_stance():
    posts = [make_post(str(i)) for i in range(3)]
    theme = "Government forces are targeting civilians"

    mock_response = make_mock_response(["support", "oppose", "neutral"])

    with patch("sensemaking.stance.posthoc_gpt.OpenAI") as MockOpenAI:
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value = mock_response

        classifier = PosthocGPTStanceClassifier(batch_size=10)
        result = classifier.classify_posts(posts, theme)

    assert [p.stance for p in result] == ["support", "oppose", "neutral"]


def test_classify_posts_batches_correctly():
    posts = [make_post(str(i)) for i in range(7)]
    theme = "Ceasefire negotiations are failing"

    batch1 = make_mock_response(["support"] * 5)
    batch2 = make_mock_response(["neutral"] * 2)

    with patch("sensemaking.stance.posthoc_gpt.OpenAI") as MockOpenAI:
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.side_effect = [batch1, batch2]

        classifier = PosthocGPTStanceClassifier(batch_size=5)
        result = classifier.classify_posts(posts, theme)

    assert len(result) == 7
    assert all(p.stance in STANCE_LABELS for p in result)
    assert mock_client.chat.completions.create.call_count == 2


def test_classify_posts_fallback_on_bad_parse():
    posts = [make_post("1", "some text")]
    theme = "Foreign intervention is escalating the conflict"

    bad_response = MagicMock()
    bad_response.choices[0].message.content = '{"stances": []}'  # wrong count

    good_response = make_mock_response(["oppose"])

    with patch("sensemaking.stance.posthoc_gpt.OpenAI") as MockOpenAI:
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.side_effect = [
            bad_response, bad_response, bad_response,  # batch retries
            good_response,                              # single fallback
        ]

        classifier = PosthocGPTStanceClassifier(batch_size=5, max_retries=3)
        result = classifier.classify_posts(posts, theme)

    assert result[0].stance == "oppose"


def test_callable_interface():
    posts = [make_post("1")]
    theme = "Peace talks are imminent"

    mock_response = make_mock_response(["neutral"])

    with patch("sensemaking.stance.posthoc_gpt.OpenAI") as MockOpenAI:
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value = mock_response

        classifier = PosthocGPTStanceClassifier()
        result = classifier(posts, theme)

    assert result[0].stance == "neutral"
