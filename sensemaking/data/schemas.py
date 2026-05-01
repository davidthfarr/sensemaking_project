from datetime import datetime

class Post:
    def __init__(self, post_id: str | int, user_id: str | int, timestamp: datetime, text: str, reply_parent_id = None, reply_parent_author = None, reply_root_id = None, reply_root_author = None, sample_type = "", embedding = None, stance = None, is_noise=False):
        if not isinstance(post_id, str) and not isinstance(post_id, int):
            raise TypeError(f"Expected 'post_id' to be str or int, got {type(post_id).__name__}")
        if not isinstance(user_id, int) and not isinstance(user_id, str):
            raise TypeError(f"Expected 'user_id' to be str or int, got {type(user_id).__name__}")
        if not isinstance(timestamp, str) and not isinstance(timestamp, datetime):
            raise TypeError(f"expected 'timestamp' to be a str or timestamp, got {type(timestamp).__name__}")
        if not isinstance(text, str):
            raise TypeError(f"expected 'text' to be str, got {type(text).__name__}")

        self.post_id = post_id
        self.user_id = user_id
        self.timestamp = timestamp
        self.text = text

        self.reply_parent_id = reply_parent_id
        self.reply_root_id = reply_root_id
        self.reply_parent_author = reply_parent_author
        self.reply_root_author = reply_root_author
        self.sample_type = sample_type

        self.stance = stance
        self.embedding = embedding
        self.is_noise = is_noise