
import pytest


@pytest.fixture
def tmp_db_path(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def sample_settings():
    return {
        "collector": {
            "schedule_interval_hours": 4,
            "keywords": ["展覽", "美術館"],
            "news_sources": [],
        },
        "analyzer": {
            "min_relevance_score": 5.0,
            "min_heat_score": 3.0,
            "daily_post_limit": 3,
            "daily_reply_limit": 10,
            "risk_keywords": ["政治"],
        },
        "generator": {
            "max_retries": 3,
            "max_post_length": 500,
            "max_reply_length": 300,
        },
        "reviewer": {"publish_hours": [10, 15, 20]},
        "publisher": {
            "retry_count": 3,
            "retry_delay_seconds": 60,
            "metrics_check_interval_hours": 24,
        },
    }


@pytest.fixture
def sample_persona():
    return {
        "name": "ARTOGO",
        "platform": "Threads",
        "tone": "輕鬆友善",
        "language": "zh-TW",
        "style_guidelines": ["用口語化的方式分享藝術觀點"],
        "forbidden_words": ["業配", "工商"],
        "example_posts": ["最近去看了展覽..."],
        "example_replies": ["這個展我也有去！"],
    }
