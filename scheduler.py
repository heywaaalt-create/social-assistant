"""Main entry point: runs the pipeline on a schedule."""

import asyncio
import logging
import os

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from src.config import init_env, load_persona, load_settings
from src.db.database import Database
from src.analyzer.strategy import StrategyAnalyzer
from src.collector.news import NewsCollector
from src.collector.threads import ThreadsCollector
from src.generator.content import ContentGenerator
from src.generator.llm_provider import create_provider
from src.pipeline import Pipeline
from src.publisher.threads import ThreadsPublisher
from src.reviewer.slack_bot import SlackReviewer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    init_env()
    settings = load_settings()
    persona = load_persona()

    # Database
    db_path = os.getenv("DATABASE_PATH", "data/social_assistant.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db = Database(db_path)
    await db.init()

    # LLM
    llm = create_provider(os.getenv("LLM_PROVIDER", "claude"))

    # Collectors
    collectors = []
    threads_token = os.getenv("THREADS_ACCESS_TOKEN")
    threads_user = os.getenv("THREADS_USER_ID")
    if threads_token and threads_user:
        collectors.append(ThreadsCollector(threads_token, threads_user))

    for src in settings["collector"].get("news_sources", []):
        if src.get("type") == "scrape":
            collectors.append(
                NewsCollector(
                    source_name=src["name"],
                    base_url=src["url"],
                    article_selector="article",
                    title_selector="h2 a, h3 a",
                    excerpt_selector="p",
                )
            )

    # Analyzer, Generator
    analyzer = StrategyAnalyzer(settings["analyzer"], persona, llm)
    generator = ContentGenerator(settings["generator"], persona, llm)

    # Slack
    from slack_sdk import WebClient

    slack_client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
    reviewer = SlackReviewer(
        slack_client, os.getenv("SLACK_REVIEW_CHANNEL", "#social-review")
    )

    # Publisher (for approved items — handled by Slack bot event loop)
    _publisher = ThreadsPublisher(threads_token or "", threads_user or "")

    # Pipeline
    pipeline = Pipeline(db, collectors, analyzer, generator, reviewer)

    # Scheduler
    scheduler = AsyncIOScheduler()
    interval = settings["collector"].get("schedule_interval_hours", 4)
    scheduler.add_job(pipeline.run, "interval", hours=interval, id="pipeline")
    scheduler.start()

    logger.info(
        "Social Assistant started. Pipeline runs every %d hours.", interval
    )

    # Run once immediately
    await pipeline.run()

    # Keep alive
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        await db.close()
        logger.info("Social Assistant stopped.")


if __name__ == "__main__":
    asyncio.run(main())
