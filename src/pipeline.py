"""Pipeline orchestrator: collect → analyze → generate → review."""

import logging

from src.analyzer.strategy import StrategyAnalyzer
from src.collector.base import BaseCollector
from src.db.database import Database
from src.generator.content import ContentGenerator
from src.models.schemas import ContentItem
from src.reviewer.slack_bot import SlackReviewer

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        db: Database,
        collectors: list[BaseCollector],
        analyzer: StrategyAnalyzer,
        generator: ContentGenerator,
        reviewer: SlackReviewer,
    ):
        self._db = db
        self._collectors = collectors
        self._analyzer = analyzer
        self._generator = generator
        self._reviewer = reviewer

    async def run(self) -> None:
        # 1. Collect
        all_items: list[ContentItem] = []
        for collector in self._collectors:
            try:
                items = await collector.collect()
                logger.info(
                    "Collected %d items from %s",
                    len(items),
                    type(collector).__name__,
                )
                all_items.extend(items)
            except Exception:
                logger.exception(
                    "Collector %s failed", type(collector).__name__
                )

        # 2. Deduplicate and save
        new_items: list[ContentItem] = []
        for item in all_items:
            if not await self._db.url_exists(item.url):
                await self._db.save_content_item(item)
                new_items.append(item)

        logger.info("New items after dedup: %d", len(new_items))

        # 3. Analyze
        suggestions = await self._analyzer.analyze(new_items)
        logger.info("Action suggestions: %d", len(suggestions))

        # 4. Generate and send for review
        item_map = {item.id: item for item in new_items}
        for suggestion in suggestions:
            item = item_map.get(suggestion.content_item_id)
            if not item:
                continue
            try:
                draft = await self._generator.generate(item, suggestion)
                await self._db.save_draft(draft)
                await self._reviewer.send_for_review(draft, item)
                logger.info("Sent draft %s for review", draft.id)
            except Exception:
                logger.exception(
                    "Failed to generate/review for item %s", item.id
                )
