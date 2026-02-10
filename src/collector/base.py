"""Base collector interface."""

from abc import ABC, abstractmethod

from src.models.schemas import ContentItem


class BaseCollector(ABC):
    @abstractmethod
    async def collect(self) -> list[ContentItem]:
        ...
