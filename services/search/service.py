# services/search/service.py
"""Search service — clean interface for web search."""

import asyncio
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from . import (
    searxng_search_results,
    fetch_webpage_content,
    get_search_config,
)


@dataclass
class SearchResult:
    """A single search result."""
    url: str
    title: str
    snippet: str
    content: Optional[str] = None


@dataclass
class SearchResponse:
    """Response from a search query."""
    query: str
    results: List[SearchResult]
    total: int
    cached: bool = False


class SearchService:
    """
    Web search service.

    Usage:
        service = SearchService()
        result = await service.search("python async patterns")
        for r in result.results:
            print(f"{r.title}: {r.url}")
    """

    def __init__(self, default_depth: int = 1, fetch_content: bool = True):
        self.default_depth = default_depth
        # Stored under a distinct name so it doesn't shadow the fetch_content() method.
        self.fetch_content_default = fetch_content

    async def search(
        self,
        query: str,
        depth: Optional[int] = None,
        fetch_content: Optional[bool] = None,
    ) -> SearchResponse:
        """
        Search the web.

        Args:
            query: Search query
            depth: Search depth (1=quick, 2=thorough, 3=comprehensive)
            fetch_content: Whether to fetch full page content

        Returns:
            SearchResponse with results
        """
        depth = depth or self.default_depth
        do_fetch = fetch_content if fetch_content is not None else self.fetch_content_default

        # searxng_search_results is synchronous (blocking I/O) and returns a
        # list of {url, title, snippet} dicts — run it off the event loop.
        raw_results = await asyncio.to_thread(searxng_search_results, query, 10 * depth)

        results = []
        for r in raw_results:
            content = None
            url = r.get("url", "")
            if do_fetch and url:
                content = await self.fetch_content(url)
            results.append(SearchResult(
                url=url,
                title=r.get("title", ""),
                snippet=r.get("snippet", ""),
                content=content,
            ))

        return SearchResponse(
            query=query,
            results=results,
            total=len(results),
        )

    async def fetch_content(self, url: str) -> Optional[str]:
        """Fetch the extracted text content from a URL, or None on failure."""
        result = await asyncio.to_thread(fetch_webpage_content, url)
        if isinstance(result, dict):
            return result.get("content") if result.get("success") else None
        return result

    def get_config(self) -> Dict[str, Any]:
        """Get current search configuration."""
        return get_search_config()
