import logging
from dataclasses import dataclass
from datetime import datetime

from duckduckgo_search import DDGS
from langchain_core.tools import tool


@tool
def get_local_datetime() -> str:
    """Return the current datetime as a string."""
    now = datetime.now().astimezone()
    logging.debug(f"[Tool] Current datetime: {now}")
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


@dataclass
class SearchResult:
    title: str
    href: str
    body: str


@tool
def search_duckduckgo(keyword: str) -> list[SearchResult]:
    """Search DuckDuckGo for a keyword and return the top results."""
    results = DDGS().text(keyword, max_results=5)
    search_results = []

    for result in results:
        search_results.append(
            SearchResult(
                title=result["title"], href=result["href"], body=result["body"]
            )
        )

    return search_results


if __name__ == "__main__":
    keyword = "python programming"
    results = search_duckduckgo.invoke(keyword)

    for result in results:
        print(
            f"Title: {result.title}\nLink: {result.href}\nDescription: {result.body}\n"
        )
