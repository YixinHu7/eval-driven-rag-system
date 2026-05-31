from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from src.ingestion.cleaner import normalize_whitespace


@dataclass(frozen=True)
class WebDocument:
    url: str
    title: str
    text: str


def fetch_web_document(url: str, timeout: float = 20.0) -> WebDocument:
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    title = _extract_title(soup)
    main_content = _extract_main_content(soup)

    markdown_text = md(str(main_content), heading_style="ATX")
    cleaned_text = normalize_whitespace(markdown_text)

    return WebDocument(
        url=url,
        title=title,
        text=cleaned_text,
    )


def _extract_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)

    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(strip=True)

    return "Untitled Document"


def _extract_main_content(soup: BeautifulSoup):
    candidates = [
        soup.find("main"),
        soup.find("article"),
        soup.find("div", class_="td-content"),
        soup.find("div", class_="content"),
    ]

    for candidate in candidates:
        if candidate is not None:
            _remove_unwanted_elements(candidate)
            return candidate

    body = soup.body or soup
    _remove_unwanted_elements(body)
    return body


def _remove_unwanted_elements(node) -> None:
    for tag in node.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()