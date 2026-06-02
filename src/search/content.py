"""Webpage content fetching with caching, PDF extraction, and summarization helpers."""

import io
import ipaddress
import json
import os
import re
import logging
import socket
from datetime import datetime, timedelta
from typing import List
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from .analytics import RateLimitError, error_logger
from .cache import (
    CONTENT_CACHE_DIR,
    content_cache_index,
    generate_cache_key,
    cleanup_cache,
)

logger = logging.getLogger(__name__)

# Prefer curl_cffi as the HTTP client: it impersonates a real browser's TLS/HTTP
# fingerprint (JA3/JA4), getting past anti-bot edges (e.g. Wikimedia's) that
# fingerprint and 403 a plain httpx client regardless of headers. Falls back to
# httpx when curl_cffi isn't installed.
try:
    from curl_cffi import requests as _cffi_requests
    from curl_cffi.requests.exceptions import RequestException as _CurlError
    _HAS_CURL_CFFI = True
except ImportError:  # pragma: no cover - optional dependency
    _cffi_requests = None
    _HAS_CURL_CFFI = False

# Browser profile to impersonate ("chrome" = latest alias in the installed
# curl_cffi); keeps the TLS fingerprint and browser headers coherent.
_IMPERSONATE = "chrome"

# Network-layer errors that mean "fetch failed", whichever client is in use.
FETCH_NETWORK_ERRORS = (
    (httpx.RequestError, _CurlError) if _HAS_CURL_CFFI else (httpx.RequestError,)
)

# Browser-like request headers for page fetches. A stale or bot-looking
# User-Agent gets 403'd by sites like Wikipedia, so present as a current
# Chrome on Windows and send the Accept/Sec-Fetch hints a real browser does.
# (Accept-Encoding stays at gzip/deflate — what httpx can decode without extra
# optional packages like brotli/zstandard.)
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

_PRIVATE_NETWORKS = (
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)


def _is_private_address(addr: ipaddress._BaseAddress) -> bool:
    return any(addr in net for net in _PRIVATE_NETWORKS) or addr.is_private or addr.is_loopback


def _resolve_hostname_ips(hostname: str) -> List[ipaddress._BaseAddress]:
    ips = []
    for family, _, _, _, sockaddr in socket.getaddrinfo(hostname, None):
        if family in (socket.AF_INET, socket.AF_INET6):
            ips.append(ipaddress.ip_address(sockaddr[0]))
    return ips


def _public_http_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    host = parsed.hostname.strip().lower()
    if host in ("localhost", "metadata.google.internal", "metadata"):
        return False
    try:
        return not _is_private_address(ipaddress.ip_address(host))
    except ValueError:
        pass
    try:
        return all(not _is_private_address(ip) for ip in _resolve_hostname_ips(host))
    except OSError:
        return False


def _follow_redirects(start_url: str, getter):
    """Follow up to 8 redirects, re-checking every hop against the SSRF blocklist.

    `getter(url)` performs a single non-redirecting GET. Manual following (rather
    than the client's own redirect support) is what lets us re-validate each hop.
    """
    current = start_url
    for _ in range(8):
        response = getter(current)
        if response.status_code not in (301, 302, 303, 307, 308):
            return response
        location = response.headers.get("location")
        if not location:
            return response
        current = urljoin(current, location)
        if not _public_http_url(current):
            raise httpx.RequestError(f"Blocked redirect to non-public URL: {current}")
    raise httpx.RequestError("Too many redirects")


def _get_public_url(url: str, *, headers: dict, timeout: int):
    """Fetch a URL with per-hop SSRF protection.

    Uses curl_cffi (browser-impersonating TLS) when available, else httpx.
    """
    if not _public_http_url(url):
        raise httpx.RequestError(f"Blocked non-public URL: {url}")

    if _HAS_CURL_CFFI:
        # impersonate sets a coherent browser header set itself, so we don't
        # pass our own headers here (avoids a UA/Sec-Ch-Ua vs TLS mismatch).
        def _get(u):
            return _cffi_requests.get(
                u, impersonate=_IMPERSONATE, timeout=timeout, allow_redirects=False
            )
        return _follow_redirects(url, _get)

    with httpx.Client(headers=headers, timeout=timeout, follow_redirects=False) as client:
        return _follow_redirects(url, client.get)

# PDF extraction (optional dependency)
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except ImportError:
    pdf_extract_text = None  # type: ignore


# ----------------------------------------------------------------------
# HTML extraction helpers
# ----------------------------------------------------------------------
def _extract_meta(soup: BeautifulSoup) -> dict:
    """Pull meta description and keywords if present."""
    description = ""
    keywords = ""
    desc_tag = soup.find("meta", attrs={"name": re.compile("description", re.I)})
    if desc_tag and desc_tag.get("content"):
        description = desc_tag["content"].strip()
    kw_tag = soup.find("meta", attrs={"name": re.compile("keywords", re.I)})
    if kw_tag and kw_tag.get("content"):
        keywords = kw_tag["content"].strip()
    return {"description": description, "keywords": keywords}


def _extract_og_image(soup: BeautifulSoup) -> str:
    """Extract the best representative image URL from meta tags.

    Only returns absolute http(s) URLs — skips relative paths and data URIs.
    """
    candidates = []
    # Open Graph image (most reliable)
    for prop in ("og:image", "og:image:url", "og:image:secure_url"):
        tag = soup.find("meta", attrs={"property": prop})
        if tag and tag.get("content", "").strip():
            candidates.append(tag["content"].strip())
    # Twitter card image
    tag = soup.find("meta", attrs={"name": "twitter:image"})
    if tag and tag.get("content", "").strip():
        candidates.append(tag["content"].strip())
    # Thumbnail meta
    tag = soup.find("meta", attrs={"name": "thumbnail"})
    if tag and tag.get("content", "").strip():
        candidates.append(tag["content"].strip())
    # Return first absolute https URL
    for url in candidates:
        if url.startswith("https://") and not url.endswith((".svg", ".ico")):
            return url
    return ""


def _extract_lists(soup: BeautifulSoup) -> List[List[str]]:
    """Return a list of lists, each inner list representing a <ul>/<ol>."""
    all_lists = []
    for lst in soup.find_all(["ul", "ol"]):
        items = [li.get_text(separator=" ", strip=True) for li in lst.find_all("li")]
        if items:
            all_lists.append(items)
    return all_lists


def _extract_tables(soup: BeautifulSoup) -> List[List[List[str]]]:
    """Return a list of tables, each table is a list of rows, each row a list of cell texts."""
    tables_data = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(separator=" ", strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if rows:
            tables_data.append(rows)
    return tables_data


def _extract_code_blocks(soup: BeautifulSoup) -> List[str]:
    """Collect text from <pre> and <code> blocks."""
    blocks = []
    for tag in soup.find_all(["pre", "code"]):
        txt = tag.get_text(separator=" ", strip=True)
        if txt:
            blocks.append(txt)
    return blocks


def _detect_js_frameworks(soup: BeautifulSoup) -> bool:
    """Very naive detection of common JS frameworks."""
    js_indicators = [
        "react", "angular", "vue", "svelte", "next", "nuxt",
        "ember", "backbone", "jquery", "polymer", "mithril",
    ]
    for script in soup.find_all("script"):
        src = script.get("src", "").lower()
        if any(fr in src for fr in js_indicators):
            return True
        if script.string:
            content = script.string.lower()
            if any(fr in content for fr in js_indicators):
                return True
    if soup.find(attrs={"data-reactroot": True}) or soup.find(attrs={"ng-app": True}):
        return True
    return False


def _empty_result(url: str, error: str = "") -> dict:
    """Build a standard failure result dict."""
    return {
        "url": url,
        "title": "",
        "content": "",
        "lists": [],
        "tables": [],
        "code_blocks": [],
        "meta_description": "",
        "meta_keywords": "",
        "js_rendered": False,
        "js_message": "",
        "success": False,
        "error": error,
    }


# ----------------------------------------------------------------------
# Main content fetcher
# ----------------------------------------------------------------------
def fetch_webpage_content(url: str, timeout: int = 5, retry_attempt: int = 0) -> dict:
    """Fetch and extract meaningful content from a webpage with caching."""
    cache_key = generate_cache_key(url)
    cache_file = CONTENT_CACHE_DIR / f"{cache_key}.cache"

    # Check cache
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            timestamp = datetime.fromisoformat(cached_data["timestamp"])
            if datetime.now() - timestamp < timedelta(hours=2):
                logger.debug(f"Content cache hit for URL: {url}")
                return cached_data["data"]
            else:
                cache_file.unlink(missing_ok=True)
                content_cache_index.pop(cache_key, None)
        except Exception as e:
            logger.warning(f"Failed to read content cache for {url}: {e}")
            cache_file.unlink(missing_ok=True)
            content_cache_index.pop(cache_key, None)

    # Fetch
    try:
        response = _get_public_url(url, headers=dict(BROWSER_HEADERS), timeout=timeout)

        if response.status_code == 429:
            raise RateLimitError(f"Rate limit hit for {url} (attempt {retry_attempt})")

        # Handle errors by status code rather than raise_for_status() so it works
        # the same for httpx and curl_cffi responses (e.g. 403 from anti-bot
        # edges, 404 missing pages).
        if response.status_code >= 400:
            error_logger.warning(f"HTTP {response.status_code} fetching {url}")
            return _empty_result(url, f"HTTP {response.status_code}")
    except FETCH_NETWORK_ERRORS as e:
        error_logger.error(f"NetworkError fetching {url} (attempt {retry_attempt}): {e}")
        return _empty_result(url, f"NetworkError: {e}")
    except RateLimitError as e:
        error_logger.error(str(e))
        return _empty_result(url, str(e))

    # PDF handling
    content_type = response.headers.get("Content-Type", "").lower()
    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        if pdf_extract_text is None:
            logger.error("pdfminer.six is not installed; cannot extract PDF text.")
            pdf_text = ""
        else:
            try:
                pdf_bytes = io.BytesIO(response.content)
                pdf_text = pdf_extract_text(pdf_bytes)
            except Exception as e:
                logger.warning(f"PDF extraction failed for {url}: {e}")
                pdf_text = ""
        result = {
            "url": url,
            "title": os.path.basename(url),
            "content": pdf_text,
            "lists": [],
            "tables": [],
            "code_blocks": [],
            "meta_description": "",
            "meta_keywords": "",
            "js_rendered": False,
            "js_message": "",
            "success": bool(pdf_text),
            "error": "" if pdf_text else "Failed to extract PDF text",
        }
        _cache_result(cache_file, cache_key, result, url)
        return result

    # HTML handling
    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        error_logger.error(f"ParseError parsing HTML from {url} (attempt {retry_attempt}): {e}")
        result = _empty_result(url, f"ParseError: {e}")
        _cache_result(cache_file, cache_key, result, url)
        return result

    title_tag = soup.find("title")
    title_text = title_tag.get_text(strip=True) if title_tag else ""
    meta_info = _extract_meta(soup)
    og_image = _extract_og_image(soup)
    js_rendered = _detect_js_frameworks(soup)
    js_message = "Page appears to be rendered by a JavaScript framework; content may be incomplete." if js_rendered else ""

    # Main textual content (heuristic)
    main_content = ""
    content_areas = soup.find_all(
        ["main", "article", "section", "div"],
        class_=re.compile("content|main|body|article|post|entry|text", re.I),
    )
    if content_areas:
        # Rank by amount of text, not document order: page chrome (e.g.
        # Wikipedia's "vector-menu-content" nav divs) also matches the class
        # regex and sits first in the DOM, so first-N-in-order would grab menus
        # instead of the article. Largest-first, skipping anything nested inside
        # an already-chosen block to avoid double-counting.
        ranked = sorted(content_areas, key=lambda a: len(a.get_text(strip=True)), reverse=True)
        chosen = []
        for area in ranked:
            if any(area in sel.descendants for sel in chosen):
                continue
            chosen.append(area)
            if len(chosen) >= 3:
                break
        main_content = " ".join(a.get_text(separator=" ", strip=True) for a in chosen)
    if not main_content:
        body = soup.find("body")
        if body:
            main_content = body.get_text(separator=" ", strip=True)

    main_content = re.sub(r"\s+", " ", main_content).strip()

    result = {
        "url": url,
        "title": title_text,
        "content": main_content,
        "lists": _extract_lists(soup),
        "tables": _extract_tables(soup),
        "code_blocks": _extract_code_blocks(soup),
        "meta_description": meta_info.get("description", ""),
        "meta_keywords": meta_info.get("keywords", ""),
        "og_image": og_image,
        "js_rendered": js_rendered,
        "js_message": js_message,
        "success": True,
        "error": "",
    }
    _cache_result(cache_file, cache_key, result, url)
    return result


def _cache_result(cache_file, cache_key: str, result: dict, url: str):
    """Write a result to the content cache."""
    try:
        cache_data = {"timestamp": datetime.now().isoformat(), "data": result}
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)
        content_cache_index[cache_key] = datetime.now()
        cleanup_cache(CONTENT_CACHE_DIR, content_cache_index, timedelta(hours=2))
    except Exception as e:
        logger.warning(f"Failed to write content cache for {url}: {e}")


# ----------------------------------------------------------------------
# Content summarization helpers
# ----------------------------------------------------------------------
def extract_key_points(text: str) -> List[str]:
    """Pull out bullet-style key points from a block of text."""
    points: List[str] = []
    bullet_pat = re.compile(r"^\s*[-*•]\s+(.*)")
    numbered_pat = re.compile(r"^\s*\d+[\.\)]\s+(.*)")
    for line in text.splitlines():
        m = bullet_pat.match(line) or numbered_pat.match(line)
        if m:
            points.append(m.group(1).strip())
    return points


def get_tldr(text: str, max_sentences: int = 3) -> str:
    """Produce a very short TL;DR by taking the first few sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    selected = [s.strip() for s in sentences if s][:max_sentences]
    return " ".join(selected)


def extract_quotes(text: str) -> List[str]:
    """Return quoted excerpts that are at least 15 characters long."""
    return [m.group(1).strip() for m in re.finditer(r'["\']([^"\']{15,}?)["\']', text)]


def extract_statistics(text: str) -> List[str]:
    """Find numbers, percentages, dates and simple measurements."""
    pattern = re.compile(
        r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(%|percent|‰|per cent|[a-zA-Z]+)?\b",
        re.IGNORECASE,
    )
    return [m.group(0).strip() for m in pattern.finditer(text)]
