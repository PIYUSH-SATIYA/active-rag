"""Live web search retriever with disk caching.

Providers (no API key required):

1. DuckDuckGo Lite HTML  (fast; rate-limited after a burst)
2. Bing HTML             (independent quota; slower)
3. ``ddgs`` package      (last resort, tries several engines internally)
4. Tavily                (only when ``TAVILY_API_KEY`` is set)

Providers are rotated per query so no single service sees the whole run, and
each query falls through to the next provider on failure.  Every successful
response is cached as JSON so runs are resumable and cheap when several
workers share a Drive cache directory.
"""

from __future__ import annotations

import base64
import hashlib
import html as html_module
import json
import os
import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Iterable, Optional
from urllib.parse import parse_qs, unquote, urlparse

from src.retrieval.base import RetrievalResult

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_WIKIPEDIA_DOMAINS = (
    "wikipedia.org",
    "wikiwand.com",
    "wiki2.org",
    "wikimedia.org",
    "dbpedia.org",
)


def _clean(text: Any) -> str:
    if text is None:
        return ""
    text = html_module.unescape(str(text))
    return re.sub(r"\s+", " ", text).strip()


def _strip_tags(fragment: str) -> str:
    return _clean(re.sub(r"<[^>]+>", " ", fragment))


class _DDGLiteParser(HTMLParser):
    """Collect ``result-link`` anchors and attach the following snippet."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[dict[str, str]] = []
        self._capture: Optional[str] = None
        self._buffer: list[str] = []
        self._href = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        attributes = {key: value or "" for key, value in attrs}
        classes = attributes.get("class", "").split()
        if tag == "a" and "result-link" in classes:
            self._capture = "link"
            self._href = attributes.get("href", "")
            self._buffer = []
        elif tag in {"td", "div"} and "result-snippet" in classes:
            self._capture = "snippet"
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._capture == "link" and tag == "a":
            self.rows.append(
                {
                    "url": self._href,
                    "title": _clean("".join(self._buffer)),
                    "body": "",
                }
            )
            self._capture = None
            self._buffer = []
        elif self._capture == "snippet" and tag in {"td", "div"}:
            if self.rows and not self.rows[-1]["body"]:
                self.rows[-1]["body"] = _clean("".join(self._buffer))
            self._capture = None
            self._buffer = []


def _decode_redirect(url: str) -> str:
    """Unwrap DuckDuckGo/Bing click-tracking URLs."""
    if not url:
        return ""
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if "duckduckgo.com" in host and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return unquote(target[0])
    if "bing.com" in host and parsed.path.startswith("/ck/"):
        target = parse_qs(parsed.query).get("u")
        if target:
            encoded = target[0]
            if encoded.startswith("a1"):
                encoded = encoded[2:]
            padding = "=" * (-len(encoded) % 4)
            try:
                return base64.urlsafe_b64decode(encoded + padding).decode(
                    "utf-8", "ignore"
                )
            except Exception:  # noqa: BLE001 - malformed tracker
                return url
    return url


def _parse_ddg_lite(html_text: str) -> list[dict]:
    parser = _DDGLiteParser()
    parser.feed(html_text)
    return [
        {
            "title": row["title"],
            "href": _decode_redirect(row["url"]),
            "body": row["body"],
        }
        for row in parser.rows
    ]


class ProviderRateLimited(RuntimeError):
    """Raised when a provider answers with a bot check instead of results."""


_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "who", "what", "when",
    "where", "which", "why", "how", "did", "does", "has", "have", "had",
    "with", "from", "that", "this", "than", "into", "its", "his", "her",
    "their", "you", "your", "can", "could", "would", "should", "about",
    "there", "these", "those", "not", "but", "all", "any", "our", "out",
    "one", "two", "also", "been", "being", "will", "may", "might", "must",
}


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) >= 3 and token not in _STOPWORDS
    }


_YAHOO_ITEM_RE = re.compile(r"<li[^>]*>.*?</li>", re.S | re.I)
_YAHOO_TITLE_RE = re.compile(r"<h3[^>]*>(.*?)</h3>", re.S | re.I)
_YAHOO_SNIPPET_RE = re.compile(
    r'<div class="compText[^"]*"[^>]*>(.*?)</div>', re.S | re.I
)
_YAHOO_HREF_RE = re.compile(r'<a[^>]+href="([^"]+)"', re.S | re.I)


def _decode_yahoo_url(url: str) -> str:
    if "r.search.yahoo.com" in url:
        match = re.search(r"RU=([^/]+)", url)
        if match:
            return unquote(match.group(1))
    return url


def _parse_yahoo(html_text: str) -> list[dict]:
    rows: list[dict] = []
    for item in _YAHOO_ITEM_RE.findall(html_text):
        if "compTitle" not in item:
            continue
        href = _YAHOO_HREF_RE.search(item)
        title = _YAHOO_TITLE_RE.search(item)
        snippet = _YAHOO_SNIPPET_RE.search(item)
        if not href or not title:
            continue
        rows.append(
            {
                "title": _strip_tags(title.group(1)),
                "href": _decode_yahoo_url(html_module.unescape(href.group(1))),
                "body": _strip_tags(snippet.group(1)) if snippet else "",
            }
        )
    return rows


_BING_BLOCK_RE = re.compile(r'<li[^>]+class="[^"]*\bb_algo\b[^"]*".*?</li>', re.S | re.I)
_BING_TITLE_RE = re.compile(
    r"<h2[^>]*>\s*<a[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>", re.S | re.I
)
_BING_SNIPPET_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.S | re.I)


def _parse_bing(html_text: str) -> list[dict]:
    rows: list[dict] = []
    for block in _BING_BLOCK_RE.findall(html_text):
        title_match = _BING_TITLE_RE.search(block)
        if not title_match:
            continue
        snippet_match = _BING_SNIPPET_RE.search(block)
        rows.append(
            {
                "title": _strip_tags(title_match.group(2)),
                "href": _decode_redirect(html_module.unescape(title_match.group(1))),
                "body": _strip_tags(snippet_match.group(1)) if snippet_match else "",
            }
        )
    return rows


class WebRetriever:
    """Search the live web and return passage-like snippets.

    Parameters
    ----------
    backend:
        ``"auto"`` (all no-key providers, then Tavily), ``"ddgs"``, or
        ``"tavily"``.  ``"auto"`` is what experiments should use.
    cache_dir:
        Directory for JSON cache files.  Point this at Drive when running on
        Colab so workers share results.
    exclude_domains:
        Domains removed from results.  Used by the ``web_nwiki`` ablation.
    fetch_pages:
        When true, fetch the top result pages and use extracted page text
        instead of the search snippet.  Disabled by default because it adds
        latency and failure modes.
    """

    name = "web"

    def __init__(
        self,
        backend: str = "auto",
        topk: int = 2,
        cache_dir: str = ".cache/web",
        exclude_domains: Optional[Iterable[str]] = None,
        fetch_pages: bool = False,
        max_page_chars: int = 900,
        max_chars: int = 700,
        include_url: bool = False,
        tavily_key: Optional[str] = None,
        min_interval: float = 0.6,
        timeout: float = 12.0,
        max_retries: int = 3,
    ) -> None:
        self.backend = backend.lower()
        if self.backend not in {"auto", "ddgs", "tavily"}:
            raise ValueError(f"Unknown web backend: {backend!r}")

        self.topk = topk
        self.cache_dir = cache_dir
        self.exclude_domains = tuple(
            d.lower().lstrip(".") for d in (exclude_domains or ())
        )
        self.fetch_pages = fetch_pages
        self.max_page_chars = max_page_chars
        self.max_chars = max_chars
        self.include_url = include_url
        self.tavily_key = tavily_key or os.environ.get("TAVILY_API_KEY")
        self.min_interval = max(0.0, float(min_interval))
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))

        self._last_request = 0.0
        self._session = None
        self._provider_cursor = 0
        self._cooldown_until: dict[str, float] = {}
        self.cooldown_seconds = 180.0
        self.stats: dict[str, Any] = {
            "queries": 0,
            "cache_hits": 0,
            "network_queries": 0,
            "errors": 0,
            "results": 0,
            "latency_ms": [],
            "providers": {},
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(
        self,
        queries: list[str],
        topk: int = 2,
        max_query_length: Optional[int] = None,
    ) -> list[list[RetrievalResult]]:
        topk = max(1, int(topk or self.topk))
        return [self._retrieve_one(self._truncate(q, max_query_length), topk) for q in queries]

    def summary(self) -> dict:
        latencies = self.stats["latency_ms"]
        return {
            "backend": self.backend,
            "queries": self.stats["queries"],
            "cache_hits": self.stats["cache_hits"],
            "network_queries": self.stats["network_queries"],
            "errors": self.stats["errors"],
            "results": self.stats["results"],
            "providers": dict(self.stats["providers"]),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1)
            if latencies
            else 0.0,
        }

    # ------------------------------------------------------------------
    # One query
    # ------------------------------------------------------------------
    def _retrieve_one(self, query: str, topk: int) -> list[RetrievalResult]:
        self.stats["queries"] += 1
        key = self._cache_key(query, topk)
        cached = self._cache_read(key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            self.stats["results"] += len(cached)
            return [RetrievalResult(**row) for row in cached][:topk]

        started = time.monotonic()
        raw = self._search_with_fallback(query, topk)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        self.stats["latency_ms"].append(elapsed_ms)
        self.stats["network_queries"] += 1

        results = self._to_results(raw, topk)
        if results:
            self._cache_write(key, query, results)
        self.stats["results"] += len(results)
        return results

    def _to_results(self, raw: list[dict], topk: int) -> list[RetrievalResult]:
        out: list[RetrievalResult] = []
        for i, row in enumerate(raw):
            url = _clean(row.get("url") or row.get("href"))
            if url and self._is_excluded(url):
                continue

            title = _clean(row.get("title"))
            snippet = _clean(
                row.get("snippet") or row.get("body") or row.get("content")
            )
            if not snippet and not title:
                continue

            if self.fetch_pages and url:
                page_text = self._fetch_page(url)
                if page_text:
                    snippet = page_text

            text = self._format_text(title, snippet, url)
            if not text:
                continue

            score = row.get("score")
            try:
                score = float(score) if score is not None else 0.0
            except (TypeError, ValueError):
                score = 0.0
            if not score:
                score = 1.0 / (i + 1)

            out.append(
                RetrievalResult(
                    doc_id=url or f"web-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}",
                    text=text,
                    score=score,
                    source="web",
                    url=url or None,
                    title=title or None,
                )
            )
            if len(out) >= topk:
                break
        return out

    def _format_text(self, title: str, snippet: str, url: str) -> str:
        if title and snippet:
            text = f"{title}. {snippet}"
        else:
            text = title or snippet
        if self.include_url and url:
            text = f"{text} [{url}]"
        text = _clean(text)
        return text[: self.max_chars]

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------
    def _provider_order(self) -> list[str]:
        """Alternate the two fast HTML providers so their quotas are shared.

        ``ddgs`` is always last because it is the slowest and least reliable.
        """
        if self.backend == "tavily":
            return ["tavily"]
        if self.backend == "ddgs":
            return ["ddgs"]

        # Rotate the starting provider so quotas are shared across a run.
        fast = ["ddg", "yahoo", "bing"]
        shift = self._provider_cursor % len(fast)
        self._provider_cursor += 1
        order = fast[shift:] + fast[:shift]

        order.append("ddgs")
        if self.tavily_key:
            order.append("tavily")
        return order

    def _looks_relevant(self, query: str, rows: list[dict]) -> bool:
        """Reject anti-bot pages that return results for an unrelated query."""
        wanted = _content_tokens(query)
        if not wanted:
            return bool(rows)
        for row in rows:
            haystack = " ".join(
                str(row.get(key) or "")
                for key in ("title", "body", "snippet", "href", "url")
            )
            if wanted & _content_tokens(haystack):
                return True
        return False

    def _search_with_fallback(self, query: str, topk: int) -> list[dict]:
        max_results = max(topk * 3, 5)
        errors: list[str] = []

        for provider in self._provider_order():
            if provider == "tavily" and not self.tavily_key:
                continue
            if time.monotonic() < self._cooldown_until.get(provider, 0.0):
                errors.append(f"{provider} cooling down")
                continue

            attempts = 1 if provider == "ddgs" else min(self.max_retries, 2)
            for attempt in range(attempts):
                try:
                    self._respect_rate_limit()
                    rows = self._search_provider(provider, query, max_results)
                    if rows and self._looks_relevant(query, rows):
                        self.stats["providers"][provider] = (
                            self.stats["providers"].get(provider, 0) + 1
                        )
                        return rows
                    if rows:
                        errors.append(f"{provider} returned irrelevant rows")
                        break
                    errors.append(f"{provider} returned no rows")
                    break
                except ProviderRateLimited as exc:
                    self._cooldown_until[provider] = (
                        time.monotonic() + self.cooldown_seconds
                    )
                    errors.append(f"{provider} rate limited: {exc}")
                    break
                except Exception as exc:  # noqa: BLE001 - network layer
                    errors.append(f"{provider} attempt {attempt + 1}: {exc}")
                    if attempt + 1 < attempts:
                        self._backoff(attempt)

        if errors:
            self.stats["errors"] += 1
            print(f"[web] no results for {query!r}: {'; '.join(errors[-2:])}")
        return []

    def _search_provider(
        self, provider: str, query: str, max_results: int
    ) -> list[dict]:
        if provider == "ddg":
            return self._search_ddg_lite(query, max_results)
        if provider == "bing":
            return self._search_bing(query, max_results)
        if provider == "yahoo":
            return self._search_yahoo(query, max_results)
        if provider == "ddgs":
            return self._search_ddgs(query, max_results)
        if provider == "tavily":
            return self._search_tavily(query, max_results)
        raise ValueError(f"Unknown provider: {provider!r}")

    def _get_session(self):
        """Lazy ``requests`` session with browser-like headers."""
        if self._session is None:
            import requests

            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": _USER_AGENT,
                    "Accept-Language": "en-US,en;q=0.9",
                }
            )
            self._session = session
        return self._session

    def _search_ddg_lite(self, query: str, max_results: int) -> list[dict]:
        response = self._get_session().post(
            "https://lite.duckduckgo.com/lite/",
            data={"q": query, "kl": "wt-wt"},
            timeout=min(self.timeout, 8.0),
        )
        if response.status_code == 202:
            raise ProviderRateLimited("ddg-lite answered 202 (bot check)")
        response.raise_for_status()
        rows = _parse_ddg_lite(response.text)
        if not rows:
            raise RuntimeError("ddg-lite returned no parsable results")
        return rows[:max_results]

    def _search_bing(self, query: str, max_results: int) -> list[dict]:
        response = self._get_session().get(
            "https://www.bing.com/search",
            params={
                "q": query,
                "count": max_results,
                "mkt": "en-US",
                "cc": "US",
                "setlang": "en-US",
            },
            timeout=min(self.timeout, 8.0),
        )
        if response.status_code in {403, 429}:
            raise ProviderRateLimited(f"bing answered {response.status_code}")
        response.raise_for_status()
        rows = _parse_bing(response.text)
        if not rows:
            raise RuntimeError("bing returned no parsable results")
        return rows[:max_results]

    def _search_yahoo(self, query: str, max_results: int) -> list[dict]:
        response = self._get_session().get(
            "https://search.yahoo.com/search",
            params={"p": query},
            timeout=min(self.timeout, 8.0),
        )
        if response.status_code in {403, 429}:
            raise ProviderRateLimited(f"yahoo answered {response.status_code}")
        response.raise_for_status()
        rows = _parse_yahoo(response.text)
        if not rows:
            raise RuntimeError("yahoo returned no parsable results")
        return rows[:max_results]

    def _search_ddgs(
        self, query: str, max_results: int, backend: str = "auto"
    ) -> list[dict]:
        try:
            from ddgs import DDGS  # type: ignore
        except ImportError:
            try:
                from duckduckgo_search import DDGS  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "Install the web search dependency: pip install ddgs"
                ) from exc

        client = None
        try:
            try:
                client = DDGS(timeout=self.timeout)
            except TypeError:
                client = DDGS()
            try:
                rows = client.text(
                    query, max_results=max_results, backend=backend
                )
            except TypeError:
                # Older duckduckgo_search releases do not expose `backend`.
                rows = client.text(query, max_results=max_results)
        finally:
            if client is not None:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
        return list(rows or [])

    def _search_tavily(self, query: str, max_results: int) -> list[dict]:
        if not self.tavily_key:
            return []

        response = self._get_session().post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.tavily_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return list(payload.get("results", []) or [])

    def _fetch_page(self, url: str) -> str:
        import requests

        try:
            response = requests.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": _USER_AGENT},
            )
            response.raise_for_status()
        except Exception:  # noqa: BLE001 - page fetch is best effort
            return ""

        raw_html = response.text
        try:
            import trafilatura  # type: ignore

            extracted = trafilatura.extract(raw_html) or ""
        except Exception:
            text = re.sub(
                r"<script.*?</script>|<style.*?</style>",
                " ",
                raw_html,
                flags=re.S | re.I,
            )
            extracted = re.sub(r"<[^>]+>", " ", text)

        extracted = _clean(extracted)
        return extracted[: self.max_page_chars]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _truncate(self, query: str, max_query_length: Optional[int]) -> str:
        if not max_query_length:
            return query
        return " ".join(str(query).split()[: int(max_query_length)])

    def _is_excluded(self, url: str) -> bool:
        host = urlparse(url).netloc.lower().split(":")[0]
        if host.startswith("www."):
            host = host[4:]
        for domain in self.exclude_domains:
            if host == domain or host.endswith("." + domain):
                return True
        return False

    def _respect_rate_limit(self) -> None:
        if self.min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.monotonic()

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(min(8.0, 1.5**attempt))

    def _cache_key(self, query: str, topk: int) -> str:
        payload = json.dumps(
            {
                "backend": self.backend,
                "query": query,
                "topk": topk,
                "exclude": sorted(self.exclude_domains),
                "fetch_pages": self.fetch_pages,
                "max_chars": self.max_chars,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def _cache_read(self, key: str) -> Optional[list[dict]]:
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            rows = payload.get("results")
            return rows if isinstance(rows, list) else None
        except (OSError, ValueError):
            return None

    def _cache_write(
        self, key: str, query: str, results: list[RetrievalResult]
    ) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        payload = {
            "query": query,
            "backend": self.backend,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "results": [
                {
                    "doc_id": row.doc_id,
                    "text": row.text,
                    "score": row.score,
                    "source": row.source,
                    "url": row.url,
                    "title": row.title,
                }
                for row in results
            ],
        }
        path = self._cache_path(key)
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            os.replace(tmp_path, path)
        except OSError as exc:
            print(f"[web] cache write failed: {exc}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
