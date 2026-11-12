import base64
import json

from src.retrieval.web import (
    WebRetriever,
    _decode_redirect,
    _parse_bing,
    _parse_ddg_lite,
    _parse_yahoo,
)


def _row(title, url, snippet):
    return {"title": title, "href": url, "body": snippet}


def _route_all_providers(retriever, search):
    """Send every provider through one fake search function."""
    retriever._search_ddg_lite = lambda query, max_results: search(
        query, max_results
    )
    retriever._search_bing = lambda query, max_results: search(query, max_results)
    retriever._search_yahoo = lambda query, max_results: search(query, max_results)
    retriever._search_ddgs = lambda query, max_results, backend="auto": search(
        query, max_results
    )


def test_cache_prevents_second_network_call(tmp_path):
    calls = {"n": 0}

    def fake_search(query, max_results):
        calls["n"] += 1
        return [
            _row("FLARE retrieval", "https://example.com/a", "flare first snippet"),
            _row("Beta", "https://other.org/b", "second snippet"),
        ]

    retriever = WebRetriever(
        backend="ddgs",
        cache_dir=str(tmp_path),
        min_interval=0,
        topk=1,
    )
    _route_all_providers(retriever, fake_search)

    first = retriever.retrieve(["what is flare"], topk=1)
    second = retriever.retrieve(["what is flare"], topk=1)

    assert calls["n"] == 1
    assert first == second
    assert first[0][0].source == "web"
    assert first[0][0].url == "https://example.com/a"
    assert retriever.stats["cache_hits"] == 1


def test_domain_exclusion_and_topk(tmp_path):
    def fake_search(query, max_results):
        return [
            _row("Wiki flare", "https://en.wikipedia.org/wiki/Flare", "wiki flare"),
            _row("Other flare", "https://example.org/page", "other flare"),
        ]

    retriever = WebRetriever(
        backend="ddgs",
        cache_dir=str(tmp_path),
        exclude_domains=["wikipedia.org"],
        min_interval=0,
    )
    _route_all_providers(retriever, fake_search)

    rows = retriever.retrieve(["flare"], topk=5)[0]
    assert len(rows) == 1
    assert rows[0].url == "https://example.org/page"


def test_max_query_length_truncates_before_search(tmp_path):
    seen = {}

    def fake_search(query, max_results):
        seen["query"] = query
        return [_row("One two", "https://a.example/x", "one two snippet")]

    retriever = WebRetriever(
        backend="ddgs", cache_dir=str(tmp_path), min_interval=0
    )
    _route_all_providers(retriever, fake_search)

    retriever.retrieve(["one two three four"], topk=1, max_query_length=2)
    assert seen["query"] == "one two"


def test_tavily_fallback_when_other_providers_fail(tmp_path):
    def failing_search(query, max_results):
        raise RuntimeError("blocked")

    def fake_tavily(query, max_results):
        return [
            {
                "title": "Tavily fallback",
                "url": "https://tavily.example/x",
                "content": "tavily fallback snippet",
                "score": 0.9,
            }
        ]

    retriever = WebRetriever(
        backend="auto",
        cache_dir=str(tmp_path),
        tavily_key="test-key",
        min_interval=0,
        max_retries=1,
    )
    _route_all_providers(retriever, failing_search)
    retriever._search_tavily = fake_tavily  # type: ignore[assignment]

    rows = retriever.retrieve(["tavily fallback"], topk=1)[0]
    assert rows[0].text.startswith("Tavily fallback")
    assert rows[0].score == 0.9


def test_cached_payload_is_valid_json(tmp_path):
    def fake_search(query, max_results):
        return [_row("Alpha snippet", "https://a.example/x", "snippet")]

    retriever = WebRetriever(
        backend="ddgs", cache_dir=str(tmp_path), min_interval=0
    )
    _route_all_providers(retriever, fake_search)
    retriever.retrieve(["alpha"], topk=1)

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["query"] == "alpha"
    assert payload["results"][0]["source"] == "web"


def test_parse_ddg_lite_pairs_titles_and_snippets():
    fixture = """
    <table>
      <tr><td>1.&nbsp;</td>
        <td><a rel="nofollow" href="https://example.org/a"
               class='result-link'>Alpha title</a></td></tr>
      <tr><td>&nbsp;</td>
        <td class='result-snippet'><b>Alpha</b> body text</td></tr>
      <tr><td>2.&nbsp;</td>
        <td><a rel="nofollow" href="https://example.org/b"
               class='result-link'>Beta title</a></td></tr>
      <tr><td>&nbsp;</td>
        <td class='result-snippet'>Beta body</td></tr>
    </table>
    """
    rows = _parse_ddg_lite(fixture)
    assert [row["title"] for row in rows] == ["Alpha title", "Beta title"]
    assert rows[0]["body"] == "Alpha body text"
    assert rows[0]["href"] == "https://example.org/a"


def test_parse_bing_extracts_h2_and_snippet():
    fixture = """
    <ol>
      <li class="b_algo">
        <h2><a href="https://example.org/a">Alpha title</a></h2>
        <div class="b_caption"><p>Alpha body text</p></div>
      </li>
      <li class="b_algo">
        <h2><a href="https://example.org/b">Beta <strong>title</strong></a></h2>
        <div class="b_caption"><p>Beta body</p></div>
      </li>
    </ol>
    """
    rows = _parse_bing(fixture)
    assert [row["title"] for row in rows] == ["Alpha title", "Beta title"]
    assert rows[0]["body"] == "Alpha body text"
    assert rows[0]["href"] == "https://example.org/a"


def test_decode_bing_redirect_url():
    target = "https://example.org/decoded"
    encoded = base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
    url = f"https://www.bing.com/ck/a?u=a1{encoded}&ntb=1"
    assert _decode_redirect(url) == target


def test_decode_ddg_redirect_url():
    url = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fx&rut=abc"
    assert _decode_redirect(url) == "https://example.org/x"


def test_parse_yahoo_extracts_title_and_snippet():
    fixture = """
    <ol class="reg searchCenterMiddle">
      <li class="first"><div class="dd algo algo-sr relsrch Sr">
        <div class="compTitle"><a href="https://r.search.yahoo.com/_ylt=x/RU=https%3a%2f%2fexample.org%2fa/RK=2">link</a>
          <h3 class="title"><span>Alpha title</span></h3></div>
        <div class="compText aAbs"><p>Alpha body text</p></div>
      </div></li>
      <li><div class="dd algo algo-sr relsrch Sr">
        <div class="compTitle"><a href="https://r.search.yahoo.com/_ylt=y/RU=https%3a%2f%2fexample.org%2fb/RK=2">link</a>
          <h3 class="title"><span>Beta title</span></h3></div>
        <div class="compText aAbs"><p>Beta body</p></div>
      </div></li>
    </ol>
    """
    rows = _parse_yahoo(fixture)
    assert [row["title"] for row in rows] == ["Alpha title", "Beta title"]
    assert rows[0]["href"] == "https://example.org/a"
    assert rows[0]["body"] == "Alpha body text"


def test_looks_relevant_rejects_unrelated_serp():
    retriever = WebRetriever(backend="ddgs", cache_dir="/tmp/unused", min_interval=0)
    garbage = [
        {"title": "Outlook help", "href": "https://zhihu.com/q/1", "body": "mail"}
    ]
    assert not retriever._looks_relevant("Who wrote Pride and Prejudice?", garbage)
    good = [
        {
            "title": "Pride and Prejudice - Wikipedia",
            "href": "https://en.wikipedia.org/wiki/Pride_and_Prejudice",
            "body": "novel by Jane Austen",
        }
    ]
    assert retriever._looks_relevant("Who wrote Pride and Prejudice?", good)
