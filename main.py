"""AG2 multi-agent research pipeline for trusted journals and universities.

This version adapts the simple AG2 example into a staged workflow:

1. A search agent looks for yesterday's research across curated sources.
2. A review agent reads candidate articles, scores them, and filters them.
3. A report agent writes a top-N research brief with source links.

The script uses a small terminal UI so the user's interests can change per run
without touching code. It exports both Markdown and HTML reports.
"""

from __future__ import annotations

import asyncio
import ast
import html
import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tavily import TavilyClient

from autogen.beta import Agent, MemoryStream, tool
from autogen.beta.config import GeminiConfig, OpenAIConfig
from autogen.beta.config.config import ModelConfig
from autogen.beta.events import TaskCompleted, TaskFailed, TaskStarted, ToolCallEvent, ToolErrorEvent

DEFAULT_INTERESTS = ["quantum", "AI", "image processing"]
DEFAULT_TOP_N = 10
DEFAULT_SEARCH_RESULTS = 6
DEFAULT_CANDIDATE_LIMIT = 20
DEFAULT_FETCH_LIMIT = 12
REPORTS_DIR = Path("reports")

TRUSTED_SOURCES = {
    "journals": [
        "nature.com",
        "science.org",
        "cell.com",
        "pnas.org",
        "ieee.org",
        "ieeexplore.ieee.org",
        "acm.org",
        "sciencedirect.com",
        "springer.com",
        "thecvf.com",
        "openreview.net",
        "arxiv.org",
    ],
    "universities": [
        "mit.edu",
        "stanford.edu",
        "berkeley.edu",
        "cmu.edu",
        "caltech.edu",
        "harvard.edu",
        "ox.ac.uk",
        "cam.ac.uk",
        "ethz.ch",
        "epfl.ch",
    ],
}

_PROVIDER_DEFAULTS = {
    "gemini": {
        "search": "gemini-2.5-flash",
        "review": "gemini-2.5-pro",
        "report": "gemini-2.5-pro",
        "env": "GEMINI_API_KEY",
    },
    "openai": {
        "search": "gpt-4o-mini",
        "review": "gpt-4o",
        "report": "gpt-4o",
        "env": "OPENAI_API_KEY",
    },
}


@dataclass(slots=True)
class ResearchPreferences:
    interests: list[str]
    target_date: date
    top_n: int
    search_results_per_query: int
    output_dir: Path

    @property
    def date_label(self) -> str:
        return f"{self.target_date.strftime('%B')} {self.target_date.day}, {self.target_date.year}"

    @property
    def filename_stem(self) -> str:
        joined = "_".join(_slugify(item) for item in self.interests[:3]) or "research"
        return f"research_brief_{self.target_date.isoformat()}_{joined}"


def build_config(role: str) -> ModelConfig:
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    if provider not in _PROVIDER_DEFAULTS:
        raise SystemExit(
            f"LLM_PROVIDER must be one of {list(_PROVIDER_DEFAULTS)}; got {provider!r}"
        )

    defaults = _PROVIDER_DEFAULTS[provider]
    override_var = f"{role.upper()}_MODEL"
    model = os.environ.get(override_var, defaults[role])
    api_key = _resolve_provider_api_key(provider)

#    if provider == "openai":
#        return OpenAIConfig(model=model, api_key=api_key)
#    return GeminiConfig(model=model, api_key=api_key)
    return OpenAIConfig(model=model, streaming = True,
        base_url="https://openrouter.ai/api/v1",
        max_completion_tokens=16384)

@tool
async def get_trusted_sources() -> dict[str, list[str]]:
    """Return the curated journal and university domains the search agent should prioritize."""
    return TRUSTED_SOURCES


@tool
async def curated_web_search(
    query: str,
    allowed_domains_csv: str = "",
    max_results: int = DEFAULT_SEARCH_RESULTS,
) -> list[dict[str, Any]]:
    """Search the web and bias results toward approved domains."""
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    domains = [item.strip() for item in allowed_domains_csv.split(",") if item.strip()]
    scoped_query = query
    if domains:
        domain_clause = " OR ".join(f"site:{domain}" for domain in domains)
        scoped_query = f"({domain_clause}) {query}"

    response = await asyncio.to_thread(
        client.search,
        query=scoped_query,
        max_results=max_results,
        topic="news",
        search_depth="advanced",
    )

    records = []
    for result in response.get("results", []):
        records.append(
            {
                "title": result.get("title"),
                "url": result.get("url"),
                "content": result.get("content"),
                "score": result.get("score"),
                "published_date": result.get("published_date") or result.get("published_at"),
            }
        )
    return records


@tool
async def fetch_url(url: str) -> str:
    """Extract readable page content from a URL."""
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = await asyncio.to_thread(client.extract, urls=url)
    results = response.get("results", [])
    if not results:
        return f"[extract returned no content for {url}]"
    return results[0].get("raw_content") or results[0].get("content") or ""


class LaneRouter:
    """Print stage progress for each AG2 stream."""

    def __init__(self) -> None:
        self._lane_widths: dict[str, int] = {}

    def attach(self, stream: MemoryStream, label: str) -> None:
        self._lane_widths.setdefault(label, max(6, len(label)))

        @stream.where(ToolCallEvent).subscribe
        async def _on_tool_call(event: ToolCallEvent) -> None:
            args = _parse_args(event.arguments)
            if event.name == "curated_web_search":
                query = args.get("query", "")
                self._print(label, f"search -> {query[:110]}")
            elif event.name == "fetch_url":
                self._print(label, f"fetch -> {args.get('url', '')}")
            elif event.name == "get_trusted_sources":
                self._print(label, "load trusted source list")
            else:
                self._print(label, f"tool -> {event.name}")

        @stream.where(ToolErrorEvent).subscribe
        async def _on_tool_error(event: ToolErrorEvent) -> None:
            self._print(label, f"tool error -> {event.name}: {event.error!r}")

        @stream.where(TaskStarted).subscribe
        async def _on_task_started(event: TaskStarted) -> None:
            self._print(label, f"start -> {event.agent_name}")

        @stream.where(TaskCompleted).subscribe
        async def _on_task_completed(event: TaskCompleted) -> None:
            size = len(event.result or "")
            self._print(label, f"done -> {event.agent_name} ({size} chars)")

        @stream.where(TaskFailed).subscribe
        async def _on_task_failed(event: TaskFailed) -> None:
            self._print(label, f"failed -> {event.agent_name}: {event.error!r}")

    def _print(self, label: str, message: str) -> None:
        width = self._lane_widths.get(label, 6)
        print(f"[{label:<{width}}] {message}", flush=True)


SEARCH_AGENT_PROMPT = """
You are a research search specialist.

Your job is to find candidate articles published on the requested date from
notable and reliable journals or universities.

Workflow:
1. Call `get_trusted_sources` first.
2. Search both journal and university domains for each interest.
3. Prefer official journal articles, institutional research announcements,
   conference proceedings, and arXiv/OpenReview entries tied to serious labs.
4. Deduplicate near-identical URLs or mirrored coverage.
5. Return JSON only. No markdown. No commentary outside JSON.

Return this exact JSON shape:
{
  "search_summary": "one paragraph",
  "articles": [
    {
      "title": "string",
      "url": "string",
      "source": "domain or publication name",
      "published_date": "YYYY-MM-DD or unknown",
      "interest": "which user interest it best matches",
      "why_candidate": "one sentence",
      "source_type": "journal or university",
      "search_query": "query used"
    }
  ]
}

Constraints:
- Focus on the requested date.
- Keep at most 20 articles.
- Exclude generic blogs, marketing pages, and pages with no research substance.
""".strip()

REVIEW_AGENT_PROMPT = """
You are a strict research reviewer.

Input will include user interests, a target date, and candidate article metadata.
Read the strongest candidates with `fetch_url`, then decide which ones are factual,
insightful, and relevant.

Scoring rules:
- factuality_score: reliability of the source and whether the page contains concrete research details.
- insight_score: novelty, technical substance, and likely value to an informed reader.
- relevance_score: fit to the user's interests.
- overall_score: your weighted final score on a 1-10 scale.

Decision rules:
- Reject anything clearly off-date, thin, duplicated, broken, or weakly sourced.
- Prefer primary sources over commentary.
- Keep at most the requested top_n articles.
- Return JSON only.

Return this exact JSON shape:
{
  "review_summary": "one paragraph",
  "top_articles": [
    {
      "rank": 1,
      "title": "string",
      "url": "string",
      "source": "string",
      "published_date": "YYYY-MM-DD or unknown",
      "matched_interests": ["AI"],
      "factuality_score": 9,
      "insight_score": 8,
      "relevance_score": 10,
      "overall_score": 9.1,
      "summary": "2-4 sentence factual summary",
      "why_it_matters": "1-2 sentences",
      "evidence_notes": "why you trust it or what caveat exists"
    }
  ],
  "rejected_articles": [
    {
      "title": "string",
      "url": "string",
      "reason": "string"
    }
  ]
}

Use `fetch_url` on the most promising items first. You do not need to fetch every single candidate.
""".strip()

REPORT_AGENT_PROMPT = """
You are a research briefing writer.

You will receive reviewed article JSON that already includes scores and summaries.
Write a polished markdown report for a busy technical reader.

Report requirements:
- Title with the target date and user interests.
- Short overview of what stood out across the set.
- A Top Articles section with ranked entries.
- For each entry include: title, source, published date, matched interests,
  a compact summary, why it matters, the score breakdown, and a markdown source link.
- End with a short "Notable Patterns" section.

Keep the writing factual, compact, and free of hype.
Return markdown only.
""".strip()


def build_search_agent() -> Agent:
    return Agent(
        name="search_agent",
        prompt=SEARCH_AGENT_PROMPT,
        config=build_config("search"),
        tools=[get_trusted_sources, curated_web_search],
    )


def build_review_agent() -> Agent:
    return Agent(
        name="review_agent",
        prompt=REVIEW_AGENT_PROMPT,
        config=build_config("review"),
        tools=[fetch_url],
    )


def build_report_agent() -> Agent:
    return Agent(
        name="report_agent",
        prompt=REPORT_AGENT_PROMPT,
        config=build_config("report"),
        tools=[],
    )


async def run_pipeline(preferences: ResearchPreferences) -> tuple[str, dict[str, Any], dict[str, Path]]:
    router = LaneRouter()

    print("\nRunning research pipeline...\n", flush=True)

    search_stream = MemoryStream()
    router.attach(search_stream, "search")
    search_agent = build_search_agent()
    search_prompt = build_search_prompt(preferences)
    search_reply = await search_agent.ask(search_prompt, stream=search_stream)
    search_payload = _load_json_payload(search_reply.body)

    review_stream = MemoryStream()
    router.attach(review_stream, "review")
    review_agent = build_review_agent()
    review_prompt = build_review_prompt(preferences, search_payload)
    review_reply = await review_agent.ask(review_prompt, stream=review_stream)
    review_payload = _load_json_payload(review_reply.body)

    report_stream = MemoryStream()
    router.attach(report_stream, "report")
    report_agent = build_report_agent()
    report_prompt = build_report_prompt(preferences, review_payload)
    report_reply = await report_agent.ask(report_prompt, stream=report_stream)
    report_markdown = report_reply.body.strip()

    output_paths = save_reports(preferences, review_payload, report_markdown)
    return report_markdown, review_payload, output_paths


def build_search_prompt(preferences: ResearchPreferences) -> str:
    interest_list = ", ".join(preferences.interests)
    return f"""
Target date: {preferences.target_date.isoformat()} ({preferences.date_label})
Interests: {interest_list}
Search results per query: {preferences.search_results_per_query}
Goal: Find strong research outputs from yesterday across trusted journals and universities.

Use separate searches for each interest across both journals and universities.
Search with date-aware phrasing like:
- "<interest> research published {preferences.date_label}"
- "<interest> new paper {preferences.date_label}"
- "<interest> university research announcement {preferences.date_label}"

Return at most {DEFAULT_CANDIDATE_LIMIT} deduplicated candidates.
""".strip()


def build_review_prompt(preferences: ResearchPreferences, search_payload: dict[str, Any]) -> str:
    interest_list = ", ".join(preferences.interests)
    candidates_json = json.dumps(search_payload, indent=2)
    return f"""
User interests: {interest_list}
Target date: {preferences.target_date.isoformat()} ({preferences.date_label})
top_n: {preferences.top_n}
Fetch at most {DEFAULT_FETCH_LIMIT} URLs.

Candidate articles JSON:
{candidates_json}

Score and rank the best articles. Keep only the strongest final set.
""".strip()


def build_report_prompt(preferences: ResearchPreferences, review_payload: dict[str, Any]) -> str:
    interest_list = ", ".join(preferences.interests)
    reviewed_json = json.dumps(review_payload, indent=2)
    return f"""
Target date: {preferences.target_date.isoformat()} ({preferences.date_label})
User interests: {interest_list}
Top N requested: {preferences.top_n}

Reviewed articles JSON:
{reviewed_json}
""".strip()


def collect_preferences() -> ResearchPreferences:
    default_date = date.today() - timedelta(days=1)

    raw_interests = input(
        f"Interests (comma-separated) [{', '.join(DEFAULT_INTERESTS)}]: "
    ).strip()
    if raw_interests:
        interests = [item.strip() for item in raw_interests.split(",") if item.strip()]
    else:
        interests = DEFAULT_INTERESTS.copy()

    raw_date = input(f"Research date YYYY-MM-DD [{default_date.isoformat()}]: ").strip()
    target_date = default_date if not raw_date else datetime.strptime(raw_date, "%Y-%m-%d").date()

    raw_top_n = input(f"Top article count [{DEFAULT_TOP_N}]: ").strip()
    top_n = DEFAULT_TOP_N if not raw_top_n else max(1, int(raw_top_n))

    raw_results = input(
        f"Search results per query [{DEFAULT_SEARCH_RESULTS}]: "
    ).strip()
    search_results = DEFAULT_SEARCH_RESULTS if not raw_results else max(3, int(raw_results))

    output_dir = REPORTS_DIR
    print(
        f"\nUsing interests={', '.join(interests)} | date={target_date.isoformat()} | top_n={top_n}",
        flush=True,
    )

    return ResearchPreferences(
        interests=interests,
        target_date=target_date,
        top_n=top_n,
        search_results_per_query=search_results,
        output_dir=output_dir,
    )


def save_reports(
    preferences: ResearchPreferences,
    review_payload: dict[str, Any],
    report_markdown: str,
) -> dict[str, Path]:
    preferences.output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = preferences.output_dir / f"{preferences.filename_stem}.md"
    html_path = preferences.output_dir / f"{preferences.filename_stem}.html"

    markdown_path.write_text(report_markdown + "\n", encoding="utf-8")
    html_path.write_text(render_html_report(preferences, review_payload), encoding="utf-8")

    return {"markdown": markdown_path.resolve(), "html": html_path.resolve()}


def render_html_report(preferences: ResearchPreferences, review_payload: dict[str, Any]) -> str:
    cards = []
    for article in review_payload.get("top_articles", []):
        matched = ", ".join(article.get("matched_interests", [])) or "general"
        cards.append(
            f"""
            <article class="card">
              <div class="rank">#{html.escape(str(article.get("rank", "?")))}</div>
              <h2>{html.escape(article.get("title", "Untitled"))}</h2>
              <p class="meta">{html.escape(article.get("source", "Unknown source"))} | {html.escape(article.get("published_date", "unknown"))}</p>
              <p><strong>Matched interests:</strong> {html.escape(matched)}</p>
              <p>{html.escape(article.get("summary", ""))}</p>
              <p><strong>Why it matters:</strong> {html.escape(article.get("why_it_matters", ""))}</p>
              <p><strong>Scores:</strong> factuality {html.escape(str(article.get("factuality_score", "")))}, insight {html.escape(str(article.get("insight_score", "")))}, relevance {html.escape(str(article.get("relevance_score", "")))}, overall {html.escape(str(article.get("overall_score", "")))}</p>
              <p><strong>Evidence notes:</strong> {html.escape(article.get("evidence_notes", ""))}</p>
              <p><a href="{html.escape(article.get("url", "#"))}">Source link</a></p>
            </article>
            """.strip()
        )

    overview = html.escape(review_payload.get("review_summary", "No review summary generated."))
    interests = html.escape(", ".join(preferences.interests))
    report_date = html.escape(preferences.date_label)

    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Research Brief - {report_date}</title>
    <style>
      :root {{
        --bg: #f4efe7;
        --panel: #fffdf8;
        --ink: #182126;
        --muted: #58656d;
        --accent: #b24c2f;
        --line: #d9cec1;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: Georgia, "Iowan Old Style", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top right, rgba(178, 76, 47, 0.14), transparent 30%),
          linear-gradient(180deg, #f8f3ec 0%, var(--bg) 100%);
      }}
      main {{
        max-width: 960px;
        margin: 0 auto;
        padding: 48px 20px 72px;
      }}
      h1 {{
        margin-bottom: 8px;
        font-size: clamp(2rem, 4vw, 3.5rem);
      }}
      .deck {{
        color: var(--muted);
        font-size: 1.05rem;
        margin-bottom: 28px;
      }}
      .summary {{
        background: var(--panel);
        border: 1px solid var(--line);
        padding: 20px;
        border-radius: 18px;
        margin-bottom: 24px;
      }}
      .grid {{
        display: grid;
        gap: 18px;
      }}
      .card {{
        background: rgba(255, 253, 248, 0.92);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(24, 33, 38, 0.06);
      }}
      .rank {{
        color: var(--accent);
        font-weight: 700;
        margin-bottom: 6px;
      }}
      .meta {{
        color: var(--muted);
      }}
      a {{
        color: var(--accent);
      }}
    </style>
  </head>
  <body>
    <main>
      <h1>Top Research Brief</h1>
      <p class="deck">{report_date} | interests: {interests}</p>
      <section class="summary">
        <h2>Overview</h2>
        <p>{overview}</p>
      </section>
      <section class="grid">
        {''.join(cards) if cards else '<p>No top articles were returned.</p>'}
      </section>
    </main>
  </body>
</html>
"""


def _load_json_payload(text: str) -> dict[str, Any]:
    candidate = text.strip()
    possible_payloads = [candidate]

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
    if fence_match:
        possible_payloads.insert(0, fence_match.group(1).strip())

    for block in _extract_json_objects(candidate):
        possible_payloads.append(block)

    seen: set[str] = set()
    for payload in possible_payloads:
        normalized = payload.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        try:
            parsed = json.loads(normalized)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(normalized)
            except (SyntaxError, ValueError):
                continue
        if isinstance(parsed, dict):
            return _normalize_agent_payload(parsed)

    preview = text.strip().replace("\n", " ")
    if len(preview) > 240:
        preview = preview[:237] + "..."
    raise ValueError(f"Agent did not return valid JSON object. Preview: {preview}")


def _extract_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    start = -1
    depth = 0
    in_string = False
    escape = False

    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            if depth == 0:
                start = index
            depth += 1
            continue

        if char == "}" and depth:
            depth -= 1
            if depth == 0 and start != -1:
                objects.append(text[start:index + 1])
                start = -1

    return objects


def _normalize_agent_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "articles" in payload or "top_articles" in payload or "review_summary" in payload:
        return payload

    tool_result = _unwrap_tool_result(payload)
    if isinstance(tool_result, list):
        articles = [_coerce_search_article(item) for item in tool_result]
        articles = [item for item in articles if item]
        if articles:
            return {
                "search_summary": "Recovered candidate articles directly from tool output.",
                "articles": articles,
            }

    if isinstance(tool_result, dict):
        nested_result = tool_result.get("result")
        if isinstance(nested_result, list):
            articles = [_coerce_search_article(item) for item in nested_result]
            articles = [item for item in articles if item]
            if articles:
                return {
                    "search_summary": "Recovered candidate articles directly from nested tool output.",
                    "articles": articles,
                }

    return payload


def _unwrap_tool_result(payload: dict[str, Any]) -> Any:
    for key, value in payload.items():
        if key.startswith("tool_") and key.endswith("_response"):
            if isinstance(value, dict) and "result" in value:
                return value["result"]
            return value
    return payload


def _coerce_search_article(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    url = item.get("url")
    title = item.get("title")
    if not url or not title:
        return None

    source = item.get("source") or _domain_from_url(url)
    interest = item.get("interest") or "unspecified"
    published_date = item.get("published_date") or "unknown"
    source_type = item.get("source_type") or _infer_source_type(source)
    why_candidate = item.get("why_candidate") or _summarize_candidate_reason(item)
    search_query = item.get("search_query") or ""

    return {
        "title": title,
        "url": url,
        "source": source,
        "published_date": published_date,
        "interest": interest,
        "why_candidate": why_candidate,
        "source_type": source_type,
        "search_query": search_query,
    }


def _domain_from_url(url: str) -> str:
    match = re.search(r"https?://([^/]+)", url)
    return match.group(1) if match else "unknown"


def _infer_source_type(source: str) -> str:
    source = source.lower()
    if any(source.endswith(domain) for domain in TRUSTED_SOURCES["universities"]):
        return "university"
    return "journal"


def _summarize_candidate_reason(item: dict[str, Any]) -> str:
    content = (item.get("content") or "").strip()
    if not content:
        return "Recovered from tool output; requires review agent validation."
    snippet = re.sub(r"\s+", " ", content)
    if len(snippet) > 140:
        snippet = snippet[:137] + "..."
    return snippet


def _parse_args(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned or "topic"


def _require_env(name: str) -> None:
    if not os.environ.get(name):
        raise SystemExit(
            f"Missing {name}. Add it to .env or export it in your shell before running."
        )


def _resolve_provider_api_key(provider: str) -> str | None:
    if provider == "gemini":
        for candidate in (
            "GOOGLE_GEMINI_API_KEY",
            "GEMINI_API_KEY",
            "AG2_GEMINI_API_KEY",
        ):
            value = os.environ.get(candidate)
            if value:
                return value
        return None

    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY")

    return None


async def main() -> None:
    load_dotenv()

    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    if provider not in _PROVIDER_DEFAULTS:
        raise SystemExit(f"LLM_PROVIDER must be one of {list(_PROVIDER_DEFAULTS)}")

    if not _resolve_provider_api_key(provider):
        if provider == "gemini":
            raise SystemExit(
                "Missing Gemini API key. Set one of GOOGLE_GEMINI_API_KEY, "
                "GEMINI_API_KEY, or AG2_GEMINI_API_KEY in .env."
            )
        _require_env(_PROVIDER_DEFAULTS[provider]["env"])
    _require_env("TAVILY_API_KEY")

    print(
        f"[provider={provider}] "
        f"search={os.environ.get('SEARCH_MODEL', _PROVIDER_DEFAULTS[provider]['search'])} "
        f"review={os.environ.get('REVIEW_MODEL', _PROVIDER_DEFAULTS[provider]['review'])} "
        f"report={os.environ.get('REPORT_MODEL', _PROVIDER_DEFAULTS[provider]['report'])}",
        flush=True,
    )
    print(
        "Multi-agent research brief generator. Press Ctrl-D or type nothing to accept defaults.\n",
        flush=True,
    )

    preferences = collect_preferences()
    report_markdown, review_payload, output_paths = await run_pipeline(preferences)

    print("\n" + "=" * 72)
    print("TOP RESEARCH REPORT")
    print("=" * 72)
    print(report_markdown)
    print("=" * 72)
    print(
        f"Saved markdown report to: {output_paths['markdown']}\n"
        f"Saved HTML report to: {output_paths['html']}\n"
        f"Top articles returned: {len(review_payload.get('top_articles', []))}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
