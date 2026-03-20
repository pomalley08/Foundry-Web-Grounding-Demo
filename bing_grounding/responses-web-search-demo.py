"""
Web Search Responses API Demo  (GA)
Compare non-reasoning vs reasoning web search modes using the Responses API
web_search tool.

Uses the Azure AI Foundry Responses API with the web_search tool.
Auth: Entra ID via DefaultAzureCredential.
Required env vars (or .env file):
    FOUNDRY_PROJECT_ENDPOINT
    NON_REASONING_MODEL          (default: gpt-4.1)
    REASONING_MODEL              (default: gpt-5-mini)
"""

import os
import sys
import time
import textwrap
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

load_dotenv()


# ---------------------------------------------------------------------------
# Tee — write to both stdout and a log file
# ---------------------------------------------------------------------------
class Tee:
    """Context manager that duplicates stdout to a file."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self._file = None
        self._original_stdout = None

    def __enter__(self):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.filepath, "w", encoding="utf-8")
        self._original_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *args):
        sys.stdout = self._original_stdout
        self._file.close()

    def write(self, data):
        self._original_stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._original_stdout.flush()
        self._file.flush()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]

NON_REASONING_MODEL = os.environ.get("NON_REASONING_MODEL", "gpt-4.1")
REASONING_MODEL = os.environ.get("REASONING_MODEL", "gpt-5-mini")

DEMO_QUERIES = [
    "What were the results of the most recent Formula 1 Grand Prix?",
    "Compare the latest GDP growth forecasts for the US, EU, and China for 2026.",
    "What are the biggest AI announcements from the past week?",
    "What is the current price of Bitcoin and how has it changed in the last 24 hours?",
]

WEB_SEARCH_TOOL = {"type": "web_search"}

OUTPUT_PREVIEW_LEN = 500  # chars to show per response
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, doubles each retry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_client() -> object:
    """Create an OpenAI client via the azure-ai-projects SDK."""
    project_client = AIProjectClient(
        endpoint=ENDPOINT,
        credential=DefaultAzureCredential(),
    )
    return project_client.get_openai_client()


def extract_citations(response) -> list[dict]:
    """Pull url_citation annotations from the response output items."""
    citations = []
    for item in (response.output or []):
        content = getattr(item, "content", None)
        if not content:
            continue
        for block in content:
            for ann in (getattr(block, "annotations", None) or []):
                if getattr(ann, "type", None) == "url_citation":
                    citations.append({"url": ann.url, "title": getattr(ann, "title", "")})
    return citations


def count_search_calls(response) -> int:
    """Count how many web_search_call items the model issued."""
    return sum(1 for item in (response.output or []) if getattr(item, "type", None) == "web_search_call")


def run_web_search(client, model: str, query: str) -> dict:
    """Send a query with web search and return structured results.

    Retries up to MAX_RETRIES times with exponential backoff on transient errors.
    """
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.perf_counter()
            response = client.responses.create(
                model=model,
                tools=[WEB_SEARCH_TOOL],
                input=query,
            )
            latency = time.perf_counter() - t0

            return {
                "model": model,
                "query": query,
                "latency_s": round(latency, 2),
                "output_text": response.output_text or "(no text returned)",
                "citations": extract_citations(response),
                "search_count": count_search_calls(response),
            }
        except Exception as e:
            last_exc = e
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            print(f"    ⚠ Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                print(f"    Retrying in {wait}s …")
                time.sleep(wait)
    # All retries exhausted – re-raise so the caller's except block handles it
    raise last_exc  # type: ignore[misc]


def print_result(result: dict) -> None:
    """Pretty-print a single result."""
    text = result["output_text"]
    preview = text[:OUTPUT_PREVIEW_LEN]
    if len(text) > OUTPUT_PREVIEW_LEN:
        preview += " … [truncated]"

    print(f"  Model:           {result['model']}")
    print(f"  Latency:         {result['latency_s']}s")
    print(f"  Search calls:    {result['search_count']}")
    print(f"  Citations ({len(result['citations'])}):")
    for c in result["citations"][:5]:
        print(f"    - {c['title'][:80]}")
        print(f"      {c['url']}")
    if len(result["citations"]) > 5:
        print(f"    … and {len(result['citations']) - 5} more")
    print(f"  Response preview:")
    for line in textwrap.wrap(preview, width=100):
        print(f"    {line}")
    print()


def print_summary_table(results: list[dict]) -> None:
    """Print a comparison table after all queries have run."""
    # Column widths
    q_w, m_w, lat_w, sc_w, cit_w = 50, 12, 10, 8, 10
    header = (
        f"{'Query':<{q_w}} {'Model':<{m_w}} {'Latency':<{lat_w}} {'Searches':<{sc_w}} {'Citations':<{cit_w}}"
    )
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("SUMMARY")
    print("=" * len(header))
    print(header)
    print(sep)

    # Group by query
    queries_seen: list[str] = []
    for r in results:
        if r["query"] not in queries_seen:
            queries_seen.append(r["query"])

    for query in queries_seen:
        q_results = [r for r in results if r["query"] == query]
        for i, r in enumerate(q_results):
            q_display = query[:q_w - 3] + "..." if len(query) > q_w else query
            if i > 0:
                q_display = ""  # don't repeat query text on 2nd row
            print(
                f"{q_display:<{q_w}} {r['model']:<{m_w}} {r['latency_s']:<{lat_w}} "
                f"{r['search_count']:<{sc_w}} {len(r['citations']):<{cit_w}}"
            )

        # Print latency delta
        if len(q_results) == 2:
            delta = q_results[1]["latency_s"] - q_results[0]["latency_s"]
            sign = "+" if delta >= 0 else ""
            print(f"{'':>{q_w}} {'Δ latency:':<{m_w}} {sign}{delta:.2f}s")
        print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outfile = Path("output") / f"responses-web-search-{stamp}.txt"

    with Tee(outfile):
        print("=" * 70)
        print("Web Search Responses API Demo (GA)")
        print(f"Non-reasoning model: {NON_REASONING_MODEL}")
        print(f"Reasoning model:     {REASONING_MODEL}")
        print(f"Queries:             {len(DEMO_QUERIES)}")
        print("=" * 70)

        client = get_client()
        all_results: list[dict] = []

        for i, query in enumerate(DEMO_QUERIES, 1):
            print(f"\n{'─' * 70}")
            print(f"Query {i}/{len(DEMO_QUERIES)}: {query}")
            print(f"{'─' * 70}")

            for model in [NON_REASONING_MODEL, REASONING_MODEL]:
                print(f"\n  ▶ Running with {model} …")
                try:
                    result = run_web_search(client, model, query)
                    all_results.append(result)
                    print_result(result)
                except Exception as e:
                    print(f"  ✗ Error with {model}: {e}\n")
                    all_results.append({
                        "model": model,
                        "query": query,
                        "latency_s": 0,
                        "output_text": f"ERROR: {e}",
                        "citations": [],
                        "search_count": 0,
                    })

        print_summary_table(all_results)

    print(f"\nOutput saved to {outfile}")


if __name__ == "__main__":
    main()
