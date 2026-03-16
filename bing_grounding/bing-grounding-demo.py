"""
Bing Grounding Agent Demo — Query Runner
Sends queries to a pre-existing Foundry Agent Service agent with Bing
grounding and measures latency, search consistency, and citation quality.

The agent must already exist — run setup-bing-agent.py first.

Designed to address specific customer concerns:
  - LATENCY: Labels first request as potential cold-start, computes warm stats.
  - CONSISTENCY: Tracks search invocation + citation rates across all queries.
  - CODE-FIRST: No portal UI required — pure SDK, works behind private networks.
  - GA ONLY: Uses only GA features (BingGroundingTool + agent_reference).

Auth: Entra ID via DefaultAzureCredential.
Required env vars (or .env file):
    FOUNDRY_PROJECT_ENDPOINT
    BING_AGENT_NAME              (default: bing-grounding-demo)
"""

import os
import sys
import time
import textwrap
import statistics
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
AGENT_NAME = os.environ.get("BING_AGENT_NAME", "bing-grounding-demo")

# Queries that *require* fresh web data — the model cannot answer from
# training data alone, which forces consistent tool invocation.
DEMO_QUERIES = [
    "What were the results of the most recent Formula 1 Grand Prix?",
    "Compare the latest GDP growth forecasts for the US, EU, and China for 2026.",
    "What are the biggest AI announcements from the past week?",
    "What is the current price of Bitcoin and how has it changed in the last 24 hours?",
]

OUTPUT_PREVIEW_LEN = 600  # chars to show per response
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, doubles each retry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_openai_client():
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
                    citations.append({
                        "url": getattr(ann, "url", ""),
                        "title": getattr(ann, "title", ""),
                    })
    return citations


def had_search_call(response) -> bool:
    """Check whether the agent actually invoked a Bing search.

    The agent service emits various output item types; we look for any
    that indicate a tool/search invocation.
    """
    for item in (response.output or []):
        item_type = getattr(item, "type", "")
        if "tool" in item_type or "search" in item_type or "bing" in item_type:
            return True
    return False


def run_query(openai_client, query: str) -> dict:
    """Send a query via the agent reference and return structured results.

    Uses tool_choice="required" to force the agent to invoke Bing Search,
    addressing the customer's concern about inconsistent tool invocation.

    Retries up to MAX_RETRIES times with exponential backoff on transient errors.
    """
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.perf_counter()
            response = openai_client.responses.create(
                input=query,
                tool_choice="required",
                extra_body={
                    "agent_reference": {
                        "name": AGENT_NAME,
                        "type": "agent_reference",
                    }
                },
            )
            latency = time.perf_counter() - t0

            citations = extract_citations(response)
            searched = had_search_call(response)

            return {
                "query": query,
                "latency_s": round(latency, 2),
                "output_text": response.output_text or "(no text returned)",
                "citations": citations,
                "search_invoked": searched or len(citations) > 0,
            }
        except Exception as e:
            last_exc = e
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            print(f"    ⚠ Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                print(f"    Retrying in {wait}s …")
                time.sleep(wait)

    # All retries exhausted
    raise last_exc  # type: ignore[misc]


def print_result(result: dict, is_first: bool) -> None:
    """Pretty-print a single result with latency context."""
    text = result["output_text"]
    preview = text[:OUTPUT_PREVIEW_LEN]
    if len(text) > OUTPUT_PREVIEW_LEN:
        preview += " … [truncated]"

    label = "  ← cold start?" if is_first else ""
    print(f"  Latency:         {result['latency_s']}s{label}")
    print(f"  Search invoked:  {'✓' if result['search_invoked'] else '✗ (no search detected!)'}")
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


def print_summary(all_results: list[dict]) -> None:
    """Print a summary table plus consistency & latency analysis."""
    q_w, lat_w, cit_w, srch_w = 50, 12, 10, 10
    header = f"{'Query':<{q_w}} {'Latency':<{lat_w}} {'Citations':<{cit_w}} {'Searched':<{srch_w}}"
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("SUMMARY")
    print("=" * len(header))
    print(header)
    print(sep)

    for r in all_results:
        q_display = r["query"][:q_w - 3] + "..." if len(r["query"]) > q_w else r["query"]
        srch = "✓" if r["search_invoked"] else "✗"
        print(f"{q_display:<{q_w}} {r['latency_s']:<{lat_w}} {len(r['citations']):<{cit_w}} {srch:<{srch_w}}")
    print(sep)

    # ----- Latency analysis -----
    latencies = [r["latency_s"] for r in all_results if r["latency_s"] > 0]
    if latencies:
        print(f"\n  Latency breakdown:")
        print(f"    First request:    {latencies[0]}s")
        if len(latencies) > 1:
            warm = latencies[1:]
            print(f"    Warm (n={len(warm)}):        "
                  f"mean={statistics.mean(warm):.2f}s  "
                  f"median={statistics.median(warm):.2f}s  "
                  f"min={min(warm):.2f}s  max={max(warm):.2f}s")
            overhead = latencies[0] - statistics.mean(warm)
            if overhead > 0:
                print(f"    Cold-start Δ:     +{overhead:.2f}s vs warm mean")

    # ----- Consistency analysis -----
    total = len(all_results)
    searched = sum(1 for r in all_results if r["search_invoked"])
    cited = sum(1 for r in all_results if len(r["citations"]) > 0)
    print(f"\n  Consistency:")
    print(f"    Search invoked:  {searched}/{total} ({searched/total*100:.0f}%)")
    print(f"    Citations found: {cited}/{total} ({cited/total*100:.0f}%)")
    if searched == total and cited == total:
        print(f"    ✓ All queries used search and returned citations.")
    else:
        if searched < total:
            print(f"    ⚠ {total - searched} query(ies) did NOT invoke search.")
        if cited < total:
            print(f"    ⚠ {total - cited} query(ies) returned NO citations.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outfile = Path("output") / f"bing-grounding-{stamp}.txt"

    with Tee(outfile):
        print("=" * 70)
        print("Bing Grounding Agent Demo  (GA – Foundry Agent Service)")
        print("=" * 70)
        print(f"  Agent:     {AGENT_NAME}")
        print(f"  Endpoint:  {ENDPOINT}")
        print(f"  Queries:   {len(DEMO_QUERIES)}")
        print(f"  Approach:  Responses API + agent_reference (code-first, no portal)")
        print("=" * 70)
        print(f"\n  Tip: Run this script multiple times to compare cold-start vs")
        print(f"  warm latency. Create the agent first with: setup-bing-agent.py\n")

        openai_client = get_openai_client()
        all_results: list[dict] = []

        for i, query in enumerate(DEMO_QUERIES, 1):
            print(f"{'─' * 70}")
            print(f"Query {i}/{len(DEMO_QUERIES)}: {query}")
            print(f"{'─' * 70}")

            try:
                result = run_query(openai_client, query)
                all_results.append(result)
                print_result(result, is_first=(i == 1))
            except Exception as e:
                print(f"  ✗ Error: {e}\n")
                all_results.append({
                    "query": query,
                    "latency_s": 0,
                    "output_text": f"ERROR: {e}",
                    "citations": [],
                    "search_invoked": False,
                })

        print_summary(all_results)

    print(f"\nOutput saved to {outfile}")


if __name__ == "__main__":
    main()
