"""
Bing Grounding Agent Demo
Demonstrates the GA approach for web grounding using a Bing Search resource
connected to a Foundry Agent Service agent.

Uses the azure-ai-projects SDK to create an agent with BingGroundingTool,
run queries via the Responses API with an agent_reference, and clean up.

Auth: Entra ID via DefaultAzureCredential.
Required env vars (or .env file):
    FOUNDRY_PROJECT_ENDPOINT
    FOUNDRY_MODEL_DEPLOYMENT_NAME
    BING_PROJECT_CONNECTION_NAME
"""

import os
import time
import textwrap

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    PromptAgentDefinition,
    BingGroundingTool,
    BingGroundingSearchToolParameters,
    BingGroundingSearchConfiguration,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
MODEL = os.environ["FOUNDRY_MODEL_DEPLOYMENT_NAME"]
BING_CONNECTION_NAME = os.environ["BING_PROJECT_CONNECTION_NAME"]

AGENT_NAME = "bing-grounding-demo"

DEMO_QUERIES = [
    "What were the results of the most recent Formula 1 Grand Prix?",
    "Compare the latest GDP growth forecasts for the US, EU, and China for 2026.",
    "What are the biggest AI announcements from the past week?",
    "What is the current price of Bitcoin and how has it changed in the last 24 hours?",
]

OUTPUT_PREVIEW_LEN = 500  # chars to show per response
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, doubles each retry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_clients():
    """Create project + OpenAI clients and resolve the Bing connection ID."""
    credential = DefaultAzureCredential()
    project_client = AIProjectClient(
        endpoint=ENDPOINT,
        credential=credential,
    )
    openai_client = project_client.get_openai_client()

    bing_connection = project_client.connections.get(BING_CONNECTION_NAME)
    print(f"Bing connection verified: {bing_connection.name}")
    print(f"Connection ID: {bing_connection.id}")

    return project_client, openai_client, bing_connection.id


def create_agent(project_client, bing_connection_id: str):
    """Create a versioned agent with the Bing grounding tool attached."""
    agent = project_client.agents.create_version(
        agent_name=AGENT_NAME,
        definition=PromptAgentDefinition(
            model=MODEL,
            instructions=(
                "You are a helpful assistant with access to Bing Search. "
                "Use the Bing grounding tool to find current information "
                "before answering. Always cite your sources."
            ),
            tools=[
                BingGroundingTool(
                    bing_grounding=BingGroundingSearchToolParameters(
                        search_configurations=[
                            BingGroundingSearchConfiguration(
                                project_connection_id=bing_connection_id,
                            )
                        ]
                    )
                )
            ],
        ),
        description="Demo agent for Bing-grounded web search.",
    )
    print(f"Agent created (id: {agent.id}, name: {agent.name}, version: {agent.version})")
    return agent


def cleanup_agent(project_client):
    """Delete the demo agent (all versions). Fails gracefully."""
    try:
        project_client.agents.delete(AGENT_NAME)
        print(f"Agent '{AGENT_NAME}' deleted.")
    except Exception as e:
        print(f"⚠ Could not delete agent '{AGENT_NAME}': {e}")


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


def run_query(openai_client, agent_name: str, query: str) -> dict:
    """Send a query via the agent reference and return structured results.

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
                        "name": agent_name,
                        "type": "agent_reference",
                    }
                },
            )
            latency = time.perf_counter() - t0

            return {
                "query": query,
                "latency_s": round(latency, 2),
                "output_text": response.output_text or "(no text returned)",
                "citations": extract_citations(response),
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


def print_result(result: dict) -> None:
    """Pretty-print a single result."""
    text = result["output_text"]
    preview = text[:OUTPUT_PREVIEW_LEN]
    if len(text) > OUTPUT_PREVIEW_LEN:
        preview += " … [truncated]"

    print(f"  Latency:         {result['latency_s']}s")
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Bing Grounding Agent Demo  (GA – Foundry Agent Service)")
    print(f"Model:   {MODEL}")
    print(f"Bing:    {BING_CONNECTION_NAME}")
    print(f"Queries: {len(DEMO_QUERIES)}")
    print("=" * 70)

    project_client, openai_client, bing_conn_id = get_clients()
    agent = create_agent(project_client, bing_conn_id)

    all_results: list[dict] = []

    try:
        for i, query in enumerate(DEMO_QUERIES, 1):
            print(f"\n{'─' * 70}")
            print(f"Query {i}/{len(DEMO_QUERIES)}: {query}")
            print(f"{'─' * 70}")

            print(f"\n  ▶ Running …")
            try:
                result = run_query(openai_client, agent.name, query)
                all_results.append(result)
                print_result(result)
            except Exception as e:
                print(f"  ✗ Error: {e}\n")
                all_results.append({
                    "query": query,
                    "latency_s": 0,
                    "output_text": f"ERROR: {e}",
                    "citations": [],
                })

        # ----- Summary table -----
        q_w, lat_w, cit_w = 50, 10, 10
        header = f"{'Query':<{q_w}} {'Latency':<{lat_w}} {'Citations':<{cit_w}}"
        sep = "-" * len(header)

        print("\n" + "=" * len(header))
        print("SUMMARY")
        print("=" * len(header))
        print(header)
        print(sep)

        for r in all_results:
            q_display = r["query"][:q_w - 3] + "..." if len(r["query"]) > q_w else r["query"]
            print(f"{q_display:<{q_w}} {r['latency_s']:<{lat_w}} {len(r['citations']):<{cit_w}}")
        print(sep)

    finally:
        cleanup_agent(project_client)


if __name__ == "__main__":
    main()
