"""
Bing Grounding Agent – Setup / Teardown
Creates (or deletes) a Foundry Agent Service agent with BingGroundingTool.

Run once to create the agent, then use bing-grounding-demo.py to query it
repeatedly. The agent persists between runs so you avoid cold-start on every
invocation.

Usage:
    python setup-bing-agent.py            # create / update the agent
    python setup-bing-agent.py --delete   # delete the agent

Auth: Entra ID via DefaultAzureCredential.
Required env vars (or .env file):
    FOUNDRY_PROJECT_ENDPOINT
    FOUNDRY_MODEL_DEPLOYMENT_NAME
    BING_PROJECT_CONNECTION_NAME
    BING_AGENT_NAME              (default: bing-grounding-demo)
"""

import os
import sys

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
AGENT_NAME = os.environ.get("BING_AGENT_NAME", "bing-grounding-demo")

# Instructions engineered to maximize search consistency and citation quality.
# Key techniques:
#   1. Explicit "ALWAYS search" directive — reduces skipped-search rate.
#   2. "MUST cite with [Source Title](URL)" — forces citation output.
#   3. "Do not answer from memory" — prevents the model from skipping the tool.
AGENT_INSTRUCTIONS = """\
You are a research assistant with access to Bing Search.

RULES — follow these strictly:
1. ALWAYS use the Bing grounding tool for EVERY query, even if you think you \
know the answer. Do NOT answer from memory or training data alone.
2. After searching, synthesize the results into a clear, well-structured answer.
3. You MUST cite every factual claim with an inline citation.
4. If search results are insufficient, say so explicitly — do NOT fabricate \
information.
5. Include the date/time context of the information when relevant.
"""


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
def create_or_update_agent():
    """Create (or update) the persistent Bing-grounded agent."""
    credential = DefaultAzureCredential()
    project_client = AIProjectClient(endpoint=ENDPOINT, credential=credential)

    # Resolve Bing connection
    bing_connection = project_client.connections.get(BING_CONNECTION_NAME)
    print(f"Bing connection: {bing_connection.name}")
    print(f"Connection ID:   {bing_connection.id}")

    agent = project_client.agents.create_version(
        agent_name=AGENT_NAME,
        definition=PromptAgentDefinition(
            model=MODEL,
            instructions=AGENT_INSTRUCTIONS,
            tools=[
                BingGroundingTool(
                    bing_grounding=BingGroundingSearchToolParameters(
                        search_configurations=[
                            BingGroundingSearchConfiguration(
                                project_connection_id=bing_connection.id,
                            )
                        ]
                    )
                )
            ],
        ),
        description="Bing-grounded web search agent (managed via setup-bing-agent.py).",
    )

    print(f"\n✓ Agent ready")
    print(f"  Name:    {agent.name}")
    print(f"  Version: {agent.version}")
    print(f"  Model:   {MODEL}")
    print(f"\nRun queries with:  python bing-grounding-demo.py")


def delete_agent():
    """Delete the agent and all its versions."""
    credential = DefaultAzureCredential()
    project_client = AIProjectClient(endpoint=ENDPOINT, credential=credential)

    try:
        project_client.agents.delete(AGENT_NAME)
        print(f"✓ Agent '{AGENT_NAME}' deleted.")
    except Exception as e:
        print(f"✗ Could not delete agent '{AGENT_NAME}': {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if "--delete" in sys.argv:
        print(f"Deleting agent '{AGENT_NAME}' …")
        delete_agent()
    else:
        print(f"Creating/updating agent '{AGENT_NAME}' …")
        print(f"  Endpoint: {ENDPOINT}")
        print(f"  Model:    {MODEL}")
        print(f"  Bing:     {BING_CONNECTION_NAME}")
        print()
        create_or_update_agent()


if __name__ == "__main__":
    main()
