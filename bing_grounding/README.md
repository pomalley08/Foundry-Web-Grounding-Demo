# Microsoft Foundry — Web Search Demo

Demonstrates two approaches to adding web search capabilities to an AI application using Microsoft Foundry:

| Script | Approach | Status | Best for |
|--------|----------|--------|----------|
| `bing-grounding-demo.py` | Bing Grounding via Foundry Agent Service | **GA** | Production use today |
| `responses-web-search-demo.py` | Responses API `web_search_preview` tool | Preview | Simpler integration (when GA) |

Both demos run a set of queries that require fresh web data, extract citations, measure latency, and save results to timestamped files in `output/`.

## Prerequisites

- **Python 3.10+**
- **Azure subscription** with:
  - A [Microsoft Foundry](https://ai.azure.com/) project with a configured endpoint
  - A deployed model (e.g., `gpt-4.1`)
  - [DefaultAzureCredential](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential) configured (e.g., `az login`)
- **For Bing Grounding demo only:**
  - A [Grounding with Bing Search](https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/tools/bing-tools) resource created and connected to your Foundry project. Can be connected to the project in the Classic Foundry Portal.

## Setup

```bash
# Clone and enter the repo
git clone <repo-url>
cd foundry-demo

# Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies (--pre is required for the azure-ai-projects prerelease SDK)
pip install -r requirements.txt
```

Copy the environment template and fill in your values:

```bash
cp .env.sample .env
```

Edit `.env` with your Azure resource details:

```dotenv
# Required for all demos
FOUNDRY_PROJECT_ENDPOINT=https://<your-account>.services.ai.azure.com/api/projects/<your-project>

# Required for Bing Grounding demo (setup-bing-agent.py + bing-grounding-demo.py)
FOUNDRY_MODEL_DEPLOYMENT_NAME=gpt-4.1
BING_PROJECT_CONNECTION_NAME=<your-bing-connection-name>
BING_AGENT_NAME=bing-grounding-demo

# Optional — Responses API demo model overrides
NON_REASONING_MODEL=gpt-4.1
REASONING_MODEL=gpt-5.2
```

## Usage

### Bing Grounding Demo (GA)

This demo uses the [Grounding with Bing Search](https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/tools/bing-tools) tool via Foundry Agent Service — the current GA approach for adding web search to agents.

```bash
# 1. Create the agent (one-time setup)
python setup-bing-agent.py

# 2. Run queries against the agent
python bing-grounding-demo.py

# 3. Run again to compare warm vs cold-start latency
python bing-grounding-demo.py

# 4. Clean up when done
python setup-bing-agent.py --delete
```

The agent is created once and persists between runs. This enables the evaluation of cold-start latency on the first invocation and compare steady-state latency.

**Output includes:**
- Per-query latency with cold-start labeling
- Search invocation tracking (did the agent actually search?)
- Citation extraction from Bing grounding annotations
- Summary table with consistency analysis (search rate, citation rate)

### Responses API Web Search Demo (Preview)

This demo uses the `web_search_preview` tool directly on the Responses API — no agent setup required. It compares a non-reasoning model against a reasoning model.

> **Note:** `web_search_preview` is currently in preview.

```bash
python responses-web-search-demo.py
```

**Output includes:**
- Side-by-side model comparison (latency, search calls, citations)
- Per-query results with response previews

### Output Files

Both demos save their full console output to timestamped files in `output/`:

```
output/bing-grounding-20260305-090635.txt
output/responses-web-search-20260305-085013.txt
```

## Project Structure

```
├── setup-bing-agent.py             # Create/delete the Bing-grounded agent
├── bing-grounding-demo.py          # Query runner for Bing grounding (GA)
├── responses-web-search-demo.py    # Responses API web_search_preview (Preview)
├── requirements.txt                # Python dependencies (uses --pre for SDK)
├── .env.sample                     # Environment variable template
└── output/                         # Auto-created; timestamped result files
```

## Key Design Decisions

- **`tool_choice="required"`** — Forces the agent to invoke Bing Search on every query, ensuring consistent grounding behavior.
- **Strong agent instructions** — The system prompt explicitly directs the agent to always search and never answer from training data alone.
- **Code-first, no portal** — Everything is done via the Python SDK. No Foundry portal UI interaction required — works behind private networks.

## References

- [Grounding agents with Bing Search tools](https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/tools/bing-tools)
- [Microsoft Foundry Responses API - Web Search](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/web-search)
- [azure-ai-projects Python SDK](https://learn.microsoft.com/en-us/python/api/overview/azure/ai-projects-readme?view=azure-python-preview)
