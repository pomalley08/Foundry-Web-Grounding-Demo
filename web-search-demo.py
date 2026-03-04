# Before running the sample:
#    pip install --pre azure-ai-projects>=2.0.0b1
#    pip install azure-identity

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

myEndpoint = "https://new-foundry-rgn10.services.ai.azure.com/api/projects/new-foundry"

project_client = AIProjectClient(
    endpoint=myEndpoint,
    credential=DefaultAzureCredential(),
)

myAgent = "bing-search-agent"
myVersion = "1"

openai_client = project_client.get_openai_client()

# Reference the agent to get a response
response = openai_client.responses.create(
    input=[{"role": "user", "content": "Pull the results of the winter olympics."}],
    extra_body={"agent_reference": {"name": myAgent, "version": myVersion, "type": "agent_reference"}},
)

print(f"Response output: {response.output_text}")



