# Azure AI Project Connection Details

## Install dependencies

```bash
pip install --pre azure-ai-projects>=2.0.0b1
pip install azure-identity
```

## Python sample

```python
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition

user_endpoint = "https://industrial-engineering-resource.services.ai.azure.com/api/projects/industrial_engineering"

project_client = AIProjectClient(
    endpoint=user_endpoint,
    credential=DefaultAzureCredential(),
)

agent_name = "<your-agent-name>"
model_deployment_name = "<your-model-deployment-name>"

# Creates an agent, bumps the agent version if parameters have changed
agent = project_client.agents.create_version(
    agent_name=agent_name,
    definition=PromptAgentDefinition(
        model=model_deployment_name,
        instructions="You are a storytelling agent. You craft engaging one-line stories based on user prompts and context.",
    ),
)

openai_client = project_client.get_openai_client()

# Reference the agent to get a response
response = openai_client.responses.create(
    input=[{"role": "user", "content": "Tell me a one line story"}],
    extra_body={"agent": {"name": agent.name, "type": "agent_reference"}},
)

print(f"Response output: {response.output_text}")
```
