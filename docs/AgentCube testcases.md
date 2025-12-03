## Test Case 1: CLI Deployment Test

- **Objective:** Verify that an agent can be packaged, built, published, and deployed into a specific Kubernetes namespace.

- **Preconditions:**

  - Kubernetes cluster is running
  - `kubectl` CLI is installed and configured
  - `agentrun` plugin is available

- **Steps:**

  1.  Run `kubectl create namespace test-agent-space`
  2.  Run `kubectl agentrun pack`
  3.  Run `kubectl agentrun build`
  4.  Run `kubectl agentrun publish`

- **Expected Result:**

  - A new namespace `test-agent-space` is created
  - The agent is successfully packaged, built, and published
  - The agent is deployed to the `test-agent-space` namespace

## Test Case 2: Code Interpreter Workflow Test

**Objective:** Validate that the Code Interpreter can handle file upload, dependency installation, model training, and artifact download with Python-SDK.

``` python
from agentcube import CodeInterpreterClient

# Create a CodeInterpreterClient instance
code_interpreter = CodeInterpreterClient(
    ttl=3600,  # Time-to-live in seconds
    image="sandbox:latest",  # Container image to use
)

try:
    # Step 1: Upload dependencies file (WriteFile API)
    code_interpreter.write_file(
        content="pandas\nnumpy\nscikit-learn\nmatplotlib",
        remote_path="/workspace/requirements.txt"
    )

    # Step 2: Install dependencies (Execute API)
    code_interpreter.execute_command("pip install -r /workspace/requirements.txt")

    # Step 3: Upload training data (WriteFile API)
    code_interpreter.upload_file(
        local_path="./data/train.csv",
        remote_path="/workspace/train.csv"
    )

    # Step 4: Train model (Execute API)
    training_code = """
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    import pickle
    df = pd.read_csv('/workspace/train.csv')
    X, y = df[['feature1', 'feature2']], df['target']
    model = LinearRegression().fit(X, y)
    pickle.dump(model, open('/workspace/model.pkl', 'wb'))
    print(f'Model R² score: {model.score(X, y):.4f}')
    """
    result = code_interpreter.run_code("python", training_code)

    print(result)

    # Step 5: Download trained model (ReadFile API)
    code_interpreter.download_file(
        remote_path="/workspace/model.pkl",
        local_path="./models/model.pkl"
    )

    print("Workflow completed successfully!")

finally:
    code_interpreter.stop()
```

## Integration Test: LangChain Agent with SiliconFlow + Code Interpreter

**Test Agent Source**

``` python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from langchain_core.tools import tool
from langchain.agents import create_agent, Tool
from langchain.chat_models import init_chat_model
import os 

api_key = os.environ.get("OPENAI_API_KEY", "")
api_base_url = os.environ.get("OPENAI_API_BASE", "")
model_name = os.environ.get("OPENAI_MODEL", "")

# Global variable for the Code Interpreter client
ci_client = None

@tool
def run_python_code(code: str) -> str:
"""Wrapper to run Python code inside Code Interpreter."""
    global ci_client
    if ci_client is None:
        from agentcube import CodeInterpreterClient
        ci_client = CodeInterpreterClient(ttl=600, image="sandbox:latest")
    
    return ci_client.run_code("python", code)


# Define tools properly
tools = [run_python_code]

# Initialize your LLM (adjust model and parameters as needed)
llm = init_chat_model(
    "DeepSeek-V3",
    model_provider="openai",
    base_url=api_base_url,
    configurable_fields="any",
    temperature=0.1
) 

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a good python code interpreter assistant.",
)

  

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global ci_client
    from agentcube import CodeInterpreterClient
    ci_client = CodeInterpreterClient(ttl=600, image="sandbox:latest")
    yield
    # Shutdown

ci_client.stop()
# FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

@app.post("/run")
async def run_agent(request: Request):
    data = await request.json()
    query = data.get("query", "")
    response = agent.invoke(query, 
                            config={
                                "configurable": {
                                        "api_key": api_key,
                                        "model": model_name,
                                }
                            })

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Deployment Instructions

Similar to **Test Case 1**, use the CLI to package, build, and publish this agent into the cluster:

``` bash
kubectl create namespace test-agent-space
kubectl agentrun pack
kubectl agentrun build
kubectl agentrun publish
```

This workflow ensures the agent is deployed into the `test-agent-space` namespace and exposed as a FastAPI service on port **8080**.

### Expected Result

- The agent is successfully packaged, built, and published via CLI into the `test-agent-space` namespace

- The FastAPI service runs on port **8080** inside the cluster

- Invoke the deployed agent:
  `shell   kubectl agentrun invoke -f ./ --header {} --payload {"What is square root of 49?"}`
  Verify response:
  `json   {       "response": "Square root of 49 is 7.0"   }`

- Continue conversation with the return `x-agentcube-session-id` in header,
  `shell   kubectl agentrun invoke -f ./ --header {x-agentcube-session-id: xxxxxxxxx} --payload {"And then plus 3?"}`
  Verify response includes continuation logic:
  `json       {           "response": "Square root of 49 is 7.0"       }`

- Wait 15 minutes (session expiry) and send:,
  `shell   kubectl agentrun invoke -f ./ --header {x-agentcube-session-id: xxxxxxxxx} --payload {"And then plus 1?"}`
  Verify agent returns incorrect or reset response due to expired session.

- The workflow executes end-to-end, with **SiliconFlow** providing reasoning and **Code Interpreter** handling code execution
