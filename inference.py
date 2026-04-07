"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""
import os
import json
from typing import Any, Dict

from openai import OpenAI
from openenv.core.mcp_client import MCPToolClient
from dotenv import load_dotenv
load_dotenv()


API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

SERVER_URL = os.getenv("OPENENV_SERVER_URL", "http://localhost:7860")

MAX_ITERATIONS = 15
TEMPERATURE = 0.2
MAX_TOKENS = 500


def run_llm_agent(env: MCPToolClient, client: OpenAI, task_list: list[str]) -> Dict[str, float]:
    """
    Runs the OpenAI-compatible LLM against the environment tasks.
    """
    scores = {}

    for task in task_list:
        # Reset the environment for the specific task
        obs = env.reset(task=task)
        meta = getattr(obs, "metadata", {}) or {}
        
        system_msg = meta.get("system_message","""
            You are a customer support agent.

            You must investigate the customer's issue before replying.

            Guidelines:
            1. Use search_kb to understand policies.
            2. Use get_order_status if the issue involves orders.
            3. Use check_payment if the issue involves payments.
            4. Only call reply_customer after collecting enough information.
            5. Hard tasks may require multiple tool calls.
        """)
        customer_query = meta.get("customer_query", "")

        # Exactly format the START block
        print(f"[START] task={task} env=customer_support model={MODEL_NAME}")

        # 1. Discover tools dynamically
        tools_raw = env.list_tools()
        openai_tools = []
        for t in tools_raw:
            schema = getattr(t, "input_schema", {})
            props = schema.get("properties", {}) if isinstance(schema, dict) else getattr(schema, "properties", {})
            reqs = schema.get("required", []) if isinstance(schema, dict) else getattr(schema, "required", [])
            
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or t.name,
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": list(reqs),
                    },
                },
            })

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": customer_query},
        ]

        grade_score = 0.0
        success = False
        step_rewards = []
        episode_ended = False
        step_count = 0

        # 2. Agent Interaction Loop
        for iteration in range(MAX_ITERATIONS):
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=openai_tools if openai_tools else None,
                    tool_choice="required" if openai_tools else None,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    parallel_tool_calls=False,
                )
            except Exception as e:
                break

            msg = completion.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            # If the model emits raw text instead of a tool call
            if not getattr(msg, "tool_calls", None):
                # Put the raw text into the message history and ask it to use a tool
                messages.append({
                    "role": "user", 
                    "content": "You must invoke a tool. Do not just reply with plain text. If you want to reply to the user, use the reply_customer tool."
                })
                continue

            # Process tool calls
            for tc in msg.tool_calls:
                step_count += 1
                fn_name = tc.function.name
                
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}
                
                try:
                    tool_result = env.call_tool(fn_name, **fn_args)
                    error_val = "null"
                except Exception as e:
                    tool_result = {"error": f"Tool execution failed: {e}"}
                    error_val = str(e).replace('\n', ' ')

                # Track reward proxy based on environment step evaluation
                reward = 0.0
                if fn_name in ["get_order_status", "check_payment"]:
                    reward = 0.2
                elif fn_name == "search_kb":
                    reward = 0.1
                elif fn_name == "escalate_ticket":
                    reward = -0.1

                if fn_name in ("reply_customer", "escalate_ticket"):
                    grade_score = tool_result.get("grade_score", 0.0)
                    if fn_name == "reply_customer":
                        if grade_score >= 0.8:
                            reward = 0.4
                        elif grade_score >= 0.5:
                            reward = 0.2
                        else:
                            reward = -0.2
                    episode_ended = True

                step_rewards.append(reward)
                done_val = "true" if episode_ended else "false"
                action_str = f"{fn_name}('{json.dumps(fn_args)}')"
                
                print(f"[STEP] step={step_count} action={action_str} reward={reward:.2f} done={done_val} error={error_val}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn_name,
                    "content": json.dumps(tool_result),
                })

                if episode_ended:
                    break
            
            if episode_ended:
                break
        
        success = grade_score > 0.0
        rewards_str = ",".join(f"{r:.2f}" for r in step_rewards)
        if not step_rewards:
            rewards_str = "0.0"

        # Exactly format the END block
        print(f"[END] success={str(success).lower()} steps={step_count} score={grade_score:.3f} rewards={rewards_str}")
        scores[task] = grade_score

    return scores


def main() -> None:
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN is not set. API calls will fail if auth is required.")

    # Initialize OpenAI client 
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

    try:
        with MCPToolClient(base_url=SERVER_URL).sync() as env:
            scores = run_llm_agent(
                env=env,
                client=client, 
                task_list=["easy", "medium", "hard"]
            )
    except Exception as e:
        print(f"\n[FATAL] Could not complete environment interaction: {e}")

if __name__ == "__main__":
    main()