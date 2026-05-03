import asyncio
import os

from autogen.beta import Agent
from autogen.beta.config import OpenAIConfig
from autogen.beta.tools import tool
from autogen import ConversableAgent, LLMConfig

config = OpenAIConfig(
    model="google/gemini-2.5-flash",   # any model from the list above
    streaming=True,
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    max_completion_tokens=1024,
)
llm_config = llm_config = LLMConfig(
    {"api_type": "google", "model": "gemini-2.5-flash", "api_key": os.environ["OPENAI_API_KEY"]}
)

@tool
def add(a: int, b: int) -> int:
    """Adds two numbers together."""
    return a + b

#Agents
# 2. Create our LLM agent
llm_agent = ConversableAgent(
    name="helpful_agent",
    system_message="You are a poetic AI assistant, respond in rhyme.",
    llm_config=llm_config,
)

agent = Agent(
    config=config,
    name="anas",
    tools=[add],
)


async def main() -> None:
    #reply = await agent.ask("What's 1 + 1?")
    #print(reply.body)
    # 3. Run the agent with a prompt and process the response
    response = my_agent.run(
    	message="In one sentence, what's the big deal about AI?",
    	max_turns=3,
    	user_input=True,
    )
    print(response.process())


if __name__ == "__main__":
    asyncio.run(main())
