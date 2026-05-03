import os
import asyncio

from autogen.beta import Agent
from autogen.beta.config import OpenAIConfig
from autogen.beta.tools import tool

from autogen import ConversableAgent, LLMConfig

# 1. Define our LLM configuration for OpenAI's gpt-5-nano
#    uses the OPENAI_API_KEY environment variable
llm_config = LLMConfig(
    {"api_type": "google", "model": "gemini-2.5-flash", "api_key": os.environ["OPENAI_API_KEY"]}
)

# 2. Create our LLM agent
my_agent = ConversableAgent(
    name="helpful_agent",
    system_message="You are a poetic AI assistant, respond in rhyme.",
    llm_config=llm_config,
)

# 3. Run the agent with a prompt and process the response
response = my_agent.run(
    message="In one sentence, what's the big deal about AI?",
    max_turns=3,
    user_input=True,
)
response.process()
