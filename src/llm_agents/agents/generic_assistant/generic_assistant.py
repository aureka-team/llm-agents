from pathlib import Path

from pydantic_ai import Agent, NativeOutput
from pydantic_ai.capabilities import ReinjectSystemPrompt
from pydantic_ai.common_tools.web_fetch import web_fetch_tool
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

from pydantic import BaseModel, StrictStr, Field

from llm_agents.meta.interfaces import LLMAgent
from llm_agents.message_history import MongoDBMessageHistory


class GenericAssistantOutput(BaseModel):
    response: StrictStr = Field(
        description="Concise and accurate answer to the user's query."
    )


agent = Agent(  # type: ignore
    # model="gpt-5.4-2026-03-05",
    model="gpt-5.4-mini-2026-03-17",
    system_prompt=LLMAgent.read_file(
        file_path=str(Path(__file__).with_name("system-prompt.md"))
    ),
    output_type=NativeOutput(GenericAssistantOutput),
    retries=3,
    tools=[
        duckduckgo_search_tool(),
        web_fetch_tool(),
    ],
    capabilities=[ReinjectSystemPrompt()],
)


class GenericAssistant(LLMAgent[None, GenericAssistantOutput]):
    def __init__(
        self,
        mongodb_message_history: MongoDBMessageHistory,
        max_concurrency: int = 10,
    ):
        super().__init__(
            agent=agent,
            max_concurrency=max_concurrency,
            mongodb_message_history=mongodb_message_history,
        )
