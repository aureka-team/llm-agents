from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai import NativeOutput
from pydantic_ai.models.openai import OpenAIChatModelSettings

from pydantic import BaseModel, StrictStr, Field

from llm_agents.meta.schema import UserContent
from llm_agents.meta.interfaces import LLMAgent
from llm_agents.message_history import MongoDBMessageHistory


class GenericAssistantOutput(BaseModel):
    response: StrictStr = Field(
        description="Concise and accurate answer to the user's query."
    )


agent = Agent(
    # model="gpt-5.4-2026-03-05",
    model="gpt-5.4-mini-2026-03-17",
    system_prompt=LLMAgent.read_file(
        file_path=str(Path(__file__).with_name("system-prompt.md"))
    ),
    output_type=NativeOutput(GenericAssistantOutput),
    model_settings=OpenAIChatModelSettings(openai_reasoning_effort="none"),
    retries=3,
)


class GenericAssistant(LLMAgent[None, GenericAssistantOutput]):
    def __init__(
        self,
        max_concurrency: int = 10,
        mongodb_message_history: MongoDBMessageHistory | None = None,
    ):
        super().__init__(
            max_concurrency=max_concurrency,
            mongodb_message_history=mongodb_message_history,
        )

    async def _generate(
        self,
        user_prompt: str,
        agent_deps: None = None,
        user_content: UserContent | None = None,
    ) -> GenericAssistantOutput:
        message_history = await self.get_history_messages()
        result = await agent.run(
            user_prompt=user_prompt,
            message_history=message_history,
        )

        await self.add_history_messages(messages=result.new_messages())
        return result.output
