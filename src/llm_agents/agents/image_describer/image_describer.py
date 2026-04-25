from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai import NativeOutput
from pydantic_ai.models.openai import OpenAIChatModelSettings

from pydantic import BaseModel, StrictStr, Field

from llm_agents.meta.schema import UserContent
from llm_agents.meta.interfaces import LLMAgent


class ImageDescriberOutput(BaseModel):
    description: StrictStr = Field(
        description="Detailed description of the provided image."
    )


agent = Agent(
    # model="gpt-5.4-2026-03-05",
    model="gpt-5.4-mini-2026-03-17",
    system_prompt=LLMAgent.read_file(
        file_path=str(Path(__file__).with_name("system-prompt.md"))
    ),
    output_type=NativeOutput(ImageDescriberOutput),
    model_settings=OpenAIChatModelSettings(openai_reasoning_effort="none"),
    retries=3,
)


class ImageDescriber(LLMAgent[None, ImageDescriberOutput]):
    def __init__(self, max_concurrency: int = 10):
        super().__init__(max_concurrency=max_concurrency)

    async def _generate(
        self,
        user_prompt: str,
        user_content: UserContent,
        agent_deps: None = None,
    ) -> ImageDescriberOutput:
        result = await agent.run(user_prompt=[user_prompt, user_content])

        return result.output
