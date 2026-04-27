from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai import NativeOutput
from pydantic_ai.models.openai import OpenAIChatModelSettings

from pydantic import BaseModel, Field
from pydantic_extra_types.language_code import LanguageAlpha2

from llm_agents.meta.interfaces import LLMAgent


class LanguageDetectorOutput(BaseModel):
    language: LanguageAlpha2 = Field(
        description="The primary language of the given text."
    )


agent = Agent(
    # model="gpt-5.4-2026-03-05",
    model="gpt-5.4-mini-2026-03-17",
    system_prompt=LLMAgent.read_file(
        file_path=str(Path(__file__).with_name("system-prompt.md"))
    ),
    output_type=NativeOutput(LanguageDetectorOutput),
    model_settings=OpenAIChatModelSettings(openai_reasoning_effort="none"),
    retries=3,
)


class LanguageDetector(LLMAgent[None, LanguageDetectorOutput]):
    def __init__(self, max_concurrency: int = 10):
        super().__init__(
            agent=agent,
            max_concurrency=max_concurrency,
        )
