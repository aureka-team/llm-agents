from pathlib import Path

from pydantic_ai import NativeOutput
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModelSettings

from pydantic import BaseModel, StrictStr, Field
from pydantic_extra_types.language_code import LanguageName

from llm_agents.meta.interfaces import LLMAgent


class LanguageTranslatorDeps(BaseModel):
    source_language: LanguageName
    target_language: LanguageName


class LanguageTranslatorOutput(BaseModel):
    translation: StrictStr = Field(description="Translation of the given text.")


agent = Agent(
    # model="gpt-5.4-2026-03-05",
    model="gpt-5.4-mini-2026-03-17",
    deps_type=LanguageTranslatorDeps,
    output_type=NativeOutput(LanguageTranslatorOutput),
    model_settings=OpenAIChatModelSettings(openai_reasoning_effort="none"),
    retries=3,
)


@agent.system_prompt
async def get_system_prompt(ctx: RunContext[LanguageTranslatorDeps]) -> str:
    system_prompt = LLMAgent.read_file(
        file_path=str(Path(__file__).with_name("system-prompt.md"))
    )

    return system_prompt.format(**ctx.deps.model_dump())


class LanguageTranslator(
    LLMAgent[LanguageTranslatorDeps, LanguageTranslatorOutput]
):
    def __init__(self, max_concurrency: int = 10):
        super().__init__(
            agent=agent,
            max_concurrency=max_concurrency,
        )
