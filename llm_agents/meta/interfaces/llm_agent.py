import os
import joblib
import asyncio

from tqdm import tqdm
from collections import deque
from itertools import zip_longest
from typing import TypeVar, Generic, TypeAlias

from pydantic_ai.mcp import MCPServer
from pydantic_ai import Agent, Tool, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import ImageUrl, BinaryContent
from pydantic import (
    BaseModel,
    StrictStr,
    PositiveInt,
    NonNegativeFloat,
    Field,
    ConfigDict,
)

from common.cache import RedisCache
from common.logger import get_logger
from common.utils.yaml_data import load_yaml


logger = get_logger(__name__)


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")


AgentDeps = TypeVar("AgentDeps", bound=BaseModel)
AgentOutput = TypeVar("AgentOutput", bound=BaseModel)
UserContent: TypeAlias = ImageUrl | BinaryContent


class Config(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: StrictStr
    temperature: NonNegativeFloat | None = None
    max_tokens: PositiveInt | None = Field(
        alias="max-tokens",
        default=None,
    )

    instructions_template: StrictStr | None = Field(
        alias="instructions-template",
        default=None,
    )


class MissingInstructionsTemplateError(Exception):
    def __init__(self):
        super().__init__("instructions_template is required when deps are set.")


class LLMAgent(Generic[AgentDeps, AgentOutput]):
    def __init__(
        self,
        conf_path: str,
        output_type: type[BaseModel],
        deps_type: type[BaseModel] | None = None,
        tools: list[Tool] = [],
        mcp_servers: list[MCPServer] = [],
        retries: int = 1,
        max_concurrency: int = 10,
        message_history_length: int = 0,  # NOTE: 0 means no history
        cache: RedisCache | None = None,
    ):
        self.max_concurrency = max_concurrency
        self.cache = cache

        self.conf = Config(**load_yaml(file_path=conf_path))
        self.agent = Agent(
            model=self.get_agent_model(model=self.conf.model),
            output_type=output_type,
            deps_type=deps_type,
            name=self.__class__.__name__,
            model_settings=self.conf.model_dump(),
            retries=retries,
            tools=tools,
            mcp_servers=mcp_servers,
        )

        @self.agent.instructions
        def get_instructions(ctx: RunContext[AgentDeps]) -> str | None:
            instructions_template = self.conf.instructions_template
            deps = ctx.deps
            if deps is None:
                return instructions_template

            if instructions_template is None:
                raise MissingInstructionsTemplateError()

            return instructions_template.format(**deps.model_dump())

        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.message_history = deque(maxlen=message_history_length)

    @staticmethod
    def get_agent_model(model: str) -> str | OpenAIModel:
        model_type = model.split(":")[0]
        if model_type != "ollama":
            return model

        return OpenAIModel(
            model_name=model.split(":")[1],
            provider=OpenAIProvider(base_url=f"http://{OLLAMA_HOST}:11434/v1"),
        )

    def _get_cache_key(
        self,
        user_prompt: str,
        agent_deps: AgentDeps | None = None,
        user_content: UserContent | None = None,
    ) -> str:
        return joblib.hash((user_prompt, agent_deps, user_content))

    async def generate(
        self,
        user_prompt: str,
        agent_deps: AgentDeps | None = None,
        user_content: UserContent | None = None,
        pbar: tqdm | None = None,
    ) -> AgentOutput:
        async with self.semaphore:
            cache_key = self._get_cache_key(
                user_prompt=user_prompt,
                agent_deps=agent_deps,
                user_content=user_content,
            )

            if self.cache is not None:
                cached_output = self.cache.load(cache_key=cache_key)
                if cached_output is not None:
                    if pbar is not None:
                        pbar.update(1)

                    return cached_output

            user_prompt = (
                [
                    user_prompt,
                    user_content,
                ]
                if user_content is not None
                else user_prompt
            )

            agent_run_result = await self.agent.run(
                user_prompt=user_prompt,
                deps=agent_deps,
                message_history=list(self.message_history)
                if self.message_history
                else None,
            )

            self.message_history.extend(agent_run_result.new_messages())
            usage = agent_run_result.usage()

            logger.debug(
                {
                    "request_tokens": usage.request_tokens,
                    "response_tokens": usage.response_tokens,
                }
            )

            agent_output = agent_run_result.output
            if self.cache is not None:
                self.cache.save(
                    obj=agent_output,
                    cache_key=cache_key,
                )

            if pbar is not None:
                pbar.update(1)

            return agent_output

    async def batch_generate(
        self,
        user_prompts: list[str],
        agent_deps_list: list[AgentDeps] = [],
        user_contents: list[UserContent] = [],
    ) -> list[AgentOutput]:
        num_user_prompts = len(user_prompts)

        num_deps = len(agent_deps_list)
        if num_deps:
            assert num_user_prompts == num_deps, (
                f"length of user_prompts and agent_deps doesn't match: "
                f"{num_user_prompts} != {num_deps}"
            )

        num_contents = len(user_contents)
        if num_contents:
            assert num_user_prompts == num_contents, (
                f"length of user_prompts and user_contents doesn't match: "
                f"{num_user_prompts} != {num_contents}"
            )

        with tqdm(
            total=len(user_prompts),
            ascii=" ##",
            colour="#808080",
        ) as pbar:
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(
                        self.generate(
                            user_prompt=user_prompt,
                            agent_deps=agent_deps,
                            user_content=user_content,
                            pbar=pbar,
                        )
                    )
                    for user_prompt, agent_deps, user_content in zip_longest(
                        user_prompts,
                        agent_deps_list,
                        user_contents,
                    )
                ]

            return [task.result() for task in tasks]
