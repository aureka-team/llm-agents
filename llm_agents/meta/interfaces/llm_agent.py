import joblib
import asyncio

from tqdm import tqdm
from collections import deque
from typing import TypeVar, Generic, TypeAlias


from pydantic_ai import Agent
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


AgentInput = TypeVar("AgentInput", bound=BaseModel)
AgentOutput = TypeVar("AgentOutput", bound=BaseModel)
UserContent: TypeAlias = ImageUrl | BinaryContent | None


class Config(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: StrictStr
    temperature: NonNegativeFloat | None = None
    max_tokens: PositiveInt | None = Field(
        alias="max-tokens",
        default=None,
    )

    system_prompt: StrictStr | None = Field(
        alias="system-prompt",
        default=None,
    )

    human_prompt_template: StrictStr | None = Field(
        alias="human-prompt-template",
        default=None,
    )


class LLMAgent(Generic[AgentInput, AgentOutput]):
    def __init__(
        self,
        conf_path: str,
        agent_input: type[BaseModel],
        agent_output: type[BaseModel],
        max_concurrency: int = 10,
        message_history_length: int = 0,  # NOTE: 0 means no history
        cache: RedisCache | None = None,
    ):
        self.max_concurrency = max_concurrency
        self.cache = cache

        self.conf = Config(**load_yaml(file_path=conf_path))
        self.agent = Agent(
            model=self.get_agent_model(model=self.conf.model),
            system_prompt=self.conf.system_prompt,
            deps_type=agent_input,
            result_type=agent_output,
            model_settings=self.conf.model_dump(),
        )

        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.message_history = deque(maxlen=message_history_length)

    @staticmethod
    def get_agent_model(model: str) -> str | OpenAIModel:
        model_type = model.split(":")[0]
        if model_type != "ollama":
            return model

        return OpenAIModel(
            model_name=model.split(":")[1],
            provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
        )

    def _get_cache_key(
        self,
        agent_input: AgentInput,
        user_content: UserContent,
    ) -> str:
        return joblib.hash(
            f"{joblib.hash(self.conf)}-{joblib.hash(agent_input)}-{joblib.hash(user_content)}"
        )

    async def generate(
        self,
        agent_input: AgentInput,
        user_content: UserContent = None,
        pbar: tqdm | None = None,
    ) -> AgentOutput:
        async with self.semaphore:
            cache_key = self._get_cache_key(
                agent_input=agent_input,
                user_content=user_content,
            )

            if self.cache is not None:
                cached_output = self.cache.load(cache_key=cache_key)
                if cached_output is not None:
                    if pbar is not None:
                        pbar.update(1)

                    return cached_output

            try:
                human_prompt = self.conf.human_prompt_template.format(
                    **agent_input.model_dump()
                )

            except KeyError as e:
                raise ValueError(f"Missing required field in agent_input: {e}")

            user_content
            agent_run_result = await self.agent.run(
                user_prompt=[
                    human_prompt,
                    user_content,
                ]
                if user_content is not None
                else human_prompt,
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

            agent_output = agent_run_result.data
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
        agent_inputs: list[AgentInput],
    ) -> list[AgentOutput]:
        with tqdm(
            total=len(agent_inputs),
            ascii=" ##",
            colour="#808080",
        ) as pbar:
            pass

            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(
                        self.generate(
                            agent_input,
                            pbar=pbar,
                        )
                    )
                    for agent_input in agent_inputs
                ]

            return [task.result() for task in tasks]
