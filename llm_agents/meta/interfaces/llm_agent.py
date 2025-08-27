import joblib
import asyncio

from tqdm import tqdm
from itertools import zip_longest
from typing import TypeVar, Generic, TypeAlias, Literal

from pydantic_ai.models import Model
from pydantic_ai.mcp import MCPServer
from pydantic_ai import (
    Agent,
    Tool,
    RunContext,
    ToolOutput,
    NativeOutput,
    PromptedOutput,
)

from pydantic_ai.messages import (
    ImageUrl,
    BinaryContent,
    ModelMessage,
    ToolReturnPart,
)

from pydantic import (
    BaseModel,
    StrictStr,
    PositiveInt,
    NonNegativeFloat,
    ConfigDict,
)

from common.cache import RedisCache
from common.logger import get_logger
from common.utils.yaml_data import load_yaml

from llm_agents.message_history import MongoDBMessageHistory


logger = get_logger(__name__)


AgentDeps = TypeVar("AgentDeps", bound=BaseModel | None)
AgentOutput = TypeVar("AgentOutput", bound=BaseModel)
UserContent: TypeAlias = ImageUrl | BinaryContent


class Reasoning(BaseModel):
    effort: Literal["minimal"]


class Config(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: StrictStr | None = None
    verbosity: Literal["low", "medium", "high"] = "low"
    reasoning: Reasoning | None = None
    temperature: NonNegativeFloat | None = None
    max_tokens: PositiveInt | None = None
    instructions_template: StrictStr | None = None


class MissingInstructionsTemplateError(Exception):
    def __init__(self):
        super().__init__("instructions_template is required when deps are set.")


class MissingModelError(Exception):
    def __init__(self):
        super().__init__("Model is required but was not provided.")


class LLMAgent(Generic[AgentDeps, AgentOutput]):
    def __init__(
        self,
        conf_path: str,
        output_type: ToolOutput | NativeOutput | PromptedOutput,
        deps_type: type[BaseModel] | None = None,
        model: Model | str | None = None,
        tools: list[Tool] = [],
        mcp_servers: list[MCPServer] = [],
        retries: int = 1,
        max_concurrency: int = 10,
        message_history_length: int = 0,  # NOTE: 0 means no history
        mongodb_message_history: MongoDBMessageHistory | None = None,
        read_only_message_history: bool = False,
        cache: RedisCache | None = None,
    ):
        self.max_concurrency = max_concurrency
        self.message_history_length = message_history_length

        self.cache = cache
        self.conf = Config(**load_yaml(file_path=conf_path))
        self.mongodb_message_history = mongodb_message_history
        self.read_only_message_history = read_only_message_history

        model = model if model is not None else self.conf.model
        if model is None:
            raise MissingModelError()

        self.agent = Agent(
            model=model,
            output_type=output_type,
            deps_type=deps_type,
            name=self.__class__.__name__,
            model_settings=self.conf.model_dump(exclude_unset=True),
            retries=retries,
            tools=tools,
            mcp_servers=mcp_servers,  # type: ignore
        )

        @self.agent.instructions  # type: ignore
        def get_instructions(ctx: RunContext[AgentDeps]) -> str | None:
            instructions_template = self.conf.instructions_template
            deps = ctx.deps
            if deps is None:
                return instructions_template

            if instructions_template is None:
                raise MissingInstructionsTemplateError()

            return instructions_template.format(**deps.model_dump())  # type: ignore

        self.message_history = []
        if self.mongodb_message_history is not None:
            messages = self.mongodb_message_history.get_messages()
            self.add_history_messages(messages=messages)

        self.semaphore = asyncio.Semaphore(max_concurrency)

    def add_history_messages(self, messages: list[ModelMessage]) -> None:
        if not len(messages):
            return

        if self.message_history_length == 0:
            return

        self.message_history.extend(messages)
        if len(self.message_history) <= self.message_history_length:
            return

        new_first_element = self.message_history[-self.message_history_length]
        if not isinstance(new_first_element.parts[0], ToolReturnPart):
            self.message_history = self.message_history[
                -self.message_history_length :
            ]

            return

        self.message_history = self.message_history[
            -(self.message_history_length + 1) :
        ]

    def _get_cache_key(
        self,
        user_prompt: str,
        agent_deps: AgentDeps | None = None,
        user_content: UserContent | None = None,
    ) -> str:
        return joblib.hash((user_prompt, agent_deps, user_content))  # type: ignore

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
                    user_prompt,  # type: ignore
                    user_content,
                ]
                if user_content is not None
                else user_prompt
            )

            agent_run_result = await self.agent.run(
                user_prompt=user_prompt,
                deps=agent_deps,  # type: ignore
                message_history=list(self.message_history)
                if self.message_history
                else None,
            )

            if not self.read_only_message_history:
                new_messages = agent_run_result.new_messages()
                self.add_history_messages(messages=new_messages)
                if self.mongodb_message_history is not None:
                    self.mongodb_message_history.add_messages(
                        messages=new_messages
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
