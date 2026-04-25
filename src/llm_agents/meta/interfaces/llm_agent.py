import joblib
import asyncio

from tqdm import tqdm  # type: ignore
from pathlib import Path

from aiocache import Cache, cached
from aiocache.serializers import PickleSerializer

from functools import lru_cache
from itertools import zip_longest
from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic

from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage

from llm_agents.config import config
from llm_agents.meta.schema import UserContent
from llm_agents.message_history import MongoDBMessageHistory


AgentDeps = TypeVar("AgentDeps", bound=BaseModel | None)
AgentOutput = TypeVar("AgentOutput", bound=BaseModel)


def get_cache_key(
    func: Any,
    _self: Any,
    *args: Any,
    **kwargs: Any,
) -> str:
    cache_key = joblib.hash(
        (
            func.__module__,
            func.__qualname__,
            args,
            kwargs,
        )
    )

    assert cache_key is not None
    return cache_key


class LLMAgent(ABC, Generic[AgentDeps, AgentOutput]):
    def __init__(
        self,
        max_concurrency: int = 10,
        mongodb_message_history: MongoDBMessageHistory | None = None,
    ):
        self.mongodb_message_history = mongodb_message_history
        self.semaphore = asyncio.Semaphore(max_concurrency)

    @lru_cache()
    @staticmethod
    def read_file(file_path: str) -> str:
        return Path(file_path).read_text()

    async def add_history_messages(self, messages: list[ModelMessage]) -> None:
        messages = [m for m in messages if m.parts[0].part_kind != "tool-call"]
        assert self.mongodb_message_history is not None
        await self.mongodb_message_history.add_messages(messages=messages)

    async def get_history_messages(self) -> list[ModelMessage]:
        assert self.mongodb_message_history is not None
        return await self.mongodb_message_history.get_messages()

    @abstractmethod
    async def _generate(
        self,
        user_prompt: str,
        agent_deps: AgentDeps | None = None,
        user_content: UserContent | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> AgentOutput:
        pass

    async def generate(
        self,
        user_prompt: str,
        agent_deps: AgentDeps | None = None,
        user_content: UserContent | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> AgentOutput:
        return await self._generate(
            user_prompt=user_prompt,
            agent_deps=agent_deps,
            user_content=user_content,
            *args,
            **kwargs,
        )

    async def generate_pbar(
        self,
        user_prompt: str,
        agent_deps: AgentDeps | None = None,
        user_content: UserContent | None = None,
        pbar: tqdm | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> AgentOutput:
        async with self.semaphore:
            output = await self.generate(
                user_prompt=user_prompt,
                agent_deps=agent_deps,
                user_content=user_content,
                *args,
                **kwargs,
            )

            if pbar is not None:
                pbar.update(1)

            return output

    @cached(
        cache=Cache.REDIS,
        endpoint=config.redis_host,
        port=config.redis_port,
        db=config.redis_db,
        serializer=PickleSerializer(),
        key_builder=get_cache_key,
        noself=True,
    )
    async def generate_cached(
        self,
        user_prompt: str,
        agent_deps: AgentDeps | None = None,
        user_content: UserContent | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> AgentOutput:
        return await self.generate(
            user_prompt=user_prompt,
            agent_deps=agent_deps,
            user_content=user_content,
            *args,
            **kwargs,
        )

    async def generate_cached_pbar(
        self,
        user_prompt: str,
        agent_deps: AgentDeps | None = None,
        user_content: UserContent | None = None,
        pbar: tqdm | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> AgentOutput:
        async with self.semaphore:
            output = await self.generate_cached(
                user_prompt=user_prompt,
                agent_deps=agent_deps,
                user_content=user_content,
                *args,
                **kwargs,
            )

            if pbar is not None:
                pbar.update(1)

            return output

    async def batch_generate(
        self,
        user_prompts: list[str],
        agent_deps_list: list[AgentDeps] = [],
        user_contents: list[UserContent] = [],
        cached_generation: bool = False,
        *args: Any,
        **kwargs: Any,
    ):

        assert user_prompts
        with tqdm(
            total=len(user_prompts),
            ascii=" ##",
            colour="#808080",
        ) as pbar:
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(
                        self.generate_pbar(
                            user_prompt=user_prompt,
                            agent_deps=agent_deps,
                            user_content=user_content,
                            pbar=pbar,
                            *args,
                            **kwargs,
                        )
                    )
                    if not cached_generation
                    else tg.create_task(
                        self.generate_cached_pbar(
                            user_prompt=user_prompt,
                            agent_deps=agent_deps,
                            user_content=user_content,
                            pbar=pbar,
                            *args,
                            **kwargs,
                        )
                    )
                    for user_prompt, agent_deps, user_content in zip_longest(
                        user_prompts,
                        agent_deps_list,
                        user_contents,
                    )
                ]

            return [task.result() for task in tasks]
