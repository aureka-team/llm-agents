from rich.console import Console

from pymongo import AsyncMongoClient
from datetime import datetime, timezone

from pydantic_core import to_jsonable_python
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelMessage,
    ModelRequest,
)

from llm_agents.config import config


console = Console()


class MongoDBMessageHistory:
    def __init__(
        self,
        session_id: str,
        mongodb_dsn: str = config.mongodb_dsn,
        mongodb_db_name: str = config.mongodb_db_name,
        mongodb_collection: str = config.mongodb_collection,
        message_limit: int | None = 50,
        save_tool_messages: bool = False,
        read_only: bool = False,
    ):
        self.client = AsyncMongoClient(
            mongodb_dsn,
            serverSelectionTimeoutMS=5000,
            retryWrites=True,
        )

        self.db = self.client[mongodb_db_name]
        self.session_id = session_id
        self.mongodb_collection = mongodb_collection
        self.message_limit = message_limit
        self.save_tool_messages = save_tool_messages
        self.read_only = read_only

    async def ensure_index(self) -> None:
        indexes = await self.db[self.mongodb_collection].index_information()
        index_name = "session_date_idx"
        if index_name in indexes:
            return

        await self.db[self.mongodb_collection].create_index(
            [
                ("session_id", 1),
                ("date", -1),
            ],
            name=index_name,
        )

    @staticmethod
    def filter_tool_message(message: ModelMessage) -> bool:
        message_part = message.parts[0]
        if isinstance(
            message_part,
            ToolCallPart,
        ) or isinstance(
            message_part,
            ToolReturnPart,
        ):
            return False

        return True

    @staticmethod
    def trim_to_last_turns(
        messages: list[ModelMessage],
        turn_limit: int | None,
    ) -> list[ModelMessage]:
        if turn_limit is None:
            return messages

        if turn_limit <= 0:
            return []

        request_indexes = [
            index
            for index, message in enumerate(messages)
            if isinstance(message, ModelRequest)
        ]

        if not request_indexes:
            return []

        start_index = request_indexes[max(0, len(request_indexes) - turn_limit)]
        trimmed_messages = messages[start_index:]

        if trimmed_messages and isinstance(trimmed_messages[-1], ModelRequest):
            return trimmed_messages[:-1]

        return trimmed_messages

    async def add_messages(self, messages: list[ModelMessage]) -> None:
        if self.read_only:
            return

        if not len(messages):
            console.log("[yellow]WARNING[/yellow] no messages to store.")
            return

        if not self.save_tool_messages:
            messages = [
                message
                for message in messages
                if self.filter_tool_message(message=message)
            ]

        messages = [
            {
                "session_id": self.session_id,
                "date": datetime.now(timezone.utc),
            }
            | to_jsonable_python(m)
            for m in messages
        ]

        await self.db[self.mongodb_collection].insert_many(messages)

    async def get_messages(self) -> list[ModelMessage] | None:
        messages = (
            self.db[self.mongodb_collection]
            .find(
                {
                    "session_id": self.session_id,
                },
                {
                    "session_id": 0,
                    "date": 0,
                },
            )
            .sort([("date", -1), ("_id", -1)])
        )

        messages = await messages.to_list()
        if not len(messages):
            return

        model_messages = ModelMessagesTypeAdapter.validate_python(
            reversed(messages)
        )

        return self.trim_to_last_turns(
            messages=model_messages,
            turn_limit=self.message_limit,
        )

    async def remove_messages(self) -> None:
        await self.db[self.mongodb_collection].delete_many(
            {"session_id": self.session_id}
        )
