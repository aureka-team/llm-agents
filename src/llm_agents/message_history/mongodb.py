from rich.console import Console

from pymongo import AsyncMongoClient
from datetime import datetime, timezone

from pydantic_core import to_jsonable_python
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelMessage

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

    async def add_messages(self, messages: list[ModelMessage]) -> None:
        if not len(messages):
            console.log("[yellow]WARNING[/yellow] no messages to store.")
            return

        messages = [
            {
                "session_id": self.session_id,
                "date": datetime.now(timezone.utc),
            }
            | to_jsonable_python(m)
            for m in messages
        ]

        await self.db[self.mongodb_collection].insert_many(messages)

    async def get_messages(self) -> list[ModelMessage]:
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

        if self.message_limit is not None:
            messages = messages.limit(self.message_limit)

        return ModelMessagesTypeAdapter.validate_python(
            reversed(await messages.to_list())
        )

    async def remove_messages(self) -> None:
        await self.db[self.mongodb_collection].delete_many(
            {"session_id": self.session_id}
        )
