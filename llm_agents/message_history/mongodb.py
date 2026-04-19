import os

from pymongo import AsyncMongoClient
from common.logger import get_logger
from datetime import datetime, timezone

from pydantic_core import to_jsonable_python
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelMessage


logger = get_logger(__name__)


MONGO_DSN = os.getenv("MONGO_DSN", "mongodb://lupai-mongo:27017")
MONGO_DATABASE = os.getenv("MONGO_DATABASE", "llm-agents")


class MongoDBMessageHistory:
    def __init__(
        self,
        session_id: str,
        mongo_dsn: str = MONGO_DSN,
        mongo_database: str = MONGO_DATABASE,
        mongo_collection: str = "message_history",
    ):
        self.client = AsyncMongoClient(
            mongo_dsn,
            serverSelectionTimeoutMS=5000,
            retryWrites=True,
        )

        self.db = self.client[mongo_database]
        self.session_id = session_id
        self.mongo_collection = mongo_collection

    async def ensure_index(self) -> None:
        indexes = await self.db[self.mongo_collection].index_information()
        index_name = "session_date_idx"
        if index_name in indexes:
            return

        await self.db[self.mongo_collection].create_index(
            [
                ("session_id", 1),
                ("date", -1),
            ],
            name=index_name,
        )

    async def add_messages(self, messages: list[ModelMessage]) -> None:
        if not len(messages):
            logger.warning("no messages to store.")
            return

        messages = [
            {
                "session_id": self.session_id,
                "date": datetime.now(timezone.utc),
            }
            | to_jsonable_python(m)
            for m in messages
        ]

        await self.db[self.mongo_collection].insert_many(messages)

    async def get_messages(self) -> list[ModelMessage]:
        messages = (
            self.db[self.mongo_collection]
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

        return ModelMessagesTypeAdapter.validate_python(
            reversed(await messages.to_list())
        )

    async def remove_messages(self) -> None:
        await self.db[self.mongo_collection].delete_many(
            {"session_id": self.session_id}
        )
