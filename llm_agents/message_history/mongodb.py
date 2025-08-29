import os

from pymongo import MongoClient
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
        mongo_collection: str = "message-history",
    ):
        self.client = MongoClient(
            mongo_dsn,
            serverSelectionTimeoutMS=5000,
            retryWrites=True,
        )

        self.db = self.client[mongo_database]

        self.session_id = session_id
        self.mongo_collection = mongo_collection

        self.ensure_index()

    def __del__(self) -> None:
        self.client.close()

    def ensure_index(self) -> None:
        indexes = self.db[self.mongo_collection].index_information()
        if "session_date_idx" in indexes:
            return

        self.db[self.mongo_collection].create_index(
            [
                ("session_id", 1),
                ("date", -1),
            ],
            name="session_date_idx",
        )

    def add_messages(self, messages: list[ModelMessage]) -> None:
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

        self.db[self.mongo_collection].insert_many(messages)

    def get_messages(self) -> list[ModelMessage]:
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
            reversed(list(messages))
        )
