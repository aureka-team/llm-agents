from pydantic import StrictInt, StrictStr
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    redis_host: StrictStr = "llm-agents-redis"
    redis_port: StrictInt = 6379
    redis_db: StrictInt = 0
    mongodb_dsn: StrictStr = "mongodb://llm-agents-mongo:27017"
    mongodb_db_name: StrictStr = "llm_agents"
    mongodb_collection: StrictStr = "message_history"


config = Config()
