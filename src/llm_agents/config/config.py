from pydantic import StrictInt, StrictStr
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    redis_host: StrictStr = "llm-agents-redis"
    redis_port: StrictInt = 6379
    redis_db: StrictInt = 0


config = Config()
