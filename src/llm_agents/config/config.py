from pydantic import Field, StrictInt, StrictStr
from pydantic_settings import BaseSettings


class RedisConf(BaseSettings):
    host: StrictStr = "llm-agents-redis"
    port: StrictInt = 6379
    db: StrictInt = 0
    namespace: StrictStr = "llm-agents"


class Config(BaseSettings):
    redis: RedisConf = Field(default_factory=RedisConf)


config = Config()
