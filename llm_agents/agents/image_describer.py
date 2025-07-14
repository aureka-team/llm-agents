from pydantic import BaseModel, StrictStr, Field

from common.cache import RedisCache

from llm_agents.conf import agents
from llm_agents.meta.interfaces import LLMAgent


class ImageDescriberOutput(BaseModel):
    description: StrictStr = Field(
        description="Detailed description of the provided image."
    )


class ImageDescriber(LLMAgent[None, ImageDescriberOutput]):
    def __init__(
        self,
        conf_path=f"{agents.__path__[0]}/image-describer.yaml",
        max_concurrency: int = 10,
        cache: RedisCache = None,
    ):
        super().__init__(
            conf_path=conf_path,
            output_type=ImageDescriberOutput,
            max_concurrency=max_concurrency,
            cache=cache,
        )
