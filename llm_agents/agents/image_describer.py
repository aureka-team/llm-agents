from pydantic import BaseModel, StrictStr

from common.cache import RedisCache

from llm_agents.conf import agents
from llm_agents.meta.interfaces import LLMAgent


class ImageDescriberInput(BaseModel):
    description_guidelines: StrictStr


class ImageDescriberOutput(BaseModel):
    description: StrictStr


class ImageDescriber(LLMAgent[ImageDescriberInput, ImageDescriberOutput]):
    def __init__(
        self,
        conf_path=f"{agents.__path__[0]}/image-describer.yaml",
        max_concurrency: int = 10,
        cache: RedisCache = None,
    ):
        super().__init__(
            conf_path=conf_path,
            agent_input=ImageDescriberInput,
            agent_output=ImageDescriberOutput,
            max_concurrency=max_concurrency,
            cache=cache,
        )
