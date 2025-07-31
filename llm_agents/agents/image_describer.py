from pydantic_ai import ToolOutput
from pydantic_ai.models import Model
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
        conf_path=f"{list(agents.__path__)[0]}/image-describer.yaml",
        model: Model | None = None,
        max_concurrency: int = 10,
        cache: RedisCache | None = None,
    ):
        super().__init__(
            conf_path=conf_path,
            output_type=ToolOutput(ImageDescriberOutput),
            model=model,
            max_concurrency=max_concurrency,
            cache=cache,
        )
