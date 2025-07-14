from pydantic import BaseModel, StrictStr, Field

from llm_agents.conf import agents
from llm_agents.meta.interfaces import LLMAgent


class GenericAssistantOutput(BaseModel):
    response: StrictStr = Field(
        description="Concise and accurate answer to the user's query."
    )


class GenericAssistant(LLMAgent[None, GenericAssistantOutput]):
    def __init__(
        self,
        conf_path: str = f"{agents.__path__[0]}/generic-assistant.yaml",
        message_history_length: int = 10,
    ):
        super().__init__(
            conf_path=conf_path,
            output_type=GenericAssistantOutput,
            message_history_length=message_history_length,
        )
