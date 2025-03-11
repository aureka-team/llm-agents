from pydantic import BaseModel, StrictStr

from common.logger import get_logger

from llm_agents.conf import experts
from llm_agents.meta.interfaces import LLMAgent


logger = get_logger(__name__)


class GenericAssistantInput(BaseModel):
    user_query: StrictStr


class GenericAssistantOutput(BaseModel):
    response: StrictStr


class GenericAssistant(LLMAgent[GenericAssistantInput, GenericAssistantOutput]):
    def __init__(
        self,
        conf_path: str = f"{experts.__path__[0]}/generic-assistant.yaml",
        message_history_length: int = 10,
    ):
        super().__init__(
            conf_path=conf_path,
            agent_input=GenericAssistantInput,
            agent_output=GenericAssistantOutput,
            message_history_length=message_history_length,
        )
