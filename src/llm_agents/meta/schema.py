from typing import TypeAlias
from pydantic_ai.messages import ImageUrl, BinaryContent


UserContent: TypeAlias = ImageUrl | BinaryContent
