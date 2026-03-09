# LLM Agents

`llm_agents` is a Python package for building typed, reusable agents on top of `pydantic-ai`.
It combines a shared `LLMAgent` base class with configuration-driven prompts, structured outputs, and example implementations for common agent patterns.

## Included agents

The repository currently includes four example agents:

- `LanguageDetector`: detect the primary language of a text.
- `LanguageTranslator`: translate text between languages.
- `GenericAssistant`: general-purpose assistant with optional tools and message history.
- `ImageDescriber`: describe an image from a URL or binary content.

## Development setup

Run the project inside the devcontainer. Jupyter is available there for the notebooks in [`notebooks/`](./notebooks).

Optional local services:

- Start Redis for agent output caching:

```bash
make redis-start
```

- Start MongoDB for persistent message history:

```bash
make mongo-start
```

- Start Ollama to run agents against a local model:

```bash
make ollama-start
```

## Environment variables

- `OPENAI_API_KEY`: required when using the default OpenAI-backed agent configs.
- `IONOS_TOKEN`: required when using [`get_ionos_model`](./llm_agents/utils/models.py).
- `OLLAMA_DSN`: optional Ollama-compatible base URL. Default: `http://llm-agents-ollama:11434/v1`.
- `MONGO_DSN`: optional MongoDB DSN for chat history. Default: `mongodb://lupai-mongo:27017`.
- `MONGO_DATABASE`: optional MongoDB database name. Default: `llm-agents`.

## Usage

Typical usage is:

1. Instantiate an agent.
2. Call `await agent.generate(...)`.
3. Read the typed output model returned by the agent.

### Minimal example

```python
import asyncio

from llm_agents.agents import (
    LanguageTranslator,
    LanguageTranslatorDeps,
)


async def main() -> None:
    translator = LanguageTranslator()
    translated = await translator.generate(
        "Bonjour tout le monde",
        agent_deps=LanguageTranslatorDeps(
            source_language="French",
            target_language="English",
        ),
    )
    print(translated.translation)


asyncio.run(main())
```

### Using a different model provider

Use the helper functions in [`llm_agents/utils/models.py`](./llm_agents/utils/models.py) to swap the underlying model:

```python
from llm_agents.agents import GenericAssistant
from llm_agents.utils.models import get_ollama_model

assistant = GenericAssistant(model=get_ollama_model("llama3.2:1b"))
```

### Message history

`GenericAssistant` can persist chat history in MongoDB:

```python
from llm_agents.agents import GenericAssistant
from llm_agents.message_history import MongoDBMessageHistory

history = MongoDBMessageHistory(session_id="demo-session")
assistant = GenericAssistant(mongodb_message_history=history)
```

## LLMAgent

`LLMAgent` is the shared base class behind the concrete agents in this repository. It keeps agent implementations small by handling the runtime wiring around `pydantic-ai`.

Main features:

- Typed outputs with Pydantic models.
- Optional typed deps interpolated into the instructions template.
- YAML-driven model settings and instructions.
- Async generation with bounded concurrency and batch execution.
- Optional Redis-based output caching.
- Optional MongoDB-backed message history.
- Support for `pydantic-ai` tools, prepared tools, and MCP servers.

### Creating a new agent

1. Create a YAML config in [`llm_agents/conf/agents`](./llm_agents/conf/agents) with model settings and an `instructions_template`.
2. Define a Pydantic output model for the structured response.
3. If the prompt needs runtime parameters, define a Pydantic deps model and reference its fields in `instructions_template`.
4. Subclass `LLMAgent` and pass `conf_path`, `output_type`, and optionally `deps_type`, tools, cache, or message history.
5. Instantiate the agent and call `await agent.generate(...)`.

```python
from pydantic import BaseModel, Field
from pydantic_ai import ToolOutput

from llm_agents.meta.interfaces import LLMAgent


class SummarizerDeps(BaseModel):
    style: str


class SummarizerOutput(BaseModel):
    summary: str = Field(description="Summary of the given text.")


class TextSummarizer(LLMAgent[SummarizerDeps, SummarizerOutput]):
    def __init__(self, conf_path: str, model=None):
        super().__init__(
            conf_path=conf_path,
            output_type=ToolOutput(SummarizerOutput),
            deps_type=SummarizerDeps,
            model=model,
        )
```

```yaml
model: openai:gpt-4o-mini-2024-07-18
temperature: 0.2
instructions_template: |-
  Summarize the user's text in a {style} style.
  Keep the response short and precise.
```

## Installation in another project

External projects can install `llm_agents` as a regular dependency.

For example, [`lupai`](https://github.com/aureka-team/lupai) uses it as a dependency.

In `requirements.txt`:

```txt
llm_agents>=<version>
```

In `uv.toml`:

```toml
[pip]
find-links = [
    "https://github.com/aureka-team/llm-agents/releases/expanded_assets/index",
]
```

## Main components

Common starting points:

- [`llm_agents.meta.interfaces.LLMAgent`](./llm_agents/meta/interfaces/llm_agent.py)
- [`llm_agents.agents.LanguageDetector`](./llm_agents/agents/language_detector.py)
- [`llm_agents.agents.LanguageTranslator`](./llm_agents/agents/language_translator.py)
- [`llm_agents.agents.GenericAssistant`](./llm_agents/agents/generic_assistant.py)
- [`llm_agents.agents.ImageDescriber`](./llm_agents/agents/image_describer.py)
- [`llm_agents.message_history.MongoDBMessageHistory`](./llm_agents/message_history/mongodb.py)

## Notebooks

Example notebooks are available in [`notebooks/01-agents`](./notebooks/01-agents):

- `01-language-detector.ipynb`
- `02-language-translator.ipynb`
- `03-generic-assistant.ipynb`
- `04-image-describer.ipynb`

## License

This project is licensed under the terms of the [`LICENSE`](./LICENSE) file.
