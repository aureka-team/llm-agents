# LLM Agents

`llm_agents` is a small Python package for building typed agents on top of `pydantic-ai`.
It is intentionally thin and mainly exists to add three capabilities on top of `pydantic-ai`: cached generation, batch generation, and message history management.

## Included agents

The repository currently ships four example agents:

- `LanguageDetector`: detects the primary language of a text.
- `LanguageTranslator`: translates text between languages.
- `ImageDescriber`: produces a detailed description for an image URL or binary image input.
- `GenericAssistant`: answers arbitrary user queries with MongoDB-backed message history.

## What This Package Adds

The core value of `llm_agents` lives in [`src/llm_agents/meta/interfaces/llm_agent.py`](./src/llm_agents/meta/interfaces/llm_agent.py), which wraps `pydantic-ai` with a small shared interface for:

- `generate_cached(...)`: Redis-backed cached generation.
- `batch_generate(...)`: concurrent batch execution with progress reporting.
- Message history management: loading and storing conversation history through `MongoDBMessageHistory`.

Everything else in the repository is mostly example agent implementations built on that base.

## Installation

Python `3.11+` is required.

Install the package locally:

```bash
pip install -e .
```

Or install it as a dependency in another project:

```txt
llm_agents>=<version>
```

If you consume GitHub release artifacts with `uv`, keep the custom release index in `uv.toml`:

```toml
[pip]
find-links = [
    "https://github.com/aureka-team/llm-agents/releases/expanded_assets/index",
]
```

## Development setup

The intended development environment is the devcontainer.

Build it with:

```bash
make devcontainer-build
```

Optional local services:

- Redis for `generate_cached(...)` and `batch_generate(..., cached_generation=True)`.
- MongoDB for `MongoDBMessageHistory`.
- Ollama for running agents against a local OpenAI-compatible endpoint.

Start them as needed:

```bash
make redis-start
make mongo-start
make ollama-start
```

Jupyter notebooks are available in [`notebooks/01-agents`](./notebooks/01-agents).

## Environment variables

- `OPENAI_API_KEY`: required for the default OpenAI-backed agents.
- `IONOS_TOKEN`: required by `llm_agents.utils.models.get_ionos_model(...)`.
- `OLLAMA_DSN`: optional Ollama-compatible base URL. Default: `http://llm-agents-ollama:11434/v1`.
- `MONGO_DSN`: optional MongoDB DSN. Default: `mongodb://llm-agents-mongo:27017`.
- `MONGO_DATABASE`: optional MongoDB database name. Default: `llm_agents`.

Redis settings are currently read from `src/llm_agents/config/config.py` via nested settings with defaults:

- host: `llm-agents-redis`
- port: `6379`
- db: `0`
- namespace: `llm-agents`

## Usage

Typical usage is:

1. Instantiate an agent.
2. Call `await agent.generate(...)`.
3. Read the typed Pydantic output.

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

### Image input

`ImageDescriber` accepts `pydantic-ai` image content types through `user_content`:

```python
import asyncio

from pydantic_ai.messages import ImageUrl

from llm_agents.agents import ImageDescriber


async def main() -> None:
    describer = ImageDescriber()
    result = await describer.generate(
        "Describe this image.",
        user_content=ImageUrl(url="https://example.com/cat.jpg"),
    )
    print(result.description)


asyncio.run(main())
```

### Message history

`GenericAssistant` is designed to run with MongoDB-backed history:

```python
from llm_agents.agents import GenericAssistant
from llm_agents.message_history import MongoDBMessageHistory

history = MongoDBMessageHistory(session_id="demo-session")
assistant = GenericAssistant(mongodb_message_history=history)
```

### Alternative model providers

[`src/llm_agents/utils/models.py`](./src/llm_agents/utils/models.py) exposes helpers for building OpenAI-compatible model objects when you add new agents or adapt the existing modules:

```python
from llm_agents.utils.models import get_ollama_model

model = get_ollama_model("llama3.2:1b")
```

## Architecture

The package is built around plain Python agent modules:

- Each agent module defines its output schema and a module-level `pydantic_ai.Agent`.
- Static prompts live in a neighboring `system-prompt.md`.
- Dynamic prompts are composed with `@agent.system_prompt` and typed deps.
- `LLMAgent` adds the shared capabilities that `pydantic-ai` does not provide directly in this package: cached generation, batch generation, and message history management.

The main base class lives in [`src/llm_agents/meta/interfaces/llm_agent.py`](./src/llm_agents/meta/interfaces/llm_agent.py).

## Creating a new agent

1. Create a package under [`src/llm_agents/agents`](./src/llm_agents/agents).
2. Add a `system-prompt.md` file next to the agent module.
3. Define a Pydantic output model and optional deps model.
4. Create a module-level `Agent(...)` with `NativeOutput(...)`.
5. Subclass `LLMAgent` and implement `_generate(...)`.

Example:

```python
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput, RunContext

from llm_agents.meta.interfaces import LLMAgent


class SummarizerDeps(BaseModel):
    style: str


class SummarizerOutput(BaseModel):
    summary: str = Field(description="Summary of the given text.")


agent = Agent(
    model="gpt-5.4-mini-2026-03-17",
    deps_type=SummarizerDeps,
    output_type=NativeOutput(SummarizerOutput),
)


@agent.system_prompt
async def get_system_prompt(ctx: RunContext[SummarizerDeps]) -> str:
    template = LLMAgent.read_file(
        str(Path(__file__).with_name("system-prompt.md"))
    )
    return template.format(**ctx.deps.model_dump())


class TextSummarizer(LLMAgent[SummarizerDeps, SummarizerOutput]):
    async def _generate(
        self,
        user_prompt: str,
        agent_deps: SummarizerDeps,
        user_content=None,
    ) -> SummarizerOutput:
        result = await agent.run(user_prompt=user_prompt, deps=agent_deps)
        return result.output
```

Example prompt file:

```md
# Role

You are a concise summarization assistant.

# Objective

Summarize the user's text in a {style} style.
```

## Notebooks

Example notebooks are available in [`notebooks/01-agents`](./notebooks/01-agents):

- `01-language-detector.ipynb`
- `02-language-translator.ipynb`
- `03-image-describer.ipynb`
- `04-generic-assistant.ipynb`

## Releases

Pushing a Git tag matching `v*` creates a release through [`.github/workflows/python-release.yml`](./.github/workflows/python-release.yml).

Example:

```bash
git tag v<version>
git push origin v<version>
```

The workflow builds the wheel, creates the GitHub release for the tag, and uploads the wheel to both the tag release and the permanent `index` release used by `uv`.

## License

This project is licensed under the terms of the [`LICENSE`](./LICENSE) file.
