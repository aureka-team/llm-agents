.PHONY: devcontainer-build


devcontainer-build:
	docker compose -f .devcontainer/docker-compose.yml build llm-agents-devcontainer


redis-start:
	docker compose -f .devcontainer/docker-compose.yml up -d llm-agents-redis

redis-stop:
	docker compose -f .devcontainer/docker-compose.yml stop llm-agents-redis

redis-flush:
	docker compose -f .devcontainer/docker-compose.yml exec llm-agents-redis redis-cli FLUSHALL


mongo-start:
	docker compose -f .devcontainer/docker-compose.yml up -d llm-agents-mongo

mongo-stop:
	docker compose -f .devcontainer/docker-compose.yml stop llm-agents-mongo

mongo-restart: mongo-stop mongo-start


ollama-start:
	docker compose -f .devcontainer/docker-compose.yml up -d llm-agents-ollama
	docker exec -it llm-agents-ollama ollama pull llama3.2
	docker exec -it llm-agents-ollama ollama pull llama3.2:1b

ollama-stop:
	docker compose -f .devcontainer/docker-compose.yml stop llm-agents-ollama
