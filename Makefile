.PHONY: install test lint run eval diagrams qdrant seed
qdrant:     ; docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
install:    ; pip install -e ".[dev]" && pre-commit install
test:       ; pytest -q
lint:       ; ruff check src tests && black --check src tests
run:        ; uvicorn agentic_rag.api:app --reload
seed:       ; python scripts/seed_corpus.py
eval:       ; python -m agentic_rag.evaluate
diagrams:   ; npx -y -p @mermaid-js/mermaid-cli mmdc -i docs/architecture.md -o docs/diagrams/arch.svg
