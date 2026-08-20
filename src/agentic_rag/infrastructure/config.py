"""Configuracao central (env/paths). Fonte unica de verdade."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    llm_provider: str = "anthropic"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_model: str = "claude-opus-4-8"        # geracao/raciocinio
    # Juiz: haiku-4-5 e o modelo do baseline versionado (models/baseline_metrics.json) e
    # ACEITA `temperature`, que o ADR 0005 usa como knob de determinismo. Cuidado ao trocar:
    # a familia Opus 4.7+ REJEITA `temperature` com 400 ("deprecated for this model"). Se
    # apontar o juiz para um desses, defina tambem `grader_temperature=None`.
    grader_model: str = "claude-haiku-4-5"
    grader_temperature: float | None = 0.0    # None = nao envia o parametro
    max_output_tokens: int = 4096
    vector_db_url: str = "http://localhost:6333"
    top_k: int = 5                            # trechos recuperados por busca (knob de precisao/recall)
    # reranking (ADR 0005, item 5): recupera um pool maior e reordena p/ os melhores k.
    # Cross-encoder deterministico (sem LLM); opt-in p/ nao forcar download do modelo.
    rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_fetch_k: int = 20                   # tamanho do pool de candidatos antes do rerank
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_llm_calls_per_item: int = 2   # orcamento de custo/latencia (RAG QA)
    # eval harness (gate de promocao): golden set + baseline versionado
    eval_dataset_path: str = "data/eval_set.jsonl"
    baseline_metrics_path: str = "models/baseline_metrics.json"
    eval_promotion_tolerance: float = 0.0  # margem p/ "bater o baseline" (>= baseline - tol)
    # estruturador de PDF (agente A): loop propor -> cortar -> avaliar -> refinar
    max_structure_iterations: int = 3      # teto de iteracoes do loop de recorte
    structure_score_threshold: float = 0.85  # para cedo quando o recorte e bom o bastante
    # --- modo auditavel (contexto regulado) ---
    # trilha append-only: uma linha por resposta, com id do trecho (nao o texto do corpus)
    audit_log_path: str = "logs/auditoria.jsonl"
    # portao de escrita: agente e read-only por padrao; habilitar e decisao consciente
    write_enabled: bool = False


settings = Settings()
