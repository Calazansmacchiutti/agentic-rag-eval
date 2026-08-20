"""Compatibilidade: a configuracao mora em `infrastructure/config.py` (composition root).

Mantido para nao quebrar imports existentes (`from agentic_rag.config import settings`).
"""
from agentic_rag.infrastructure.config import Settings, settings

__all__ = ["Settings", "settings"]
