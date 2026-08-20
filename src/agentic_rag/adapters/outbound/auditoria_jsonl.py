"""Adapter de auditoria: trilha append-only em JSONL.

Escolha deliberada de formato: uma linha por evento, texto puro, sem banco.
- append-only por construcao (abrimos em modo "a"), que e o requisito de uma trilha;
- inspecionavel com `tail` e diffavel, sem ferramenta;
- trocar por Postgres depois e escrever outro adapter - o dominio nao muda.

O log guarda o ID do trecho, nunca o texto: a trilha nao replica o corpus (que pode ser
confidencial) e ainda assim permite reconstruir a decisao.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from agentic_rag.domain.entities import EventoAuditoria


class AuditoriaJSONL:
    """Implementa PortaAuditoria. Seguro entre threads dentro de um processo."""

    def __init__(self, caminho: str | Path):
        self.caminho = Path(caminho)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self._trava = threading.Lock()

    def registrar(self, evento: EventoAuditoria) -> None:
        linha = json.dumps(evento.to_dict(), ensure_ascii=False)
        with self._trava, self.caminho.open("a", encoding="utf-8") as f:
            f.write(linha + "\n")

    def listar(self, usuario: str | None = None, limite: int = 100) -> list[dict]:
        """Le os eventos mais recentes. Linha corrompida e ignorada, nao derruba a leitura."""
        if not self.caminho.exists():
            return []
        eventos: list[dict] = []
        with self.caminho.open(encoding="utf-8") as f:
            for linha in f:
                linha = linha.strip()
                if not linha:
                    continue
                try:
                    ev = json.loads(linha)
                except json.JSONDecodeError:
                    continue
                if usuario is None or ev.get("usuario") == usuario:
                    eventos.append(ev)
        return eventos[-limite:]


class AuditoriaMemoria:
    """Implementacao em memoria, para teste e para rodar sem escrever em disco."""

    def __init__(self):
        self.eventos: list[dict] = []

    def registrar(self, evento: EventoAuditoria) -> None:
        self.eventos.append(evento.to_dict())

    def listar(self, usuario: str | None = None, limite: int = 100) -> list[dict]:
        base = [e for e in self.eventos if usuario is None or e.get("usuario") == usuario]
        return base[-limite:]
