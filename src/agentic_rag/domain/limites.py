"""Travas do agente: orcamento de turnos e portao de escrita.

Inspirado em `turn_limits.py` e `write_gate.py` do harness de producao usado como molde.
Num agente com tool-use o modelo decide quantas voltas dar - sem teto explicito, um caso
degenerado vira laco e custo. E toda acao com efeito colateral precisa de autorizacao
declarada, nao implicita.

Dominio puro: sem I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class OrcamentoEsgotado(RuntimeError):
    """Levantada quando o agente tenta ultrapassar o teto de chamadas."""


@dataclass
class LimiteDeTurnos:
    """Contador explicito de chamadas de LLM por pergunta.

    Preferimos estourar com erro claro a truncar em silencio: um agente que para de
    buscar sem avisar produz resposta pior sem que ninguem perceba.
    """

    maximo: int
    usados: int = 0

    def consumir(self, quantos: int = 1) -> None:
        if self.usados + quantos > self.maximo:
            raise OrcamentoEsgotado(
                f"orcamento de {self.maximo} chamadas de LLM esgotado (usadas: {self.usados})"
            )
        self.usados += quantos

    @property
    def restantes(self) -> int:
        return max(self.maximo - self.usados, 0)

    @property
    def esgotado(self) -> bool:
        return self.restantes == 0


@dataclass
class PortaoDeEscrita:
    """Autoriza (ou nao) acoes com efeito colateral.

    Neste projeto o agente e read-only por padrao: ele consulta e explica, nunca altera
    estado. O portao existe para que essa escolha fique EXPLICITA no codigo e no log, e
    para que habilitar escrita amanha seja uma decisao consciente, com lista de permissao.
    """

    habilitado: bool = False
    operacoes_permitidas: frozenset[str] = field(default_factory=frozenset)
    negadas: list[str] = field(default_factory=list)

    def autorizar(self, operacao: str) -> bool:
        """Registra e decide. Retorna False em vez de levantar: negar e fluxo normal."""
        if self.habilitado and operacao in self.operacoes_permitidas:
            return True
        self.negadas.append(operacao)
        return False
