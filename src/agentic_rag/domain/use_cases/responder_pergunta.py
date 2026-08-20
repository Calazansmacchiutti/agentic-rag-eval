"""Caso de uso central: responder uma pergunta com fundamentacao auditavel.

Fluxo: guardrail de entrada -> loop agentic (retrieve/reason) -> resposta estruturada ->
guardrail de saida -> registro de auditoria.

O que este arquivo NAO faz: falar com Anthropic, com Qdrant ou com HTTP. Ele so conhece
as portas. E o que permite testar a regra inteira com dublês, sem rede.
"""
from __future__ import annotations

from dataclasses import dataclass

from agentic_rag.domain import guardrails
from agentic_rag.domain.entities import Escopo, EventoAuditoria, Resposta, Trecho
from agentic_rag.domain.limites import LimiteDeTurnos, OrcamentoEsgotado
from agentic_rag.domain.ports.outbound import PortaAuditoria, PortaLLM, PortaRecuperacao

FERRAMENTA_BUSCA = {
    "name": "search",
    "description": (
        "Busca trechos relevantes na base de documentos por similaridade vetorial. "
        "Use antes de responder."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Consulta de busca."}},
        "required": ["query"],
    },
}


@dataclass
class ResultadoConsulta:
    """O que sai do caso de uso: resposta + lastro + veredito, tudo junto.

    `permitido=False` NAO e erro: e o sistema recusando com motivo, que e um desfecho
    legitimo e esperado em contexto regulado.
    """

    resposta: Resposta
    trechos: list[Trecho]
    permitido: bool
    decisao: str
    motivo: str = ""
    turnos_usados: int = 0

    @property
    def fontes(self) -> list[str]:
        """Rotulos legiveis dos trechos efetivamente citados."""
        return [self.trechos[i].fonte for i in self.resposta.citations if 0 <= i < len(self.trechos)]


def _numerar(trechos: list[Trecho], inicio: int) -> str:
    """Numera a partir de `inicio` p/ os indices casarem com as citacoes acumuladas."""
    linhas = [f"[{i}] {t.texto}" for i, t in enumerate(trechos, start=inicio)]
    return "\n".join(linhas) if linhas else "(sem resultados)"


def _dedup(trechos: list[Trecho], vistos: set[str]) -> list[Trecho]:
    """Curador deterministico: descarta repetidos no batch e entre rodadas.

    O loop tende a re-recuperar os mesmos trechos, o que dilui a precisao de contexto
    (medida no eval) sem agregar informacao. Custa 0 chamadas de LLM e preserva a ordem.
    """
    saida = []
    for t in trechos:
        chave = t.id_conteudo
        if not t.texto.strip() or chave in vistos:
            continue
        vistos.add(chave)
        saida.append(t)
    return saida


def responder(
    pergunta: str,
    *,
    recuperacao: PortaRecuperacao,
    llm: PortaLLM,
    sistema: str,
    versao_prompt: str = "sem-versao",
    escopo: Escopo | None = None,
    auditoria: PortaAuditoria | None = None,
    max_chamadas: int = 2,
) -> ResultadoConsulta:
    """Executa o fluxo completo e devolve resultado ja validado pelos guardrails."""
    escopo = escopo or Escopo()
    limite = LimiteDeTurnos(maximo=max(max_chamadas, 1))

    def _registrar(res: ResultadoConsulta) -> ResultadoConsulta:
        if auditoria is not None:
            auditoria.registrar(EventoAuditoria(
                pergunta=pergunta,
                resposta=res.resposta.answer,
                grounded=res.resposta.grounded,
                confidence=res.resposta.confidence,
                trechos=[t.id_conteudo for t in res.trechos],
                versao_prompt=versao_prompt,
                modelo=llm.modelo_corrente,
                usuario=escopo.usuario,
                papel=escopo.papel,
                decisao=res.decisao,
                motivo=res.motivo,
            ))
        return res

    # --- guardrail de entrada: barra antes de gastar LLM ---
    entrada = guardrails.checar_pergunta(pergunta)
    if not entrada.permitido:
        vazia = Resposta(answer=entrada.motivo, citations=[], grounded=False, confidence=0.0)
        return _registrar(ResultadoConsulta(vazia, [], False, entrada.decisao, entrada.motivo))

    # --- loop agentic: o modelo decide quando buscar ---
    mensagens: list[dict] = [{"role": "user", "content": pergunta}]
    contextos: list[Trecho] = []
    vistos: set[str] = set()

    try:
        while limite.restantes > 1:  # a ultima chamada fica reservada p/ a saida estruturada
            limite.consumir()
            resp = llm.loop_ferramentas(
                sistema=sistema, mensagens=mensagens, ferramentas=[FERRAMENTA_BUSCA]
            )
            mensagens.append({"role": "assistant", "content": resp.content})
            usos = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            if not usos:
                break  # respondeu sem (mais) busca; forcamos a saida estruturada abaixo
            resultados = []
            for uso in usos:
                base = len(contextos)
                # o escopo vira FILTRO de metadado: a restricao acontece antes do LLM ler
                achados = recuperacao.buscar(uso.input["query"], filtros=escopo.filtros or None)
                novos = _dedup(achados, vistos)
                contextos.extend(novos)
                resultados.append({
                    "type": "tool_result", "tool_use_id": uso.id, "content": _numerar(novos, base),
                })
            mensagens.append({"role": "user", "content": resultados})
    except OrcamentoEsgotado:
        pass  # segue para a resposta estruturada com o contexto que houver

    # --- saida estruturada ---
    mensagens.append({
        "role": "user",
        "content": "Com base no contexto recuperado, responda agora em JSON estruturado.",
    })
    resposta: Resposta = llm.resposta_estruturada(
        sistema=sistema, mensagens=mensagens, schema=Resposta
    )

    # --- guardrail de saida: sem citacao valida, nao passa ---
    saida = guardrails.checar_resposta(resposta, contextos)
    return _registrar(ResultadoConsulta(
        resposta=resposta,
        trechos=contextos,
        permitido=saida.permitido,
        decisao=saida.decisao,
        motivo=saida.motivo,
        turnos_usados=limite.usados,
    ))
