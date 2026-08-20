"""Estruturador de PDF (multi-agente).

Camada compartilhada (probe + schemas) que entende o COMPORTAMENTO de cada PDF, e
dois agentes que consomem esse perfil:
  - chunk_agent: descobre o recorte ideal para RAG (loop propor -> cortar -> avaliar).
  - extract_agent: extrai campos estruturados.

A base analitica (probe) e deterministica: le sinais de layout reais (fontes, blocos,
tabelas, TOC) antes de qualquer LLM. O LLM entra so para propor/avaliar estrategia.
"""
