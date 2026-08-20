"""CLI do Agente B (extracao de campos).

Uso:
  python -m agentic_rag.pdf.extract_cli doc.pdf \
      --field "titulo=titulo do documento" \
      --field "data=data de emissao" \
      --required titulo

  --no-llm  -> dry-run (monta o contexto, nao chama o LLM)
  --json    -> despeja o ExtractionResult completo
"""
from __future__ import annotations

import argparse
import sys

from agentic_rag.pdf.schemas import FieldSpec


def _parse_fields(field_args: list[str], required: list[str]) -> list[FieldSpec]:
    req = set(required)
    specs = []
    for raw in field_args:
        if "=" not in raw:
            raise SystemExit(f"--field invalido (use name=descricao): {raw!r}")
        name, desc = raw.split("=", 1)
        name = name.strip()
        specs.append(FieldSpec(name=name, description=desc.strip(), required=name in req))
    return specs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extrai campos estruturados de um PDF (agente B).")
    ap.add_argument("pdf")
    ap.add_argument("--field", action="append", default=[], help="name=descricao (repetivel)")
    ap.add_argument("--required", action="append", default=[], help="nome de campo obrigatorio (repetivel)")
    ap.add_argument("--no-llm", action="store_true", help="dry-run sem chamar o LLM")
    ap.add_argument("--ocr", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if not args.field:
        raise SystemExit("informe ao menos um --field name=descricao")
    fields = _parse_fields(args.field, args.required)

    from agentic_rag.pdf import extract_agent

    res = extract_agent.extract(args.pdf, fields, use_llm=not args.no_llm, ocr=args.ocr)

    if args.json:
        print(res.model_dump_json(indent=2))
        return 0

    print(f"# extracao | {res.notes}")
    for f in res.fields:
        pg = f" (p{f.page})" if f.page is not None else ""
        print(f"  {f.name}: {f.value!r}{pg}  [conf {f.confidence:.2f}]")
    if res.missing_required:
        print("  FALTANDO (obrigatorios):", ", ".join(res.missing_required))
    return 0


if __name__ == "__main__":
    sys.exit(main())
