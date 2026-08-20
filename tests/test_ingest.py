from agentic_rag.ingest import chunk


def test_chunk_overlap():
    chunks = chunk("a" * 2000, size=800, overlap=100)
    assert len(chunks) >= 2
    assert all(len(c) <= 800 for c in chunks)
