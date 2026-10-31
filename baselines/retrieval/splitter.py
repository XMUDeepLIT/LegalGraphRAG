"""Token windows for the retrieval baselines."""
from typing import List


def split_tokens(text: str, chunk_tokens: int, overlap_tokens: int, encoding_name: str = "cl100k_base") -> List[str]:
    """Split ``text`` into ``cl100k_base`` windows of ``chunk_tokens`` with ``overlap_tokens``."""
    import tiktoken

    if chunk_tokens <= overlap_tokens:
        raise ValueError("chunk_tokens must be greater than overlap_tokens")
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    if not tokens:
        return []
    chunks: List[str] = []
    start = 0
    step = chunk_tokens - overlap_tokens
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunks.append(encoding.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start += step
    return chunks
