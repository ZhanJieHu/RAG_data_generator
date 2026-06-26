import json
import os

def chunk_text(text: str, top_char: int) -> list[dict]:
    """Split text into chunks of `top_char` characters.

    Discards the final chunk if it is shorter than `top_char`.
    Returns a list of dicts: {"text": str, "char_count": int}.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + top_char
        segment = text[start:end]
        if len(segment) < top_char:
            break  # discard tail shorter than top_char
        chunks.append({"text": segment, "char_count": top_char})
        start = end
    return chunks


def process_book(file_path: str, top_char: int) -> list[dict]:
    """Read a .txt/.md book file and return chunked result."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Book file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return chunk_text(text, top_char)


def write_chunks_jsonl(chunks: list[dict], output_path: str) -> str:
    """Write chunks to a JSONL file. Returns the output path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return output_path
