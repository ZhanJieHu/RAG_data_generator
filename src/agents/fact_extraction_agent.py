"""Fact extraction agent — extracts candidate facts from ChunkRecords."""

import os
import re

from openai import OpenAI

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "fact_extraction.md")


def _load_prompt() -> str:
    with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _parse_facts_from_text(text: str) -> list[str]:
    """Parse facts from model output.

    Supports two formats:
      1. Lines starting with ``- `` (bullet list)
      2. Lines matching ``N. `` (numbered list)

    Stops at the first blank line after content begins.
    """
    facts = []
    in_facts_section = False
    for line in text.splitlines():
        stripped = line.strip()

        # Detect section header
        if re.match(r"^(事实|Facts|facts|提取的事实)[：:]", stripped):
            in_facts_section = True
            continue
        if stripped.upper() == "FACTS:":
            in_facts_section = True
            continue
        if stripped in ("NO_FACTS", "无"):
            return []

        # Bullet or numbered line
        bullet_match = re.match(r"^- (.+)", stripped)
        number_match = re.match(r"^\d+[\.、]\s*(.+)", stripped)
        if bullet_match:
            facts.append(bullet_match.group(1))
            in_facts_section = True
        elif number_match:
            facts.append(number_match.group(1))
            in_facts_section = True
        elif in_facts_section and stripped == "":
            break  # blank line ends the facts section

    return facts


def extract_facts(
    chunk_record: dict,
    api_key: str,
    model: str = "deepseek-v4-flash",
    base_url: str = "https://api.deepseek.com",
) -> list[dict]:
    """Extract 0–3 candidate FactRecords from a single ChunkRecord.

    Parameters
    ----------
    chunk_record : dict
        A dict with at least a ``"text"`` key (the chunk text).
    api_key : str
        DeepSeek API key.
    model : str
        Model name.
    base_url : str
        API base URL.

    Returns
    -------
    list[dict]
        List of FactRecords, each ``{"chunk": str, "fact": str}``.
    """
    chunk_text = chunk_record.get("text", "")
    if not chunk_text or not chunk_text.strip():
        return []

    system_prompt = _load_prompt()
    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk_text},
            ],
            temperature=0.1,
        )
    except Exception as e:
        print(f"    [error] API call failed: {e}", file=__import__("sys").stderr)
        return []

    content = response.choices[0].message.content or ""
    facts = _parse_facts_from_text(content)

    return [
        {"chunk": chunk_text, "fact": fact}
        for fact in facts
        if isinstance(fact, str) and fact.strip()
    ][:3]
