"""QCFA generation agent — generates a QCFA record from a FactRecord."""

import os
import re

from openai import OpenAI

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "qcfa_generation.md")


def _load_prompt() -> str:
    with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _parse_qcfa_from_text(text: str) -> tuple[str, str]:
    """Parse QUESTION / ANSWER from model output."""
    question = ""
    answer = ""
    current_section = None
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^QUESTION[：:]", stripped, re.IGNORECASE):
            current_section = "q"
            question = stripped.split("：", 1)[-1].split(":", 1)[-1].strip()
        elif re.match(r"^ANSWER[：:]", stripped, re.IGNORECASE):
            current_section = "a"
            answer = stripped.split("：", 1)[-1].split(":", 1)[-1].strip()
        elif current_section == "q" and stripped:
            question += " " + stripped
        elif current_section == "a" and stripped:
            answer += " " + stripped
        elif not stripped:
            current_section = None
    return question.strip(), answer.strip()


def generate_qcfa(
    fact_record: dict,
    api_key: str,
    model: str = "deepseek-v4-flash",
    base_url: str = "https://api.deepseek.com",
) -> dict | None:
    """Generate a QCFARecord from a FactRecord.

    Parameters
    ----------
    fact_record : dict
        A dict with ``"chunk"`` and ``"fact"`` keys.
    api_key : str
        DeepSeek API key.
    model : str
        Model name.
    base_url : str
        API base URL.

    Returns
    -------
    dict | None
        QCFARecord ``{"question": str, "answer": str, "chunk": str, "fact": str}``,
        or ``None`` on failure.
    """
    chunk_text = fact_record.get("chunk", "")
    fact_text = fact_record.get("fact", "")
    if not chunk_text or not fact_text:
        return None

    system_prompt = _load_prompt()
    client = OpenAI(api_key=api_key, base_url=base_url)

    user_message = (
        f"===\n事实：{fact_text}\n"
        f"===\n原文文段：{chunk_text}\n"
        f"===\n请根据上述事实生成 QUESTION 和 ANSWER。"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
        )
    except Exception as e:
        print(f"    [error] API call failed: {e}", file=__import__("sys").stderr)
        return None

    content = response.choices[0].message.content or ""
    question, answer = _parse_qcfa_from_text(content)

    if not question or not answer:
        return None

    return {
        "question": question,
        "answer": answer,
        "chunk": chunk_text,
        "fact": fact_text,
    }
