"""Tests for fact_extraction_agent."""

import unittest
from unittest.mock import patch, MagicMock

from src.agents.fact_extraction_agent import extract_facts, _parse_facts_from_text


def _make_text_response(text: str):
    """Build a mock response that returns the given text content."""
    resp = MagicMock()
    resp.choices = [MagicMock(finish_reason="stop", message=MagicMock(content=text))]
    return resp


class TestParseFactsFromText(unittest.TestCase):
    """Tests for _parse_facts_from_text()."""

    def test_bullet_list(self):
        text = "FACTS:\n- 徽州商人经营盐业。\n- 徽商在扬州从事木材贸易。"
        self.assertEqual(
            _parse_facts_from_text(text),
            ["徽州商人经营盐业。", "徽商在扬州从事木材贸易。"],
        )

    def test_numbered_list(self):
        text = "FACTS：\n1. 徽州商人经营盐业。\n2. 徽商在扬州从事木材贸易。"
        self.assertEqual(
            _parse_facts_from_text(text),
            ["徽州商人经营盐业。", "徽商在扬州从事木材贸易。"],
        )

    def test_chinese_numbered(self):
        text = "事实：\n1、徽州商人经营盐业。\n2、徽商在扬州从事木材贸易。"
        self.assertEqual(
            _parse_facts_from_text(text),
            ["徽州商人经营盐业。", "徽商在扬州从事木材贸易。"],
        )

    def test_no_facts(self):
        text = "NO_FACTS"
        self.assertEqual(_parse_facts_from_text(text), [])

    def test_no_facts_chinese(self):
        text = "无"
        self.assertEqual(_parse_facts_from_text(text), [])

    def test_empty_text(self):
        self.assertEqual(_parse_facts_from_text(""), [])

    def test_stops_at_blank_line(self):
        text = "FACTS:\n- 事实一\n- 事实二\n\n后面还有内容"
        self.assertEqual(
            _parse_facts_from_text(text), ["事实一", "事实二"]
        )

    def test_section_header_chinese(self):
        text = "提取的事实：\n- 徽商贩卖茶叶"
        self.assertEqual(
            _parse_facts_from_text(text), ["徽商贩卖茶叶"]
        )


class TestExtractFacts(unittest.TestCase):
    """Tests for extract_facts()."""

    def setUp(self):
        self.api_key = "test-key"
        self.chunk_text = "徽州商人经营盐业，获利颇丰。"

    @patch("src.agents.fact_extraction_agent.OpenAI")
    def test_extract_single_fact(self, mock_openai):
        """Returns one FactRecord from a bullet list response."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_text_response(
            "FACTS:\n- 徽州商人经营盐业。"
        )
        mock_openai.return_value = mock_client

        result = extract_facts({"text": self.chunk_text}, api_key=self.api_key)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["chunk"], self.chunk_text)
        self.assertEqual(result[0]["fact"], "徽州商人经营盐业。")

    @patch("src.agents.fact_extraction_agent.OpenAI")
    def test_extract_multiple_facts(self, mock_openai):
        """Returns multiple FactRecords."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_text_response(
            "FACTS:\n- 徽州商人经营盐业。\n- 徽商在扬州从事木材贸易。"
        )
        mock_openai.return_value = mock_client

        result = extract_facts({"text": self.chunk_text}, api_key=self.api_key)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["fact"], "徽州商人经营盐业。")
        self.assertEqual(result[1]["fact"], "徽商在扬州从事木材贸易。")

    @patch("src.agents.fact_extraction_agent.OpenAI")
    def test_no_facts(self, mock_openai):
        """NO_FACTS response returns empty list."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_text_response("NO_FACTS")
        mock_openai.return_value = mock_client

        result = extract_facts({"text": self.chunk_text}, api_key=self.api_key)

        self.assertEqual(result, [])

    @patch("src.agents.fact_extraction_agent.OpenAI")
    def test_api_exception(self, mock_openai):
        """API exception is caught, returns empty list."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_openai.return_value = mock_client

        result = extract_facts({"text": self.chunk_text}, api_key=self.api_key)

        self.assertEqual(result, [])

    @patch("src.agents.fact_extraction_agent.OpenAI")
    def test_empty_chunk_text(self, mock_openai):
        """Empty chunk text skips API call."""
        result = extract_facts({"text": ""}, api_key=self.api_key)

        self.assertEqual(result, [])
        mock_openai.assert_not_called()

    @patch("src.agents.fact_extraction_agent.OpenAI")
    def test_whitespace_chunk(self, mock_openai):
        """Whitespace-only chunk skips API call."""
        result = extract_facts({"text": "   \n  "}, api_key=self.api_key)

        self.assertEqual(result, [])
        mock_openai.assert_not_called()

    @patch("src.agents.fact_extraction_agent.OpenAI")
    def test_missing_text_key(self, mock_openai):
        """Chunk without 'text' key skips API call."""
        result = extract_facts({"char_count": 100}, api_key=self.api_key)

        self.assertEqual(result, [])
        mock_openai.assert_not_called()

    @patch("src.agents.fact_extraction_agent.OpenAI")
    def test_capped_at_3_facts(self, mock_openai):
        """More than 3 facts in response → only first 3 kept."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_text_response(
            "FACTS:\n- 事实一\n- 事实二\n- 事实三\n- 事实四\n- 事实五"
        )
        mock_openai.return_value = mock_client

        result = extract_facts({"text": self.chunk_text}, api_key=self.api_key)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["fact"], "事实一")
        self.assertEqual(result[2]["fact"], "事实三")


class TestExtractFactsIntegration(unittest.TestCase):
    """Light integration tests."""

    def test_prompt_file_exists(self):
        """Prompt markdown file exists and is non-empty."""
        import os
        path = os.path.join(
            os.path.dirname(__file__), "..", "src", "prompts", "fact_extraction.md",
        )
        self.assertTrue(os.path.isfile(path))
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertGreater(len(content), 100)


if __name__ == "__main__":
    unittest.main()
