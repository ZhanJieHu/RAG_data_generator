"""Tests for qcfa_generation_agent."""

import unittest
from unittest.mock import patch, MagicMock

from src.agents.qcfa_generation_agent import generate_qcfa, _parse_qcfa_from_text


def _make_text_response(text: str):
    resp = MagicMock()
    resp.choices = [MagicMock(finish_reason="stop", message=MagicMock(content=text))]
    return resp


class TestParseQcfaFromText(unittest.TestCase):
    """Tests for _parse_qcfa_from_text()."""

    def test_standard_format(self):
        text = "QUESTION:\n徽商中的多数人主要通过什么商业活动发家？\n\nANSWER:\n主要通过长途贩运活动发家。"
        q, a = _parse_qcfa_from_text(text)
        self.assertEqual(q, "徽商中的多数人主要通过什么商业活动发家？")
        self.assertEqual(a, "主要通过长途贩运活动发家。")

    def test_chinese_colon(self):
        text = "QUESTION：\n徽商经营什么？\n\nANSWER：\n盐业。"
        q, a = _parse_qcfa_from_text(text)
        self.assertEqual(q, "徽商经营什么？")
        self.assertEqual(a, "盐业。")

    def test_multi_line_answer(self):
        text = "QUESTION:\n问题？\n\nANSWER:\n第一句。\n第二句。"
        q, a = _parse_qcfa_from_text(text)
        self.assertEqual(q, "问题？")
        self.assertEqual(a, "第一句。 第二句。")

    def test_no_answer(self):
        text = "QUESTION:\n问题？"
        q, a = _parse_qcfa_from_text(text)
        self.assertEqual(q, "问题？")
        self.assertEqual(a, "")

    def test_empty_text(self):
        q, a = _parse_qcfa_from_text("")
        self.assertEqual(q, "")
        self.assertEqual(a, "")


class TestGenerateQcfa(unittest.TestCase):
    """Tests for generate_qcfa()."""

    def setUp(self):
        self.api_key = "test-key"
        self.fact_record = {
            "chunk": "徽商中的多数人通过长途贩运活动发家。",
            "fact": "徽商中的多数人通过长途贩运活动发家。",
        }

    @patch("src.agents.qcfa_generation_agent.OpenAI")
    def test_generate_success(self, mock_openai):
        """Normal case returns a QCFARecord."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_text_response(
            "QUESTION:\n徽商主要通过什么发家？\n\nANSWER:\n长途贩运。"
        )
        mock_openai.return_value = mock_client

        result = generate_qcfa(self.fact_record, api_key=self.api_key)

        self.assertIsNotNone(result)
        self.assertEqual(result["question"], "徽商主要通过什么发家？")
        self.assertEqual(result["answer"], "长途贩运。")
        self.assertEqual(result["chunk"], self.fact_record["chunk"])
        self.assertEqual(result["fact"], self.fact_record["fact"])

    @patch("src.agents.qcfa_generation_agent.OpenAI")
    def test_empty_fact_text(self, mock_openai):
        """Empty fact text returns None."""
        result = generate_qcfa({"chunk": "text", "fact": ""}, api_key=self.api_key)
        self.assertIsNone(result)
        mock_openai.assert_not_called()

    @patch("src.agents.qcfa_generation_agent.OpenAI")
    def test_empty_chunk_text(self, mock_openai):
        """Empty chunk text returns None."""
        result = generate_qcfa({"chunk": "", "fact": "fact"}, api_key=self.api_key)
        self.assertIsNone(result)
        mock_openai.assert_not_called()

    @patch("src.agents.qcfa_generation_agent.OpenAI")
    def test_missing_keys(self, mock_openai):
        """Missing keys returns None."""
        result = generate_qcfa({}, api_key=self.api_key)
        self.assertIsNone(result)
        mock_openai.assert_not_called()

    @patch("src.agents.qcfa_generation_agent.OpenAI")
    def test_api_exception(self, mock_openai):
        """API exception returns None."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_openai.return_value = mock_client

        result = generate_qcfa(self.fact_record, api_key=self.api_key)
        self.assertIsNone(result)


class TestGenerateQcfaIntegration(unittest.TestCase):
    """Light integration tests."""

    def test_prompt_file_exists(self):
        """Prompt markdown file exists and is non-empty."""
        import os
        path = os.path.join(
            os.path.dirname(__file__), "..", "src", "prompts", "qcfa_generation.md",
        )
        self.assertTrue(os.path.isfile(path))
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertGreater(len(content), 100)


if __name__ == "__main__":
    unittest.main()
