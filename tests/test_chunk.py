"""Tests for document_processor chunking logic."""

import json
import os
import tempfile
import unittest

from src.document_processor import chunk_text, process_book, write_chunks_jsonl


class TestChunkText(unittest.TestCase):

    def test_exact_multiple(self):
        """Text length is an exact multiple of top_char."""
        text = "A" * 3000
        result = chunk_text(text, 1000)
        self.assertEqual(len(result), 3)
        for chunk in result:
            self.assertEqual(chunk["text"], "A" * 1000)
            self.assertEqual(chunk["char_count"], 1000)

    def test_tail_discarded(self):
        """Short trailing segment (< top_char) is discarded."""
        text = "A" * 2500  # 2 full chunks + 500 tail
        result = chunk_text(text, 1000)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["char_count"], 1000)
        self.assertEqual(result[1]["char_count"], 1000)

    def test_shorter_than_top_char(self):
        """Entire text shorter than top_char → empty result."""
        result = chunk_text("short", 1000)
        self.assertEqual(result, [])

    def test_empty_text(self):
        """Empty input → empty result."""
        result = chunk_text("", 100)
        self.assertEqual(result, [])

    def test_exact_boundary(self):
        """Text exactly equal to top_char → one chunk."""
        text = "B" * 500
        result = chunk_text(text, 500)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["char_count"], 500)

    def test_non_ascii_chinese(self):
        """Chinese characters (multi-byte) should work correctly."""
        text = "徽商" * 300  # 600 Chinese chars
        result = chunk_text(text, 200)
        self.assertEqual(len(result), 3)
        for chunk in result:
            self.assertEqual(len(chunk["text"]), 200)
            self.assertEqual(chunk["char_count"], 200)


class TestProcessBook(unittest.TestCase):

    def test_process_book_integration(self):
        """Integration test: write a temp file, process it, verify chunks."""
        content = "Hello World!\n" * 50  # 750 chars roughly
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(content)
            tmp_path = f.name

        try:
            chunks = process_book(tmp_path, top_char=200)
            self.assertGreater(len(chunks), 0)
            for chunk in chunks:
                self.assertIn("text", chunk)
                self.assertIn("char_count", chunk)
                self.assertEqual(chunk["char_count"], 200)
        finally:
            os.unlink(tmp_path)

    def test_file_not_found(self):
        """Non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            process_book("/nonexistent/book.txt", 1000)


class TestWriteChunksJsonl(unittest.TestCase):

    def test_write_and_read(self):
        """Write chunks to JSONL, then read back to verify."""
        chunks = [
            {"text": "chunk one", "char_count": 9},
            {"text": "chunk two", "char_count": 9},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test.jsonl")
            result = write_chunks_jsonl(chunks, out_path)

            self.assertEqual(result, out_path)
            self.assertTrue(os.path.isfile(out_path))

            with open(out_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)

            for i, line in enumerate(lines):
                data = json.loads(line)
                self.assertEqual(data, chunks[i])


if __name__ == "__main__":
    unittest.main()
