"""Unit tests for Day 1 tokenization helpers."""

from __future__ import annotations

import unittest
from io import StringIO
from typing import Any, cast
from unittest.mock import patch

import torch

from transformers_week import day01_tokenization
from transformers_week.day01_tokenization import (
    ENGLISH_MODEL_NAME,
    MULTILINGUAL_MODEL_NAME,
    TokenizationExplanation,
    build_parser,
    decode_input_ids,
    explain_tokenization,
    get_recommended_model_name,
    get_special_tokens_info,
    load_tokenizer,
    main,
    tokenize_texts,
)


class FakeTokenizer:
    cls_token = "[CLS]"
    cls_token_id = 101
    sep_token = "[SEP]"
    sep_token_id = 102
    pad_token = "[PAD]"
    pad_token_id = 0
    vocab_size = 999
    model_max_length = 512

    def __call__(
        self,
        texts: list[str] | str,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int = 512,
        return_tensors: str | None = None,
    ) -> dict[str, torch.Tensor] | dict[str, list[int]]:
        del padding, truncation
        normalized_texts = [texts] if isinstance(texts, str) else texts
        ids: list[list[int]] = []
        masks: list[list[int]] = []
        for text in normalized_texts:
            encoded = [len(token) for token in text.split()]
            encoded = encoded[:max_length]
            ids.append(encoded)
        max_len = max(len(row) for row in ids)
        for row in ids:
            padding_needed = max_len - len(row)
            masks.append([1] * len(row) + [0] * padding_needed)
            row.extend([0] * padding_needed)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(ids),
                "attention_mask": torch.tensor(masks),
            }

        return {
            "input_ids": ids[0],
            "token_type_ids": [0] * len(ids[0]),
            "attention_mask": masks[0],
        }

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [len(token) for token in tokens]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(f"T{token_id}" for token_id in token_ids)


class Day01TokenizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = FakeTokenizer()

    def test_get_recommended_model_name_for_english(self) -> None:
        self.assertEqual(get_recommended_model_name("en"), ENGLISH_MODEL_NAME)
        self.assertEqual(get_recommended_model_name("english"), ENGLISH_MODEL_NAME)

    def test_get_recommended_model_name_for_russian(self) -> None:
        self.assertEqual(get_recommended_model_name("ru"), MULTILINGUAL_MODEL_NAME)
        self.assertEqual(get_recommended_model_name("multilingual"), MULTILINGUAL_MODEL_NAME)

    def test_get_recommended_model_name_rejects_unknown_language(self) -> None:
        with self.assertRaises(ValueError):
            get_recommended_model_name("de")

    def test_load_tokenizer_rejects_empty_model_name(self) -> None:
        with self.assertRaises(ValueError):
            load_tokenizer("   ")

    def test_load_tokenizer_delegates_to_auto_tokenizer(self) -> None:
        fake_loaded = object()
        with patch.object(
            day01_tokenization.AutoTokenizer,
            "from_pretrained",
            return_value=fake_loaded,
        ) as mocked:
            loaded = load_tokenizer("distilbert-base-uncased")

        mocked.assert_called_once_with("distilbert-base-uncased")
        self.assertIs(loaded, fake_loaded)

    def test_tokenize_texts_returns_batched_tensors(self) -> None:
        encoded = tokenize_texts(
            ["great movie", "bad"],
            self.tokenizer,
            max_length=4,
        )
        self.assertEqual(tuple(encoded["input_ids"].shape), (2, 2))
        self.assertEqual(tuple(encoded["attention_mask"].shape), (2, 2))

    def test_tokenize_texts_accepts_single_string_input(self) -> None:
        encoded = tokenize_texts("great movie", self.tokenizer, max_length=4)
        self.assertEqual(tuple(encoded["input_ids"].shape), (1, 2))
        self.assertEqual(tuple(encoded["attention_mask"].shape), (1, 2))

    def test_tokenize_texts_rejects_empty_batches(self) -> None:
        with self.assertRaises(ValueError):
            tokenize_texts([], self.tokenizer)

    def test_tokenize_texts_rejects_non_string_items(self) -> None:
        bad_item = cast(Any, 7)
        with self.assertRaises(TypeError):
            tokenize_texts(["great", bad_item], self.tokenizer)

    def test_tokenize_texts_rejects_blank_input(self) -> None:
        with self.assertRaises(ValueError):
            tokenize_texts(["great movie", "   "], self.tokenizer)

    def test_tokenize_texts_rejects_non_positive_max_length(self) -> None:
        with self.assertRaises(ValueError):
            tokenize_texts(["great movie"], self.tokenizer, max_length=0)

    def test_decode_input_ids_accepts_tensor(self) -> None:
        decoded = decode_input_ids(torch.tensor([5, 2]), self.tokenizer)
        self.assertEqual(decoded, "T5 T2")

    def test_decode_input_ids_accepts_sequence(self) -> None:
        decoded = decode_input_ids([4, 1], self.tokenizer)
        self.assertEqual(decoded, "T4 T1")

    def test_explain_tokenization_returns_expected_details(self) -> None:
        explanation = explain_tokenization("amazing movie", self.tokenizer)
        self.assertEqual(explanation.tokens, ["amazing", "movie"])
        self.assertEqual(explanation.token_ids, [7, 5])
        self.assertEqual(explanation.token_count, 2)
        self.assertEqual(explanation.decoded_text, "T7 T5")

    def test_explain_tokenization_rejects_blank_text(self) -> None:
        with self.assertRaises(ValueError):
            explain_tokenization("   ", self.tokenizer)

    def test_get_special_tokens_info_returns_core_fields(self) -> None:
        info = get_special_tokens_info(self.tokenizer)
        self.assertEqual(info["cls_token"], "[CLS]")
        self.assertEqual(info["sep_token_id"], 102)
        self.assertEqual(info["pad_token_id"], 0)

    def test_tokenization_explanation_to_pretty_json(self) -> None:
        explanation = TokenizationExplanation(
            original_text="hello",
            tokens=["hello"],
            token_ids=[5],
            token_count=1,
            decoded_text="hello",
        )
        payload = explanation.to_pretty_json()

        self.assertIn('"original_text": "hello"', payload)
        self.assertIn('"token_count": 1', payload)

    def test_build_parser_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])

        self.assertEqual(args.language, "en")
        self.assertEqual(args.max_length, 128)
        self.assertEqual(
            args.texts,
            [
                "This movie was absolutely amazing!",
                "Terrible movie, waste of time.",
                "Pretty good, I liked it.",
            ],
        )

    def test_demo_prints_expected_sections(self) -> None:
        stream = StringIO()
        with (
            patch.object(day01_tokenization, "load_tokenizer", return_value=self.tokenizer),
            patch("sys.stdout", new=stream),
        ):
            day01_tokenization._demo(["great movie", "bad movie"], language="en", max_length=8)

        output = stream.getvalue()
        self.assertIn("Model name: distilbert-base-uncased", output)
        self.assertIn("=== Single text tokenization ===", output)
        self.assertIn("=== Batch tokenization ===", output)
        self.assertIn("=== Special tokens ===", output)
        self.assertIn("=== Tokenization explanation ===", output)

    def test_main_forwards_cli_args_to_demo(self) -> None:
        with (
            patch(
                "sys.argv",
                ["transformers-day1", "--language", "ru", "--max-length", "64", "text"],
            ),
            patch.object(day01_tokenization, "_demo") as mocked_demo,
        ):
            main()

        mocked_demo.assert_called_once_with(["text"], language="ru", max_length=64)


if __name__ == "__main__":
    unittest.main()
