"""Day 1: tokenization utilities for the transformers homework project.

This module keeps the first-day code production-ready:
* type hints;
* validation and explicit errors;
* reusable pure functions;
* a small CLI demo suitable for showing the homework progress.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch
from transformers import AutoTokenizer

ENGLISH_MODEL_NAME = "distilbert-base-uncased"
MULTILINGUAL_MODEL_NAME = "distilbert-base-multilingual-cased"
TokenizerLike = Any


@dataclass(frozen=True)
class TokenizationExplanation:
    """Human-readable explanation of how a text was tokenized."""

    original_text: str
    tokens: list[str]
    token_ids: list[int]
    token_count: int
    decoded_text: str

    def to_pretty_json(self) -> str:
        """Serialize the explanation for CLI/debug output."""

        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def get_recommended_model_name(language: str = "en") -> str:
    """Return the recommended model name for the given language code."""

    normalized = language.strip().lower()
    if normalized in {"en", "eng", "english"}:
        return ENGLISH_MODEL_NAME
    if normalized in {"ru", "rus", "russian", "multi", "multilingual"}:
        return MULTILINGUAL_MODEL_NAME
    raise ValueError(
        f"Unsupported language '{language}'. Use 'en' for English or 'ru' for Russian/multilingual."
    )


def load_tokenizer(model_name: str) -> TokenizerLike:
    """Load a Hugging Face tokenizer by model name."""

    if not model_name or not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")
    return AutoTokenizer.from_pretrained(model_name)


def _normalize_texts(texts: Sequence[str] | str) -> list[str]:
    """Normalize a single string or a sequence of strings into a non-empty list."""

    normalized = [texts] if isinstance(texts, str) else list(texts)

    if not normalized:
        raise ValueError("At least one text is required for tokenization.")
    if any(not isinstance(text, str) for text in normalized):
        raise TypeError("All items passed to tokenize_texts must be strings.")
    if any(not text.strip() for text in normalized):
        raise ValueError("Texts must not be empty or whitespace-only.")
    return normalized


def tokenize_texts(
    texts: Sequence[str] | str,
    tokenizer: TokenizerLike,
    *,
    max_length: int = 128,
    return_tensors: str = "pt",
) -> Any:
    """Tokenize one or multiple texts with padding and truncation enabled."""

    normalized_texts = _normalize_texts(texts)
    if max_length <= 0:
        raise ValueError("max_length must be a positive integer.")

    return tokenizer(
        normalized_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors=return_tensors,
    )


def decode_input_ids(input_ids: Sequence[int] | torch.Tensor, tokenizer: TokenizerLike) -> str:
    """Decode a 1D list/tensor of token ids back into text."""

    if isinstance(input_ids, torch.Tensor):
        normalized_ids = input_ids.detach().cpu().tolist()
    else:
        normalized_ids = list(input_ids)
    return cast(str, tokenizer.decode(normalized_ids))


def get_special_tokens_info(tokenizer: TokenizerLike) -> dict[str, Any]:
    """Return the special tokens most relevant for day 1."""

    return {
        "cls_token": tokenizer.cls_token,
        "cls_token_id": tokenizer.cls_token_id,
        "sep_token": tokenizer.sep_token,
        "sep_token_id": tokenizer.sep_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
    }


def explain_tokenization(text: str, tokenizer: TokenizerLike) -> TokenizationExplanation:
    """Explain how the tokenizer splits a text into tokens and ids."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string.")

    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    decoded_text = tokenizer.decode(token_ids)
    return TokenizationExplanation(
        original_text=text,
        tokens=tokens,
        token_ids=token_ids,
        token_count=len(tokens),
        decoded_text=decoded_text,
    )


def _demo(texts: list[str], language: str, max_length: int) -> None:
    """Run a small CLI demo for the day 1 assignment."""

    model_name = get_recommended_model_name(language)
    tokenizer = load_tokenizer(model_name)

    print(f"Model name: {model_name}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Model max length: {tokenizer.model_max_length}")
    print()

    single_text = texts[0]
    single_tokens = tokenizer(single_text)
    input_ids = single_tokens["input_ids"]

    print("=== Single text tokenization ===")
    print(f"Text: {single_text}")
    print(f"Raw tokens payload: {single_tokens}")
    print(f"Number of tokens: {len(input_ids)}")
    print(f"Decoded back: {tokenizer.decode(input_ids)}")
    print()

    batch_tokens = tokenize_texts(texts, tokenizer, max_length=max_length)
    print("=== Batch tokenization ===")
    print(f"Input IDs shape: {tuple(batch_tokens['input_ids'].shape)}")
    print(f"Attention mask:\n{batch_tokens['attention_mask']}")
    print()

    print("=== Special tokens ===")
    print(json.dumps(get_special_tokens_info(tokenizer), ensure_ascii=False, indent=2))
    print()

    print("=== Tokenization explanation ===")
    print(explain_tokenization(single_text, tokenizer).to_pretty_json())


def build_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for the Day 1 demo."""

    parser = argparse.ArgumentParser(
        description="Run the Day 1 tokenization demo for the transformers homework project."
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language choice: 'en' for English, 'ru' for Russian/multilingual.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length for padded batch tokenization.",
    )
    parser.add_argument(
        "texts",
        nargs="*",
        default=[
            "This movie was absolutely amazing!",
            "Terrible movie, waste of time.",
            "Pretty good, I liked it.",
        ],
        help="Texts to tokenize. If omitted, built-in examples are used.",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""

    args = build_parser().parse_args()
    _demo(args.texts, language=args.language, max_length=args.max_length)


if __name__ == "__main__":  # pragma: no cover
    main()
