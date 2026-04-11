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
DEFAULT_SINGLE_TEXT = "This movie was absolutely amazing!"
DEFAULT_BATCH_TEXTS = [
    "This movie was great!",
    "Terrible movie, waste of time.",
]
DEFAULT_EXPLANATION_TEXT = "Transformers are amazing!"
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
    max_length: int = 128,
    tokenizer: TokenizerLike | None = None,
    *,
    return_tensors: str = "pt",
    language: str = "en",
) -> Any:
    """Tokenize one or multiple texts with padding and truncation enabled."""

    normalized_texts = _normalize_texts(texts)
    if max_length <= 0:
        raise ValueError("max_length must be a positive integer.")

    resolved_tokenizer = tokenizer
    if resolved_tokenizer is None:
        model_name = get_recommended_model_name(language)
        resolved_tokenizer = load_tokenizer(model_name)

    return resolved_tokenizer(
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


def _demo(
    texts: list[str],
    language: str,
    max_length: int,
    *,
    single_text: str,
    explanation_text: str,
) -> None:
    """Run a small CLI demo for the day 1 assignment."""

    model_name = get_recommended_model_name(language)
    tokenizer = load_tokenizer(model_name)

    print(f"Model name: {model_name}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Model max length: {tokenizer.model_max_length}")
    print()

    single_tokens = tokenizer(single_text)
    input_ids = single_tokens["input_ids"]

    print("=== Single text tokenization ===")
    print(f"Text: {single_text}")
    print(f"Raw tokens payload: {single_tokens}")
    print(f"Input IDs: {input_ids}")
    print(f"Number of tokens: {len(input_ids)}")
    print(f"Decoded back: {tokenizer.decode(input_ids)}")
    print()

    batch_tokens = tokenize_texts(texts, max_length=max_length, tokenizer=tokenizer)
    print("=== Batch tokenization ===")
    print(f"Input IDs shape: {tuple(batch_tokens['input_ids'].shape)}")
    print(f"Attention mask:\n{batch_tokens['attention_mask']}")
    print()

    print("=== Special tokens ===")
    special_tokens = get_special_tokens_info(tokenizer)
    print(f"CLS token: {special_tokens['cls_token']} (ID: {special_tokens['cls_token_id']})")
    print(f"SEP token: {special_tokens['sep_token']} (ID: {special_tokens['sep_token_id']})")
    print(f"PAD token: {special_tokens['pad_token']} (ID: {special_tokens['pad_token_id']})")
    print()

    print('=== Single text with return_tensors="pt" ===')
    single_with_tensors = tokenizer(single_text, return_tensors="pt")
    print(f"Input IDs: {single_with_tensors['input_ids']}")
    print(f"Decoded: {tokenizer.decode(single_with_tensors['input_ids'][0])}")
    print()

    print("=== Tokenization explanation ===")
    explanation = explain_tokenization(explanation_text, tokenizer)
    print(f"Исходный текст: {explanation.original_text}")
    print(f"Токены: {explanation.tokens}")
    print(f"IDs: {explanation.token_ids}")
    print(f"Количество: {explanation.token_count}")


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
        "--single-text",
        default=DEFAULT_SINGLE_TEXT,
        help="Single-text example used for the initial tokenization walkthrough.",
    )
    parser.add_argument(
        "--explanation-text",
        default=DEFAULT_EXPLANATION_TEXT,
        help="Text passed to explain_tokenization(...).",
    )
    parser.add_argument(
        "texts",
        nargs="*",
        default=DEFAULT_BATCH_TEXTS,
        help="Batch texts to tokenize. If omitted, the homework examples are used.",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""

    args = build_parser().parse_args()
    _demo(
        args.texts,
        language=args.language,
        max_length=args.max_length,
        single_text=args.single_text,
        explanation_text=args.explanation_text,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
