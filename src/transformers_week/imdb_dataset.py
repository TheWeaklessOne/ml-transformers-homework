"""Download and convert the IMDb movie reviews dataset into repo-friendly CSV files."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import re
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve

IMDB_SOURCE_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
LABEL_BY_DIRECTORY = {"neg": 0, "pos": 1}
LABEL_NAMES = {"0": "negative", "1": "positive"}
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
BR_TAG_PATTERN = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)


def normalize_review_text(text: str) -> str:
    """Apply lightweight deterministic cleanup to the raw IMDb review text."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = BR_TAG_PATTERN.sub("\n", normalized)
    return normalized.strip()


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    """Extract a tar archive while preventing path traversal."""

    destination = destination.resolve()
    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()
        if member_path != destination and destination not in member_path.parents:
            raise RuntimeError(f"Archive member escapes destination: {member.name}")
    if "filter" in inspect.signature(tarfile.TarFile.extractall).parameters:
        archive.extractall(destination, filter="data")
        return
    archive.extractall(destination)


def _iter_review_records(split_dir: Path) -> list[dict[str, int | str]]:
    """Collect labeled review records from one IMDb split directory."""

    records: list[dict[str, int | str]] = []
    for directory_name, label in LABEL_BY_DIRECTORY.items():
        review_dir = split_dir / directory_name
        if not review_dir.is_dir():
            raise FileNotFoundError(f"Expected review directory at {review_dir}")

        for review_file in sorted(review_dir.glob("*.txt")):
            records.append(
                {
                    TEXT_COLUMN: normalize_review_text(review_file.read_text(encoding="utf-8")),
                    LABEL_COLUMN: label,
                }
            )
    if not records:
        raise ValueError(f"No review records found under {split_dir}")
    return records


def write_split_csv(split_dir: Path, destination: Path) -> int:
    """Write one IMDb split to CSV and return the number of written rows."""

    records = _iter_review_records(split_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[TEXT_COLUMN, LABEL_COLUMN])
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def prepare_imdb_dataset(output_dir: Path, *, force: bool = False) -> dict[str, object]:
    """Download the official IMDb archive and convert train/test splits into CSV files."""

    output_dir = output_dir.resolve()
    train_csv = output_dir / "train.csv"
    test_csv = output_dir / "test.csv"
    metadata_path = output_dir / "metadata.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_targets = [train_csv, test_csv, metadata_path]
    if any(path.exists() for path in existing_targets) and not force:
        raise FileExistsError(
            "One or more IMDb output files already exist. Re-run with force=True to replace them."
        )
    if force:
        for path in existing_targets:
            if path.exists():
                path.unlink()

    with tempfile.TemporaryDirectory(prefix="imdb-download-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        archive_path = temp_dir / "aclImdb_v1.tar.gz"
        urlretrieve(IMDB_SOURCE_URL, archive_path)

        with tarfile.open(archive_path, mode="r:gz") as archive:
            _safe_extract(archive, temp_dir)

        dataset_root = temp_dir / "aclImdb"
        train_rows = write_split_csv(dataset_root / "train", train_csv)
        test_rows = write_split_csv(dataset_root / "test", test_csv)

    metadata: dict[str, object] = {
        "source_url": IMDB_SOURCE_URL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "text_column": TEXT_COLUMN,
        "label_column": LABEL_COLUMN,
        "labels": LABEL_NAMES,
        "files": {
            train_csv.name: train_rows,
            test_csv.name: test_rows,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser used to fetch the dataset into the repo."""

    parser = argparse.ArgumentParser(
        description="Download the official IMDb movie reviews dataset into data/imdb CSV files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/imdb"),
        help="Directory where train.csv, test.csv, and metadata.json should be written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory if it already exists.",
    )
    return parser


def main() -> None:
    """CLI entrypoint for dataset preparation."""

    args = build_parser().parse_args()
    metadata = prepare_imdb_dataset(args.output_dir, force=args.force)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
