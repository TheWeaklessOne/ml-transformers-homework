"""Unit tests for IMDb dataset preparation helpers."""

from __future__ import annotations

import csv
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from shutil import copyfile
from tarfile import TarInfo
from types import SimpleNamespace
from unittest.mock import Mock, patch

from transformers_week.imdb_dataset import (
    LABEL_COLUMN,
    TEXT_COLUMN,
    _iter_review_records,
    _safe_extract,
    build_parser,
    main,
    normalize_review_text,
    prepare_imdb_dataset,
    write_split_csv,
)


class ImdbDatasetTests(unittest.TestCase):
    def test_normalize_review_text_replaces_html_breaks(self) -> None:
        normalized = normalize_review_text("Great movie!<br /><br />Loved it.\r\n")
        self.assertEqual(normalized, "Great movie!\n\nLoved it.")

    def test_iter_review_records_reads_pos_and_neg_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            split_dir = Path(temp_dir_name)
            neg_dir = split_dir / "neg"
            pos_dir = split_dir / "pos"
            neg_dir.mkdir()
            pos_dir.mkdir()
            (neg_dir / "0_1.txt").write_text("bad movie", encoding="utf-8")
            (pos_dir / "1_9.txt").write_text("great movie", encoding="utf-8")

            records = _iter_review_records(split_dir)

        self.assertEqual(records[0][LABEL_COLUMN], 0)
        self.assertEqual(records[0][TEXT_COLUMN], "bad movie")
        self.assertEqual(records[1][LABEL_COLUMN], 1)
        self.assertEqual(records[1][TEXT_COLUMN], "great movie")

    def test_iter_review_records_requires_expected_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            split_dir = Path(temp_dir_name)
            (split_dir / "neg").mkdir()

            with self.assertRaises(FileNotFoundError):
                _iter_review_records(split_dir)

    def test_iter_review_records_rejects_empty_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            split_dir = Path(temp_dir_name)
            (split_dir / "neg").mkdir()
            (split_dir / "pos").mkdir()

            with self.assertRaises(ValueError):
                _iter_review_records(split_dir)

    def test_write_split_csv_creates_expected_header_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            split_dir = temp_dir / "train"
            destination = temp_dir / "train.csv"
            (split_dir / "neg").mkdir(parents=True)
            (split_dir / "pos").mkdir(parents=True)
            (split_dir / "neg" / "0_1.txt").write_text("bad<br />movie", encoding="utf-8")
            (split_dir / "pos" / "1_9.txt").write_text("great movie", encoding="utf-8")

            written_rows = write_split_csv(split_dir, destination)

            with destination.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(written_rows, 2)
        self.assertEqual(rows[0][LABEL_COLUMN], "0")
        self.assertEqual(rows[0][TEXT_COLUMN], "bad\nmovie")
        self.assertEqual(rows[1][LABEL_COLUMN], "1")
        self.assertEqual(rows[1][TEXT_COLUMN], "great movie")

    def test_safe_extract_uses_filter_when_supported(self) -> None:
        archive = Mock()
        archive.getmembers.return_value = [TarInfo(name="aclImdb/train/neg/0_1.txt")]
        with (
            tempfile.TemporaryDirectory() as temp_dir_name,
            patch(
                "transformers_week.imdb_dataset.inspect.signature",
                return_value=SimpleNamespace(parameters={"filter": object()}),
            ),
        ):
            destination = Path(temp_dir_name)
            _safe_extract(archive, destination)

        archive.extractall.assert_called_once_with(destination.resolve(), filter="data")

    def test_safe_extract_falls_back_without_filter_support(self) -> None:
        archive = Mock()
        archive.getmembers.return_value = [TarInfo(name="aclImdb/train/neg/0_1.txt")]
        with (
            tempfile.TemporaryDirectory() as temp_dir_name,
            patch(
                "transformers_week.imdb_dataset.inspect.signature",
                return_value=SimpleNamespace(parameters={}),
            ),
        ):
            destination = Path(temp_dir_name)
            _safe_extract(archive, destination)

        archive.extractall.assert_called_once_with(destination.resolve())

    def test_safe_extract_rejects_path_traversal(self) -> None:
        archive = Mock()
        archive.getmembers.return_value = [TarInfo(name="../../evil.txt")]

        with (
            tempfile.TemporaryDirectory() as temp_dir_name,
            self.assertRaises(RuntimeError),
        ):
            _safe_extract(archive, Path(temp_dir_name))

    def test_prepare_imdb_dataset_writes_csv_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            source_root = temp_dir / "source" / "aclImdb"
            for split in ("train", "test"):
                for label_dir, text in (("neg", "bad movie"), ("pos", "great movie")):
                    target_dir = source_root / split / label_dir
                    target_dir.mkdir(parents=True, exist_ok=True)
                    (target_dir / f"{split}_{label_dir}.txt").write_text(text, encoding="utf-8")

            archive_path = temp_dir / "aclImdb_v1.tar.gz"
            import tarfile

            with tarfile.open(archive_path, mode="w:gz") as archive:
                archive.add(source_root, arcname="aclImdb")

            output_dir = temp_dir / "output"

            def fake_urlretrieve(url: str, destination: Path) -> tuple[str, object]:
                del url
                copyfile(archive_path, destination)
                return str(destination), None

            with patch("transformers_week.imdb_dataset.urlretrieve", side_effect=fake_urlretrieve):
                metadata = prepare_imdb_dataset(output_dir, force=True)

            self.assertEqual(metadata["files"], {"train.csv": 2, "test.csv": 2})
            self.assertTrue((output_dir / "train.csv").exists())
            self.assertTrue((output_dir / "test.csv").exists())
            self.assertTrue((output_dir / "metadata.json").exists())

            with (output_dir / "metadata.json").open("r", encoding="utf-8") as handle:
                saved_metadata = handle.read()
            self.assertIn(
                '"source_url": "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"',
                saved_metadata,
            )

    def test_prepare_imdb_dataset_rejects_existing_outputs_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            output_dir = Path(temp_dir_name)
            (output_dir / "train.csv").write_text("text,label\nx,1\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                prepare_imdb_dataset(output_dir, force=False)

    def test_build_parser_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])

        self.assertEqual(args.output_dir, Path("data/imdb"))
        self.assertFalse(args.force)

    def test_main_prints_metadata_json(self) -> None:
        with (
            patch("sys.argv", ["python", "--force"]),
            patch(
                "transformers_week.imdb_dataset.prepare_imdb_dataset",
                return_value={"files": {"train.csv": 25000, "test.csv": 25000}},
            ),
            patch("sys.stdout", new=StringIO()) as stream,
        ):
            main()

        self.assertIn('"train.csv": 25000', stream.getvalue())


if __name__ == "__main__":
    unittest.main()
