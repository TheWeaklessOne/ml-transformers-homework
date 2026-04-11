# ML Transformers Homework

Репозиторий для недельного проекта по теме **«Трансформеры: sentiment analysis за 7 дней»**.

## Что уже готово

- `TRANSFORMERS_HOMEWORK_PLAN.md` — подробный недельный план с чеклистами
- `src/transformers_week/day01_tokenization.py` — аккуратная реализация Дня 1
- `src/transformers_week/imdb_dataset.py` — утилита для скачивания и подготовки IMDb в CSV
- `tests/test_day01_tokenization.py` — unit-тесты для ключевых функций
- `tests/test_imdb_dataset.py` — unit-тесты для подготовки датасета
- `pyproject.toml` / `requirements.txt` — зависимости и базовая упаковка проекта

## Датасет проекта

В репозиторий намеренно добавлен только **IMDb Movie Reviews** как основной датасет проекта.

- путь: `data/imdb/`
- файлы: `train.csv`, `test.csv`, `metadata.json`
- формат: две колонки — `text`, `label`
- метки: `0 = negative`, `1 = positive`

CSV-файлы сгенерированы из официального архива Stanford IMDb и подходят для дальнейших дней проекта.

### Как пересобрать IMDb локально

```bash
python -m transformers_week.imdb_dataset --force
```

## День 1: что реализовано

В проект уже добавлены:

- загрузка рекомендованного токенизатора для английского или русского/мультиязычного режима;
- функция `tokenize_texts(...)` для батчевой токенизации в стиле задания;
- функция `explain_tokenization(...)` для наглядного разбора текста на токены;
- функция `get_special_tokens_info(...)` для вывода специальных токенов;
- CLI-демо, которое буквально повторяет ключевые примеры из формулировки домашнего задания:
  - `"This movie was absolutely amazing!"`
  - `"This movie was great!"`
  - `"Terrible movie, waste of time."`
  - `explain_tokenization("Transformers are amazing!", tokenizer)`
  - отдельный пример `tokenizer(text, return_tensors="pt")`

## Быстрый старт

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

После editable install пакет можно запускать без ручного `PYTHONPATH`.

## Запуск демо Дня 1

```bash
transformers-day1
```

Пример с русским / мультиязычным токенизатором:

```bash
transformers-day1 --language ru "Трансформеры отлично подходят для NLP"
```

Если нужно пересобрать IMDb и затем сразу проверить Day 1 end-to-end:

```bash
python -m transformers_week.imdb_dataset --force
make check
```

## Запуск тестов

```bash
pytest
```

## Quality tooling

В репозитории настроены:

- `ruff` — lint + formatting
- `mypy` — static type checking
- `pytest` + `pytest-cov` + `coverage` — тесты и покрытие
- `pre-commit` — локальные проверки перед коммитом
- GitHub Actions CI — автоматическая проверка на Python 3.10 / 3.11 / 3.12

### Установка dev-инструментов

```bash
python -m pip install -r requirements-dev.txt
```

### Полный локальный quality check

```bash
make check
```

### Отдельные команды

```bash
make lint
make format-check
make typecheck
make test
make coverage
make smoke
```

### Подключение pre-commit

```bash
make pre-commit-install
```

## Текущая структура

```text
.
├── README.md
├── TRANSFORMERS_HOMEWORK_PLAN.md
├── data/
│   └── imdb/
│       ├── README.md
│       ├── metadata.json
│       ├── test.csv
│       └── train.csv
├── Makefile
├── requirements-dev.txt
├── pyproject.toml
├── requirements.txt
├── src/
│   └── transformers_week/
│       ├── __init__.py
│       ├── day01_tokenization.py
│       └── imdb_dataset.py
└── tests/
    ├── test_day01_tokenization.py
    └── test_imdb_dataset.py
```
