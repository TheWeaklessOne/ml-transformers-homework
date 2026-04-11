# ML Transformers Homework

Репозиторий для недельного проекта по теме **«Трансформеры: sentiment analysis за 7 дней»**.

## Что уже готово

- `TRANSFORMERS_HOMEWORK_PLAN.md` — подробный недельный план с чеклистами
- `src/transformers_week/day01_tokenization.py` — аккуратная реализация Дня 1
- `tests/test_day01_tokenization.py` — unit-тесты для ключевых функций
- `pyproject.toml` / `requirements.txt` — зависимости и базовая упаковка проекта

## День 1: что реализовано

В проект уже добавлены:

- загрузка рекомендованного токенизатора для английского или русского/мультиязычного режима;
- функция `tokenize_texts(...)` для батчевой токенизации;
- функция `explain_tokenization(...)` для наглядного разбора текста на токены;
- функция `get_special_tokens_info(...)` для вывода специальных токенов;
- CLI-демо, которое можно показать как аккуратный результат первого дня.

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

## Запуск тестов

```bash
python -m unittest discover -s tests -v
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
├── pyproject.toml
├── requirements.txt
├── src/
│   └── transformers_week/
│       ├── __init__.py
│       └── day01_tokenization.py
└── tests/
    └── test_day01_tokenization.py
```
