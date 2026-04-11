# IMDb dataset for the transformers week project

Этот каталог намеренно версионируется в репозитории как основной датасет проекта.

Содержимое:

- `train.csv` — 25 000 размеченных отзывов
- `test.csv` — 25 000 размеченных отзывов
- `metadata.json` — краткая сводка по источнику и числу строк

Формат CSV:

- `text` — текст отзыва
- `label` — метка тональности (`0 = negative`, `1 = positive`)

Источник: официальный Stanford IMDb Large Movie Review Dataset  
URL: <https://ai.stanford.edu/~amaas/data/sentiment/>

Подготовка в репозитории выполняется командой:

```bash
python -m transformers_week.imdb_dataset --force
```
