# README

## Severstal Defect Detection — Local Interface & REST API

Проект для мультиметочной классификации и сегментации дефектов стали (4 класса: царапины, трещины, расслоения, пятна).
Два интерфейса: локальный GUI на PySide6 и HTTP API на FastAPI. Они используют одну и ту же пару инференс-фабрик.

---

## 1. Требования

- Python 3.10+
- Список библиотек из файла requirements_inference.txt

Данные модели должны лежать по путям, указанным в `config.json` (`h_load_path`) и в `segmantation_base/config.json` (`checkpoint`).

---

## 2. Конфигурация

Используются **два независимых конфига**:

| Файл | Назначение | Кем читается |
|------|-----------|--------------|
| `config.json` (в корне) | Классификатор: модель, размер входа, порог, имена/цвета классов, чекпоинт | `Config.from_json` |
| `segmantation_base/config.json` | Сегментатор: чекпоинт, `target_size`, `transform_type`, порог, классы | `InferenceConfig.from_json` |

Правки в файлах применяются при следующем запуске, то есть оба интерфейса перечитывают их только на старте.

---

## 3. Локальный интерфейс (`local_interface.py`)

### Запуск

```bash
python local_interface.py
```

### Устройство

1. Читает `config.json` через `Config.from_json`.
2. Собирает словарь `classifier_config` только из полей, которые нужны фабрике: `device`, `num_classes`, `in_height`, `in_width`, `model_name`, `model_mode`, `h_load_path`, `transform_mean`, `transform_std`, `threshold`, `class_names`, `class_colors`, `batch_size`.
3. Вызывает:
   - `return_classifier(classifier_config)` $\rightarrow$ `(predict, class_names)`: `predict` принимает `PIL.Image` или список `PIL.Image`, возвращает `(probs, labels)`.
   - `return_segmenter(InferenceConfig.from_json("segmantation_base/config.json"))` $\rightarrow$ `segment(images)`: принимает `PIL.Image` или список `PIL.Image`, возвращает RGB-overlay `[H,W,3]` (или `[N,H,W,3]`).
4. Открывает окно `ImageViewer`.

5. Порог фиксированный на основе обучения (`self.threshold = 0.5`). Чтобы поменять измените `Config.threshold` (классификатор) и/или `InferenceConfig.threshold` (сегментатор). Метки и предсказания пересчитываются только на этапе загрузки/классификации.

6. Классификация идёт батчами по `BATCH_SIZE = 32` (`_batch_classify`); сегментация с тем же батчем (`_batch_save_segmented`). Сегментированные файлы сохраняются с суффиксом из активных классов: `image_царапины_трещины.jpg`, либо `image_none.jpg`, если дефектов нет.

---

## 4. REST API (`rest_api.py` + `rest_api_main.py`)

### Запуск

```bash
python rest_api_main.py
```

### Шаблоны

Jinja2-шаблоны ожидаются в `templates/` рядом с `rest_api.py`:

``` text
templates/
├── upload.html
└── result.html
```

### Маршруты

Маршруты завязаны на «префикс прокси» вида `/{process}/ports/{ports}/`:

| Метод | Путь | Что делает |
|-------|------|-----------|
| GET | `/{process}/ports/{ports}/` | Форма загрузки |
| POST | `/{process}/ports/{ports}/upload/` | Принимает `file`, классифицирует + сегментирует, отдаёт HTML с результатом |
| POST | `/{process}/ports/{ports}/download/` | Принимает `image_data` (base64 data-URL) + `filename`, отдаёт PNG вложением |

### Устройство

1. На импорте `rest_api.py`:
   - Читает `config.json` (`Config`) и `segmantation_base/config.json` (`InferenceConfig`).
   - Собирает `classifier_config`.
   - Вызывает `return_classifier` и `return_segmenter` — один раз, дальше переиспользует.
2. `upload_file`:
   - Проверяет размер (`MAX_FILE_SIZE = 10 MB`), `content_type`, что PIL-открытие проходит.
   - Классификатор возвращает `probs, labels` по одному изображению.
   - Сегментатор возвращает либо PIL, либо ndarray.
   - Результат кодируется в base64 и передаётся в `result.html`.
3. `download_file` распаковывает base64 из формы и отдаёт как вложение. Никакого состояния между запросами не сохраняется.

---

## 5. Общая архитектура инференса

Оба входа (`GUI` и `API`) не знают ничего о внутренностях моделей — только про две фабрики:

``` text
inference/
├── classifier.py -> return_classifier(config_dict) -> (predict, class_names)
└── segmenter.py  -> return_segmenter(config_dict)  -> segment(images)
```

**`return_classifier`**:

- Строит модель через `models.create_model(config, num_classes, (in_height, in_width))`.
- Загружает `state_dict` из `config.h_load_path`.
- (Опционально) замораживает параметризованную часть квантовых слоёв через `freeze_for_inference` при `freeze_quantum=True`.
- Применяет `Resize + ToTensor + Normalize(mean, std)`.
- Возвращает callable, который на `PIL.Image`/списке изображений отдаёт `(probs, labels)` и работает под `torch.no_grad()`.

**`return_segmenter`**:

- Строит `SegmentationInference` из `segmantation_base/inference.py`.
- `output_type` (`overlay` по умолчанию) определяет форму выхода: `[H,W,3]` uint8 overlay или `[C,H,W]` / `[H,W,C]` пробы/маски.
- Внутри разделенный инференс (`crop` или `resize`), батчи ограничены `_tile_batch_size`.

**Если меняется архитектура модели, то трогать GUI и API не нужно, достаточно обновить конфиг и веса.**
