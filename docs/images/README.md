# docs/images/

Этот каталог содержит демонстрационные изображения для документации репозитория и README.

## Структура

```
docs/images/
├── README.md                       (этот файл)
├── architecture.png                Рис. 14 ВКР — общая архитектура подсистемы
├── ros2_graph.png                  Рис. 15 ВКР — ROS 2-граф узлов и топиков
├── enhancement_demo_exdark.png     split-view: оригинал → NIGHT-улучшение
├── enhancement_demo_reside.png     split-view: оригинал → FOG-улучшение
├── enhancement_demo_smoke.png      split-view: оригинал → SMOKE-улучшение
└── mchs_detections.png             кадр из видео МЧС с детекциями + track-ID
```

## Указание для пользователя репозитория

Перед публикацией репозитория необходимо добавить сюда:

* `architecture.png` — взять из ВКР Главы 3 (Рисунок 14);
* `ros2_graph.png` — взять из ВКР Главы 3 (Рисунок 15);
* `enhancement_demo_*.png` — сгенерировать через `tools/make_comparison.py`:

```bash
python3 tools/make_comparison.py \
    --input path/to/exdark_sample.jpg \
    --output docs/images/enhancement_demo_exdark.png \
    --modes off night
```

* `mchs_detections.png` — взять кадр из обработанного видео МЧС с нарисованными bbox и track-ID.

После добавления картинок раскомментировать ссылки на них в главном `README.md`.

## Требования к изображениям

* Формат: PNG или JPG;
* Максимальная ширина: 1600 px (для оптимального отображения в GitHub README);
* Сжатие: PNG-8 для скриншотов, JPEG q=85 для фотографий;
* Размер файла: до 500 КБ на изображение, чтобы репозиторий оставался компактным;
* В именах файлов — только латиница и подчёркивания.
