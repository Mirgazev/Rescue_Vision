# Rescue Vision - подсистема технического зрения мобильного спасательного робота

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![ROS 2](https://img.shields.io/badge/ROS_2-Foxy-orange.svg)](https://docs.ros.org/en/foxy/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red.svg)](https://pytorch.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.14-purple.svg)](https://docs.ultralytics.com)
[![mAP@0.5](https://img.shields.io/badge/mAP%400.5-0.8905-success.svg)](experiments/README.md)

Программное обеспечение подсистемы технического зрения мобильного робота для повышения устойчивости обнаружения людей в условиях ухудшенной видимости (низкая освещённость, туман, дым, осадки). Сборка реализована как комбинация эвристического классификатора условий видимости, набора классических алгоритмов улучшения изображения, дообученного детектора YOLO11-pose и трекера ByteTrack с интеграцией в ROS 2 Foxy.

Репозиторий сопровождает выпускную квалификационную работу бакалавра по направлению **15.03.06 «Мехатроника и робототехника»**, профиль «Автономные роботы», РТУ МИРЭА.

**Автор:** Миргазев Марат Айратович, группа КРБО-03-22
**Институт:** Институт искуcственного интеллекта
**Кафедра:** Проблем управления
**Научный руководитель:** Волкова М. А., к.т.н., доцент КПУ
**Дата защиты:** 29 мая 2026 г.

---

## Главные результаты

Финальная конфигурация (модель v4 + автоматический классификатор сцены) проверена на пяти публичных наборах данных и одном реальном видеоматериале. Все целевые показатели технического задания достигнуты или перевыполнены.

| Показатель ТЗ | Требование | Достигнуто | Источник |
|---|:---:|:---:|---|
| mAP@0.5 в условиях низкой освещённости | ≥ 0,77 | **0,8905** | ExDark, 607 изображений, режим OFF |
| Точность автоматического выбора режима | ≥ 80 % | **90,5 %** | 911 размеченных кадров, 5 классов |
| Производительность на потоковом сценарии | ≥ 20 FPS | **23,6 … 62,0** | MOT17, видео МЧС, RTX 2060 |
| mAP@0.5 на ручной разметке МЧС | — | **0,82** | 60 кадров, 119 GT-боксов, OFF |
| Стабильность трекинга (MOT17) | — | **IDF1 = 0,74; MOTA = 0,77** | см. подраздел 4.7 ВКР |

Главный экспериментальный вывод работы: для модели, прошедшей качественный этап обучения на смешанной выборке ExDark + RESIDE + COCO, принудительное применение runtime-улучшения изображения даёт минимальный или отрицательный вклад в качество детекции на in-distribution данных. Автоматический классификатор сцены при этом сохраняет полезность как механизм контролируемого вмешательства: на out-of-distribution и реальном видеоматериале его поведение приближается к режиму OFF, что выражает консервативный компромисс между устойчивостью и сложностью обработки.

## Архитектура подсистемы

```
RealSense / rosbag / MP4
        │
        ▼
┌─────────────────────────┐
│  ImprovedSceneAnalyzer  │   статистические признаки кадра,
│  (5 признаков + дерево  │   иерархическое дерево решений,
│   решений + история)    │   сглаживание окном из 5 кадров
└─────────┬───────────────┘
          │ mode ∈ {OFF, NIGHT, FOG, SMOKE, RAIN}
          ▼
┌─────────────────────────┐
│   RescueEnhancer        │   NIGHT: CLAHE + γ + linear stretch
│   (5 режимов с единым   │   FOG:   Dark Channel Prior
│    BGR-интерфейсом)     │   SMOKE: Single-Scale Retinex
│                         │   RAIN:  медианный + билатеральный
└─────────┬───────────────┘   OFF:   без обработки
          │ enhanced BGR
          ▼
┌─────────────────────────┐
│ YOLO11-pose v4 (FP16)   │   дообученные веса, imgsz 768,
│ + ByteTrack             │   conf 0,25, iou 0,5, person class 0
└─────────┬───────────────┘
          │
          ▼
ROS 2 топики:
  /scene/mode           SceneMode         (RELIABLE + TRANSIENT_LOCAL)
  /detection/persons    PersonDetection   (RELIABLE)
  /detection/poses      Detection2DArray  (RELIABLE)
  /vision/debug/image   sensor_msgs/Image (BEST_EFFORT)
  /vision/performance   DiagnosticArray   (RELIABLE)
```

Подробное обоснование выбора монолитной архитектуры (вместо распределения функциональных блоков по отдельным узлам), описание QoS-профилей и схема привязки к платформе Unitree H1 приведены в [`docs/architecture.md`](docs/architecture.md) и в подразделе 3.6 ВКР.

## Структура репозитория

```
rescue_vision_bachelor/
├── README.md                         главный файл (этот документ)
├── LICENSE                           MIT
├── requirements.txt                  Python-зависимости, точные версии
├── .gitignore
│
├── rescue_vision/                    основной ament_python-пакет ROS 2
│   ├── rescue_vision/
│   │   ├── rescue_vision_node.py         Листинг А.4 — основной узел конвейера
│   │   ├── operator_visualization_node.py Листинг А.5 — узел оператора
│   │   ├── scene_analyzer.py             Листинг Б.1 — ImprovedSceneAnalyzer
│   │   ├── rescue_enhancer.py            Листинг В.1 — 5 режимов улучшения
│   │   ├── base_detector.py              Листинг Г.1 — обёртка YOLO11 + ByteTrack
│   │   ├── enhancement_modes.py          IntEnum {OFF, NIGHT, FOG, SMOKE, RAIN, AUTO}
│   │   ├── vision_pro_v5.py              shim для обратной совместимости с Листингами
│   │   ├── config.py                     VisionConfig dataclass
│   │   ├── main.py                       standalone-запуск без ROS 2
│   │   └── metrics_logger.py             CSV-логгер метрик
│   ├── launch/
│   │   ├── pipeline_realsense.launch.py  Листинг А.6 — запуск с RealSense D435i
│   │   ├── pipeline_rosbag.launch.py     Листинг А.7 — воспроизведение rosbag
│   │   └── pipeline_sim.launch.py        проектный шаблон (не использован в ВКР)
│   ├── config/
│   │   ├── default_params.yaml           параметры узла Таблицы 7 ВКР
│   │   └── bytetrack.yaml                конфигурация трекера
│   ├── resource/rescue_vision
│   ├── package.xml                       Листинг А.8
│   └── setup.py                          Листинг А.9
│
├── rescue_vision_msgs/               пакет собственных сообщений (ament_cmake)
│   ├── msg/
│   │   ├── SceneMode.msg                 Листинг А.2
│   │   └── PersonDetection.msg           Листинг А.3
│   ├── package.xml
│   └── CMakeLists.txt
│
├── experiments/                      обучение и валидация
│   ├── README.md                         полные таблицы по итерациям v1-v4
│   ├── configs/
│   │   ├── v1.yaml                       Таблица 20 ВКР, строка v1
│   │   ├── v2.yaml                       Таблица 20 ВКР, строка v2
│   │   ├── v3.yaml                       Таблица 20 ВКР, строка v3 (отклонена)
│   │   └── v4.yaml                       Таблица 20 ВКР, строка v4 (финальная)
│   └── results/
│       ├── exdark_results.csv            Таблица 11 (15 строк)
│       ├── reside_results.csv            Таблица 12 (15 строк)
│       ├── coco_results.csv              Таблица 13 (10 строк)
│       ├── smoketest_results.csv         Таблица 14 (4 строки)
│       ├── mot17_detection_results.csv   Таблица 15 (2 строки)
│       ├── mot17_tracking_results.csv    Таблица 16 (2 строки)
│       ├── mchs_manual_gt_results.csv    Таблица 17 (5 строк)
│       ├── mchs_auto_label_results.csv   Таблица 18 (5 строк)
│       ├── auto_classifier_confusion_matrix.csv  Таблица 19
│       ├── hyperparam_grid_conf_imgsz.csv        Таблицы 21-23
│       └── training_iterations_summary.csv       Таблица 20 (сводная)
│
├── tools/                            утилиты
│   ├── evaluate_map_fps.py               Листинг Д.1 — расчёт mAP/FPS/IDF1
│   ├── gt_labeler_tkinter.py             ручная разметка GT для МЧС-видео
│   ├── make_comparison.py                split-view сравнение режимов
│   └── synthesize_smoke.py               синтез SmokeTest по модели атм. рассеяния
│
├── lab_work/                         методические указания (Приложение Ж ВКР)
│   └── classifier_lab.md                 полная методичка SceneAnalyzer
│
├── docs/                             документация
│   ├── architecture.md                   архитектура + QoS-профили
│   ├── installation.md                   установка ROS 2 Foxy + Python
│   ├── usage.md                          команды Листингов Е.1-Е.5
│   └── images/                           демонстрационные изображения
│
└── scripts/                          вспомогательные скрипты
```

## Быстрый старт

Подробные команды приведены в [`docs/usage.md`](docs/usage.md). Здесь — минимальный сценарий запуска на ROS 2 Foxy с камерой Intel RealSense D435i.

```bash
# 1. Системные пакеты (Ubuntu 20.04 + ROS 2 Foxy)
sudo apt install -y \
    ros-foxy-vision-msgs ros-foxy-cv-bridge \
    ros-foxy-realsense2-camera ros-foxy-diagnostic-msgs

# 2. Python-зависимости
python3 -m pip install -r requirements.txt

# 3. Сборка ROS 2 workspace
mkdir -p ~/ros2_ws/src
cp -r rescue_vision rescue_vision_msgs ~/ros2_ws/src/
cd ~/ros2_ws
colcon build --packages-select rescue_vision_msgs && source install/setup.bash
colcon build --packages-select rescue_vision      && source install/setup.bash

# 4. Запуск конвейера
ros2 launch rescue_vision pipeline_realsense.launch.py \
    model_path:=<project_root>/weights/best.pt \
    enhance_mode:=auto \
    use_half_precision:=true
```

Веса дообученной модели v4 (`best.pt`, ~ 39 МБ) не включаются в репозиторий из-за ограничения размера и публикуются как Release-артефакт ([Releases](https://github.com/Mirgazev/Rescue_Vision/releases) репозитория).

## Standalone-режим (без ROS 2)

```bash
python3 -m rescue_vision.rescue_vision.main \
    --source path/to/video.mp4 \
    --model weights/best.pt \
    --enhance auto \
    --half --show
```

## Итерации обучения

В работе выполнены четыре итерации дообучения базовой модели `yolo11m-pose.pt`. Сводная таблица:

| Версия | Данные | Эпохи | Ключевые изменения | mAP@0.5 ExDark / RESIDE / COCO | Статус |
|---|---|:---:|---|---|---|
| **Default** | COCO-pretrain | — | базовая модель Ultralytics | 0,7242 / 0,4403 / 0,5204 | базовый уровень |
| **v1** | ExDark ~70 % + RESIDE ~20 % + COCO ~10 % | до 200 (early stop ~29-59) | freeze=10, lr0=0,001, базовые аугментации | 0,8112 / 0,5827 / 0,5225 | первый устойчивый прирост |
| **v2** | ExDark ~80 % + RESIDE / COCO ~20 % | 137 (остановлено вручную) | амп=True, cache=False, убраны AdamW и blur | 0,8725 / 0,6654 / 0,5534 | стабильная |
| **v3** | та же выборка + RescueEnhancer p=0,5 | 60 | кастомный trainer, патч load_image | 0,7678 / 0,6265 / 0,5214 | **отклонена** (distribution mismatch) |
| **v4** | безопасный enhance-aware | 25 | старт с v2, mosaic=0, mixup=0, p=0,25, freeze=22, lr0=1e-4 | **0,8905 / 0,6749 / 0,5646** | **финальная** |

Полное обсуждение в [`experiments/README.md`](experiments/README.md).

## Аппаратная и программная база

Все приведённые в работе значения FPS и времени инференса измерены на одной рабочей станции в соответствии с уровнем платформы, указанным в техническом задании:

| Компонент | Значение |
|---|---|
| CPU | 13th Gen Intel Core i5-13420H |
| GPU | NVIDIA GeForce RTX 2060, 6 ГБ |
| RAM | 15 GiB |
| OS | Ubuntu 24.04.4 LTS |
| ROS 2 | Foxy Fitzroy |
| Python | 3.13.11 |
| PyTorch / torchvision | 2.10.0 / 0.25.0 |
| Ultralytics | 8.4.14 |
| OpenCV | 4.13.0.92 |
| CUDA (в PyTorch) | 12.8 |
| Размер inference | imgsz = 768 |
| Точность | FP16 |

## Что НЕ входит в работу

Эти направления вынесены за пределы текущей ВКР и рассматриваются как смежные задачи дальнейшей интеграции. Обозначены явно во избежание неоднозначностей при защите:

* физическая или программная симуляция платформы (Gazebo и иные симуляторы) в рамках ВКР не использовалась; все экспериментальные результаты получены на реальной видеокамере Intel RealSense D435i, записанных rosbag-сценариях, MP4-файлах публичных датасетов и видеоматериале МЧС России;
* низкоуровневые контуры управления Unitree H1 (стабилизация походки, планирование траектории, управление суставами) рассматриваются как направления дальнейшего развития работы;
* проектирование аппаратной части мобильной платформы не входило в задачи ВКР;
* оценка ID-метрик трекинга (IDF1, MOTA, ID switches, Fragmentation) выполнена только на MOT17 в подразделе 4.7 ВКР;
* MATLAB и его инструментарий в работе не использовались.

## Цитирование

Если репозиторий или его результаты используются в вашей работе, ссылайтесь на текст ВКР:

```
Миргазев М. А. Программное обеспечение мобильного робота для выполнения
спасательных операций : выпускная квалификационная работа бакалавра /
МИРЭА — Российский технологический университет, институт искусственного интеллекта, кафедра проблем управления;
науч. рук. Волкова М. А. — Москва, 2026.
```

## Лицензия

Исходный код пакетов `rescue_vision` и `rescue_vision_msgs` распространяется по лицензии MIT (см. файл [`LICENSE`](LICENSE)). Веса дообученной модели, текст ВКР и методические указания не входят в это разрешение и принадлежат автору работы и РТУ МИРЭА.

## Контакты

email: marat.mirgazev@mai.ru
