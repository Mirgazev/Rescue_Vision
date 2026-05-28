# Команды запуска и диагностики

Документ содержит полный набор команд для развёртывания, запуска и диагностики подсистемы технического зрения `rescue_vision`. Соответствует **Листингам Е.1–Е.5** Приложения Е ВКР. Все команды приведены для дистрибутива **ROS 2 Foxy Fitzroy** на Ubuntu 20.04 (минимальная поддерживаемая) или 22.04 / 24.04 (рекомендуется). Поле `<project_root>` обозначает абсолютный путь к корню репозитория после клонирования.

## Предварительная установка

### Системные пакеты ROS 2

```bash
sudo apt update
sudo apt install -y \
    ros-foxy-rclpy \
    ros-foxy-sensor-msgs \
    ros-foxy-vision-msgs \
    ros-foxy-diagnostic-msgs \
    ros-foxy-cv-bridge \
    ros-foxy-realsense2-camera \
    ros-foxy-rosbag2 \
    ros-foxy-rosbag2-storage-mcap \
    python3-colcon-common-extensions
```

Если требуется поддержка камеры Intel RealSense D435i вне ROS 2, дополнительно установить пакеты SDK:

```bash
sudo apt install -y librealsense2-utils librealsense2-dev
```

### Python-зависимости

```bash
cd <project_root>
python3 -m pip install -r requirements.txt
```

### Сборка colcon workspace

```bash
mkdir -p ~/ros2_ws/src
cp -r <project_root>/rescue_vision ~/ros2_ws/src/
cp -r <project_root>/rescue_vision_msgs ~/ros2_ws/src/

cd ~/ros2_ws
source /opt/ros/foxy/setup.bash

# Сначала собираем пакет сообщений, затем основной пакет
colcon build --packages-select rescue_vision_msgs
source install/setup.bash
colcon build --packages-select rescue_vision
source install/setup.bash
```

### Получение весов модели

Финальные веса `best.pt` (модель v4, ≈ 39 МБ) не входят в репозиторий из-за ограничения GitHub на размер файла. Скачать их можно из раздела GitHub Releases репозитория:

```bash
mkdir -p <project_root>/weights
# После публикации Release v1.0:
# wget https://github.com/Mirgazev/Rescue_Vision/releases/download/v1.0/best.pt \
#      -O <project_root>/weights/best.pt
```

---

## Листинг Е.1. Запуск pipeline на реальной видеокамере Intel RealSense

Сценарий используется при подключённой к рабочей станции (или бортовому компьютеру) камере Intel RealSense D435i.

```bash
# Активация окружения
source /opt/ros/foxy/setup.bash
source ~/ros2_ws/install/setup.bash

# Запуск pipeline с источником RealSense
ros2 launch rescue_vision pipeline_realsense.launch.py \
    model_path:=<project_root>/weights/best.pt \
    device:=cuda:0 \
    imgsz:=768 \
    conf_threshold:=0.25 \
    iou_threshold:=0.5 \
    enhance_mode:=auto \
    use_half_precision:=true \
    tracker_config:=bytetrack.yaml \
    publish_debug_image:=false
```

При работе на CPU (без GPU NVIDIA) заменить `device:=cuda:0` на `device:=cpu` и установить `use_half_precision:=false`. Производительность снизится приблизительно в 5–10 раз; для отладочных целей этот режим всё равно работоспособен.

Опциональные аргументы launch-файла:

* `launch_camera:=false` — не запускать realsense2_camera (если уже запущен внешне);
* `launch_operator_hud:=true` — параллельно запустить `operator_visualization_node` с HUD-окном;
* `input_topic:=/your_camera/image_raw` — переопределить топик источника.

## Листинг Е.2. Проверка ROS 2-графа в реальном времени

Команды для контроля состояния pipeline после его запуска. Каждая выполняется в отдельном терминале с активированным workspace.

```bash
# Перечень активных узлов
ros2 node list

# Перечень активных топиков
ros2 topic list

# Частота публикации топика камеры (должна быть около 30 Hz)
ros2 topic hz /camera/color/image_raw

# Текущий режим классификатора сцены и значения признаков
ros2 topic echo /scene/mode

# Диагностические показатели производительности (FPS, latency)
ros2 topic echo /vision/performance

# Информация об интерфейсах конкретного узла
ros2 node info /rescue_vision_node

# Поток детекций (без массивов keypoints для краткости)
ros2 topic echo /detection/persons --no-arr
```

Если `ros2 topic hz /camera/color/image_raw` показывает значение ниже 25 Hz, проблема обычно связана с одним из:

* подключение RealSense через USB 2.x вместо USB 3.0 — заменить кабель и порт;
* перегрев камеры при длительной работе — отключить и охладить;
* недостаточная производительность хоста — снизить разрешение драйвера через параметры `pipeline_realsense.launch.py`.

## Листинг Е.3. Воспроизведение записанного rosbag-сценария

Используется для воспроизведения ранее записанных сессий — для отладки, демонстрации и количественной валидации. В работе использовался формат `mcap` (рекомендован для ROS 2 Foxy и выше).

```bash
ros2 launch rescue_vision pipeline_rosbag.launch.py \
    bag_path:=<project_root>/bags/test_episode \
    model_path:=<project_root>/weights/best.pt \
    enhance_mode:=auto \
    rate:=1.0 \
    loop:=false
```

Параметры:

| Параметр | Назначение |
|---|---|
| `bag_path` | путь к каталогу rosbag (без расширения `.mcap`) |
| `rate` | множитель скорости воспроизведения (1,0 — реальное время) |
| `loop` | `true` — циклическое воспроизведение, `false` — однократное |
| `publish_debug_image` | по умолчанию `true` для отладки |
| `launch_operator_hud` | по умолчанию `true` — запустить HUD |

## Листинг Е.4. Запись нового rosbag-сценария

Применяется для сбора материала для последующей отладки и для подготовки набора кадров для ручной разметки.

```bash
# Запись всех ключевых топиков подсистемы
ros2 bag record \
    /camera/color/image_raw \
    /camera/color/camera_info \
    /scene/mode \
    /detection/persons \
    /detection/poses \
    /vision/performance \
    -o <project_root>/bags/new_episode \
    --storage mcap
```

Размер записанного rosbag-файла зависит от продолжительности и разрешения. Для одночасовой сессии с FullHD-камерой при частоте 30 Hz без сжатия можно ожидать около 80–100 ГБ. Рекомендуется включить уровень компрессии командой `--compression-mode file --compression-format zstd` (потребуется пакет `ros-foxy-rosbag2-compression-zstd`).

## Листинг Е.5. Динамическое изменение параметров узла

ROS 2 предоставляет сервис `set_parameters`, через который параметры узла можно менять без его перезапуска. Эта возможность реализована в `rescue_vision_node` через `add_on_set_parameters_callback` и активно используется в ходе экспериментальной части ВКР для быстрого сравнения режимов на одной и той же видеопоследовательности.

```bash
# Переключение режима улучшения на лету
ros2 param set /rescue_vision_node enhance_mode off
ros2 param set /rescue_vision_node enhance_mode auto
ros2 param set /rescue_vision_node enhance_mode night

# Изменение порога уверенности
ros2 param set /rescue_vision_node conf_threshold 0.20

# Изменение IoU-порога
ros2 param set /rescue_vision_node iou_threshold 0.45

# Включение отладочного изображения для оператора
ros2 param set /rescue_vision_node publish_debug_image true

# Перечень всех параметров узла с текущими значениями
ros2 param list /rescue_vision_node
ros2 param get /rescue_vision_node conf_threshold
ros2 param get /rescue_vision_node enhance_mode
```

Полный список параметров узла приведён в файле `rescue_vision/config/default_params.yaml`.

---

## Запуск без ROS 2 (только для отладки)

Если задача — быстрая проверка алгоритмической части на отдельных MP4-файлах или одиночных изображениях, ROS 2 не обязателен. Поднимать workspace и собирать пакет в этом случае не требуется; достаточно установленного Python-окружения.

```bash
cd <project_root>
python3 -m rescue_vision.rescue_vision.main \
    --source path/to/video.mp4 \
    --model weights/best.pt \
    --enhance auto \
    --imgsz 768 \
    --conf 0.25 \
    --device cuda \
    --half \
    --show
```

Аргументы CLI:

| Аргумент | Назначение |
|---|---|
| `--source` | Путь к MP4 или индекс камеры (0, 1, ...) |
| `--model` | Путь к .pt весам YOLO |
| `--enhance` | `off`/`night`/`fog`/`smoke`/`rain`/`auto` |
| `--imgsz` | Размер input для YOLO (по умолчанию 768) |
| `--conf` | Порог уверенности (по умолчанию 0.25) |
| `--iou` | Порог IoU (по умолчанию 0.5) |
| `--device` | `cuda` / `cpu` |
| `--half` | Включить FP16 inference |
| `--tracker` | Конфиг трекера (по умолчанию `bytetrack.yaml`) |
| `--show` | Открыть окно превью OpenCV |
| `--save` | Путь для сохранения выходного MP4 |

---

## Расчёт метрик на тестовой выборке

Утилита `tools/evaluate_map_fps.py` (соответствует Листингу Д.1 ВКР) рассчитывает mAP@0.5, Precision, Recall, F1 и FPS для произвольной выборки. Поддерживаются форматы аннотаций: ExDark (`.txt` с префиксом `Person`), MOT17 (`gt.txt`), RESIDE / YOLO (`labels/*.txt`), Pascal VOC (`.xml`), COCO (`annotations.json`).

Пример полного прогона для воспроизведения строки `v4 / OFF` Таблицы 11 ВКР (ExDark):

```bash
python3 tools/evaluate_map_fps.py \
    --dataset exdark \
    --root /path/to/ExDark \
    --model <project_root>/weights/best.pt \
    --imgsz 768 \
    --conf 0.25 \
    --iou 0.5 \
    --enhance off \
    --half \
    --output <project_root>/experiments/results/exdark_v4_off.json
```

После завершения прогона JSON-файл содержит блоки `config`, `metrics`, `timings` и `counts`. Использование одного и того же скрипта для всех таблиц обеспечивает методологическую согласованность измерений.

Аналогично для RESIDE / FOG (Таблица 12):

```bash
python3 tools/evaluate_map_fps.py \
    --dataset reside \
    --root /path/to/RESIDE \
    --model <project_root>/weights/best.pt \
    --enhance fog \
    --half \
    --output <project_root>/experiments/results/reside_v4_fog.json
```

Для COCO val2017 (Таблица 13):

```bash
python3 tools/evaluate_map_fps.py \
    --dataset coco \
    --root /path/to/COCO/val2017 \
    --model <project_root>/weights/best.pt \
    --enhance off \
    --half \
    --output <project_root>/experiments/results/coco_v4_off.json
```

Для MOT17 (Таблицы 15-16):

```bash
python3 tools/evaluate_map_fps.py \
    --dataset mot17 \
    --root /path/to/MOT17 \
    --model <project_root>/weights/best.pt \
    --enhance off \
    --half \
    --output <project_root>/experiments/results/mot17_v4_off.json
```

---

## Типичные неполадки и их решения

| Сообщение об ошибке | Причина | Решение |
|---|---|---|
| `ModuleNotFoundError: rclpy` | не активирован ROS 2 setup.bash | `source /opt/ros/foxy/setup.bash` |
| `Package 'rescue_vision' not found` | не собран workspace | `cd ~/ros2_ws && colcon build && source install/setup.bash` |
| `RuntimeError: CUDA out of memory` | imgsz слишком большой для VRAM | снизить `imgsz` до 640 или отключить FP16 |
| `No frames received from RealSense` | проблема с USB или питанием | заменить кабель на USB 3.0, проверить блок питания |
| `ImportError: pyrealsense2` | не установлен SDK | `pip install pyrealsense2` |
| Низкий FPS на ExDark | холодный старт CUDA | прогрев модели 10–20 dummy-инференсами выполняется автоматически в `BaseDetector._warmup()` |
| Расхождение mAP с таблицей | `iou_threshold` или `conf_threshold` отличается | привести в соответствие с разделом 4.1 ВКР |
| `Could not find a package configuration file provided by "realsense2_camera"` | не установлен ROS 2-пакет камеры | `sudo apt install ros-foxy-realsense2-camera` |
| Параметр `tracker_config` не находится | путь к bytetrack.yaml относительный | использовать абсолютный путь или положить в `share/rescue_vision/config/` |

При возникновении других проблем — открыть issue в репозитории с приложенным выводом команды `ros2 doctor --report` и фрагментом лога узла.
