# Установка и первоначальная настройка

Документ описывает полный процесс развёртывания подсистемы `rescue_vision` на чистой системе. Целевой дистрибутив — **Ubuntu 20.04 / 22.04 / 24.04 LTS** с дистрибутивом **ROS 2 Foxy Fitzroy**. Все приведённые команды протестированы на референсной аппаратной конфигурации (см. `docs/architecture.md`, раздел 6).

## 1. Системные требования

### Минимальные

* OS: Ubuntu 20.04 LTS (или совместимый Debian-based дистрибутив)
* CPU: 4 ядра, x86_64
* RAM: 8 ГБ
* Диск: 10 ГБ свободного места
* Python: 3.10 или новее

### Рекомендуемые

* OS: Ubuntu 22.04 / 24.04 LTS
* CPU: 6+ ядер
* GPU: NVIDIA GeForce GTX 1650 или новее, ≥ 4 ГБ VRAM (для GPU-инференса)
* RAM: 16 ГБ
* Диск: SSD, 50 ГБ
* Python: 3.13

### Для воспроизведения результатов ВКР

* GPU: NVIDIA GeForce RTX 2060, 6 ГБ VRAM (точная конфигурация Таблицы 5 ВКР)

## 2. Установка ROS 2 Foxy

Если ROS 2 Foxy ещё не установлен, выполнить базовую инсталляцию по официальной инструкции:

```bash
# Локаль (на чистой системе может отсутствовать)
sudo apt update
sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Добавление репозитория ROS 2
sudo apt install -y software-properties-common curl
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Установка ROS 2 Foxy desktop + dev tools
sudo apt update
sudo apt install -y ros-foxy-desktop python3-argcomplete
sudo apt install -y ros-dev-tools python3-colcon-common-extensions

# Источник окружения
echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

Проверка установки:

```bash
ros2 --help
ros2 doctor --report
```

## 3. ROS 2-пакеты, необходимые для rescue_vision

```bash
sudo apt install -y \
    ros-foxy-rclpy \
    ros-foxy-sensor-msgs \
    ros-foxy-vision-msgs \
    ros-foxy-diagnostic-msgs \
    ros-foxy-cv-bridge \
    ros-foxy-realsense2-camera \
    ros-foxy-realsense2-camera-msgs \
    ros-foxy-rosbag2 \
    ros-foxy-rosbag2-storage-default-plugins \
    ros-foxy-launch-ros \
    ros-foxy-launch-xml \
    ros-foxy-launch-yaml
```

Для записи в формате MCAP (рекомендуется, более эффективный, чем sqlite3):

```bash
sudo apt install -y ros-foxy-rosbag2-storage-mcap
```

## 4. NVIDIA-драйверы и CUDA (опционально, для GPU-инференса)

Для использования GPU необходима совместимая версия CUDA. PyTorch 2.10 требует **CUDA 12.x**. Установка:

```bash
# Проверка наличия NVIDIA-драйвера
nvidia-smi

# Если драйвера нет — установить через ubuntu-drivers
sudo ubuntu-drivers autoinstall
sudo reboot

# CUDA Toolkit устанавливается автоматически вместе с PyTorch-пакетом из pip;
# отдельная инсталляция CUDA Toolkit для запуска PyTorch не требуется.
```

После перезагрузки команда `nvidia-smi` должна показать GPU и драйвер версии не ниже 525 (для CUDA 12.x).

## 5. Intel RealSense SDK (опционально, для штатного запуска на роботе)

Если планируется работа с физической камерой Intel RealSense D435i:

```bash
# Подключение репозитория Intel
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | \
    sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] \
    https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | \
    sudo tee /etc/apt/sources.list.d/librealsense.list

# Установка SDK
sudo apt update
sudo apt install -y librealsense2-utils librealsense2-dev
```

Проверка подключения камеры:

```bash
realsense-viewer
```

## 6. Клонирование репозитория и Python-зависимости

```bash
# Клонирование
git clone https://github.com/Mirgazev/Rescue_Vision.git
cd Rescue_Vision

# Создание виртуального окружения (рекомендуется)
python3 -m venv .venv
source .venv/bin/activate

# Установка Python-пакетов
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
```

Проверка работоспособности базового стека:

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"
python3 -c "from ultralytics import YOLO; print('Ultralytics:', YOLO.__module__)"
```

## 7. Сборка ROS 2 workspace

```bash
# Создание workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Симлинк на пакеты репозитория
ln -s ~/Rescue_Vision/rescue_vision .
ln -s ~/Rescue_Vision/rescue_vision_msgs .

# Сборка
cd ~/ros2_ws
source /opt/ros/foxy/setup.bash

# 1) Сначала собираем пакет сообщений
colcon build --packages-select rescue_vision_msgs
source install/setup.bash

# 2) Затем собираем основной пакет
colcon build --packages-select rescue_vision
source install/setup.bash

# Добавление source-команд в ~/.bashrc для удобства
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

Проверка успешной сборки:

```bash
ros2 pkg list | grep rescue
# Должно вывести:
#   rescue_vision
#   rescue_vision_msgs
```

## 8. Скачивание весов модели

Финальные веса дообученной модели v4 (`best.pt`, ≈ 39 МБ) не входят в репозиторий — они опубликованы как Release-артефакт:

```bash
mkdir -p ~/Rescue_Vision/weights
cd ~/Rescue_Vision/weights

# После публикации Release v1.0:
# wget https://github.com/Mirgazev/Rescue_Vision/releases/download/v1.0/best.pt
```

## 9. Первый запуск (smoke-test)

```bash
# Активация окружения
source /opt/ros/foxy/setup.bash
source ~/ros2_ws/install/setup.bash

# Проверка работоспособности на тестовом MP4 (без ROS 2)
cd ~/Rescue_Vision
python3 -m rescue_vision.rescue_vision.main \
    --source assets/videos/sample.mp4 \
    --model weights/best.pt \
    --enhance auto \
    --show
```

Если окно OpenCV с детекциями открылось — установка успешна.

## 10. Решение типичных проблем установки

| Проблема | Решение |
|---|---|
| `E: Unable to locate package ros-foxy-desktop` | проверить, что добавлен ROS 2 apt-репозиторий (шаг 2) |
| `Failed to fetch ... 404 Not Found` | дистрибутив Ubuntu может не поддерживать Foxy напрямую — использовать Ubuntu 20.04 или 22.04 |
| `nvidia-smi: command not found` | установить NVIDIA-драйверы: `sudo ubuntu-drivers autoinstall` + reboot |
| `RuntimeError: CUDA error: no kernel image is available` | несовместимость CUDA + PyTorch + GPU compute capability — обновить драйвер |
| `Could not find package configuration file provided by "realsense2_camera"` | пакет камеры не установлен: `sudo apt install ros-foxy-realsense2-camera` |
| `colcon build` падает с ошибкой по `numpy` | использовать NumPy < 2.0 (см. requirements.txt) |
| Кириллица в путях вызывает ошибки в Ultralytics | использовать только английские пути в `weights/`, `data/`, `runs/` |
| `ImportError: cannot import name 'CvBridge'` | установлен Python-пакет cv_bridge для другого ROS-дистрибутива — переустановить через apt |

## 11. Что дальше

После успешной установки переходите к **`docs/usage.md`** для команд запуска (Листинги Е.1–Е.5 ВКР) и к **`experiments/README.md`** для воспроизведения экспериментальных результатов главы 4 ВКР.
