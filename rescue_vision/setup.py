"""setup.py для пакета rescue_vision (ament_python).

Соответствует Листингу А.9 ВКР Миргазева М.А., 2026.
"""
from glob import glob
from setuptools import setup, find_packages

package_name = 'rescue_vision'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test', 'tests']),
    data_files=[
        # ament index resource marker
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        # манифест пакета
        ('share/' + package_name, ['package.xml']),
        # launch-файлы
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        # конфигурационные YAML-файлы
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Marat Mirgazev',
    maintainer_email='mirgazev.ma@edu.mirea.ru',
    description=(
        'Rescue robot vision pipeline: adaptive scene-aware image '
        'enhancement + YOLO11-pose + ByteTrack as ROS 2 Foxy node.'
    ),
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Основной узел конвейера (Листинг А.4)
            'rescue_vision_node = rescue_vision.rescue_vision_node:main',
            # HUD оператора (Листинг А.5)
            'operator_visualization_node = '
            'rescue_vision.operator_visualization_node:main',
            # Standalone-запуск без ROS 2 (через MP4Detector)
            'rescue_vision_standalone = rescue_vision.main:main',
        ],
    },
)
