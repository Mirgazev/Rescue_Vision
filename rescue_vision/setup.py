from setuptools import setup, find_packages
from glob import glob

package_name = 'rescue_vision'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Marat Mirgazev',
    maintainer_email='92t89nh84w@privaterelay.appleid.com',
    description='Adaptive rescue robot vision pipeline.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'rescue_vision_node = rescue_vision.rescue_vision_node:main',
            'operator_visualization_node = rescue_vision.operator_visualization_node:main',
            'rescue_vision_standalone = rescue_vision.main:main',
        ],
    },
)
