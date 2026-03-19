from setuptools import setup
import os
from glob import glob

package_name = 'fruitninja'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),  # ← this line is missing
],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gohan',
    description='FruitNinja - UR3e fruit cutting robot',
    license='MIT',
    entry_points={
        'console_scripts': [
            'planning_scene = fruitninja.planning_scene:main',
            'movement = fruitninja.movement:main',
        ],
    },
)