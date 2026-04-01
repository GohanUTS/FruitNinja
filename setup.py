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
    (os.path.join('share', package_name, 'urdf'),    glob('urdf/*')),
    (os.path.join('share', package_name, 'meshes'), glob('trolley/*.dae')),
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
            'reset = fruitninja.movement:reset_main',
            'gui = fruitninja.gui:main',
            'gui_sim = fruitninja.gui_sim:main',
            'gui_ur3e = fruitninja.gui_ur3e:main',
            'grid_mover = fruitninja.grid_mover:main',
        ],
    },
)