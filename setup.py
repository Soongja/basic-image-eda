import io
import os
from setuptools import setup, find_packages

from basic_image_eda.__version__ import __version__

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

REQUIRED = [
    # 'opencv-python',
    # 'numpy',
    # 'matplotlib',
    # 'tqdm',
    # 'skimage.io',
]

setup(
    name='basic-image-eda',
    description='image dataset eda tool to check basic information of images.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Soongja/basic-image-eda',
    version=__version__,
    author='Seungjae Kim',
    author_email='sjn735@gmail.com',
    licence='MIT',
    packages=find_packages(exclude=["assets", "test_data", "tests"]),
    install_requires=REQUIRED,
    entry_points={
        'console_scripts': [
            'basic-image-eda = basic_image_eda.eda:main'
        ]
    }
)
