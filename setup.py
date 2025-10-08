#!/usr/bin/env python3
# Copyright 2025 torchtangents contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup script for torchtangents."""

import os
from setuptools import setup, find_packages


def get_version():
    """Get version from package __init__.py."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, 'torchtangents', '__init__.py')) as fp:
        for line in fp:
            if line.startswith('__version__') and '=' in line:
                version = line[line.find('=') + 1:].strip(' \'"\n')
                if version:
                    return version
    raise ValueError('`__version__` not found in `torchtangents/__init__.py`')


def get_long_description():
    """Get long description from README."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(current_dir, 'README.md'), encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "PyTorch implementation of Neural Tangents for computing empirical and infinite-width NTKs."


INSTALL_REQUIRES = [
    'torch>=2.0.0',
    'numpy>=1.20.0',
]

TESTS_REQUIRES = [
    'pytest>=6.0',
    'pytest-cov',
    'numpy>=1.20.0',  # Required for tests
]

EXAMPLES_REQUIRES = [
    'matplotlib>=3.5.0',
    'numpy>=1.20.0',
]

BENCHMARKS_REQUIRES = [
    'numpy>=1.20.0',
    'matplotlib>=3.5.0',
    'seaborn>=0.11.0',
    'pandas>=1.3.0',
    'psutil>=5.8.0',
]

COMPARISON_REQUIRES = [
    'jax>=0.3.0',
    'neural-tangents>=0.6.0',
]

EXTRAS_REQUIRE = {
    'testing': TESTS_REQUIRES,
    'examples': EXAMPLES_REQUIRES,
    'benchmarks': BENCHMARKS_REQUIRES,
    'comparison': COMPARISON_REQUIRES,
    'all': TESTS_REQUIRES + EXAMPLES_REQUIRES + BENCHMARKS_REQUIRES + COMPARISON_REQUIRES,
    'dev': TESTS_REQUIRES + EXAMPLES_REQUIRES + [
        'black',
        'isort',
        'flake8',
    ],
}

setup(
    name='torchtangents',
    version=get_version(),
    license='Apache 2.0',
    author='TorchTangents Contributors',
    author_email='torchtangents@example.com',
    description='PyTorch implementation of Neural Tangents for computing empirical and infinite-width NTKs',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/torchtangents',
    packages=find_packages(exclude=('tests', 'examples', 'benchmarks', 'docs', 'neural-tangents-main')),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='neural networks, neural tangent kernel, pytorch, machine learning, deep learning, infinite width',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/torchtangents/issues',
        'Source': 'https://github.com/yourusername/torchtangents',
        'Documentation': 'https://github.com/yourusername/torchtangents#readme',
    },
)
