import re
from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
init_py = (ROOT / "torchtangents" / "__init__.py").read_text(encoding="utf-8")
version_match = re.search(r'^__version__ = ["\']([^"\']+)["\']', init_py, re.M)
if not version_match:
    raise RuntimeError("Unable to find __version__ in torchtangents/__init__.py")
version = version_match.group(1)

long_description = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="torchtangents",
    version=version,
    description="PyTorch-native Neural Tangent Kernels (NTK) and Infinite-Width Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Joshua DeOliveira",
    author_email="jcdeoliveira@wpi.edu",
    url="https://github.com/deoliveirajoshua/torchtangents",
    project_urls={
        "Source Code": "https://github.com/deoliveirajoshua/torchtangents",
        "Bug Tracker": "https://github.com/deoliveirajoshua/torchtangents/issues",
    },
    license="Apache-2.0",
    packages=find_packages(include=["torchtangents", "torchtangents.*"]),
    package_data={"torchtangents": ["py.typed"]},
    include_package_data=True,
    install_requires=[
        "torch>=2.2",
        "numpy>=1.20",
        "typing_extensions>=4.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
        "parity": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "numpy<2",
            "optree",
            "jax[cuda12]==0.4.27",
            "neural-tangents==0.6.5",
        ],
        "benchmark": [
            "tqdm>=4.60.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "tqdm>=4.60.0",
        ],
        "all": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "tqdm>=4.60.0",
            "numpy<2",
            "optree",
            "jax[cuda12]==0.4.27",
            "neural-tangents==0.6.5",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords="pytorch, neural-tangent-kernel, ntk, nngp, infinite-width, gaussian-processes, deep-learning",
    python_requires=">=3.9",
)
