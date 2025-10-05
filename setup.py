from setuptools import setup, find_packages

setup(
    name="torchtangents",
    version="0.0.0",
    description="PyTorch port of Neural Tangents",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1",
    ],
    packages=find_packages(
        where=".",
        include=["torchtangents*"],
    ),
)
