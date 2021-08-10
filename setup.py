#!/usr/bin/env python3

"""Setup polyphone dis"""

from setuptools import find_packages, setup

def readme():
    """Retrieves the readme content."""
    with open("README.md", "r") as f:
        content = f.read()
    return content

setup(
    name="polyphonesdis",
    version="0.1.0",
    description="A codebase for music information retrieval",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0 License",
    ],
    install_requires=[
        "numpy", "simplejson", "yacs"
    ],)