"""
@File    :   setup.py
@Time    :   2025/11/16 09:37:10
@Author  :   Lin
@Version :   1.0
@Desc    :   None
copyright USTC
"""

import pybind11
from setuptools import Extension, find_packages, setup

math_parser_module = Extension(
    name="parser.EMath",
    sources=["cpp/parser.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=["-std=c++20"],
)

setup(
    name="parser",
    ext_modules=[math_parser_module],
    packages=find_packages(),
    package_data={"parser": ["*.so", "*.pyi"]},
)
