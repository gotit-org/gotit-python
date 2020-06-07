from distutils.dir_util import remove_tree
from setuptools import setup, Extension
import pybind11
import os


ext_modules = [
    Extension(
        name='ColorPy',
        sources=['Color.cpp'],
        include_dirs=[
            pybind11.get_include(),
        ],
        language='c++'
    )
]

setup(ext_modules=ext_modules)

# delete build & temp directory
file_paths = ['temp', 'build']
for file_path in file_paths:
    if os.path.isdir(file_path):
        remove_tree(file_path)