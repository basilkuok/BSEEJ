from setuptools import setup, find_packages

setup(
    name="bseej",
    version="0.1",
    packages=find_packages(include=["BSEEJ", "BSEEJ.*"]),
    py_modules=["bseej", "utilities"],
    entry_points={"console_scripts": ["bseej=bseej:main"]},
)
