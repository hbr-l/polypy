from setuptools import find_packages, setup

setup(
    name="polypy",
    packages=find_packages(
        exclude=["scripts", "doc", "profiling", "lab", "tests", "examples"]
    ),
)
