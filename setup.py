import re

from setuptools import find_packages, setup


def read_requirements(filename: str):
    with open(filename) as fn:
        reqs = fn.read().splitlines()
        return [
            r
            for r in reqs
            if not re.match(r".*\.txt.*requirements.*", r, re.IGNORECASE)
        ]


setup(
    name="polypy",
    version="0.0.0",
    install_requirements=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "examples": read_requirements("requirements-examples.txt"),
    },
    packages=find_packages(exclude=["docs", "profiling", "tests", "examples"]),
)
