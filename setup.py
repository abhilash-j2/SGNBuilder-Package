from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="SGNBuilder",
    version="0.1.0",
    install_requires=requirements,
    packages=find_packages(),
    author="Abhilash Janardhanan",
    author_email="jabhilash7@gmail.com"
)