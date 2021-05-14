import pip
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

try:
    # pip >=20
    from pip._internal.network.session import PipSession
    from pip._internal.req import parse_requirements
except ImportError:
    try:
        # 10.0.0 <= pip <= 19.3.1
        from pip._internal.download import PipSession
        from pip._internal.req import parse_requirements
    except ImportError:
        # pip <= 9.0.3
        from pip.download import PipSession
        from pip.req import parse_requirements


try:
    requirements = parse_requirements('requirements.txt')
except:
    # new versions of pip requires a session
    requirements = parse_requirements(
        'requirements.txt', session=PipSession())

setup(
    name="SGNBuilder",
    version="0.1.1",
    url='https://github.com/abhilash-j2/SGNBuilder-Package',
    install_requires=[str(requirement.requirement) for requirement in requirements],
    packages=find_packages(),
    include_package_data=True,
    author="Abhilash Janardhanan",
    author_email="jabhilash7@gmail.com"
)