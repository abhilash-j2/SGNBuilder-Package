import pip
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

links = []
requires = []

try:
    requirements = pip.req.parse_requirements('requirements.txt')
except:
    # new versions of pip requires a session
    requirements = pip.req.parse_requirements(
        'requirements.txt', session=pip.download.PipSession())

for item in requirements:
    # we want to handle package names and also repo urls
    if getattr(item, 'url', None):  # older pip has url
        links.append(str(item.url))
    if getattr(item, 'link', None): # newer pip has link
        links.append(str(item.link))
    if item.req:
        requires.append(str(item.req))


setup(
    name="SGNBuilder",
    version="0.1.1",
    url='https://github.com/abhilash-j2/SGNBuilder-Package',
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True,
    author="Abhilash Janardhanan",
    author_email="jabhilash7@gmail.com"
)