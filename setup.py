import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "oxnnet",
    version = "0.1.0",
    author = "Padraig Looney",
    author_email = "padraig.looney@gmail.com",
    description = (" "
                                   "to the cheese shop a5 pypi.org."),
    license = "LGPL",
    keywords = "example documentation tutorial",
    url = "http://packages.python.org/an_example_pypi_project",
    packages=['oxnnet', 'oxnnet.model', 'tests'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: LGPL version 3",
    ],
)
