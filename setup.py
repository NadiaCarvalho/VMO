from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'SyVMO - '
LONG_DESCRIPTION = 'A package that calculates a (Synchronous) Variable Markov Oracle from features'

setup(
    name="syvmo",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Nádia Carvalho",
    author_email="nadiacarvalho118@gmail.com",
    license='MIT',
    packages=find_packages(),
    install_requires=[],
    keywords='vmo, syvmo',
    classifiers= [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
    ]
)
